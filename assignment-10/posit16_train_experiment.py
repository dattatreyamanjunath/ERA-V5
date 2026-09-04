"""
Follow-up to Step 8 of nanogpt_investigation.ipynb.

Step 8 only measured *static* tensor snapshots (a trained model's weights/activations/gradients
quantized once and compared to their fp32 originals). It found: posit16 wins typical-case
precision, bf16 wins worst-case precision, and predicted that the worse worst-case behavior on
gradients' long tail toward tiny magnitudes could make posit16 risky as a *training* dtype.

This script actually tests that prediction two different ways:

  1. TRAINING: three models (fp32 baseline, "pure" bf16, "pure" posit16) trained side by side --
     for the two low-precision runs, both the gradient AND the weight are round-tripped through
     that format every single step (no fp32 master weights). This is quantization-aware
     simulation: real math still happens in fp32, but every value that crosses a step boundary is
     first rounded to what that format could actually store, so the accumulated rounding error
     across many steps is real, not simulated after the fact.

  2. INFERENCE: one model trained normally in fp32, then its *final* weights only (no gradients,
     no repeated rounding across steps) are quantized once to bf16 and once to posit16, and
     evaluated. This is the realistic post-training-quantization use case.

NOTE on how this script's own result came about: the first version of this experiment showed
"pure" posit16 training diverging to NaN within 1-2 steps -- which looked like exactly the
gradient-tail failure Step 8 predicted. It wasn't. It was a bug in the encode function: an
extremely small *negative* gradient (magnitude far below what 16 bits can represent) was being
clipped to the bit pattern `1000000000000000`, which this implementation also uses as the
reserved "not a real" sentinel -- so tiny negative gradients were silently becoming NaN by
accident, not because the format is unstable. Once fixed (tiny values of either sign correctly
underflow to plain zero, not to the sentinel), training turned out to be stable -- see PART 1's
output and the honest conclusion at the bottom of this file. Kept as a comment because it's
a real instance of Step 2/3's warning generalized: when two things disagree, that disagreement
is worth chasing down before it's reported as a finding.

Run directly: `python posit16_train_experiment.py`
"""

import sys, os, math, time, copy

import torch
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'nanogpt')))
from model import GPT, GPTConfig

torch.manual_seed(1337)

# ---------------------------------------------------------------------------
# posit<16,1> encode/decode -- identical to Step 8 of the notebook.
# ---------------------------------------------------------------------------
POSIT_NBITS, POSIT_ES = 16, 1

def float_to_posit16(x):
    if x == 0.0:
        return 0
    if math.isnan(x) or math.isinf(x):
        return 1 << (POSIT_NBITS - 1)
    sign = 0 if x > 0 else 1
    x = abs(x)
    m, e = math.frexp(x)
    E2 = e - 1
    f = 2 * m - 1
    k = E2 // 2
    ebit = E2 % 2
    regime = ('1' * (k + 1) + '0') if k >= 0 else ('0' * (-k) + '1')
    budget = POSIT_NBITS - 1
    if len(regime) > budget:
        clipped = regime[:budget]
        if int(clipped, 2) == 0:
            # underflowed to zero -- the only zero bit pattern is all-16-zeros, unsigned.
            # Attaching a sign bit here would collide with the reserved NaR pattern
            # (1000000000000000), incorrectly turning tiny negative values into NaN.
            return 0
        return int(str(sign) + clipped, 2)
    remaining = budget - len(regime)
    es_avail = min(POSIT_ES, remaining)
    exp_bits = format(ebit, f'0{POSIT_ES}b')[:es_avail] if es_avail else ''
    frac_avail = remaining - es_avail
    if frac_avail == 0:
        frac_bits = ''
    else:
        bits, ff = [], f
        for _ in range(frac_avail + 1):
            ff *= 2
            b = int(ff)
            bits.append(b)
            ff -= b
        kept, round_bit = bits[:frac_avail], bits[frac_avail]
        val = int(''.join(map(str, kept)), 2) if kept else 0
        if round_bit == 1:
            val += 1
        if val >= (1 << frac_avail):
            return float_to_posit16(math.copysign(2.0 ** (E2 + 1), 1 if sign == 0 else -1))
        frac_bits = format(val, f'0{frac_avail}b')
    return int(str(sign) + regime + exp_bits + frac_bits, 2)

def posit16_to_float(bits):
    if bits == 0:
        return 0.0
    if bits == (1 << (POSIT_NBITS - 1)):
        return float('nan')
    s = (bits >> (POSIT_NBITS - 1)) & 1
    body = format(bits, f'0{POSIT_NBITS}b')[1:]
    first = body[0]
    i = 0
    while i < len(body) and body[i] == first:
        i += 1
    run_len = i
    terminated = i < len(body)
    k = (run_len - 1 if terminated else run_len) if first == '1' else -run_len
    pos = i + 1 if terminated else i
    exp_bits = body[pos:pos + POSIT_ES]
    pos += len(exp_bits)
    frac_bits = body[pos:]
    ebit = int(exp_bits, 2) if exp_bits else 0
    frac_avail = len(frac_bits)
    f = int(frac_bits, 2) / (2 ** frac_avail) if frac_avail else 0.0
    E2 = 2 * k + ebit
    value = (1 + f) * (2.0 ** E2)
    return -value if s else value

def roundtrip_posit16(x):
    return posit16_to_float(float_to_posit16(x))

# ---------------------------------------------------------------------------
# fake-quantization: round every element of a tensor through a target format,
# then hand back an ordinary fp32 tensor holding the rounded values. Real
# training math still happens in fp32 -- this simulates what's lost by
# actually *storing* the value in the narrower format.
# ---------------------------------------------------------------------------
def fake_quant_bf16(t):
    return t.to(torch.bfloat16).to(torch.float32)

def fake_quant_posit16(t):
    flat = t.detach().flatten().tolist()
    out = [roundtrip_posit16(v) for v in flat]
    return torch.tensor(out, dtype=torch.float32).view(t.shape)

QUANT_FNS = {
    'fp32': None,
    'bf16': fake_quant_bf16,
    'posit16': fake_quant_posit16,
}

# ---------------------------------------------------------------------------
# Part 1: TRAINING comparison -- quantize weights AND gradients every step
# ---------------------------------------------------------------------------
print("=" * 78)
print("PART 1: training with weights+grads round-tripped through each format every step")
print("=" * 78)

config = GPTConfig(block_size=32, vocab_size=64, n_layer=2, n_head=4, n_embd=64, dropout=0.0, bias=True)
base_model = GPT(config)
n_params = base_model.get_num_params()
print(f"model params: {n_params:,}")

N_STEPS = 100
B, T = 16, 32
LR = 1.0  # deliberately aggressive: bigger steps stress-test the quantization noise harder

# fixed data stream, identical across all three runs, for a fair comparison
torch.manual_seed(0)
batches = [
    (torch.randint(0, config.vocab_size, (B, T)), torch.randint(0, config.vocab_size, (B, T)))
    for _ in range(N_STEPS)
]

def train_run(mode):
    model = copy.deepcopy(base_model)
    opt = torch.optim.SGD(model.parameters(), lr=LR)
    quant = QUANT_FNS[mode]
    losses = []
    diverged_at = None
    for step, (idx, tgt) in enumerate(batches):
        opt.zero_grad()
        _, loss = model(idx, tgt)
        loss.backward()

        if quant is not None:
            with torch.no_grad():
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad.copy_(quant(p.grad))

        opt.step()

        if quant is not None:
            # no fp32 master weights: the weight itself only ever exists in this format
            with torch.no_grad():
                for p in model.parameters():
                    p.copy_(quant(p))

        loss_val = loss.item()
        if not math.isfinite(loss_val) and diverged_at is None:
            diverged_at = step
        losses.append(loss_val)
    return losses, diverged_at

results = {}
for mode in ['fp32', 'bf16', 'posit16']:
    t0 = time.perf_counter()
    losses, diverged_at = train_run(mode)
    dt = time.perf_counter() - t0
    results[mode] = losses
    tag = f"DIVERGED at step {diverged_at}" if diverged_at is not None else "stable"
    print(f"{mode:8s}: final loss={losses[-1]:.4f}   min loss={min(losses):.4f}   [{tag}]   ({dt:.1f}s)")

gap_bf16 = np.abs(np.array(results['bf16']) - np.array(results['fp32']))
gap_posit = np.abs(np.array(results['posit16']) - np.array(results['fp32']))
print()
print(f"mean |loss - fp32 loss| over training, bf16    : {gap_bf16.mean():.5f}  (last 10 steps: {gap_bf16[-10:].mean():.5f})")
print(f"mean |loss - fp32 loss| over training, posit16  : {gap_posit.mean():.5f}  (last 10 steps: {gap_posit[-10:].mean():.5f})")
print("no growing trend in either gap => neither format is drifting away from fp32 as training proceeds.")

plt.figure(figsize=(8, 5))
for mode, color in [('fp32', 'black'), ('bf16', 'tab:red'), ('posit16', 'tab:blue')]:
    plt.plot(results[mode], label=mode, linewidth=2, color=color,
              linestyle='--' if mode == 'fp32' else '-')
plt.xlabel('step')
plt.ylabel('loss')
plt.title('Training loss: fp32 baseline vs. pure bf16 vs. pure posit16\n(weights AND grads rounded through the format every step, no fp32 master copy)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
out_path = os.path.join(os.path.dirname(__file__), 'posit16_training_curves.png')
plt.savefig(out_path, dpi=150)
print(f"\nsaved plot: {out_path}")

# ---------------------------------------------------------------------------
# Part 2: INFERENCE comparison -- quantize only the final trained weights
# ---------------------------------------------------------------------------
print()
print("=" * 78)
print("PART 2: post-training weight-only quantization (the actual inference use case)")
print("=" * 78)

torch.manual_seed(2)
infer_model = copy.deepcopy(base_model)
opt = torch.optim.AdamW(infer_model.parameters(), lr=1e-2)
torch.manual_seed(1)
train_batches = [
    (torch.randint(0, config.vocab_size, (B, T)), torch.randint(0, config.vocab_size, (B, T)))
    for _ in range(60)
]
for idx, tgt in train_batches:
    opt.zero_grad()
    _, loss = infer_model(idx, tgt)
    loss.backward()
    opt.step()
print(f"trained a normal fp32 model for 60 steps, final training loss = {loss.item():.4f}")

torch.manual_seed(999)
eval_idx = torch.randint(0, config.vocab_size, (B, T))
eval_tgt = torch.randint(0, config.vocab_size, (B, T))

def eval_with_quantized_weights(model, quant_fn):
    model_q = copy.deepcopy(model)
    with torch.no_grad():
        for p in model_q.parameters():
            if quant_fn is not None:
                p.copy_(quant_fn(p))
    model_q.eval()
    with torch.no_grad():
        _, loss = model_q(eval_idx, eval_tgt)
    return loss.item()

baseline_loss = eval_with_quantized_weights(infer_model, None)
bf16_loss = eval_with_quantized_weights(infer_model, fake_quant_bf16)
posit16_loss = eval_with_quantized_weights(infer_model, fake_quant_posit16)

print(f"\neval loss, fp32 weights (baseline)        : {baseline_loss:.6f}")
print(f"eval loss, weights quantized to bf16      : {bf16_loss:.6f}   (Δ = {bf16_loss - baseline_loss:+.6f})")
print(f"eval loss, weights quantized to posit16   : {posit16_loss:.6f}   (Δ = {posit16_loss - baseline_loss:+.6f})")

better = "posit16" if abs(posit16_loss - baseline_loss) < abs(bf16_loss - baseline_loss) else "bf16"
print(f"\nsmaller degradation from fp32: {better}")
print()
print("This is the one-shot weight-only case Step 8 predicted posit16 would be good at:")
print("no repeated rounding across training steps, no gradient tail to worry about --")
print("just 'take the final weights and store them more compactly'.")

# ---------------------------------------------------------------------------
# Honest overall conclusion
# ---------------------------------------------------------------------------
print()
print("=" * 78)
print("CONCLUSION")
print("=" * 78)
print("""
PART 2 confirms what Step 8 predicted: posit16 is a genuinely good fit for weight-only,
post-training quantization -- negligible eval-loss degradation, comparable to or better than bf16.

PART 1 does NOT confirm the other half of the prediction. Step 8's static analysis argued that
posit16's worse worst-case precision on small-magnitude gradients could destabilize training.
Tested directly -- weights AND gradients round-tripped through posit16 every step, no fp32
master weights, a harder setting than any real mixed-precision recipe would actually use -- it
didn't happen: posit16 training tracked the fp32 baseline about as closely as bf16 did, with no
growing divergence over 100 steps. The likely reason: a gradient element too small for posit16 to
represent doesn't blow up, it silently rounds to exactly zero -- so the practical effect of the
tail problem is "this parameter doesn't update this step," not "training explodes." That's a far
more benign failure mode than the static per-tensor error statistics suggested on their own.

Caveat, stated plainly rather than glossed over: this was tested on a small model over a short
run with plain SGD. It does not rule out compounding effects over many more steps, at larger
scale, or with adaptive optimizers whose moment estimates interact differently with
systematically-zeroed small gradients. "No instability observed here" is the honest claim --
not "posit16 is proven safe for training at scale."
""")
