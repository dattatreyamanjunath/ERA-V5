# Assignment 10 — NanoGPT Internals, Verified Instead of Trusted

Eight questions about [karpathy/nanoGPT](https://github.com/karpathy/nanogpt)'s actual training step — every forward/backward pass, gradient, and quantization claim below is a real run against `model.py`, not a description of one.

Notebook: [`nanogpt_investigation.ipynb`](nanogpt_investigation.ipynb). Runs top to bottom, no errors, plots embedded inline. Separate script: [`posit16_train_experiment.py`](posit16_train_experiment.py). All numbers in this README are copied from actual local execution logs (Apple Silicon / MPS backend).

## What's in the notebook

- **Named shapes** through one full forward pass — embeddings, per-head attention reshape, MLP widen/narrow, logits — one line of meaning per tensor.
- **A hand-verified gradient**: float64 finite differences vs. `backward()` on a weight used by every token (`c_attn`), not an embedding row that only fires for tokens present in the batch.
- **A weight nudge vs. the gradient's prediction**, swept across four step sizes — including the step size small enough that the *measurement* itself hits float64's precision floor, reported as exactly that rather than as a mismatch.
- **Gradient accumulation broken on purpose**: two models trained in parallel on identical data, one correctly weighting each micro-batch by its size, one using the "average of the averages" bug, plotted together.
- **Grad norm logged every step for 80 steps**, with the leading-indicator step (grad norm moves, loss barely reacts, then catches up next step) picked out programmatically.
- **MFU computed honestly**: nanoGPT's own `estimate_mfu()`, fed a *real* measured wall-clock time on this machine — not an assumed one.
- **0.1 written out by hand** in fp32, bf16, and fp8 E4M3 — binary expansion, normalization, rounding, and decode, as arithmetic text, not a `torch` dtype cast.
- **A custom tapered-precision 16-bit format (posit⟨16,1⟩)**, implemented from scratch and measured against bf16 on this model's real weights, activations, and gradients.

## Results

| Check | Result |
|---|---|
| Hand-verified gradient (`c_attn.weight[3,5]`), backprop vs. finite-difference | 5.0471041e-05 vs. 5.0471183e-05 (relative diff 2.8e-06) |
| Weight-nudge agreement at lr=1e-3 | actual ΔL=−2.5464e-12, predicted ΔL=−2.5473e-12 (rel. error 3.6e-04) |
| Weight-nudge at lr=1e-7 | actual ΔL rounds to exactly 0 — float64 precision floor, not a disagreement |
| Gradient accumulation: correct vs. "average of averages" (unequal micro-batches) | final loss 4.8010 vs. 4.8412 (max gap over training: 0.1357) |
| Grad-norm-leads-loss example | step 0→1: grad norm 1.217→0.660 (Δ=−0.557), loss barely moves (+0.016), then drops −0.025 the next step |
| Measured MFU (0.83M-param model, batch=8, seq_len=128, on Apple Silicon/MPS) | 0.126% of A100 bf16 peak |
| 0.1 in fp32 / bf16 / fp8 E4M3, relative error | 0.0000% / 0.0977% / 1.5625% |
| posit16 vs. bf16, median (typical-case) relative error, real model tensors | posit16 wins on all 4 populations tested (weights, activations, 2 gradient tensors) |
| posit16 vs. bf16, max (worst-case) relative error, same tensors | bf16 wins on all 4 — up to ~10× tighter worst case |
| posit16 "pure" training (weights+grads round-tripped every step, no fp32 master) vs. fp32/bf16, 100 steps | stable, final loss 4.1592 vs. fp32's 4.1592 and bf16's 4.1593; mean gap from fp32 *smaller* than bf16's (2e-5 vs. 1.1e-4) |
| posit16 weight-only post-training quantization, eval loss vs. fp32 baseline | Δ = −0.000001 (bf16's Δ = +0.000010) |

A few of these are worth a sentence of context.

**The finite-difference check** was picked to succeed for a specific reason: an earlier attempt on a token-embedding row gave a numerical gradient of exactly 0 against a real backprop gradient of −2.4e-3, a 100% "mismatch" — not because backprop was wrong, but because that weight's effect on the loss was too small to survive fp32 rounding at any usable epsilon. Switching to a weight every token actually uses, and computing in float64, is what makes the agreement above meaningful instead of accidental.

**The weight-nudge sweep** was left with all four step sizes on purpose, including the one that "fails." At `lr=1e-7` the actual loss change underflows to exactly 0 in float64, which would read as total disagreement if reported as a bare relative error. The honest read — spelled out explicitly in the notebook — is that the measurement floor was reached, not that the gradient stopped being correct. This is the same failure mode as the embedding-row check above, deliberately kept visible rather than edited out.

**The gradient-accumulation bug** was made to actually bite by using lopsided micro-batch partitions (`[1, 1, 22]`, `[2, 20, 2]`, ...) rather than roughly-equal ones, since equal-sized micro-batches make the "average of averages" bug numerically invisible. With 1-sequence micro-batches getting the same vote as 22-sequence ones, the two training curves visibly separate.

**MFU came out at 0.13%, nowhere near 40%.** The honest reason isn't a flaw in the model code — this ran on Apple Silicon (MPS), not an A100, so the 312 TFLOP/s bf16-peak denominator in `estimate_mfu()` is the wrong yardstick entirely. On top of that mismatch, the model/batch here (0.83M params, batch=8) is too small to saturate any accelerator's matmul units, everything ran in fp32 rather than bf16, and there's no Flash Attention fused kernel or `torch.compile()` in play. On real A100/H100 hardware with those four things fixed, 40%+ is realistic for this exact code — none of the preconditions held in this run.

**posit16 (a custom tapered-precision 16-bit format)** was built to test whether "accuracy fades with significance" beats bf16's fixed relative precision. It does, consistently, on the *typical* value in every real tensor tested — nanoGPT's weights and activations mostly sit within a few octaves of magnitude 1, where posit16's variable-length "regime" field is still short and leaves more fraction bits than bf16's fixed 7. It loses just as consistently on the *worst case*, because gradients have a long tail toward tiny magnitudes where the regime eats the fraction budget entirely — bf16's fixed mantissa width caps its worst case everywhere; posit16's doesn't.

**Testing that prediction with actual training code caught a real bug before it became a false finding.** The first version of `posit16_train_experiment.py` showed "pure" posit16 training diverging to NaN within one step — which looked like exact confirmation of the tail-precision concern above. It wasn't: an extremely small *negative* gradient was being encoded to the bit pattern reserved for posit's "not a real" sentinel, so tiny negative gradients were silently becoming NaN by accident, not because the format is unstable. Fixed (verified against a 200,000-sample wide-dynamic-range sweep with zero spurious NaNs afterward) and rerun, training was stable — posit16 tracked the fp32 baseline about as closely as bf16 did, with no growing divergence over 100 steps. The likely reason: a gradient too small for posit16 to represent rounds to exactly zero, not to something huge — a far more benign failure mode than the static per-tensor error statistics alone suggested. Stated plainly in that script's own conclusion: this is "no instability observed at this scale," not "proven safe for training at scale" — small model, short run, plain SGD. The weight-only, post-training quantization case (no gradients, no repeated rounding) held up exactly as predicted: negligible degradation, on par with or slightly better than bf16 — the genuinely good, lower-risk use for this format.

## Running it

```bash
pip install torch numpy matplotlib jupyter nbformat
jupyter nbconvert --to notebook --execute --inplace nanogpt_investigation.ipynb
python posit16_train_experiment.py
```

Both run on CPU or Apple Silicon (MPS) fine. The notebook takes well under a minute. `posit16_train_experiment.py` takes ~1–2 minutes — posit16 encode/decode is pure-Python bit manipulation (no vectorized/library implementation exists to call), and Part 1 quantizes every weight and gradient element through it on every one of 100 training steps.
