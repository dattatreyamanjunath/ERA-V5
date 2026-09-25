"""
revgpt.py — a ~20M-parameter GPT with optional *reversible* residual streams.

Every variant uses exactly the same parameters (10 layers of attention + MLP,
d_model=256, tied GPT-2 embeddings). What changes is how the sub-layers are
composed into the residual stream, and therefore whether activations must be
stored for the backward pass.

  baseline     x <- x + F(x)                          (standard pre-LN GPT, stores everything)
  euler_fp     x <- x + F(x), inverted by fixed-point  (same forward as baseline; the inverse
               iteration x = y - F(x)                  is only approximate)
  coupling     y1 = x1 + Attn(x2); y2 = x2 + MLP(y1)   (RevNet / Reformer additive coupling =
                                                        symplectic Euler on a 2-stream system)
  midpoint     x_{k+1} = x_{k-1} + F_k(x_k)            (leapfrog / explicit midpoint, Chang et al. 2018)
  momentum     v <- g*v + (1-g)*F(x); x <- x + v        (Momentum ResNet, Sander et al. 2021)

For every reversible variant the forward pass runs under no_grad and keeps only
the *final* stream state. The backward pass walks the layers in reverse, rebuilds
each layer's input from its output, re-runs that single layer with autograd on and
back-propagates through it. Activation memory is therefore O(1) in depth.
"""
import math, time, json, gc, os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------- model parts

class Attn(nn.Module):
    """Pre-LN causal self-attention sub-layer: returns the residual *delta*."""
    def __init__(self, d, h, n_layer):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.h = h
        nn.init.normal_(self.proj.weight, std=0.02 / math.sqrt(2 * n_layer))

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(self.ln(x)).split(C, dim=2)
        q, k, v = (t.view(B, T, self.h, C // self.h).transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(y.transpose(1, 2).reshape(B, T, C))


class MLP(nn.Module):
    """Pre-LN GELU MLP sub-layer: returns the residual *delta*."""
    def __init__(self, d, n_layer):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.fc = nn.Linear(d, 4 * d, bias=False)
        self.proj = nn.Linear(4 * d, d, bias=False)
        nn.init.normal_(self.proj.weight, std=0.02 / math.sqrt(2 * n_layer))

    def forward(self, x):
        return self.proj(F.gelu(self.fc(self.ln(x)), approximate="tanh"))


def _vjp(fn, x, g):
    """Recompute fn(x) with autograd on; back-prop g. Returns (fn(x) detached, dL/dx).
    Parameter grads of fn are accumulated into their .grad as a side effect."""
    with torch.enable_grad():
        x = x.detach().requires_grad_(True)
        y = fn(x)
        torch.autograd.backward(y, g.to(y.dtype))
    return y.detach(), x.grad


# ------------------------------------------------------ reversible step types
# A step maps a tuple of stream tensors to a new tuple.
#   forward(state)            -> state'                  (called under no_grad)
#   reverse(state', grad')    -> (state, grad)           (rebuilds input, does the VJP)

class CouplingStep:
    """y1 = x1 + F(x2);  y2 = x2 + G(y1)."""
    def __init__(self, f, g): self.f, self.g = f, g
    def forward(self, s):
        x1, x2 = s
        y1 = x1 + self.f(x2)
        y2 = x2 + self.g(y1)
        return (y1, y2)
    def reverse(self, s, gs):
        y1, y2 = s; gy1, gy2 = gs
        gout, dy1 = _vjp(self.g, y1, gy2)      # G(y1) and J_G^T gy2
        x2 = y2 - gout
        gy1 = gy1 + dy1
        fout, dx2 = _vjp(self.f, x2, gy1)      # F(x2) and J_F^T gy1
        x1 = y1 - fout
        return (x1, x2), (gy1, gy2 + dx2)


class MidpointStep:
    """Leapfrog: (x_{k-1}, x_k) -> (x_k, x_{k-1} + F(x_k))."""
    def __init__(self, f): self.f = f
    def forward(self, s):
        p, q = s
        return (q, p + self.f(q))
    def reverse(self, s, gs):
        q, r = s; gq, gr = gs
        fout, dq = _vjp(self.f, q, gr)
        p = r - fout
        return (p, q), (gr, gq + dq)


class MomentumStep:
    """v' = g v + (1-g) F(x);  x' = x + v'."""
    def __init__(self, f, gamma): self.f, self.gm = f, gamma
    def forward(self, s):
        x, v = s
        v = self.gm * v + (1 - self.gm) * self.f(x)
        return (x + v, v)
    def reverse(self, s, gs):
        x1, v1 = s; gx1, gv1 = gs
        x = x1 - v1                                  # exact, no F needed
        gv_tot = gv1 + gx1
        fout, dx = _vjp(self.f, x, (1 - self.gm) * gv_tot)
        v = (v1 - (1 - self.gm) * fout) / self.gm    # divides by gamma: error x (1/gamma) per step
        return (x, v), (gx1 + dx, self.gm * gv_tot)


class EulerFPStep:
    """Plain residual x' = x + F(x); inverse by K fixed-point iterations x <- x' - F(x).
    Only exact if F is a contraction and K is large enough."""
    def __init__(self, f, iters): self.f, self.k = f, iters
    def forward(self, s):
        (x,) = s
        return (x + self.f(x),)
    def reverse(self, s, gs):
        (y,), (gy,) = s, gs
        x = y
        with torch.no_grad():
            for _ in range(self.k):
                x = y - self.f(x)
        _, dx = _vjp(self.f, x, gy)
        return (x,), (gy + dx,)


class RevFunction(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(ctx, steps, *state):
        with torch.no_grad():
            for st in steps:
                state = st.forward(state)
        ctx.steps = steps
        ctx.save_for_backward(*state)    # the ONLY activations kept: final stream state
        return tuple(state)

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, *grads):
        state = ctx.saved_tensors
        grads = tuple(torch.zeros_like(s) if g is None else g for s, g in zip(state, grads))
        with torch.no_grad():
            for st in reversed(ctx.steps):
                state, grads = st.reverse(state, grads)
        return (None, *grads)


# ---------------------------------------------------------------------- GPT

class GPT(nn.Module):
    def __init__(self, vocab=50304, block=512, d=256, n_layer=10, n_head=8,
                 variant="baseline", gamma=0.9, fp_iters=3, ce_chunk=1024):
        super().__init__()
        self.variant, self.gamma, self.fp_iters, self.ce_chunk = variant, gamma, fp_iters, ce_chunk
        self.wte = nn.Embedding(vocab, d)
        self.wpe = nn.Embedding(block, d)
        self.attn = nn.ModuleList(Attn(d, n_head, n_layer) for _ in range(n_layer))
        self.mlp = nn.ModuleList(MLP(d, n_layer) for _ in range(n_layer))
        self.ln_f = nn.LayerNorm(d)
        self.lm_head = nn.Linear(d, vocab, bias=False)
        self.lm_head.weight = self.wte.weight          # weight tying
        nn.init.normal_(self.wte.weight, std=0.02)
        nn.init.normal_(self.wpe.weight, std=0.01)
        for m in list(self.attn) + list(self.mlp):
            for n, p in m.named_parameters():
                if p.dim() == 2 and "proj" not in n:
                    nn.init.normal_(p, std=0.02)

    def sublayers(self):
        for a, m in zip(self.attn, self.mlp):
            yield a; yield m

    def steps(self):
        v = self.variant
        if v == "coupling":
            return [CouplingStep(a, m) for a, m in zip(self.attn, self.mlp)]
        if v == "midpoint":
            return [MidpointStep(f) for f in self.sublayers()]
        if v == "momentum":
            return [MomentumStep(f, self.gamma) for f in self.sublayers()]
        if v == "euler_fp":
            return [EulerFPStep(f, self.fp_iters) for f in self.sublayers()]
        raise ValueError(v)

    def init_state(self, x):
        v = self.variant
        if v == "coupling": return (x, x)
        if v == "midpoint": return (x, x)                # x_{-1} = x_0 = embedding
        if v == "momentum": return (x, torch.zeros_like(x))
        return (x,)

    def readout(self, state):
        v = self.variant
        if v == "coupling": return 0.5 * (state[0] + state[1])
        if v == "midpoint": return state[1]
        return state[0]

    def body(self, x, reversible=True):
        if self.variant == "baseline":
            for f in self.sublayers():
                x = x + f(x)
            return x
        state = self.init_state(x)
        if reversible:
            state = RevFunction.apply(self.steps(), *state)
        else:   # same maths, plain autograd (stores activations) — used for gradient checks
            for st in self.steps():
                state = st.forward(state)
        return self.readout(state)

    def forward(self, idx, targets=None, reversible=True):
        B, T = idx.shape
        x = self.wte(idx) + self.wpe(torch.arange(T, device=idx.device))
        h = self.ln_f(self.body(x, reversible)).view(B * T, -1)
        if targets is None:
            return self.lm_head(h[-T:])
        return chunked_ce(h, targets.view(-1), self.lm_head.weight, self.ce_chunk)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


def _ce_chunk(h, w, y):
    return F.cross_entropy((h @ w.t()).float(), y, reduction="sum")

def chunked_ce(h, y, w, chunk):
    """Cross-entropy over the 50k vocab without ever materialising all B*T*V logits:
    each chunk's logits are recomputed in backward (checkpointed). Used by ALL variants
    so that the block activations — not the logits — decide the memory footprint."""
    tot = 0.0
    for i in range(0, h.shape[0], chunk):
        tot = tot + checkpoint(_ce_chunk, h[i:i + chunk], w, y[i:i + chunk], use_reentrant=False)
    return tot / h.shape[0]


# ---------------------------------------------------------------------- data

def load_fineweb(n_train=50_000_000, n_val=2_000_000, cache="/content/data"):
    """GPT-2 tokenised FineWeb shards (the ones used by modded-nanogpt / llm.c)."""
    from huggingface_hub import hf_hub_download
    def shard(name):
        p = hf_hub_download("kjj0/fineweb10B-gpt2", name, repo_type="dataset", local_dir=cache)
        hdr = np.fromfile(p, dtype=np.int32, count=256)
        assert hdr[0] == 20240520, "bad magic"
        return np.memmap(p, dtype=np.uint16, mode="r", offset=1024, shape=(int(hdr[2]),))
    tr = shard("fineweb_train_000001.bin")[: n_train + 1]
    va = shard("fineweb_val_000000.bin")[: n_val + 1]
    return (torch.from_numpy(tr.astype(np.int32)), torch.from_numpy(va.astype(np.int32)))


# ------------------------------------------------------------------ training

def amp_dtype():
    # T4 (sm75) *reports* bf16 support but only emulates it, and SDPA then falls back to the
    # O(T^2) math kernel. Use bf16 only on Ampere+ (sm80+), fp16 + GradScaler otherwise.
    ok = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    return torch.bfloat16 if ok else torch.float16

def get_batch(data, i, B, T, dev=DEVICE):
    n = B * T
    s = (i * n) % (len(data) - n - 1)
    buf = data[s: s + n + 1].to(dev, non_blocking=True).long()
    return buf[:-1].view(B, T), buf[1:].view(B, T)

@torch.no_grad()
def evaluate(model, val, B, T, n_batches=20):
    model.eval()
    dt = amp_dtype(); tot = 0.0
    for i in range(n_batches):
        x, y = get_batch(val, i, B, T)
        with torch.autocast("cuda", dtype=dt):
            tot += model(x, y, reversible=False).item()   # no_grad: nothing stored anyway
    model.train()
    return tot / n_batches

def train(variant, B, T=512, total_tokens=50_000_000, lr=1e-3, warmup_frac=0.03,
          train_data=None, val_data=None, log_every=50, eval_every=None, seed=1337,
          model_kw=None, label=None, max_steps=None):
    torch.manual_seed(seed)
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    model = GPT(block=T, variant=variant, **(model_kw or {})).to(DEVICE)
    dt = amp_dtype()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1,
                            fused=True)
    scaler = torch.amp.GradScaler("cuda", enabled=(dt == torch.float16))
    steps = total_tokens // (B * T)
    if max_steps: steps = min(steps, max_steps)
    warm = max(1, int(warmup_frac * steps))
    lr_at = lambda s: lr * (s + 1) / warm if s < warm else \
        lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))
    hist = dict(step=[], tokens=[], loss=[], tok_s=[], val_step=[], val_loss=[])
    ema = None; t_start = None; tok_timed = 0
    label = label or f"{variant}_B{B}"
    print(f"[{label}] params={model.n_params()/1e6:.2f}M  B={B} T={T}  steps={steps}  "
          f"tokens/step={B*T}  lr={lr}  amp={dt}")
    for s in range(steps):
        for g in opt.param_groups: g["lr"] = lr_at(s)
        x, y = get_batch(train_data, s, B, T)
        torch.cuda.synchronize(); t0 = time.time()
        with torch.autocast("cuda", dtype=dt):
            loss = model(x, y)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize(); dtm = time.time() - t0
        if s >= 5:                              # skip warm-up / allocator steps for speed
            tok_timed += B * T
            t_start = (t_start or 0) + dtm
        l = loss.item()
        if not math.isfinite(l):
            print(f"[{label}] non-finite loss at step {s} — diverged"); break
        ema = l if ema is None else 0.9 * ema + 0.1 * l
        if s % log_every == 0 or s == steps - 1:
            ts = tok_timed / t_start if t_start else float("nan")
            hist["step"].append(s); hist["tokens"].append((s + 1) * B * T)
            hist["loss"].append(l); hist["tok_s"].append(ts)
            print(f"[{label}] step {s:5d}/{steps}  loss {l:.4f}  ema {ema:.4f}  "
                  f"lr {lr_at(s):.2e}  {ts/1e3:7.1f}k tok/s  "
                  f"peak {torch.cuda.max_memory_allocated()/2**30:.2f} GiB")
        if eval_every and s > 0 and s % eval_every == 0:
            hist["val_step"].append(s); hist["val_loss"].append(evaluate(model, val_data, min(B, 32), T))
    val = evaluate(model, val_data, min(B, 32), T)
    hist["val_step"].append(steps - 1); hist["val_loss"].append(val)
    res = dict(label=label, variant=variant, B=B, T=T, steps=steps, lr=lr,
               tokens_seen=(s + 1) * B * T, params_M=round(model.n_params() / 1e6, 2),
               final_train_loss_ema=round(ema, 4), final_val_loss=round(val, 4),
               tok_per_s=round(tok_timed / t_start), wall_s=round(t_start, 1),
               peak_mem_GiB=round(torch.cuda.max_memory_allocated() / 2**30, 3),
               peak_reserved_GiB=round(torch.cuda.max_memory_reserved() / 2**30, 3),
               gpu=torch.cuda.get_device_name(), amp=str(dt), hist=hist)
    print(f"[{label}] DONE  val {val:.4f}  train(ema) {ema:.4f}  "
          f"{res['tok_per_s']/1e3:.1f}k tok/s  peak {res['peak_mem_GiB']:.2f} GiB  wall {res['wall_s']:.0f}s")
    del model, opt; gc.collect(); torch.cuda.empty_cache()
    return res


# --------------------------------------------------------- memory probing

def try_batch(variant, B, T=512, n_steps=2, model_kw=None):
    """Run n_steps full training steps at batch B. Returns peak GiB or None on OOM."""
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    model = opt = x = y = loss = None
    try:
        model = GPT(block=T, variant=variant, **(model_kw or {})).to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)
        for _ in range(n_steps):
            x = torch.randint(0, 50257, (B, T), device=DEVICE); y = torch.roll(x, -1, 1)
            with torch.autocast("cuda", dtype=amp_dtype()):
                loss = model(x, y)
            loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated() / 2**30
    except torch.cuda.OutOfMemoryError:
        return None
    finally:
        del model, opt, x, y, loss
        gc.collect(); torch.cuda.empty_cache()

def max_batch(variant, T=512, start=16, model_kw=None, multiple=8, verbose=True):
    """Doubling search until OOM, then bisection (to a multiple of `multiple`)."""
    lo, hi, B, log = 0, None, start, []
    while hi is None:
        m = try_batch(variant, B, T, model_kw=model_kw); log.append((B, m))
        if verbose: print(f"  {variant:9s} B={B:5d}: " + (f"{m:.2f} GiB" if m else "OOM"))
        if m is None: hi = B
        else: lo, B = B, B * 2
    while hi - lo > multiple:
        B = (lo + hi) // 2 // multiple * multiple
        if B in (lo, hi): break
        m = try_batch(variant, B, T, model_kw=model_kw); log.append((B, m))
        if verbose: print(f"  {variant:9s} B={B:5d}: " + (f"{m:.2f} GiB" if m else "OOM"))
        if m is None: hi = B
        else: lo = B
    return lo, log


# ------------------------------------------------------- correctness checks

def grad_check(variant, B=2, T=64, dtype=torch.float64, dev="cpu", **kw):
    """Compare reversible (recompute) grads vs plain autograd grads for the same weights."""
    torch.manual_seed(0)
    m = GPT(block=T, variant=variant, n_layer=4, d=64, n_head=4, vocab=512, **kw).to(dev, dtype)
    x = torch.randint(0, 512, (B, T), device=dev); y = torch.randint(0, 512, (B, T), device=dev)
    m(x, y, reversible=False).backward()
    ref = {n: p.grad.clone() for n, p in m.named_parameters()}
    m.zero_grad()
    m(x, y, reversible=True).backward()
    num = sum(((p.grad - ref[n]) ** 2).sum() for n, p in m.named_parameters())
    den = sum((ref[n] ** 2).sum() for n, p in m.named_parameters())
    return (num / den).sqrt().item()

@torch.no_grad()
def reconstruction_error(model, x):
    """Run the stack forward, then invert every step; report ||x_rebuilt - x|| / ||x||."""
    emb = model.wte(x) + model.wpe(torch.arange(x.shape[1], device=x.device))
    s0 = model.init_state(emb); s = s0
    steps = model.steps()
    for st in steps: s = st.forward(s)
    for st in reversed(steps):
        s, _ = st.reverse(s, tuple(torch.zeros_like(t) for t in s))
    num = sum(((a - b) ** 2).sum() for a, b in zip(s, s0)).sqrt()
    return (num / emb.norm()).item()
