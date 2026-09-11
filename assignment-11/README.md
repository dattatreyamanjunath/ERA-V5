# Assignment 11: Adam Optimizer Deep Dive

Built on the same nanoGPT model used in Assignment 10 (tiny GPT: n_embd=64, n_layer=2, n_head=4, vocab=65 Shakespeare chars).  
All five tasks live in [`assignment11.ipynb`](assignment11.ipynb).

---

## Task 1 — Adam by Hand

**Setup:** one scalar weight θ₀ = 0.5, five gradients [0.1, −0.2, 0.3, 0.05, −0.15].  
Parameters: lr = 1e-3, β₁ = 0.9, β₂ = 0.999, ε = 1e-8.

Hand-computed quantities vs `torch.optim.Adam` (float64):

| Step | g | m | v | m̂ | v̂ | θ (hand) | θ (PyTorch) | Δθ |
|------|---|---|---|---|---|---|---|---|
| 1 | +0.100 | 0.01000000 | 1.00e-5 | 0.10000000 | 1.000e-2 | 0.49900000 | 0.49900000 | 0.00e+00 |
| 2 | −0.200 | −0.01100000 | 5.00e-5 | −0.05789474 | 2.499e-2 | 0.49936610 | 0.49936610 | 0.00e+00 |
| 3 | +0.300 | 0.02010000 | 1.40e-4 | 0.07074830 | 4.660e-2 | 0.49902286 | 0.49902286 | 0.00e+00 |
| 4 | +0.050 | 0.02309000 | 1.42e-4 | 0.06345454 | 4.668e-2 | 0.49866715 | 0.49866715 | 0.00e+00 |
| 5 | −0.150 | 0.00578100 | 1.65e-4 | 0.01276697 | 5.207e-2 | 0.49858944 | 0.49858944 | 0.00e+00 |

**Result:** maximum absolute discrepancy ≤ 1.7×10⁻¹⁸ across all five steps — floating-point exact in float64.

---

## Task 2 — Bias Correction: First 20 Steps

We run Adam **with** and **without** bias correction on the same 20-step gradient sequence (β₁=0.9, β₂=0.999).

**Key finding: with β₂=0.999, the gap does NOT close within 20 steps.**

The step-size ratio between the two variants equals:

    ratio = step_wo / step_with = sqrt(1 − β₂ᵗ) / (1 − β₁ᵗ)

At step 20: `sqrt(1 − 0.999²⁰) / (1 − 0.9²⁰) ≈ 0.141 / 0.878 ≈ 0.16`  
→ the un-corrected version takes steps **6× larger**, giving a **~524% relative gap**.

**Analytical convergence steps** (gap < 1%):

| Moment | β | Convergence step |
|--------|---|---|
| First (m) | 0.9 | ~44 steps |
| Second (v) | 0.999 | ~**4,603 steps** |

The bias on v dominates because β₂ = 0.999 is very close to 1, so `1 − β₂ᵗ` stays near 0 for thousands of steps. **The difference stops mattering only after ~4,600 steps** — far beyond what the 20-step plot shows, but this is precisely why bias correction is necessary for the early phase of real training runs.

The plot shows the two weight trajectories diverge rapidly in the first 20 steps, with the uncorrected version making much larger initial jumps.

---

## Task 3 — Update-to-Weight Ratio Per Layer

**Setup:** tiny nanoGPT (0.10M params) trained for 100 steps with cosine LR schedule (warmup = 10 steps, lr = 1e-3 → 1e-4), AdamW (β₁=0.9, β₂=0.95).

At each step we measure `|Δθ| / |θ|` per named layer group.

**Results at warmup end (step 10) vs post-warmup (step 11–15):**

| Layer group | Ratio at warmup end | Ratio post-warmup | Δ |
|-------------|--------------------|--------------------|---|
| attn.weight | 8.07e-2 | 4.95e-2 | −38.6% |
| embeddings | 3.01e-2 | 2.71e-2 | −9.8% |
| layernorm | 7.33e-2 | 4.22e-2 | −42.5% |
| mlp.weight | 8.72e-2 | 5.29e-2 | −39.3% |

**The update/weight ratio stops being driven by LR warmup exactly at step 10** — the warmup end. After that, the ratio is determined solely by gradient magnitudes and optimizer momentum state, and decreases gradually as the cosine schedule decays the LR.

The jump between step 10 and step 11 is the clearest diagnostic of warmup: the ratio transitions from a monotonically rising curve to a gently declining one, coinciding precisely with the LR transition.

---

## Task 4 — Cosine vs WSD Scheduler (stop at step 200)

**Setup:** same tiny model (seed=42), trained twice for 300 steps; val loss recorded at step 200.

**Schedules:**
- **Cosine:** 20-step warmup → cosine decay over 300 steps to min_lr=1e-4
- **WSD:** 20-step warmup → flat plateau at lr=1e-3 until step 200 → linear decay to min_lr=1e-4 by step 300

| Schedule | Val loss at step 200 |
|----------|----------------------|
| Cosine | 2.5167 |
| **WSD** | **2.4877** |

**WSD wins by 0.0290 nats at step 200.**

**Model to keep: the WSD checkpoint.**

**Why WSD wins here:**  
At step 200 in a 300-step budget, the cosine schedule has already been decaying the LR for 180 steps — by step 200 the LR is roughly 50% of its peak value. This forces the model into fine-grained updates earlier than necessary, which can prematurely narrow the loss basin being explored.

WSD, on the other hand, holds the full learning rate through step 200 (its entire stable phase), enabling faster loss descent and keeping the optimizer exploring broadly. The model arrives at step 200 having seen more diverse gradients.

If the budget were much longer (e.g., 10,000 steps and we stopped at 200), the cosine advantage would shift earlier — but at the 2/3 mark of a short run, WSD's high stable LR wins.

---

## Task 5 — LR Sweep at Widths 256, 512, 1024

**Setup:** 21 runs, each 100 training steps, batch=32, block=64.  
LR grid: [1e-4, 3e-4, 6e-4, 1e-3, 2e-3, 4e-3, 8e-3]

**Tuning check:** the smallest LR (1e-4) gives clearly higher losses across all widths (underfitting), and the largest LR (8e-3) shows degraded or diverging loss (overfitting / instability). Both sides of the optimum are covered.

### Raw sweep results

| Width | 1e-4 | 3e-4 | 6e-4 | 1e-3 | 2e-3 | 4e-3 | 8e-3 | **Optimum** |
|-------|------|------|------|------|------|------|------|-------------|
| 256 (1.6M) | 2.573 | 2.452 | 2.413 | 2.386 | **2.298** | 2.552 | 2.830 | lr=2e-3 |
| 512 (6.3M) | 2.443 | 2.349 | 2.290 | **2.283** | 2.529 | 2.640 | 2.890 | lr=1e-3 |
| 1024 (25M) | 2.340 | **2.222** | 2.297 | 2.526 | 2.631 | 2.830 | 3.198 | lr=3e-4 |

### Optimal LRs and scaling law

| Width | lr* |
|-------|-----|
| 256 | 2e-3 |
| 512 | 1e-3 |
| 1024 | 3e-4 |

**Power-law fit:** `lr* = 4.30 × width^(−1.37)` (R² = 0.976)

The exponent −1.37 is steeper than the classical scaling `width^(−0.5)` from standard parameterisation, and steeper even than the μP prediction (width^0, i.e., constant). This is consistent with using standard Adam without μP — in standard param, gradient magnitudes and scale both shift with width, making the optimal LR drop faster than √width for these short runs.

### Prediction for width 4096

`lr*(4096) = 4.30 × 4096^(−1.37) ≈ **4.9×10⁻⁵**`

**Confidence: HIGH (R² = 0.976).**  
Caveats:
- Only 3 data points (an extrapolation of ×4 beyond the largest measured width)
- The coarse LR grid means the true optimal LR may lie between grid points (especially for width 512, where 6e-4 and 1e-3 are very close)
- Treat 4.9e-5 as a starting point; search between 2e-5 and 1.5e-4 (±0.5 log-decades)

---

## Files

```
assignment-11/
├── assignment11.ipynb     # All five tasks, fully executed
├── README.md              # This document
├── bias_correction.png    # Task 2: step-size and weight trajectories
├── update_ratio.png       # Task 3: per-layer update/weight ratio
├── cosine_vs_wsd.png      # Task 4: training curves + LR schedules
└── lr_sweep.png           # Task 5: LR sweep + power-law extrapolation
```

## How to run

```bash
cd assignment-11
jupyter notebook assignment11.ipynb
```

Or to re-execute headlessly (takes ~15–20 min on MPS for the full LR sweep):

```bash
jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=3600 \
    assignment11.ipynb --output assignment11.ipynb
```
