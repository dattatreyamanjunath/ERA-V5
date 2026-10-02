# Assignment 14: Dense GPT -> Mixture-of-Experts (sparse upcycling)

Trained on a Colab T4 (driven through Claude in Chrome). Notebook with outputs: [moe_upcycling_executed.ipynb](moe_upcycling_executed.ipynb). Raw numbers: [results.json](results.json).

## Setup
- Data: Tiny Shakespeare (1.1M chars, char-level, vocab 65), 90/10 train/val split.
- Model: 6-layer GPT, d=384, 6 heads, ctx 256, **10.8M params**, plain `nn.Linear` MLPs. Batch 64x256 tokens, AdamW, cosine LR, bf16/fp16 autocast.
- **Phase 1 (dense):** 2500 steps, lr 1e-3.
- **Conversion:** every block's MLP becomes an MoE layer with 8 experts, top-2 routing. Each expert is a copy of the trained dense MLP (+1% noise to break symmetry); the router is a fresh small-random `Linear(384, 8)`. Result: **60.4M total params, 17.9M active per token**. Switch-style load-balancing loss (weight 0.01).
- **Phase 2 (MoE):** 2000 more steps at lr 3e-4. A **control** copy of the dense model gets the same 2000 steps and schedule.

## Results
| | step | train loss | val loss |
|---|---|---|---|
| Dense, start | 0 | 4.301 | 4.293 |
| Dense, best val | 1000 | 1.184 | 1.546 |
| Dense, end of phase 1 | 2500 | 0.594 | 2.164 |
| **MoE right after conversion** | 2500 | **0.590** | 2.153 |
| MoE, end | 4500 | **0.095** | 3.843 |
| Dense control, end | 4500 | 0.175 | 3.717 |

- The conversion is function-preserving: loss right after it matches the dense model (train 0.594 -> 0.590).
- After conversion the MoE keeps training: **train loss falls 0.590 -> 0.095**, faster than the dense control (0.175), which is what the extra capacity buys.

![loss curve](loss_curve.png)

## Caveat: this run overfits
Tiny Shakespeare is only ~1M characters. Phase 1 alone sees ~40M tokens (about 36 epochs), so the dense model already overfits: val loss bottoms out at 1.546 around step 1000, then rises. Past the conversion, both the MoE and the control just memorise the training set, so **validation loss goes up while train loss goes down**. The MoE memorises faster than the control (more capacity), which is why its val is slightly worse (3.84 vs 3.72). The train-loss claim above holds; no generalization gain is claimed. A fair test of whether upcycling improves held-out loss would need a larger corpus or dropout, and a shorter dense phase (stopped near step 1000).
