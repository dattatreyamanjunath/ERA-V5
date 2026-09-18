# Assignment 14: ZeRO Distributed Training Simulation

We simulate 32 virtual GPUs (CPU threads) running a small GPT-style transformer and walk through all four distributed training strategies — DDP, ZeRO-1, ZeRO-2, and ZeRO-3 — measuring how memory and communication change at each stage.

---

## The Model

A 4-layer MiniGPT with ~3.7M parameters. Small enough to run on a laptop, large enough to make the memory math meaningful.

---

## What We're Measuring

Modern mixed-precision Adam training needs **16 bytes per parameter**:

| What | Dtype | Bytes/param |
|------|-------|-------------|
| Parameters | fp16 | 2 |
| Gradients | fp16 | 2 |
| Master weights | fp32 | 4 |
| Adam momentum (m) | fp32 | 4 |
| Adam variance (v) | fp32 | 4 |

ZeRO's entire job is to ask: does every GPU really need all 16 bytes? The answer turns out to be no — you just need the right communication to compensate.

---

## Stage-by-Stage Observations

### DDP (Baseline)

Every GPU holds a complete copy of everything: params, gradients, optimizer states. After the backward pass, gradients are all-reduced so all GPUs stay in sync, and then each one runs an identical optimizer step locally.

- **Memory per GPU**: 56.7 MB
- **Communication**: one all-reduce on gradients = **2P** (where P ≈ one model copy in fp16)
- This is the simplest setup, but it scales terribly — doubling the model size doubles memory on every single GPU.

---

### ZeRO Stage 1 — Shard the Optimizer States

The insight here is that optimizer states (master weights + Adam m + v = 12 bytes/param) are only needed locally by the GPU that runs the optimizer update. So we split them across GPUs — each GPU owns 1/32 of the optimizer state and updates only that slice.

- **Memory per GPU**: 15.5 MB (**3.7× less than DDP**)
- **Communication**: all-reduce on gradients (2P) + all-gather to redistribute updated params (1P) = **3P total**
- The bad news: this is actually *more* communication than DDP for *less* memory saving than later stages. ZeRO-1 is the weakest trade of the three stages and rarely the right stopping point in practice.

---

### ZeRO Stage 2 — Also Shard the Gradients

After the backward pass, each GPU only needs the gradient slice that corresponds to its optimizer shard. So instead of all-reducing the full gradient (which leaves every GPU with a full copy it doesn't need), we do a **reduce-scatter** — each GPU ends up with only its 1/N slice.

- **Memory per GPU**: 8.6 MB (**6.6× less than DDP**)
- **Communication**: reduce-scatter on gradients (1P) + all-gather on updated params (1P) = **2P total**

This is the result worth pausing on. **ZeRO-2 cuts memory by 6.6× at literally the same communication cost as DDP.** The reduce-scatter replaces the first half of the all-reduce, and the all-gather replaces the second half — so the total bytes transferred don't change. There is no good reason to use DDP over ZeRO-2 for large models. It's a free lunch.

---

### ZeRO Stage 3 — Also Shard the Parameters

The most aggressive stage. Each GPU holds only 1/32 of the model parameters. Before the forward pass, the full parameters are all-gathered layer by layer and then discarded. During the backward pass, gradients are reduce-scattered. The optimizer step is purely local.

- **Memory per GPU**: 1.77 MB (**32× less than DDP** — exactly 1/N, the theoretical maximum)
- **Communication**: all-gather before forward (1P) + all-gather during backward (1P) + reduce-scatter (1P) = **3P total**

The 32× memory reduction is real, but so is the cost. The extra all-gather now sits on the **critical path of the forward pass** — the model literally cannot run until the parameters arrive. This is harder to overlap with computation compared to ZeRO-2, where the communication only happens after the backward pass. ZeRO-3 is the right choice when the model simply doesn't fit on a single GPU any other way.

---

## The Communication Story: P → 2P → 3P

Let P = the cost of one ring all-reduce over the full model (≈ 7 MB in our case):

```
DDP     → 2P   one all-reduce on gradients
ZeRO-1  → 3P   same all-reduce + one all-gather for updated params
ZeRO-2  → 2P   reduce-scatter + all-gather (same total as DDP, different shape)
ZeRO-3  → 3P   all-gather (fwd) + reduce-scatter + all-gather (bwd)
```

ZeRO-2 = DDP in communication cost is the key insight from the original DeepSpeed paper. ZeRO-1 and ZeRO-3 both pay 3P — ZeRO-1 because it still does a full all-reduce and then adds a gather, ZeRO-3 because it has to fetch parameters before every forward pass.

---

## Memory Scaling with More GPUs

ZeRO-3's memory saving scales linearly with GPU count — 32 GPUs means 32× savings, 64 GPUs means 64× savings. DDP gets no benefit at all. This is why ZeRO-3 becomes increasingly attractive as cluster size grows: the more GPUs you add, the cheaper each one becomes to run.

---

## A Note on the Timing Numbers

The optimizer step showed >1 second in the simulation, dwarfing everything else. That's a CPU simulation artifact — we loop over 32 "GPU" copies sequentially in Python. In real distributed training, each GPU runs its own optimizer step in parallel, so the effective optimizer time is the cost for one GPU's shard, not 32 in series. The forward (~27ms), backward (~84ms), and communication (~64ms) timings are the meaningful ones for understanding relative overhead.

---

## When to Use Which

| Strategy | Use when |
|----------|----------|
| DDP | Model fits comfortably on one GPU, you want the simplest setup |
| ZeRO-1 | Optimizer states are the bottleneck, but gradients fit fine |
| ZeRO-2 | Model is large but fits in GPU memory — best memory/communication trade-off |
| ZeRO-3 | Model doesn't fit on a single GPU at all — LLMs, large VLMs |
