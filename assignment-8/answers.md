# Assignment answers — Session 8

## Question 1 — live app link and GitHub repo

- **Live app:** _(Netlify URL)_
- **GitHub repo:** https://github.com/dattatreyamanjunath/ERA-V5 — code in
  [`assignment-8/`](https://github.com/dattatreyamanjunath/ERA-V5/tree/main/assignment-8)

46 mechanisms, 2014 → 2026, in launch order. 42 of the 46 dates are arXiv v1 submission dates
re-checked against the arXiv API by [`scripts/verify_dates.py`](scripts/verify_dates.py); the
committed output is in [`date-verification.txt`](date-verification.txt). The remaining four have no
paper behind them and are individually labelled in the app and the README.

---

## Question 2 — what does the timeline actually show?

Six things that are invisible in a list and obvious once the mechanisms are in date order.

### 1. The compute panic ended in May 2022, and it was ended by a kernel, not by an idea

Between April 2019 and September 2020 the field produced Sparse Transformer, Reformer, Longformer,
Linformer, linear attention, BigBird and Performer — seven separate attacks on T² in eighteen
months. Then FlashAttention (27 May 2022) made **exact** attention 2–4× faster with O(T) memory, and
that entire research line went quiet for roughly three years.

None of those methods was refuted. They were made unnecessary. In a list, FlashAttention looks like
a footnote because it does not change the mathematics at all. On a timeline it is the event that
explains a three-year gap, and it carries a second-order effect that is easy to miss: because a
fused kernel cannot cheaply express an arbitrary per-pair attention bias, FlashAttention quietly
selected *for* RoPE and ALiBi and *against* Shaw-style relative position tables. A hardware
constraint decided a modelling argument.

### 2. The bill people cared about switched in 2023, and you can date the switch

Everything before 2023 attacks FLOPs. Almost everything in 2023 — GQA (May), attention sinks
(September), Mistral's sliding window (October) — attacks the KV cache. Nothing about attention
itself changed in between. *Deployment* changed. Once models were serving millions of concurrent
conversations, the per-user private cache became the binding constraint, and the field pivoted
within about twelve months.

This is also the clearest illustration of the lesson's two-bills framing: GQA does nothing for
compute, and FlashAttention does nothing for cache size. Reading them as competing "efficient
attention" methods, which a taxonomy encourages, gets both wrong.

### 3. Position is a repair cycle, and DroPE is what the end of one looks like

Learned table (May 2017) → sinusoidal (Jun 2017) → relative (Mar 2018) → RoPE (Apr 2021) → Position
Interpolation (Jun 2023) → NTK-aware (Jun 2023) → YaRN (Aug 2023) → DroPE (Dec 2025).

Every entry from PI onward is a patch on RoPE, and they arrive in a cluster — PI, NTK-aware and
YaRN land within ten weeks of each other in mid-2023, which is what a field looks like when
everyone hits the same wall simultaneously. Then DroPE arrives and argues the patches were the wrong
move because the positional embedding *itself* was the obstacle: drop it entirely after pretraining
and recalibrate. That is the shape of a field about to abandon an assumption rather than patch it
again.

### 4. Good ideas wait years for their engineering

The delta rule enters linear attention in **February 2021** (Schlag et al.). Then nothing happens for
**three years and four months**, until June 2024, when someone finds a way to parallelize it over
sequence length via the WY representation. Gated DeltaNet follows six months after that, and
production deployment (Qwen3-Next) six months after that.

The idea was never the bottleneck — trainability was. In a list the delta rule looks like a 2024
invention. The timeline shows a 2021 idea sitting on a shelf because it could not be trained in
parallel, which reframes "parallel trainability" from an implementation detail into a first-class
architectural constraint. The same pattern appears with linear attention generally: proposed in 2020,
genuinely deployed in 2025.

### 5. February 2025 shows simultaneous invention, which is the strongest available signal

**NSA** (16 Feb 2025, DeepSeek) and **MoBA** (18 Feb 2025, Moonshot) are two labs arriving at the
same core idea — block-level sparsity trained into the model rather than bolted on at inference —
and publishing **two days apart**. Independent simultaneous invention means the problem was fully
ripe and the pieces were lying in the open.

Nothing about a list would tell you those two are the same moment. It is the single best predictor on
the whole timeline that the direction was about to become mainstream, which it did: DSA in September
2025, then DeepSeek-V4's CSA in April 2026.

### 6. The oscillation the lesson predicted is real, and it is visible

> first it wants exactness, then it wants memory back, then it wants length, then it wants memory again

Laid out by date, that reads:

**exactness** (2014–17) → **compute** (2018–20) → **position solved, and exact attention gets cheap
again** (2021–22) → **decode memory** (2023) → **recurrent state returns** (2024) → **sparsity
returns, now trained in** (2025) → **compression as the primary lever** (2026).

The field has abandoned fixed-size state twice and come back for it twice — each time with the
missing piece added. First gating (so it can forget), then the delta rule (so it can correct). The
same is true of sparsity: fixed patterns in 2019, abandoned after FlashAttention, back in 2025 as
learned and trained-in selection. Techniques do not die on this timeline. They wait for the piece
they were missing.

**What that suggests comes next**, extrapolating rather than claiming knowledge: the fixed-size state
gives way to a *growing-but-sublinear* one — Log-Linear Attention (Jun 2025) is already prototyping
exactly that with O(log T) state; the schedule ratio stops being inherited (3:1 and DDDGDDDG are both
currently copied rather than measured) and becomes something searched per layer; and compression
moves earlier still, out of the KV cache and into tokenisation, since the cheapest token is the one
never spent. The honest caveat is that this timeline's own record says the next move often comes
from a kernel rather than an equation, and nobody predicts those.

---

## Mechanisms not covered in the session

The session's minimum list was standard attention, absolute learned positions, sinusoidal, RoPE,
ALiBi, MQA, GQA, sliding window, attention sinks, NTK-aware scaling, YaRN, linear attention, the
delta rule and Gated DeltaNet, MLA, sparse/top-k attention, DeepSeek's compressed sparse attention,
and DroPE. All are in the app. These are the additions, each with its date and primary source:

| Mechanism | Date | Source | Why it belongs |
|---|---|---|---|
| Additive (Bahdanau) attention | 1 Sep 2014 | [arXiv:1409.0473](https://arxiv.org/abs/1409.0473) | The origin. Attention exists three years before the Transformer. |
| Multiplicative / dot-product scoring | 17 Aug 2015 | [arXiv:1508.04025](https://arxiv.org/abs/1508.04025) | Where the dot product replaced an MLP score — the decision that made attention one GEMM. |
| Relative position representations | 6 Mar 2018 | [arXiv:1803.02155](https://arxiv.org/abs/1803.02155) | The absolute → relative pivot that RoPE later completes. |
| Transformer-XL segment recurrence | 9 Jan 2019 | [arXiv:1901.02860](https://arxiv.org/abs/1901.02860) | The first answer to "what crosses a chunk boundary" — direct ancestor of the Memory Stream. |
| Reformer (LSH attention) | 13 Jan 2020 | [arXiv:2001.04451](https://arxiv.org/abs/2001.04451) | First *learned router decides where to look*, six years before NSA/DSA make it work. |
| Linformer | 8 Jun 2020 | [arXiv:2006.04768](https://arxiv.org/abs/2006.04768) | Instructive failure: linear, but structurally unusable for autoregressive decoding. |
| BigBird | 28 Jul 2020 | [arXiv:2007.14062](https://arxiv.org/abs/2007.14062) | The argument that a sparse attention graph needs short *diameter*, not just few edges. |
| Performer (FAVOR+) | 30 Sep 2020 | [arXiv:2009.14794](https://arxiv.org/abs/2009.14794) | The theoretical bound on how wrong linear attention is. |
| Random Feature Attention | 3 Mar 2021 | [arXiv:2103.02143](https://arxiv.org/abs/2103.02143) | Introduced recency **gating** — ancestor of the decay in Mamba and Gated DeltaNet. |
| FLASH / gated attention unit | 21 Feb 2022 | [arXiv:2202.10447](https://arxiv.org/abs/2202.10447) | Origin of chunkwise local-quadratic/global-linear training, and of "gated attention". |
| **FlashAttention** | **27 May 2022** | [arXiv:2205.14135](https://arxiv.org/abs/2205.14135) | **The most consequential omission.** It ended the sparse-attention wave without changing the mathematics, and it biased the field toward RoPE/ALiBi. See point 1 above. |
| H3 | 28 Dec 2022 | [arXiv:2212.14052](https://arxiv.org/abs/2212.14052) | Diagnosed *why* fixed-state models fail at language: recall and comparison. |
| Position Interpolation | 27 Jun 2023 | [arXiv:2306.15595](https://arxiv.org/abs/2306.15595) | The "interpolate, don't extrapolate" pivot that NTK-aware and YaRN both build on. |
| Mamba | 1 Dec 2023 | [arXiv:2312.00752](https://arxiv.org/abs/2312.00752) | Input-dependent selectivity — the reason gating is now mandatory in linear models. |
| Griffin | 29 Feb 2024 | [arXiv:2402.19427](https://arxiv.org/abs/2402.19427) | Where "different layers get different memory systems" became mainstream — the template DDDGDDDG follows. |
| Infini-attention | 10 Apr 2024 | [arXiv:2404.07143](https://arxiv.org/abs/2404.07143) | Clearest statement of "compress the evicted past instead of deleting it". |
| Quest | 16 Jun 2024 | [arXiv:2406.10774](https://arxiv.org/abs/2406.10774) | Names the real top-k problem — cheap candidate *proposal* — and solves it with page bounds. |
| Lightning Attention (MiniMax-01) | 14 Jan 2025 | [arXiv:2501.08313](https://arxiv.org/abs/2501.08313) | The 456B existence proof that licensed the hybrid wave. |
| MoBA | 18 Feb 2025 | [arXiv:2502.13189](https://arxiv.org/abs/2502.13189) | NSA's independent twin, two days later. See point 5 above. |
| Log-Linear Attention | 5 Jun 2025 | [arXiv:2506.04761](https://arxiv.org/abs/2506.04761) | A third option between O(1) state and O(T) cache: O(log T) hierarchical state. |
| Learned attention sinks (gpt-oss) | 8 Aug 2025 | [arXiv:2508.10925](https://arxiv.org/abs/2508.10925) | Where the 2023 sink *diagnosis* became a designed-in *component*. |
| Qwen3-Next 3:1 hybrid | 11 Sep 2025 | [vLLM blog](https://blog.vllm.ai/2025/09/11/qwen3-next.html) | Gated DeltaNet in a production MoE — where D-D-D-G became a shipped default. |
| Kimi Delta Attention | 30 Oct 2025 | [arXiv:2510.26692](https://arxiv.org/abs/2510.26692) | First hybrid claiming it *beats* full attention, and it drops RoPE from its attention layers entirely. |
| The Feb 2026 cluster | 3–16 Feb 2026 | model cards; [Raschka survey](https://magazine.sebastianraschka.com/p/a-dream-of-spring-for-open-weight) | Four labs, four different attention stacks, one month, all competitive — evidence the choice is workload-dependent, not solved. |

### And one correction, offered in the spirit the brief asked for

**`DroPE` and `DRoPE` are two different papers nine months apart, and the session understates what
is known about the first one.**

[arXiv:2503.15029](https://arxiv.org/abs/2503.15029) — *DRoPE: Directional Rotary Position Embedding*
(19 Mar 2025) — is an autonomous-driving trajectory paper and is the **first search hit** for "DroPE
arXiv". It is not the Session 8 mechanism.

The Session 8 one is [arXiv:2512.12167](https://arxiv.org/abs/2512.12167), *Extending the Context of
Pretrained LLMs by Dropping Their Positional Embeddings* (Sakana AI, v1 13 Dec 2025), with open code
at [SakanaAI/DroPE](https://github.com/SakanaAI/DroPE).

Section 9 says *"the available record does not establish the exact DroPE algorithm or which rotary
dimensions it changes."* The paper does establish it, and the answer to the second half is **all of
them**: the method's claim is that positional embeddings are a training scaffold rather than a
permanent requirement, so they are dropped from every layer and the model is recalibrated for under
1% of its pretraining budget. That matches the V4 cookbook's *"positional recalibration: DroPE,
applied before annealing"* exactly, and it supports the section's correct point that this is a
training-time procedure and not an inference-time switch. The caution was well-placed; it was just
aimed at a gap a public paper had already closed.

---

## Question 3 — shared on LinkedIn/X/Medium

_(optional — not yet posted)_
