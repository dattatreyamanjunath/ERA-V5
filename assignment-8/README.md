# Attention, in order

**Every attention mechanism from the ERA V5 Session 8 lesson, arranged by launch date, with the
problem each one was answering and an honest account of what it costs.**

- **Live app:** _(Netlify URL — see below)_
- **Source:** [`assignment-8/`](https://github.com/dattatreyamanjunath/ERA-V5/tree/main/assignment-8)

---

## What this is

Session 8's assignment asked for a visual explainer covering every attention mechanism from the
session, ordered **chronologically by launch date** — not by teaching order, not grouped by family —
with each mechanism framed as the answer to a problem that existed at that moment, and with pros and
cons written honestly.

The brief was explicit that the animations are not the point:

> Your job is to be right about the dates, right about the trade-offs, and clear about the story.

So this repo treats **date provenance as the deliverable**, not as decoration. Every date is checked
against a primary source by a script that anyone can re-run, and the app has a *Show date provenance*
toggle that puts the source and the basis for the date next to every single mechanism.

## What's in the app

| Section | What it does |
|---|---|
| **Standard attention** | Six tokens, four dimensions, stepped through Q/K/V → scores → scale → mask → softmax → weighted sum. Every number is computed live in the browser, not drawn. Turn the causal mask off and watch weight leak onto future tokens. |
| **The two bills** | Compute (∝ T²) against KV cache (∝ T), on one slider — because most of the timeline only makes sense once you know which bill a mechanism is paying down. |
| **The timeline** | 46 mechanisms, 2014 → 2026, on one chronological axis, grouped into seven eras that name what the field was optimising for at the time. Filterable by which bill each one attacks. Click any mechanism for problem / mechanism / buys / gives up / when you'd pick it. |
| **Calculators** | Five claims from the lesson, recomputed live: softmax-off regrouping, the delta rule, the KV cache formula, MHA vs GQA vs MQA, sequence compression + top-k, and depth schedules. |
| **What you can only see in date order** | The written answer to assignment Question 2. |
| **Corrections** | Two findings that came out of checking the dates. |

Every mechanism card is required to state what it **gives up**. If a technique appears with only
advantages, it has not been understood yet — that was the brief's standard and it is enforced here.

---

## How the dates were established

**Every date in this app is an arXiv v1 submission date** — the `published` field from the arXiv
API — unless the mechanism has no paper, in which case the official release post is used and the row
is labelled accordingly.

This matters more than it sounds. Three specific traps that catch people and agents alike:

- **arXiv abstract pages show the *latest* revision at the top.** *Attention Is All You Need* has a
  v7 dated **2 Aug 2023**. Citing that would place the Transformer after GPT-4. YaRN's latest
  revision is dated **6 Feb 2026**, nearly two and a half years after its v1.
- **Conference year ≠ publication date.** Log-Linear Attention appeared at ICLR 2026 but was posted
  **5 Jun 2025** — a seven-month error if you cite the venue.
- **Blog release ≠ paper.** Mistral 7B was released by blog on 27 Sep 2023 and the paper landed
  10 Oct 2023. Both are given rather than silently choosing one.

### Re-run the verification yourself

```bash
cd assignment-8
python3 scripts/verify_dates.py
```

It reads every `arxiv:` id out of `js/mechanisms.js`, queries the arXiv API, and asserts the stored
date equals the API's `published` field. It exits non-zero if any date disagrees. No dependencies.

The committed output of the last run is [`date-verification.txt`](date-verification.txt).
**Current status: 42 of 46 dates machine-verified, 0 mismatches.** The remaining 4 have no paper and
are documented individually below.

---

## Two things worth flagging

### 1. `DroPE` and `DRoPE` are different papers, nine months apart

Searching "DroPE arXiv" returns **arXiv:2503.15029** first. That paper is real and it is **not** the
one from Session 8.

| | |
|---|---|
| ✗ **DRoPE**, [arXiv:2503.15029](https://arxiv.org/abs/2503.15029), 19 Mar 2025 | *Directional Rotary Position Embedding for Efficient Agent Interaction Modeling.* Adapts RoPE to encode agent **headings** for autonomous-driving trajectory models. Evaluated on Waymo Open Motion and Argoverse 2. Nothing to do with LLM context extension. |
| ✓ **DroPE**, [arXiv:2512.12167](https://arxiv.org/abs/2512.12167), 13 Dec 2025 | *Extending the Context of Pretrained LLMs by Dropping Their Positional Embeddings.* Sakana AI. Removes RoPE after pretraining and recalibrates for <1% of the pretraining budget. **This is the Session 8 one.** Code: [SakanaAI/DroPE](https://github.com/SakanaAI/DroPE) |

This is precisely the failure the brief warned about — an agent half-remembering a technique and
confidently attaching the wrong paper to it.

### 2. The lesson understates what is publicly known about DroPE

Section 9 of the lesson says:

> What the available record does not establish: the exact DroPE algorithm or which rotary dimensions
> it changes.

The public record does establish it. The Sakana AI paper (arXiv:2512.12167, v1 13 Dec 2025) gives
the full method and the code is open. The answer to "which rotary dimensions does it change" is
**all of them** — the method's central claim is that positional embeddings are a *training scaffold*
rather than a permanent architectural requirement, so they are dropped from every layer and the
model is recalibrated for under 1% of its pretraining budget.

That is fully consistent with the V4 cookbook line *"positional recalibration: DroPE, applied before
annealing"*, and with the section's correct insistence that this is a training-time procedure and
not an inference-time switch. The epistemic caution in that section was right — it was just aimed at
a gap a public paper had already closed.

---

## Full sources table

46 mechanisms. `Date type` says whether the date is an arXiv v1 submission date or an official
release post. This table is generated from `js/mechanisms.js` by
`node scripts/gen_sources_table.js`, so it cannot drift from what the app shows.

<!-- BEGIN SOURCES -->
| # | Date | Mechanism | Date type | Primary source |
|---|------|-----------|-----------|----------------|
| 1 | `2014-09-01` | Additive (Bahdanau) attention | arXiv v1 | [Bahdanau, Cho & Bengio, arXiv:1409.0473](https://arxiv.org/abs/1409.0473) |
| 2 | `2015-08-17` | Multiplicative / dot-product attention | arXiv v1 | [Luong, Pham & Manning, arXiv:1508.04025](https://arxiv.org/abs/1508.04025) |
| 3 | `2017-05-08` | Learned absolute position embeddings | arXiv v1 | [Gehring et al., ConvS2S, arXiv:1705.03122](https://arxiv.org/abs/1705.03122) |
| 4 | `2017-06-12` | Scaled dot-product & multi-head attention | arXiv v1 | [Vaswani et al., arXiv:1706.03762](https://arxiv.org/abs/1706.03762) |
| 5 | `2017-06-12` | Sinusoidal position encoding | arXiv v1 | [Vaswani et al., arXiv:1706.03762 §3.5](https://arxiv.org/abs/1706.03762) |
| 6 | `2018-03-06` | Relative position representations | arXiv v1 | [Shaw, Uszkoreit & Vaswani, arXiv:1803.02155](https://arxiv.org/abs/1803.02155) |
| 7 | `2019-01-09` | Segment recurrence + relative positions (Transformer-XL) | arXiv v1 | [Dai et al., arXiv:1901.02860](https://arxiv.org/abs/1901.02860) |
| 8 | `2019-04-23` | Sparse Transformer (strided / fixed patterns) | arXiv v1 | [Child et al., arXiv:1904.10509](https://arxiv.org/abs/1904.10509) |
| 9 | `2019-11-06` | Multi-Query Attention (MQA) | arXiv v1 | [Shazeer, arXiv:1911.02150](https://arxiv.org/abs/1911.02150) |
| 10 | `2020-01-13` | Reformer (LSH attention) | arXiv v1 | [Kitaev, Kaiser & Levskaya, arXiv:2001.04451](https://arxiv.org/abs/2001.04451) |
| 11 | `2020-04-10` | Sliding window + global attention (Longformer) | arXiv v1 | [Beltagy, Peters & Cohan, arXiv:2004.05150](https://arxiv.org/abs/2004.05150) |
| 12 | `2020-06-08` | Linformer (low-rank K/V projection) | arXiv v1 | [Wang et al., arXiv:2006.04768](https://arxiv.org/abs/2006.04768) |
| 13 | `2020-06-29` | Linear attention ("Transformers are RNNs") | arXiv v1 | [Katharopoulos et al., arXiv:2006.16236](https://arxiv.org/abs/2006.16236) |
| 14 | `2020-07-28` | BigBird (window + global + random) | arXiv v1 | [Zaheer et al., arXiv:2007.14062](https://arxiv.org/abs/2007.14062) |
| 15 | `2020-09-30` | Performer (FAVOR+) | arXiv v1 | [Choromanski et al., arXiv:2009.14794](https://arxiv.org/abs/2009.14794) |
| 16 | `2021-02-22` | The delta rule in linear transformers | arXiv v1 | [Schlag, Irie & Schmidhuber, arXiv:2102.11174](https://arxiv.org/abs/2102.11174) |
| 17 | `2021-03-03` | Random Feature Attention (and gating) | arXiv v1 | [Peng et al., arXiv:2103.02143](https://arxiv.org/abs/2103.02143) |
| 18 | `2021-04-20` | Rotary Position Embedding (RoPE) | arXiv v1 | [Su et al., RoFormer, arXiv:2104.09864](https://arxiv.org/abs/2104.09864) |
| 19 | `2021-08-27` | ALiBi (attention with linear biases) | arXiv v1 | [Press, Smith & Lewis, arXiv:2108.12409](https://arxiv.org/abs/2108.12409) |
| 20 | `2022-02-21` | Gated attention unit + chunked linear attention (FLASH) | arXiv v1 | [Hua et al., arXiv:2202.10447](https://arxiv.org/abs/2202.10447) |
| 21 | `2022-05-27` | FlashAttention (IO-aware exact attention) | arXiv v1 | [Dao et al., arXiv:2205.14135](https://arxiv.org/abs/2205.14135) |
| 22 | `2022-12-28` | H3 — diagnosing why state models fail at language | arXiv v1 | [Fu et al., arXiv:2212.14052](https://arxiv.org/abs/2212.14052) |
| 23 | `2023-05-22` | Grouped-Query Attention (GQA) | arXiv v1 | [Ainslie et al., arXiv:2305.13245](https://arxiv.org/abs/2305.13245) |
| 24 | `2023-06-27` | Position Interpolation (PI) | arXiv v1 | [Chen et al., arXiv:2306.15595](https://arxiv.org/abs/2306.15595) |
| 25 | `2023-06-30` | NTK-aware scaled RoPE | release/post | [u/bloc97, r/LocalLLaMA (no paper)](https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/) |
| 26 | `2023-08-31` | YaRN | arXiv v1 | [Peng et al., arXiv:2309.00071](https://arxiv.org/abs/2309.00071) |
| 27 | `2023-09-29` | Attention sinks (StreamingLLM) | arXiv v1 | [Xiao et al., arXiv:2309.17453](https://arxiv.org/abs/2309.17453) |
| 28 | `2023-10-10` | Sliding window in a production decoder (Mistral 7B) | arXiv v1 | [Jiang et al., Mistral 7B, arXiv:2310.06825](https://arxiv.org/abs/2310.06825) |
| 29 | `2023-12-01` | Mamba (selective state space) | arXiv v1 | [Gu & Dao, arXiv:2312.00752](https://arxiv.org/abs/2312.00752) |
| 30 | `2024-02-29` | Griffin — local attention + gated linear recurrence | arXiv v1 | [De et al., arXiv:2402.19427](https://arxiv.org/abs/2402.19427) |
| 31 | `2024-04-10` | Infini-attention (compressive memory) | arXiv v1 | [Munkhdalai, Faruqui & Gopal, arXiv:2404.07143](https://arxiv.org/abs/2404.07143) |
| 32 | `2024-05-07` | Multi-head Latent Attention (MLA) | arXiv v1 | [DeepSeek-AI, DeepSeek-V2, arXiv:2405.04434](https://arxiv.org/abs/2405.04434) |
| 33 | `2024-06-10` | DeltaNet parallelized over sequence length | arXiv v1 | [Yang, Wang, Zhang & Kim, arXiv:2406.06484](https://arxiv.org/abs/2406.06484) |
| 34 | `2024-06-16` | Quest — query-aware top-k page selection | arXiv v1 | [Tang et al., arXiv:2406.10774](https://arxiv.org/abs/2406.10774) |
| 35 | `2024-12-09` | Gated DeltaNet | arXiv v1 | [Yang, Kautz & Hatamizadeh, arXiv:2412.06464](https://arxiv.org/abs/2412.06464) |
| 36 | `2025-01-14` | Lightning Attention at frontier scale (MiniMax-01) | arXiv v1 | [MiniMax et al., arXiv:2501.08313](https://arxiv.org/abs/2501.08313) |
| 37 | `2025-02-16` | Native Sparse Attention (NSA) | arXiv v1 | [Yuan et al. (DeepSeek), arXiv:2502.11089](https://arxiv.org/abs/2502.11089) |
| 38 | `2025-02-18` | MoBA (Mixture of Block Attention) | arXiv v1 | [Lu et al. (Moonshot AI), arXiv:2502.13189](https://arxiv.org/abs/2502.13189) |
| 39 | `2025-06-05` | Log-Linear Attention | arXiv v1 | [Guo et al., arXiv:2506.04761](https://arxiv.org/abs/2506.04761) |
| 40 | `2025-08-08` | Learned attention sinks + alternating SWA (gpt-oss) | arXiv v1 | [OpenAI, gpt-oss model card, arXiv:2508.10925](https://arxiv.org/abs/2508.10925) |
| 41 | `2025-09-11` | Gated DeltaNet 3:1 hybrid in production (Qwen3-Next) | release/post | [vLLM day-0 support blog, 2025-09-11](https://blog.vllm.ai/2025/09/11/qwen3-next.html) |
| 42 | `2025-09-29` | DeepSeek Sparse Attention (DSA) | release/post | [DeepSeek-V3.2-Exp release, 2025-09-29](https://api-docs.deepseek.com/news/news250929) |
| 43 | `2025-10-30` | Kimi Delta Attention (Kimi Linear) | arXiv v1 | [Kimi Team, arXiv:2510.26692](https://arxiv.org/abs/2510.26692) |
| 44 | `2025-12-13` | DroPE — drop the positional embeddings, then recalibrate | arXiv v1 | [Gelberg, Eguchi, Akiba & Cetin (Sakana AI), arXiv:2512.12167](https://arxiv.org/abs/2512.12167) |
| 45 | `2026-02-15` | Hybrid attention goes mainstream — and the field splits | release/post | [Raschka, "A Dream of Spring for Open-Weight LLMs" (Jan–Feb 2026 survey)](https://magazine.sebastianraschka.com/p/a-dream-of-spring-for-open-weight) |
| 46 | `2026-04-26` | Compressed Sparse Attention + Heavily Compressed Attention (DeepSeek-V4) | arXiv v1 | [DeepSeek-AI, arXiv:2606.19348](https://arxiv.org/abs/2606.19348) |
<!-- END SOURCES -->

### Dates with no paper behind them

These four cannot be machine-checked and are the weakest links on the page. Saying so is the point.

<!-- BEGIN NOPAPER -->
**30 Jun 2023 — NTK-aware scaled RoPE**

THE WEAKEST DATE ON THIS PAGE, and flagged as such. There is no paper — the primary source is a Reddit post in late June 2023. The citable corroboration is YaRN (arXiv:2309.00071), which credits it as "bloc97, 2023" and reproduces the formula. Treat the day as approximate; the month and year are solid.

<https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/>

---

**11 Sept 2025 — Gated DeltaNet 3:1 hybrid in production (Qwen3-Next)**

No arXiv paper for Qwen3-Next specifically. Dated from the vLLM day-0 support post (2025-09-11), which coincides with Qwen’s own announcement week. The Qwen3 technical report (arXiv:2505.09388, 2025-05-14) is a different, earlier model and must not be used as this date.

<https://blog.vllm.ai/2025/09/11/qwen3-next.html>

---

**29 Sept 2025 — DeepSeek Sparse Attention (DSA)**

Official DeepSeek API news post dated 2025-09-29, plus the V3.2-Exp technical report in the GitHub release. There is no arXiv paper for DSA itself, so the release post is the primary source.

<https://api-docs.deepseek.com/news/news250929>

---

**15 Feb 2026 — Hybrid attention goes mainstream — and the field splits**

Dated from the individual model releases surveyed there: Qwen3.5 2026-02-15, Ling 2.5 2026-02-16, GLM-5 and MiniMax M2.5 2026-02-12, Qwen3-Coder-Next 2026-02-03. This row is a cluster, not one launch, and is labelled as such rather than pretending it is a single dated event.

<https://magazine.sebastianraschka.com/p/a-dream-of-spring-for-open-weight>
<!-- END NOPAPER -->

### Other sources used

- Session 8 lesson, *Modern Attention Variants*, ERA V5 — the scope of what to cover, the KV-cache
  formula and its worked example (48 layers / 8 KV heads / head_dim 128 / bf16 → 6.44 GB at 32,768
  tokens), the `D D D G D D D G` schedule and its 8.0× cache / 1.41× compute figures, and the
  softmax-off (140 = 140) and delta-rule (40 + 15 = 55 vs 95) worked examples. The app reproduces
  all of these numerically rather than restating them.
- Sebastian Raschka, [*A Dream of Spring for Open-Weight LLMs: 10 Architectures from Jan–Feb 2026*](https://magazine.sebastianraschka.com/p/a-dream-of-spring-for-open-weight)
  — used only for the 2026 model-release cluster, with individual release dates cross-checked
  against model cards.

---

## Running it locally

No build step, no dependencies. It is plain HTML, CSS and vanilla JS.

```bash
cd assignment-8 && python3 -m http.server 8899
```

Then open <http://localhost:8899>.

## Repo layout

```
assignment-8/
├── index.html                 the page
├── css/style.css              theme, layout, responsive rules
├── js/mechanisms.js           THE single source of truth: 46 mechanisms + dates + sources
├── js/attention.js            scaled dot-product attention, actually computed
├── js/timeline.js             chronological axis, filtering, detail drawer, live demo
├── js/widgets.js              the five calculators
├── scripts/verify_dates.py    re-checks every arXiv date against the arXiv API
├── scripts/gen_sources_table.js  regenerates this README's table from mechanisms.js
├── date-verification.txt      committed output of the last verification run
├── answers.md                 written answers to the assignment questions
└── netlify.toml               publish config (no build command)
```

Every view in the app is rendered from `js/mechanisms.js`. A date can therefore only be wrong in one
place, and the verification script checks exactly that place.
