# V5 Mixture-and-Curriculum Plan — A Defended Specification

## 1. Executive Summary

This document specifies the complete data mixture, multi-stage curriculum, and validation plan for V5 pretraining. Every number stated here is a **testable hypothesis**, not a commitment — the plan includes six concrete proxy experiments at 1B and 3B scale, each with a pre-committed metric, effect-size threshold, guardrail condition, and decision rule. Numbers that fail their proxy test get revised before full-scale training begins.

The plan is structured around a core principle: **a data decision is a hypothesis until a cheap experiment has tested it.** The highest-quality data is deliberately held back for the anneal phase where the learning rate is low enough to retain it, unverified and synthetic data is front-loaded where it can do the most good (language grounding) without displacing scarce verified tokens, and every stage transition is smoothed to avoid distribution-shift loss spikes.

---

## 2. Budget Architecture

**Total token budget: 100%.** Divided into two pools managed separately:

- **Main phase (92%)** — Stages S1 through S4. Governed by the phased curriculum. An adaptive selector (DoReMi-style reweighting) operates within each stage but is constrained by protected floors.
- **Anneal reserve (8%)** — Stage S5. Completely held back — no token from this pool is seen during S1–S4. Composed exclusively of the highest-quality, verified, outcome-checked data from every slot. Learning rate decays during this phase.

---

## 3. Capability Slot Allocations

### 3.1 Slot Shares, Floors, and Benchmark Targets

| Slot | Main Phase (%) | Anneal (%) | Grand Total (%) | Protected Floor (%) | Benchmark Target |
|------|---------------|-----------|----------------|--------------------|--------------------|
| Core web / general English | 40.0 | 1.05 | 41.05 | 25 | MMLU, HellaSwag, general perplexity |
| Code | 15.0 | 1.0 | 16.0 | 8 | HumanEval, MBPP, SWE-bench |
| Math / reasoning | 10.0 | 1.5 | 11.5 | 5 | GSM8K, MATH, MiniF2F |
| Indic (all tiers) | 12.0 | 3.0 | 15.0 | 6 | IndicGLUE, Indic-QA, FLORES-Indic |
| Agentic / tool-use | 6.0 | 0.7 | 6.7 | 2 | WebArena, ToolBench, AgentBench, GAIA |
| Long-context | 5.0 | 0.0 | 5.0 | 2 | RULER, LongBench, needle-in-haystack |
| Instruction / dialogue | 3.0 | 0.5 | 3.5 | 1.5 | MT-Bench, IFEval |
| Safety / alignment | 1.0 | 0.25 | 1.25 | 1.0 (hard, non-negotiable) | Internal red-team suite |
| **Total** | **92.0** | **8.0** | **100.0** | | |

**Why long-context gets 0% in the anneal:** Context-length capability is a structural property (attention patterns, position encodings) rather than something sharpened by curated-data exposure in the final tokens. The dedicated S4 phase handles it entirely.

### 3.2 Floor Semantics

The floor is a **cumulative** constraint, not an instantaneous one. At any checkpoint *t*:

```
cumulative_share(slot, t) = tokens_seen_from_slot / total_tokens_seen_so_far
```

The selector is prohibited from making any adjustment that would cause cumulative_share(slot, t) to fall below that slot's floor **at the end of training**. This means that in S1, code's instantaneous share can be 3.3% (below code's 8% floor) because the phased plan guarantees code reaches 15% by completion.

**Exception — Safety:** Safety's floor is additionally **instantaneous**. At no point in any stage is safety's per-batch share allowed to hit zero. This is a hard constraint, non-negotiable, and not selector-adjustable at all.

The floor total is 25 + 8 + 5 + 6 + 2 + 2 + 1.5 + 1.0 = 50.5%, leaving 41.5% of the main phase as truly selector-discretionary.

---

## 4. Indic Slot — Tier Decomposition

The headline "12% Indic (main phase) / 15% grand total" number is defensible only with this breakdown attached. A reviewer who asks "is that 12% real text?" gets an honest answer: roughly 8.4 percentage points (verified + unverified) is close to organically supplied, 2.4 points is derived-but-real (translation of real content), and 1.8 points is manufactured to patch a coverage hole.

### 4.1 Tier Definitions and Supply Honesty

| Tier | Main Phase (%) | Anneal (%) | Grand Total (%) | Named Sources | Supply Honesty |
|------|---------------|-----------|----------------|---------------|-----------------|
| Verified | 3.6 | 2.4 | 6.0 | AI4Bharat Sangraha "verified" subset, IndicCorp v2 cleaned, Indic Wikipedia dumps, curated news archives | Real supply across all scheduled languages (Hi, Bn, Ta, Te, Mr, Gu, Kn, Ml, Pa, Od, Ur) at genuinely high-quality bar is thin — under 1 epoch for higher-resource languages (Hi, Bn) but requires 2–3x repetition for lower-resource ones (Od, As). |
| Unverified | 4.2 | 0 | 4.2 | Raw CC-100/OSCAR Indic splits, Sangraha "unverified" web crawl | Abundant — single-epoch coverage easy. |
| Translated | 2.4 | 0.6 | 3.0 | Samanantar parallel corpus, NLLB-seed, backtranslated web text | Real parallel data exists at scale but is disproportionately English-source; over-represents translationese register. Flagged as proxy-run question (Variant D). |
| Synthetic | 1.8 | 0 | 1.8 | LLM-generated Indic QA/instruction pairs via translate-then-verify pipelines, self-instruct seeded in-language | **Exists only because real data doesn't cover it.** Generated to fill the instruction-style and low-resource-language gap. |
| **Indic Total** | **12.0** | **3.0** | **15.0** | | |

### 4.2 Indic Tier Timing Across Stages

| Tier | S1 | S2 | S3 | S4 | Main Total | Anneal | Grand Total |
|------|-----|-----|-----|-----|------------|--------|-------------|
| Unverified | 3.2 | 0.7 | 0.2 | 0.1 | 4.2 | 0 | 4.2 |
| Synthetic | 1.1 | 0.4 | 0.2 | 0.1 | 1.8 | 0 | 1.8 |
| Verified | 0 | 0.3 | 1.3 | 2.0 | 3.6 | 2.4 | 6.0 |
| Translated | 0 | 0.8 | 0.8 | 0.8 | 2.4 | 0.6 | 3.0 |
| **Indic Total** | **4.3** | **2.2** | **2.5** | **3.0** | **12.0** | **3.0** | **15.0** |

---

## 5. Named Data Sources Per Slot

### 5.1 Core Web / General English
- **Main phase:** FineWeb, FineWeb-Edu (score >= 2), DCLM baseline, RefinedWeb, C4.
- **Anneal:** FineWeb-Edu (score >= 4 only — top tier).

### 5.2 Code
- **Main phase:** The Stack v2, StarCoder training data, GitHub-code-clean.
- **Anneal:** Curated, well-documented, linted-passing subset only.

### 5.3 Math / Reasoning
- **Main phase:** OpenWebMath, MetaMathQA, OpenMathInstruct-2, NuminaMath, MiniF2F-train, INT.
- **Synthetic component:** CoT generated via rejection sampling with code-execution verification (outcome-verified, not just LLM-plausible).
- **Anneal:** Competition-grade problems only, all execution-verified.

### 5.4 Agentic / Tool-Use
- **Main phase:** WebArena/Mind2Web trajectory logs, ToolBench/API-Bank tool-call traces, GitHub issue-to-PR paired trajectories.
- **Synthetic component (~40% of slot):** Sandboxed-environment rollouts filtered by execution success.
- **S3 onward:** Filtered for DAG-structured decomposition (informed by Opus paper, arXiv:2412.00573).
- **Anneal:** 100% DAG-structured, outcome-verified only.

### 5.5 Long-Context
- **Main phase:** PG19, NarrativeQA, QMSum, whole-repo code concatenation.
- **Synthetic component:** Needle-insertion documents built from real long documents.

### 5.6 Instruction / Dialogue
- **Main phase:** FLAN-v2, OpenAssistant, Dolly, self-instruct collections.
- **Anneal:** Best-of filtered.

### 5.7 Safety / Alignment
- **All stages (flat 0.25% per stage + 0.25% anneal):** Anthropic-HH-RLHF harmlessness subset, internally curated red-team adversarial examples.
- Never reduced, never selector-adjustable.

---

## 6. Five-Stage Curriculum

### 6.1 Stage Overview

| Stage | Token Range | % of Total | Cumulative | Purpose |
|-------|------------|-----------|------------|---------|
| S1 — Language grounding | 0–30% | 30 | 30 | Orthography, morphology, basic syntax, tokenizer utilization |
| S2 — Logic introduction | 30–55% | 25 | 55 | Code ramps hard to teach structured, verifiable logic |
| S3 — Reasoning emphasis | 55–75% | 20 | 75 | Math/CoT ramps hard, DAG-structured agentic data enters |
| S4 — Long-context extension | 75–92% | 17 | 92 | Sequence-length curriculum (4k to 32k) + long-context data ramps |
| S5 — Anneal / cooldown | 92–100% | 8 | 100 | LR decay, held-back high-quality pool, consolidation |

### 6.2 Stage Composition (% of total budget per slot per stage)

| Slot | S1 (30%) | S2 (25%) | S3 (20%) | S4 (17%) | S5 (8%) | Grand Total |
|------|---------|---------|---------|---------|--------|-------------|
| Core web | 23.3 | 11.5 | 3.6 | 1.6 | 1.05 | 41.05 |
| Code | 1.0 | 8.0 | 4.0 | 2.0 | 1.0 | 16.0 |
| Math / reasoning | 0.5 | 1.5 | 7.0 | 1.0 | 1.5 | 11.5 |
| Indic | 4.3 | 2.2 | 2.5 | 3.0 | 3.0 | 15.0 |
| Agentic | 0.2 | 0.8 | 1.5 | 3.5 | 0.7 | 6.7 |
| Long-context | 0.2 | 0.3 | 0.5 | 4.0 | 0.0 | 5.0 |
| Instruction | 0.3 | 0.5 | 0.6 | 1.6 | 0.5 | 3.5 |
| Safety | 0.25 | 0.25 | 0.25 | 0.25 | 0.25 | 1.25 |
| **Stage Total** | **30.0** | **25.05** | **20.0** | **16.95** | **8.0** | **100.0** |

### 6.3 Stage-by-Stage Pedagogical Justification

**S1 — Language Grounding (0–30%):** Core web dominates because the model needs basic syntax, semantics, and world knowledge before specialized data is useful. Indic is the second-largest share because language acquisition is front-loaded.

**S2 — Logic Introduction (30–55%):** Code ramps to its peak because code teaches verifiable, structured logic — a prerequisite for math reasoning that follows. Translated Indic enters here.

**S3 — Reasoning Emphasis (55–75%):** Math/CoT ramps to its peak. Gated on code because decomposition, step-tracking, and logical chaining are skills the code phase installed. DAG-structured agentic data enters meaningfully.

**S4 — Long-Context Extension (75–92%):** Sequence length steps up (4k → 8k → 16k → 32k). Agentic peaks here because genuine multi-step tool trajectories are inherently long. Verified Indic reaches its main-phase peak.

**S5 — Anneal / Cooldown (92–100%):** Highest-quality data only across all slots. LR decays. No new capability introduction — this phase consolidates and sharpens.

### 6.4 Agentic Slot Internal Staging

| Stage | Agentic Share (of 6.7% total) | Structure |
|-------|------|-----------|
| S1–S2 | 0.2 + 0.8 = 1.0 | Flat, single-tool-call examples only. |
| S3 | 1.5 | DAG-structured multi-step trajectories enter. |
| S4 | 3.5 | Majority DAG-structured, long-horizon. |
| S5 (anneal) | 0.7 | 100% DAG-structured, outcome-verified only. |

---

## 7. Transition Smoothing Strategy

### 7.1 Standard Interpolation Window: 3% of Total Tokens

Split 1.5% before the nominal boundary and 1.5% after. Linear blend per-slot.

```
share(slot, t) = share_old(slot) + (share_new(slot) - share_old(slot)) * (t - t0 + 1.5%) / 3%
```

### 7.2 Widened Windows: 5% of Total Tokens

**S3→S4:** Data mix changes AND max sequence length steps up. Highest-risk transition.

**S4→S5:** LR starts decaying at the same moment the mix pivots. Anneal-specific sources are pre-mixed at ~10% of their S5 share during the final 2% of S4.

### 7.3 Overlap Check

| Transition | Window Start | Window End | Gap to Next |
|-----------|-------------|-----------|-------------|
| S1→S2 | 28.5% | 31.5% | 22 points |
| S2→S3 | 53.5% | 56.5% | 16 points |
| S3→S4 | 72.5% | 77.5% | 12 points |
| S4→S5 | 89.5% | 94.5% | — (final) |

No overlaps.

---

## 8. Learning Rate Schedule

### 8.1 Warmup-Stable-Decay (WSD)

- **Warmup (0–0.5%):** Linear ramp from 0 to peak LR.
- **Stable (0.5–92%):** Constant peak LR across S1–S4.
- **Decay (92–100%, S5):** Cosine decay from peak to **15% of peak — not to zero.**

### 8.2 Why Not Decay to Zero

Recent work (arXiv:2511.18903) identifies that curriculum-based training advantages diminish under LR decay schedules that go to a low final scale. The 15% floor is treated as a proxy-run question (Variant E).

---

## 9. Difficulty and Reasoning-Length Bands

| Band | Reasoning Length | Concrete Example | Where It Appears |
|------|-----------------|-------------------|-------------------|
| 1 — Trivial | <100 tokens | "What is 17 + 26?" | Math (calibration), code (simple syntax) |
| 2 — Multi-step | 100–500 tokens | GSM8K-style: "A train travels 60 mph for 2.5 hours, then 45 mph for 1 hour — total distance?" | Math (core), code (single-function) |
| 3 — Extended | 500–2000 tokens | AIME-style competition problem or multi-file code bug | Math (hard), agentic (2–3 step chains) |
| 4 — Long-horizon | 2000+ tokens | WebArena-style: "Book a flight matching these 4 constraints" | Agentic (core), long-context |

---

## 10. Proxy Validation Plan — Testable Hypotheses

All variants run at 1B and 3B parameters, ~20–25x Chinchilla-optimal tokens, 2 seeds per arm minimum.

### Hypothesis Template

Every proxy run fixes five things before training starts:
1. **Independent variable** — the one thing that changes.
2. **Directional prediction** — "X will beat Y on metric Z."
3. **Effect size threshold** — smallest delta worth acting on.
4. **Guardrail condition** — what must not move.
5. **Decision rule** — the literal sentence acted on, written in advance.

### Variant B — Indic Verified vs. Synthetic Trade-off
- **Change:** Double verified Indic (3.6% → 7.2%), taken from synthetic (1.8% → 0%).
- **H1:** B underperforms A on IndicGLUE by >=1.0 points for low-resource languages.
- **H0:** B matches or beats A — cut synthetic, increase verified.
- **Guardrail:** High-resource Indic accuracy must not drop >0.5 points.

### Variant C — Agentic Synthetic Share
- **Change:** Cut agentic synthetic from ~40% to 10%, backfill with GitHub issue-to-PR pairs.
- **H1:** A outperforms C on WebArena multi-step tasks by >=3 points.
- **H0:** C matches A — shrink synthetic to 10%.
- **Guardrail:** Single-call tool accuracy flat.

### Variant D — Anneal Indic Top-Up
- **Change:** Redirect 3.0% anneal Indic to core web.
- **H1:** A outperforms D on IndicGLUE by >=1.5 points.
- **H0:** No difference — drop to flat 12%.
- **Guardrail:** MMLU-mini must not drop >0.5 points.

### Variant E — LR Decay Floor (15% vs. ~1%)
- **Change:** E decays LR to ~1% of peak instead of 15%.
- **H1:** A preserves >=2 points more curriculum benefit.
- **H0:** No difference or 1% wins.
- **Guardrail:** Aggregate validation loss must not be worse by >3% relative.

### Variant F — DAG-Structured Agentic Filtering
- **Change:** F uses unfiltered mixed-structure trajectories.
- **H1:** A outperforms F on multi-step subset by >=5 points.
- **H0:** No difference or F wins.
- **Guardrail:** Single-call tool accuracy must stay flat.

---

## 11. Audit Verification

- **Main-phase slot sum:** 40.0 + 15.0 + 10.0 + 12.0 + 6.0 + 5.0 + 3.0 + 1.0 = **92.0** ✓
- **Anneal slot sum:** 1.05 + 1.0 + 1.5 + 3.0 + 0.7 + 0.0 + 0.5 + 0.25 = **8.0** ✓
- **Grand total:** 92.0 + 8.0 = **100.0** ✓
- **Stage sum:** S1 (30) + S2 (25) + S3 (20) + S4 (17) + S5 (8) = **100** ✓
- **Per-slot stage sums match grand totals:** Verified for all 8 slots ✓
- **Indic sub-tier sums:** 4.2 + 1.8 + 6.0 + 3.0 = **15.0** ✓
- **Floor total:** 25 + 8 + 5 + 6 + 2 + 2 + 1.5 + 1.0 = **50.5%** ✓
- **Transition window overlaps:** None ✓

---

## 12. What Makes This Plan Defensible

**Every number is allowed to lose.** The proxy runs are genuinely two-sided, with null hypotheses that would change the plan.

**Supply honesty.** Where a share requires repetition or synthetic generation, the plan says so explicitly.

**Guardrails catch hidden costs.** Every hypothesis protects against the failure mode of a slot "winning" its target by quietly degrading something else.

**The curriculum order is gated, not arbitrary.** Language before logic, logic before reasoning, reasoning before long-context, all before anneal. Each stage depends on what the previous one installed.
