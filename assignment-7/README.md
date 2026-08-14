# Reversible Kronecker Embeddings (RKE)

**An exactly invertible byte-level embedding, used as both the input layer and the output head.**

Kronecker Embeddings ([arXiv:2605.29459](https://arxiv.org/abs/2605.29459)) replace a transformer's
$|V| \times d_{\text{model}}$ lookup table with a deterministic byte codec plus one learned projection,
cutting input-side parameters by 94%. The method is not invertible: given an embedding you cannot recover
the bytes, which rules out weight tying and any byte-level output.

This work adds exact, closed-form invertibility at the same complexity, proves it, verifies it numerically,
and derives a tied output head from it. Input + output parameters go from **1073.8 M to 0.426 M**, and the
head becomes a gather instead of a matmul (**455× fewer multiply-accumulates**).

Nothing here is validated by training. Every number below is either a proof about a linear map or a
measurement of an untrained implementation. See [Status](#status).

---

## Contents

| File | What it is |
|---|---|
| `README.md` | This brief |
| `reversible-kronecker-embeddings.html` | Construction and proofs, with six measured figures |
| `training-strategy.html` | Phased training plan, parameter groups, gates, diagnostics |
| `reversible_kronecker.py` | Reference implementation: codec, exact decoder, ETF codebook, spectral mixer, round-trip self-test |

```bash
python reversible_kronecker.py     # self-test at d = 768, 2048, 4096
```

---

## 1. Where the information is actually lost

The published pipeline is

$$b \;\longrightarrow\; \kappa(b) \in \mathbb{R}^{8192} \;\longrightarrow\; \text{z-norm} \;\longrightarrow\; \kappa(b)\,\mathbf{W}_{\text{proj}} \in \mathbb{R}^{4096}$$

with

$$\kappa(b) \;=\; \frac{1}{\sqrt L}\sum_{p=1}^{L}\mathbf{c}_{b_p}\otimes\mathbf{p}_p, \qquad D = 256 \cdot P.$$

**Lemma 1.** $\kappa$ is injective. Under the index map $\iota(v,p) = v\cdot P + p$, the term
$\mathbf{c}_v \otimes \mathbf{p}_p$ *is* the basis vector $\mathbf{e}_{\iota(v,p)}$. Group coordinates into
position slots $\Sigma_p = \{\iota(v,p) : v \in \mathcal{B}\}$; slot $p$ receives a contribution from byte
position $p$ and nothing else. Then $\|\kappa(b)\|_2 = 1$ exactly, $L$ is the count of nonzero slots, and
$b_p = \arg\max_v \kappa(b)[\iota(v,p)]$.

The z-normalisation is also invertible on the codec image: its mean and variance are functions of $L$ alone
($m = \sqrt{L}/D$, $\sigma^2 = (1 - L/D)/D$), and an increasing affine map preserves both the argmax and the
zero/nonzero slot pattern.

**So reversibility is lost in exactly one place: the dense trained map $\mathbb{R}^{8192} \to \mathbb{R}^{4096}$.**
The byte encoding does not need redesigning. One matrix needs structure.

### The dead end

**Proposition 2.** For Lebesgue-almost every $\mathbf{W} \in \mathbb{R}^{D \times d}$, the composite is injective
on the finite set of admissible byte strings. *(The set of failing $\mathbf{W}$ is a finite union of
$(D-1)d$-dimensional subspaces, hence null.)*

This is true and useless. No closed form (recovery becomes sparse reconstruction — OMP or $\ell_1$, iterative
and conditionally correct). No margin (injectivity in exact arithmetic says nothing about bf16). Not preserved
by training (nothing in the loss keeps $\mathbf{W}$ away from the null set; that the original method needs a
self-entropy regulariser against subspace collapse suggests the drift is real).

We want a structural guarantee that is an invariant of training.

---

## 2. Construction

$$\boxed{\;\mathbf{e}(b) \;=\; \operatorname{flat}\!\big(T\,Z(b)\big)\,Q, \qquad Z(b)_{p,:} = \tfrac{1}{\sqrt L}\,g_{b_p}\;}$$

| Component | Shape | Constraint | Role |
|---|---|---|---|
| $G$ codebook | $256 \times s$ | rows unit-norm, pairwise distinct | byte identity |
| $T$ position mixer | $P \times P$ | $F^{*}\operatorname{diag}(w)F$, $w > 0$ | position kernel |
| $Q$ output mixer | $d \times d$ | orthogonal (Householder product) | cross-slot entangling |

with $d = P \cdot s$. Decoding runs the same path backwards: apply $Q^{\top}$, apply $T^{-1}$, read $L$ from
the slot norms, take a 256-way argmax per slot.

The whole design space is one idea: **per-position contributions must land in mutually orthogonal subspaces,
with each per-position map injective and equal-norm.** $T$ and $Q$ are invertible additions on top that restore
expressivity without costing anything.

### Why a 256-way codebook and not a bit code

A cheaper instantiation exists: give each of the 8 bits of each byte an orthonormal direction and read signs.
It needs only $d \ge 8P$ and has zero coherence. Rejected — a case flip is a single bit, so `run` / `RUN` would
land at cosine 0.75, and case separation is one of the properties byte-level embedding exists to preserve. Byte
values are categorical, not ordinal: `0x7F` and `0x80` are adjacent integers and unrelated characters.

---

## 3. Theorems

**Theorem 3 (exact reverse determinism).** If every row of $G$ has norm $r$ and no two rows are equal, $T$ is
invertible and $Q$ is orthogonal, then $\mathcal{R}(\mathbf{e}(b)) = b$ for every admissible $b$. Distinctness
is also *necessary*.

> *Proof.* $QQ^{\top} = I$ returns $TZ(b)$ exactly; $T^{-1}$ returns $Z(b)$ exactly. Equal norms give
> $\|Z_{p,:}\| = r/\sqrt{L} > 0$ for $p \le L$ and $0$ beyond, so $\hat L = L$. For bytes,
> $\langle Z_{p,:}, g_v\rangle = \tfrac{1}{\sqrt L}\langle g_{b_p}, g_v\rangle \le r^2/\sqrt{L}$ by
> Cauchy–Schwarz, with equality iff $g_v = \lambda g_{b_p}$, $\lambda \ge 0$; equal norms force $\lambda = 1$
> and distinctness forces $v = b_p$. Necessity: if $g_u = g_v$, two strings differing only at one position by
> $u$ vs $v$ have identical $\mathbf{e}$. ∎

The proof invokes no RIP constant, no incoherence assumption, no sparsity threshold, no probability. **Unit rows
and distinct rows suffice** — and both are enforceable by projection after every optimiser step.

**Theorem 4 (minimal width).** A valid codebook exists iff $s \ge 2$. Exact reversibility is available whenever
$d_{\text{model}} \ge 2P$ — 64 dimensions at $P = 32$ — independently of $|V|$ and $D$. *(For $s = 1$,
$rS^0 = \{-r, +r\}$ has two elements, so 256 distinct equal-norm rows cannot exist.)*

So $s$ does not decide *whether* inversion works. It decides how robustly.

**Theorem 5 (decoding margin).** With coherence $\mu = \max_{u \ne v}\langle g_u, g_v\rangle / r^2$, decoding of
$\hat{\mathbf{e}} = \mathbf{e}(b) + \eta$ is exact whenever

$$\|\eta\|_2 \;<\; \frac{r\,(1-\mu)}{2\sqrt{L}\;\|T^{-1}\|_2}.$$

> *Proof sketch.* After both undo steps $\hat Z = Z(b) + E$ with $\|E\|_F \le \|T^{-1}\|_2 \|\eta\|_2$. Length
> recovery survives below $r/(2\sqrt L)$. For bytes,
> $\langle \hat Z_{p,:},\, g_{b_p} - g_v\rangle \ge r^2(1-\mu)/\sqrt{L} - 2r\varepsilon_p$, strictly positive
> under the hypothesis. ∎

**Theorem 6 (Welch ceiling).** $\mu \ge \sqrt{(256-s)/(255s)}$, equality iff the rows form an equiangular tight
frame.

**Theorem 7 (distortion floor).** Preserving the original codec's inner products exactly requires 256 mutually
orthogonal $g_v$, i.e. $s \ge 256$, i.e. $d \ge 256P = D$ — the uncompressed case. **Any compression incurs
distortion, and the distortion is exactly $\mu$.** An ETF codebook is therefore the minimum-distortion
reversible compression; everything else is a choice of where to sit on that curve.

General inner-product formula (from orthogonality of $Q$):

$$\langle\mathbf{e}(b), \mathbf{e}(b')\rangle = \frac{1}{\sqrt{LL'}}\sum_{p,q} R_{pq}\,\langle g_{b_p}, g_{b'_q}\rangle, \qquad R = T^{\top}T.$$

---

## 4. The position mixer: one vector, two roles

Take $T$ circulant, $T = F^{*}\operatorname{diag}(w)F$. Then $R = T^{\top}T$ is circulant, so
$R_{pq} = \rho(p-q)$ with

$$\rho(\delta) \;=\; \frac{1}{P}\sum_{k=0}^{P-1} w_k^2\,\omega^{k\delta} \qquad\text{i.e.}\qquad \rho = \mathcal{F}^{-1}(w^2).$$

**Corollary 8 (shift response).** For $b$ with distinct bytes and $b'$ its shift by $\delta$:
$\cos(\mathbf{e}(b), \mathbf{e}(b')) = \rho(\delta)/\rho(0)$.

The original codec is the flat case $w \equiv 1 \Rightarrow \rho = \delta_0$, which makes `run` and `␣run`
**exactly orthogonal** — and space-prefixed duplicates are the most saturated pattern in a BPE vocabulary. It
does the same to the shared `-tion` in `nation` (positions 3–5) and `creation` (positions 5–7), which is the
original paper's own stated limitation.

The only thing invertibility asks of $w$ is **positivity**. So the spectral weights are simultaneously the
invertibility condition and the geometry knob. Normalising $\max_k w_k = 1$, the condition number
$\chi = 1/\min_k w_k$ is simultaneously the numerical conditioning and the margin-shrink factor of Theorem 5.
One knob, both roles, no hidden trade-off. Default $\chi = 8$.

Verified numerically ($\chi = 8$):

| $\delta$ | theory $\rho(\delta)/\rho(0)$ | measured |
|---|---|---|
| 0 | 1.000 | 1.000 |
| 1 | 0.680 | 0.658 |
| 2 | 0.228 | 0.206 |
| 3 | 0.031 | 0.037 |

The small gap is expected: a real prefix is a *linear* shift, not the cyclic shift the theory assumes, so the
shifted string is also one byte longer.

---

## 5. The tied output head

Tying means the head is the **adjoint** of the encoder, not its inverse. Since
$\mathbf{e} = \operatorname{flat}(TAG)Q$ is linear in the occupancy matrix $A \in \mathbb{R}^{P \times 256}$:

$$\langle \mathbf{e}(A), \mathbf{h}\rangle = \langle TAG, H\rangle_F = \langle A,\; \underbrace{T H G^{\top}}_{S}\rangle_F, \qquad H = \operatorname{unflat}(\mathbf{h}Q^{\top})$$

One score matrix $S \in \mathbb{R}^{P \times 256}$ serves the whole vocabulary; each token's logit is a
gather-and-sum over its own bytes:

$$\text{logit}(t) \;=\; \frac{\alpha}{\sqrt{L_t}}\sum_{p=1}^{L_t} S\big[p,\, b^{(t)}_p\big] \;+\; \beta_t$$

> **The head does *not* factorise into per-slot byte distributions.** It scores whole tokens and softmaxes over
> $V$, which is exactly an ordinary tied head. The byte structure only makes the logits cheap to compute; the
> independence assumption $P(b) = \prod_p P(b_p)$ never enters the loss. That assumption applies only to
> open-vocabulary byte generation — deliberately out of scope.

`T` uses the transpose, not the inverse, so no conditioning penalty is paid at the head.

### Three things the tying breaks, and the fixes

1. **No unigram prior.** With an equal-norm codebook, $\|\mathbf{e}(t)\|$ is nearly constant (measured std 2.3%
   of the mean). A free table encodes frequency in row norm; this one structurally cannot. Fix: the per-token
   bias $\beta_t$, initialised to $\log \hat p(t)$ from corpus counts. **Not optional.**
2. **Rank is set by byte length, not by $d$.** The span is
   $\operatorname{span}\{T\mathbf{e}_p\}_{p \le L_{\max}} \otimes \operatorname{span}(G)$, so
   $\operatorname{rank} = \min(d,\; L_{\max}\!\cdot s)$. Measured 1764 of 4096 on a set capped at 20 bytes with
   $s = 128$. **Design rule: set $P = L_{\max}$ of the actual tokenizer.** Setting $P$ larger wastes softmax
   capacity — the opposite of the natural instinct.
3. **Byte-similar tokens are hard to separate.** On the input side the body disambiguates `compute` / `commute`
   downstream; at the head there is no downstream. Worst near-duplicate pair measured at cosine 0.954, leaving
   31% of the embedding norm as logit-gap budget. Fix if binding: untie $G_{\text{out}}$ (+33 K params) or add a
   rank-$k$ per-token residual ($k \approx 8$, 1 M params).

---

## 6. Measurements

All on uniformly random byte strings of length 1–32 — the full admissible domain, not a curated vocabulary.

### Round-trip exactness

| $d_{\text{model}}$ | $s$ | $\mu$ | Welch floor | fp32 | fp16 | bf16 | params |
|---|---|---|---|---|---|---|---|
| 768 | 24 | 0.319 | 0.195 | 10000/10000 | 10000/10000 | 10000/10000 | 0.055 M |
| 2048 | 64 | 0.161 | 0.109 | 10000/10000 | 10000/10000 | 10000/10000 | 0.148 M |
| **4096** | **128** | **0.093** | **0.063** | **10000/10000** | **10000/10000** | **10000/10000** | **0.295 M** |

Exactness in **bf16** is the result that matters — that is the precision at which a generic trained projection
would be hopeless, and the precision models are actually trained in.

### Codebook: offline optimisation captures nearly all the headroom

| $s$ | random init $\mu$ | offline-optimised | Welch ceiling | margin $1-\mu$ |
|---|---|---|---|---|
| 24 | 0.720 | 0.319 | 0.195 | 0.280 → **0.681** (of 0.805) |
| 128 | 0.352 | 0.093 | 0.063 | 0.648 → **0.907** (of 0.937) |

93% of available headroom at $s = 128$, with **no task loss and no data**. This is why $G$ is frozen (§7).

### Surface-form geometry

| Pair | Kind | Original codec | RKE ($\chi=8$, $d=4096$) |
|---|---|---|---|
| separate / seperate | substitution | 0.875 | 0.867 |
| realize / realise | substitution | 0.857 | 0.871 |
| compute / commute | substitution | 0.857 | 0.863 |
| receive / recieve | transposition | 0.714 | **0.921** |
| mistake / mistkae | transposition | 0.714 | **0.897** |
| color / colour | insertion | 0.730 | **0.862** |
| nation / creation | suffix offset | **0.000** | **0.153** |
| run / ␣run | space prefix | **0.000** | **0.621** |
| run / RUN | case | 0.000 | −0.070 |

The two exact zeros are structural defects of a position-hard kernel. Both are repaired; case separation
survives; `compute` / `commute` stays high, which is inherent to surface-form encoding and is *not* fixed here.

### Robustness (Theorem 5 vs empirical)

| $L$ | guaranteed $\|\eta\|_2$ | empirical breakdown | ratio |
|---|---|---|---|
| 2 | 0.0401 | 0.352 | 9× |
| 8 | 0.0200 | 0.190 | 10× |
| 16 | 0.0142 | 0.132 | 9× |
| 32 | 0.0100 | 0.311 | 31× |

Signal $\|\mathbf{e}\| \approx 0.590$. The bound is worst-case, so ~10× conservative. The $L = 32$ uptick is
real: at full length every slot is occupied, so the length-thresholding step has nothing to get wrong.

### Head cost

| | Ops per predicted position |
|---|---|
| Standard untied head ($V \times d$ matmul) | 536.9 M MAC |
| RKE score matrix ($T H G^{\top}$) | 1.18 M MAC |
| RKE vocabulary gather ($V \times \bar L$, $\bar L = 6.2$) | 0.82 M adds |

**455× fewer MACs.** Fast-path logits verified against $\langle\mathbf{e}(t), \mathbf{h}\rangle$ over ~18k
tokens: max absolute error $9.9\times10^{-8}$ in fp32, correlation 1.0000000000.

The gather is memory-bandwidth bound, so expect the head to stop being the bottleneck rather than to become free.

### Parameter budget ($V = 131{,}072$, $d = 4096$, $P = 32$)

| Component | Params |
|---|---|
| $G$ codebook (frozen) | 0.033 M |
| $w$ spectral weights, $\alpha$ | ~0 |
| $Q$ (64 Householder reflections) | 0.262 M |
| $\beta_t$ per-token bias | 0.131 M |
| **Total input + output** | **0.426 M** |
| *vs* standard BPE, untied | 1073.8 M |

---

## 7. Training strategy

Full plan in `training-strategy.html`. Summary below.

### Settled before any run

- **$G$ is frozen** at an offline ETF construction. Theorem 6 makes the optimum a property of $(256, s)$ alone;
  §6 shows offline optimisation captures 93% of the headroom. Since $Q$ is orthogonal it cannot alter *relative*
  angles between byte directions, so training $G$ means letting the loss decide which bytes are similar — the
  co-occurrence collapse byte-level embedding exists to avoid.
- **The head scores whole tokens.** No factorised distribution, no independence assumption.

Trained: transformer body, $Q$, $w$, $\alpha$, $\beta_t$.

### Parameter groups

| Group | Params | Trainable | LR | Weight decay | Note |
|---|---|---|---|---|---|
| $G$ codebook | 32,768 | frozen | — | — | unfreeze only as an ablation |
| $w$ spectral | 32 | yes | ×1 | **none** | decay drives $w \to w_{\min}$, raising $\chi$, shrinking the margin |
| $Q$ Householder | 262,144 | yes | ×1 | **none** | vectors normalised before use |
| $\beta_t$ token bias | 131,072 | yes | **×3 – ×10** | none | the only sparse group |
| $\alpha$ logit scale | 1 | yes | ×1 | none | calibrate at init |
| Body | — | yes | ×1 | standard | unchanged |

**Gradient regimes.** RKE moves the whole geometry out of the sparse-gradient regime. At a 500 k-token batch
with $\bar L = 6.2$: $G$ rows see ~24 k gradient contributions per parameter per step, $w$ and $Q$ ~$10^6$,
alongside the body at $5 \times 10^5$. A median BPE row under Zipf sees ~0.35. Rare-token undertraining simply
does not arise for the geometry.

**The exception is $\beta_t$** — one scalar per vocabulary entry, so it inherits exactly the Zipf sparsity the
rest of the design escapes (~4 contributions per parameter per step at the mean, far fewer at the tail).

> **Pre-empted mistake.** A large effective batch means *low* gradient variance, which conventionally licenses a
> **higher** learning rate, not a lower one — and under Adam magnitude is normalised away regardless. Do not
> reduce LR on $G$, $w$ or $Q$ on the theory that they "see too much data." If instability appears there, the
> cause is almost certainly the manifold constraint fighting the optimiser state.

**Riemannian handling.** $Q$ lives on $O(d)$. Renormalising after an Adam step while leaving momentum in the
ambient space means the optimiser accumulates velocity the projection then silently discards. Project momentum
onto the tangent space before the update. This is a bug class, not a hyperparameter — it presents as unexplained
early instability.

### Initialisation

- $\beta_t \leftarrow \log \hat p(t)$ from corpus token counts.
- $\alpha$ such that initial logit std matches a standard head; verify first-step loss ≈ unigram entropy, not
  $\log|V|$.
- $Q$ random orthogonal (or identity for a diagnostic-friendly first run — with $Q = I$ slots are axis-aligned
  and every intermediate tensor is interpretable).
- $w$ to the taper giving $\chi \approx 8$. $\chi = 1$ reproduces the original hard-slot codec and is the
  control arm, not the default.

### Implementation

Precompute a `uint8` byte table of shape $(V, P)$ plus lengths — 4.2 MB, replacing a 537 M-parameter matrix.

```python
# forward, per predicted position
H     = (h @ Q.T).view(P, s)          # undo the orthogonal mixer
S     = (T @ H) @ G.T                 # (P, 256) — the whole head
flat  = S.reshape(-1)                 # P*256 = 8192 entries
idx   = byte_table * P + arange(P)    # (V, P) precomputed
logit = (flat[idx] * mask).sum(-1) * alpha / sqrt(L) + beta
```

`T` is circulant — apply as an FFT pair, $O(P \log P)$, never materialise it. `Q` is a Householder product —
apply reflection-by-reflection, $O(md)$. Keep $S$, the gather-sum, and CE in fp32; master weights fp32. Backward
through the gather is a scatter-add into $S$.

### Gates

Ordered so the experiment most likely to invalidate everything runs early and cheap.

| Gate | Test | Criterion | Cost |
|---|---|---|---|
| **0** | Full-vocabulary round-trip in bf16; truncation collision audit | 100.000% exact, zero collisions | no compute |
| **1** | Overfit a small memorisation set | training loss < 0.05; trained encoder still round-trips | hours |
| **2** | **Arm C vs arm B, 3 seeds, 124 M** | **C within ~0.02 nats of B** | **12 runs** |
| **3** | Sweep $\chi \in \{1,2,4,8,32\}$ | some $\chi > 1$ beats $\chi = 1$, else drop $T$ | sweeps |
| **4** | Spelling-perturbation protocol | ≥ published (top-1 55.5%, mean KL 0.79) | 1 large run |

**Phase 2 arms:**

| Arm | Input | Head | Isolates |
|---|---|---|---|
| A | BPE table, tied | tied | baseline |
| B | Kronecker, dense $\mathbf{W}_{\text{proj}}$ | untied standard | the published result |
| **C** | **RKE, $T = I$, $Q = I$** | untied standard | **cost of block-orthogonality alone** |
| D | RKE full | tied to $G$ | cost of tying the head |

**C vs B is the load-bearing comparison.** It changes exactly one thing: a dense trained projection becomes a
block-orthogonal constrained one. Everything else — $T$, $Q$, the tied head — is an invertible addition on top
of C and cannot remove information. If C loses materially, the correct next question is how much of the block
constraint can be relaxed while keeping a closed-form inverse, not how to tune around it.

### Diagnostics beyond a normal run

| Metric | Healthy | Meaning if it drifts |
|---|---|---|
| Round-trip exactness on held-out vocab, in training precision | 100% | **the canary** — invertibility invariant violated somewhere in the optimiser path |
| $\chi = \max_k w_k / \min_k w_k$ | ≤ 8 | softplus floor not binding; margin shrinking |
| $\|Q^{\top}Q - I\|_F$ | < 1e-5 | Householder parameterisation drifted; decoder no longer exact |
| Spread of $\|\mathbf{e}(t)\|$ | std < 5% | $w$ creating length/repetition-dependent norm, leaking into logits |
| Top singular values of induced table | ≈ $\min(d, L_{\max}s)$ | collapse = softmax bottleneck arriving |
| Confusion mass on high-cosine pairs | flat | byte-similarity limit binding |
| $\beta_t$ vs empirical $\log\hat p(t)$ | correlated | $\beta$ absorbing gradient that belongs to the geometry |

### Failure modes

| Symptom | Cause | Action |
|---|---|---|
| Loss starts at $\log\|V\|$ and plateaus | $\alpha$ miscalibrated or $\beta_t$ uninitialised | recalibrate; check first-step loss vs unigram entropy |
| Unstable early, no NaN | momentum not projected onto tangent space of $O(d)$ | Riemannian projection on $Q$ |
| Errors concentrate on byte-similar tokens | output-side discriminability limit | untie $G_{\text{out}}$ or add rank-$k$ residual |
| Plateaus above baseline, everything else healthy | rank ceiling: $P > L_{\max}$ | reduce $P$, raising $s$ |
| Round-trip canary fails mid-run | $w$ hit the floor, or $Q$ drifted off manifold | **stop** — the head is the encoder's adjoint |
| Rare tokens underpredicted | $\beta_t$ LR too low | raise the $\beta$ group LR |

---

## 8. Status

**Proved:** exact invertibility (Thm 3, with necessity), minimal width (Thm 4), decoding margin (Thm 5), Welch
ceiling (Thm 6), distortion floor (Thm 7), shift response (Cor 8).

**Measured:** round-trip exactness in fp32/fp16/bf16 at three widths; codebook coherence against Welch; surface-
form geometry against the original codec; noise robustness against the bound; head cost and fast-path
correctness; rank; norm spread; near-duplicate separability.

**Not established:**

1. **That it trains as well.** The published 2.5% advantage came from a *dense, unconstrained* projection.
   Everything here constrains it. Gate 2 exists to find out in twelve runs rather than at scale. This is the
   load-bearing uncertainty.
2. **Reversibility at depth.** $\mathcal{R}$ inverts $\mathbf{e}$, not a layer-12 hidden state. Information about
   the bytes is non-increasing through the body, and should be.
3. **Truncation behaviour on a real vocabulary.** Zero collisions on random strings, but real tokenizers contain
   long whitespace runs and Indic word-pieces. Gate 0 catches this.
4. **Open-vocabulary byte generation.** Out of scope. That is where the independence assumption genuinely bites
   (after `q`, `u` is near-certain) and it needs vocabulary reranking or a small autoregressive-over-positions
   head.
5. **Wall-clock speedup.** 455× on MACs; the gather is bandwidth bound.

---

*ERA-V5 / assignment-7. Built on Kronecker Embeddings (arXiv:2605.29459). The reversible construction, the
tied-head derivation, Theorems 3–8 and all measurements here are new and unreviewed — a design proposal with a
verified implementation, not an established result.*
