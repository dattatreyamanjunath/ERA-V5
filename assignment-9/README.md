# Assignment 9 — Making the Training Harness Correct and Observable

Notebook: [`llm_harness.ipynb`](llm_harness.ipynb) — runs top to bottom (`Runtime → Run all` in Colab, or executed locally as below). Trained and logged locally on CPU; no Colab upload needed to grade this.

Starting point: a three-line next-token training harness —

```python
hidden = model(tokens)
logits = output_head(hidden)
loss = cross_entropy(
    logits[:, :-1].reshape(-1, vocab_size),
    tokens[:, 1:].reshape(-1),
)
```

— rebuilt piece by piece so every step prints something verifiable: a named shape, a token string next to its target string, a token count that visibly changes under a mask. All numbers below are copied straight from an actual local run's logs (`llm_harness.ipynb`, cells 5–16), nothing hand-typed.

## On the warning

> A target shift in the incorrect direction can produce a beautiful loss curve.

Took this literally: the notebook trains two identical models on a tiny corpus for 80 steps, one on the correct objective (`logits[:, :-1]` vs. `tokens[:, 1:]`, predict the next token) and one on the classic off-by-one (`logits[:, :-1]` vs. `tokens[:, :-1]`, predict the token that was just fed in — a copy task, not a language model). The loss numbers alone do not give the bug away:

| step | correct (predict t+1) | buggy (predict t) |
|---|---|---|
| 0  | 10.8210 | **10.5677** (lower — a free head start from the identity shortcut through the residual stream) |
| 20 | 8.9209  | 8.3005 |
| 40 | 7.2907  | 7.1012 |
| 60 | 5.8126  | **5.9136** (correct has now caught up) |
| 79 | 4.4528  | 4.6383 |

The buggy objective starts out *lower* — exactly the "beautiful loss curve" the warning describes — because copying a token is a strictly easier function than predicting the next one. On this run it's overtaken by step 60 only because the demo corpus is small enough to memorize; on real, non-repeating data the copy task's head start wouldn't go away. Either way, no loss number at any step tells you which curve is which. What does: printing the actual input/target strings for the buggy run —

```
0   'A'        ->  'A'         <-- input == target
1   ' watched' ->  ' watched'  <-- input == target
2   ' pot'     ->  ' pot'      <-- input == target
```

— every pair identical, obvious on sight, invisible in the loss. Item 2 below is this same check applied to the real harness, confirming it does *not* have this bug.

## Part 1 — the seven numbers

| # | Check | Result |
|---|---|---|
| 1 | Loss on a 19-token sample sentence, harness wired as given | **10.9496** |
| 2 | Input→target string pairs verified by eye (`'The'→' quick'`, `' quick'→' brown'`, …, `' April'→'.'`) | **18 / 18 correctly shifted, 0 misaligned** |
| 3 | Contributing tokens, batch of 2 padded sequences: unmasked → masked | **26 → 15** (loss 10.9315 → 10.9088) |
| 4 | Packed 2-document loss: naive (attends + counts the boundary) vs. fully fixed (blocked attention + boundary label masked) | **10.8468 (n=20) → 10.8316 (n=19)** |
| 5 | Perplexity of the untrained model vs. vocab size | **54,344.6** vs. **50,259** (ratio 1.081) |
| 6 | Total parameters, tied vs. untied output head | **7,234,688 → 13,667,840** (Δ = 6,433,152 = vocab_size × d_model, exactly) |
| 7 | Peak memory, ordinary vs. hand-written chunked cross-entropy (same inputs, isolated subprocesses) | **598.23 MB → 325.93 MB** (1.84×) |

Item 7 note: the first version of this measurement polled RSS on a background thread for both calls *within one process*, and two runs of it disagreed about which method used more memory — PyTorch's CPU allocator doesn't return freed memory to the OS, so whichever function ran second inherited a polluted baseline. Fixed by running each measurement in its own fresh subprocess; the 1.84× result above then reproduced consistently across repeated runs (598–617 MB vs. 326 MB every time). Worth stating since it's the same lesson as the headline warning, one level up: an unverified measurement can look fine and still be wrong.

## Part 2 — the two losses

A second head (`head_t2`, untied) reads the same trunk hidden states and predicts `t+2` instead of `t+1`; both losses are logged separately and summed for the optimizer step, trained 300 steps on Tiny Shakespeare:

| | loss (mean, last 20 steps) |
|---|---|
| `head_t1` (predict t+1) | **5.657** |
| `head_t2` (predict t+2) | **5.932** |

Both start within 0.04 of each other near `ln(vocab_size) ≈ 10.83` (10.813 vs. 10.857 at step 0 — neither head has learned anything yet), then `head_t1` pulls ahead and the gap widens over training (0.275 by the last 20 steps, up from ~0.04 at the start). Predicting two tokens ahead from the same hidden state is a strictly harder, higher-entropy problem than predicting the very next one — every extra step into the future is another chance for the target to depend on something other than the current position — so `head_t2` both converges to a worse asymptote and gets there more slowly, sharing the same trunk representation the whole time.

## Running it

```bash
pip install torch tiktoken psutil matplotlib
jupyter nbconvert --to notebook --execute llm_harness.ipynb
```

CPU is fine (~6–7 minutes total, mostly Part 2's 300-step loop); a GPU runtime gets an exact CUDA peak-memory reading in item 7 instead of the subprocess/psutil fallback.
