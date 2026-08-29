# Assignment 9 — Making the Training Harness Correct and Observable

A from-scratch decoder-only transformer and next-token training harness, rebuilt so that every stage prints something verifiable — a named tensor shape, a token string next to its target, a token count that visibly changes under a mask — instead of just a loss number to trust.

Notebook: [`llm_harness.ipynb`](llm_harness.ipynb). Runs top to bottom (`Runtime → Run all` in Colab, or executed locally as below); all numbers in this README are copied from an actual local execution log.

## What's in the notebook

A small GPT-style model (128-dim, 4 layers, 4 heads) with a real GPT-2 BPE tokenizer, built to keep the training harness legible:

- **Named shapes.** Every tensor from raw token ids through the flattened logits/targets fed to `cross_entropy` is printed with a one-line meaning per axis.
- **String-level shift check.** Input/target token id pairs are decoded back to actual sub-word strings and printed side by side, so a shift bug is visible as a misaligned word, not an off-by-one buried in integers.
- **A worked example of that check catching something.** Two identical models are trained side by side for 80 steps — one on the correct next-token objective, one on the classic bug of predicting the token that was just fed in (a copy task, not a language model). The buggy objective's loss starts out *lower*, purely because copying is an easier function to learn; nothing about the loss curve gives it away. Printing the input/target strings does, immediately.
- **Padding masks**, verified by showing the number of tokens contributing to the loss shrink when pad positions are excluded via `ignore_index`.
- **Document packing**, with two unrelated documents concatenated into one sequence and the boundary handled two ways — masking just the loss label, and properly blocking attention across the boundary — compared against doing neither.
- **A perplexity sanity check**: an untrained model's loss should sit close to `ln(vocab_size)`.
- **Tied vs. untied output head**, comparing total parameter counts.
- **Peak memory**, comparing ordinary cross-entropy against a hand-written chunked version that never materializes the full `(batch, seq_len, vocab_size)` logits tensor at once.
- **A second prediction head** bolted onto the same trunk, trained to predict `t+2` instead of `t+1`, with both losses logged separately.

## Results

| Check | Result |
|---|---|
| Loss on a 19-token sample sentence | 10.9496 |
| Shift correctness, 18 input→target string pairs | 18 / 18 correctly shifted, 0 misaligned |
| Shift-bug demo: correct vs. copy-task loss at step 0 | 10.8210 vs. **10.5677** (buggy starts lower, for free) |
| Contributing tokens, padded batch: unmasked → masked | 26 → 15 (loss 10.9315 → 10.9088) |
| Packed 2-document loss: naive vs. fully fixed (blocked attention + masked boundary) | 10.8468 (n=20) → 10.8316 (n=19) |
| Perplexity of the untrained model vs. vocab size | 54,344.6 vs. 50,259 (ratio 1.081) |
| Parameters, tied vs. untied output head | 7,234,688 → 13,667,840 (Δ = 6,433,152 = vocab_size × d_model) |
| Peak memory, ordinary vs. chunked cross-entropy | 598.23 MB → 325.93 MB (1.84×) |
| Second-head training, mean loss over the last 20 of 300 steps | `t+1`: 5.657, `t+2`: 5.932 |

A couple of these are worth a sentence of context:

**The shift-bug demo** trains long enough that the correct objective actually overtakes the buggy one by step 60, because the demo corpus is small enough to memorize — a property of that toy corpus, not evidence the bug is safe. On real, non-repeating data the copy task's head start would persist. The point stands either way: at no step does the loss number tell you which curve is which.

**Peak memory** was trickier to measure honestly than expected. An earlier version polled process RSS on a background thread for both the ordinary and chunked runs within a single process, and two runs of it disagreed about which one used more memory — PyTorch's CPU allocator doesn't return freed memory to the OS, so whichever function ran second inherited a polluted baseline. Fixed by measuring each version in its own fresh subprocess, which reproduces the same ~1.84× result on every rerun.

**The second head** starts within 0.04 of `head_t1` near `ln(vocab_size) ≈ 10.83`, then falls behind as training progresses, ending 0.275 lower in accuracy (higher in loss) after 300 steps. Predicting two tokens ahead from the same hidden state is a strictly harder, higher-entropy problem than predicting the very next one, so it both converges to a worse asymptote and gets there more slowly.

## Running it

```bash
pip install torch tiktoken psutil matplotlib
jupyter nbconvert --to notebook --execute llm_harness.ipynb
```

CPU is fine (~6–7 minutes total, mostly the 300-step second-head training loop); a GPU runtime gets an exact CUDA peak-memory reading instead of the subprocess/psutil fallback.
