# Assignment 9 — Making the Training Harness Correct and Observable

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dattatreyamanjunath/ERA-V5/blob/main/assignment-9/llm_harness.ipynb)

Starting point: a three-line next-token training harness —

```python
hidden = model(tokens)
logits = output_head(hidden)
loss = cross_entropy(
    logits[:, :-1].reshape(-1, vocab_size),
    tokens[:, 1:].reshape(-1),
)
```

— rebuilt piece by piece so that every step prints something you can verify by eye instead of trusting the code.

## Part 1 — the harness

1. **Every tensor shape, named.** A `describe()` helper prints each tensor's shape with a one-line meaning per axis (batch, seq_len, d_model, vocab_size), all the way from raw `tokens` through the shifted, flattened `flat_logits`/`flat_targets` that go into `cross_entropy`.
2. **Verify the shift with strings.** The GPT-2 BPE tokenizer (`tiktoken`) decodes each input/target id pair to real sub-word strings, printed side by side, so an off-by-one is visible as a word, not buried in integers.
3. **Mask padding.** Two sequences of different lengths are padded to a common length; the loss is computed with and without an `ignore_index=-100` mask on the pad targets, and the count of tokens actually contributing to the loss is shown to shrink.
4. **Pack two documents, mask the boundary.** Two unrelated documents are concatenated into one sequence. Naively, the model can both attend across the boundary and is asked to predict document B's first token from document A's last token — a fabricated dependency. The notebook fixes both (a document-aware attention mask baked into the model, and `-100` on the boundary label) and explains why the effect is small at random initialization but real and growing once training starts.
5. **Perplexity sanity check.** An untrained model's loss should sit near `ln(vocab_size)`, i.e. perplexity near `vocab_size` — the softmax of small random logits is close to uniform. The notebook checks this and explains what a bug (loss far from that value) would look like.
6. **Tied vs. untied head.** Compares total parameter counts with `output_head.weight` tied to the token embedding vs. given its own matrix; the difference is exactly `vocab_size * d_model`.
7. **Peak memory: ordinary vs. chunked cross-entropy.** A hand-written chunked cross-entropy (in the spirit of fused linear-cross-entropy kernels) never materializes the full `(batch, seq_len, vocab_size)` logits tensor at once. Peak memory is measured exactly via `torch.cuda.max_memory_allocated()` on GPU, or approximated via `psutil` RSS polling on CPU (not `tracemalloc`, which doesn't see native tensor memory at all).

## Part 2 — a second head, predicting `t+2`

A second output head is bolted onto the same transformer trunk, reading the same hidden states but trained to predict the token **two** positions ahead instead of one. Both losses are logged separately and summed for the optimizer step, then plotted over ~300 training steps on the public-domain Tiny Shakespeare corpus. The `t+2` head consistently trails the `t+1` head and the gap widens over training — explained via the higher irreducible entropy of predicting further into the future from the same hidden state.

## Running it

Open [`llm_harness.ipynb`](llm_harness.ipynb) in Colab via the badge above, or run it locally with `torch`, `tiktoken`, `psutil`, and `matplotlib` installed. CPU is fine for Part 1; Part 2's training loop is faster with a T4 GPU (`Runtime → Change runtime type`).
