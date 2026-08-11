"""
tdes.packing
============
Turns a lane's TRAIN-role token sequences into fixed-length packed
training sequences, following the lane's PackingPolicy (config.py).

Pipeline:
  1. Split oversized documents (longer than seq_len) into fragments if
     the policy allows; otherwise truncate. A split fragment carries
     (parent_doc_id, fragment_index, is_continuation) so the learning
     ledger can always trace a loss-bearing token back to exactly one
     source document AND offset, even across a split.
  2. Chunk the (possibly-fragmented) document list, in doc_id-ascending
     order, into groups of policy.group_size.
  3. Solve each group's bin packing EXACTLY via binpack.exact_min_bins
     (group_size <= config.DP_EXACT_MAX_GROUP by construction), or via
     first_fit_decreasing above that threshold (not expected to trigger
     given our policy group sizes, kept as a documented safety net).
  4. Materialize each bin as one packed sequence: token ids, block-
     diagonal attention mask, per-document position ids (reset to 0 at
     each document boundary), and a loss mask that is 0 for padding and
     for each document's first token (BOS has no prediction target of
     its own), 1 everywhere else within a TRAIN document.
  5. Right-pad the final sequence to exactly seq_len.

INVARIANT (checked by tests/test_packing.py and audit.py): every
position with loss_mask == 1 traces to exactly one TRAIN-role document
id and one token offset within it, via the returned PackedSequence's
`token_provenance` list.
"""

from __future__ import annotations
import dataclasses
from typing import Dict, List, Optional, Tuple

from . import config, binpack, firewall


@dataclasses.dataclass
class DocTokens:
    doc_id: str
    role: str
    lane: str
    tokens: List[int]        # includes tokenizer BOS/EOS


@dataclasses.dataclass
class Fragment:
    doc_id: str
    parent_doc_id: str
    fragment_index: int
    is_continuation: bool
    tokens: List[int]
    token_offset_in_parent: int   # offset of tokens[0] within the parent doc's token stream


@dataclasses.dataclass
class TokenProvenanceEntry:
    doc_id: str          # fragment id if split, else original doc id
    parent_doc_id: str
    token_offset: int    # offset within the fragment/document


@dataclasses.dataclass
class PackedSequence:
    seq_id: str
    lane: str
    group_index: int
    bin_index: int
    input_ids: List[int]
    attention_block_ids: List[int]   # same block id => may attend to each other; padding gets block -1
    position_ids: List[int]
    loss_mask: List[int]
    token_provenance: List[Optional[TokenProvenanceEntry]]  # None for padding
    doc_ids_present: List[str]


def split_document(doc: DocTokens, seq_len: int, min_fragment_len: int, allow_split: bool) -> List[Fragment]:
    n = len(doc.tokens)
    if n <= seq_len:
        return [Fragment(doc.doc_id, doc.doc_id, 0, False, doc.tokens, 0)]

    if not allow_split:
        # policy forbids splitting: truncate to seq_len and record the
        # discard via a synthetic marker fragment covering only the kept span.
        return [Fragment(doc.doc_id, doc.doc_id, 0, False, doc.tokens[:seq_len], 0)]

    fragments: List[Fragment] = []
    start = 0
    idx = 0
    while start < n:
        end = min(start + seq_len, n)
        # if the final fragment would be shorter than min_fragment_len,
        # merge it backward into the previous fragment by pulling the
        # boundary back (only if a previous fragment exists to absorb it).
        if (n - start) < min_fragment_len and fragments:
            prev = fragments[-1]
            extra = doc.tokens[start:n]
            merged_tokens = prev.tokens + extra
            if len(merged_tokens) <= seq_len:
                fragments[-1] = Fragment(prev.doc_id, prev.parent_doc_id, prev.fragment_index,
                                          prev.is_continuation, merged_tokens, prev.token_offset_in_parent)
                break
        frag_id = f"{doc.doc_id}::frag{idx}"
        fragments.append(Fragment(frag_id, doc.doc_id, idx, idx > 0, doc.tokens[start:end], start))
        start = end
        idx += 1
    return fragments


def _group_chunks(items: List, group_size: int) -> List[List]:
    return [items[i:i + group_size] for i in range(0, len(items), group_size)]


def pack_lane(lane: str, role: str, docs: List[DocTokens], seq_len: int) -> List[PackedSequence]:
    """Packs all TRAIN-role documents of one lane. Raises via
    firewall.admit_for_training if any document's role is not TRAIN --
    packing is a loss-bearing operation and must go through the gate."""
    policy = config.PACKING_POLICY[lane]
    for d in docs:
        firewall.admit_for_training(d.role, d.doc_id)

    docs_sorted = sorted(docs, key=lambda d: d.doc_id)

    # Step 1: split/truncate oversized documents.
    fragments: List[Fragment] = []
    for d in docs_sorted:
        fragments.extend(split_document(d, seq_len, policy.min_fragment_len, policy.allow_split))
    fragments.sort(key=lambda f: f.doc_id)  # deterministic ordering (tie_break: doc_id_asc)

    # Step 2/3: chunk into groups, solve each group's bin packing.
    groups = _group_chunks(fragments, policy.group_size)
    packed_sequences: List[PackedSequence] = []
    seq_counter = 0

    for g_idx, group in enumerate(groups):
        lengths = [len(f.tokens) for f in group]
        if len(group) <= config.DP_EXACT_MAX_GROUP:
            num_bins, assignment = binpack.exact_min_bins(lengths, seq_len)
        else:
            num_bins, assignment = binpack.first_fit_decreasing(lengths, seq_len)

        # also respect max_docs_per_sequence: if the exact/FFD solution
        # over-fills a bin's document count, deterministically resplit
        # via FFD-with-cap (simple, declared behavior)
        bins: Dict[int, List[int]] = {}
        for item_idx, b in enumerate(assignment):
            bins.setdefault(b, []).append(item_idx)
        bins = _enforce_doc_cap(bins, policy.max_docs_per_sequence, lengths)

        for b_idx in sorted(bins.keys()):
            item_indices = bins[b_idx]
            frags_in_bin = [group[i] for i in item_indices]
            seq = _materialize_sequence(lane, g_idx, b_idx, frags_in_bin, seq_len, seq_counter)
            packed_sequences.append(seq)
            seq_counter += 1

    return packed_sequences


def _enforce_doc_cap(bins: Dict[int, List[int]], max_docs: int, lengths: List[int]) -> Dict[int, List[int]]:
    """If a bin from the DP/FFD solver holds more documents than the
    lane's max_docs_per_sequence, deterministically split it into
    multiple bins of at most max_docs items (in the same relative
    order). Renumbers bins densely and deterministically."""
    out: Dict[int, List[int]] = {}
    next_bin = 0
    for b_idx in sorted(bins.keys()):
        items = bins[b_idx]
        if len(items) <= max_docs:
            out[next_bin] = items
            next_bin += 1
        else:
            for i in range(0, len(items), max_docs):
                out[next_bin] = items[i:i + max_docs]
                next_bin += 1
    return out


def _materialize_sequence(lane: str, g_idx: int, b_idx: int, frags: List[Fragment],
                           seq_len: int, seq_counter: int) -> PackedSequence:
    input_ids: List[int] = []
    block_ids: List[int] = []
    position_ids: List[int] = []
    loss_mask: List[int] = []
    provenance: List[Optional[TokenProvenanceEntry]] = []
    doc_ids_present: List[str] = []

    for block_id, frag in enumerate(frags):
        doc_ids_present.append(frag.doc_id)
        for pos, tok in enumerate(frag.tokens):
            if len(input_ids) >= seq_len:
                break
            input_ids.append(tok)
            block_ids.append(block_id)
            position_ids.append(pos)  # per-document reset: position 0 at each fragment start
            provenance.append(TokenProvenanceEntry(frag.doc_id, frag.parent_doc_id, pos))
            is_bos_position = (pos == 0)
            loss_mask.append(0 if is_bos_position else 1)

    pad_len = seq_len - len(input_ids)
    if pad_len > 0:
        # right padding (config.PACKING_POLICY[...].pad_side == "right"
        # for every lane -- pinned; left-padding provides no semantic
        # benefit here since attention/loss are governed by explicit
        # masks rather than by padding position, see design notes)
        input_ids.extend([0] * pad_len)          # PAD_ID == 0
        block_ids.extend([-1] * pad_len)          # -1 => attends to nothing
        position_ids.extend([0] * pad_len)
        loss_mask.extend([0] * pad_len)
        provenance.extend([None] * pad_len)

    seq_id = f"{lane}:g{g_idx}:b{b_idx}:{seq_counter}"
    return PackedSequence(
        seq_id=seq_id, lane=lane, group_index=g_idx, bin_index=b_idx,
        input_ids=input_ids, attention_block_ids=block_ids, position_ids=position_ids,
        loss_mask=loss_mask, token_provenance=provenance, doc_ids_present=doc_ids_present,
    )


def build_attention_mask(block_ids: List[int]) -> List[List[int]]:
    """Materializes the full block-diagonal causal mask for one
    sequence: position j is visible from position i iff j <= i (causal)
    AND block_ids[i] == block_ids[j] (same document) AND block_ids[i] != -1
    (not padding). Returned as a dense 0/1 matrix -- fine at seq_len=128."""
    n = len(block_ids)
    mask = [[0] * n for _ in range(n)]
    for i in range(n):
        if block_ids[i] == -1:
            continue
        for j in range(i + 1):
            if block_ids[j] == block_ids[i]:
                mask[i][j] = 1
    return mask
