"""
tdes.tokenizer
==============
Byte-level BPE, vocab size 512 (256 byte base + specials + merges).

CONTAMINATION NOTE (important, not decorative): the tokenizer is trained
ONLY on TRAIN-role documents. If it were trained on anything touching
EVAL, the merge table would encode eval-specific subword structure and
eval sequences would tokenize unusually efficiently -- a firewall breach
that leaves no trace in any batch, loss mask, or ledger, because it lives
in the vocabulary itself rather than in any single training example. This
mirrors the OPUS proxy-leakage concern in opus.py. Enforced here by
`train_bpe` accepting only pre-filtered TRAIN text and by the freeze hash
covering the training-corpus manifest hash (so tokenizer identity is
bound to *which* documents trained it, not just to the resulting merges).

FULL-FUNCTION FREEZE: the tokenizer hash pins:
  - the merge table (ordered list of byte-pair merges)
  - special tokens and their ids
  - the NFC-normalization flag
  - the pre-tokenization rule name ("byte": raw UTF-8 bytes, no word
    splitting before BPE)
  - the deterministic tie-break rule name
  - the content hash of the TRAIN-role manifest used to train it

Any change anywhere in that list invalidates the hash, and therefore
every shard built with it -- which is the point: [PASS]
tokenizer_hash_verified should certify the whole tokenization FUNCTION,
not just "the merge table happens to match".

Deterministic merge selection: BPE must repeatedly pick "the most
frequent adjacent pair". Ties are broken lexicographically on the pair
tuple of byte values -- declared once here (config.TOKENIZER_MERGE_TIEBREAK)
so a different implementation of "most frequent" can never silently
produce a different merge table from the same corpus.
"""

from __future__ import annotations
import dataclasses
import unicodedata
from typing import Dict, List, Tuple

from . import hashing, config


PAD_ID, BOS_ID, EOS_ID, UNK_ID = 0, 1, 2, 3
NUM_SPECIALS = len(config.TOKENIZER_SPECIALS)
BYTE_BASE = 256


def _prep(text: str) -> bytes:
    if config.TOKENIZER_NFC:
        text = unicodedata.normalize("NFC", text)
    return text.encode("utf-8")


@dataclasses.dataclass
class Tokenizer:
    merges: List[Tuple[int, int]]           # ordered list of (a, b) -> new_id, in application order
    merge_to_id: Dict[Tuple[int, int], int]  # (a, b) -> new token id
    vocab_size: int
    train_manifest_hash: str
    freeze_hash: str = ""

    def compute_freeze_hash(self) -> str:
        payload = {
            "merges": self.merges,
            "specials": config.TOKENIZER_SPECIALS,
            "nfc": config.TOKENIZER_NFC,
            "pretokenize": config.TOKENIZER_PRETOKENIZE,
            "tiebreak": config.TOKENIZER_MERGE_TIEBREAK,
            "vocab_size": self.vocab_size,
            "train_manifest_hash": self.train_manifest_hash,
        }
        return hashing.hash_object(payload)

    def freeze(self) -> None:
        self.freeze_hash = self.compute_freeze_hash()

    def verify(self) -> bool:
        return self.freeze_hash == self.compute_freeze_hash()

    # -- encode/decode ----------------------------------------------------
    # ID SPACE (fixed, non-overlapping, used consistently everywhere):
    #   [0, NUM_SPECIALS)                       special tokens
    #   [NUM_SPECIALS, NUM_SPECIALS+256)         raw byte values (id = byte + NUM_SPECIALS)
    #   [NUM_SPECIALS+256, vocab_size)           merges, assigned in training order
    def encode(self, text: str) -> List[int]:
        b = _prep(text)
        ids = [byte_val + NUM_SPECIALS for byte_val in b]  # start directly in final id space
        for (a, bb) in self.merges:
            merged_id = self.merge_to_id[(a, bb)]
            ids = _apply_merge(ids, a, bb, merged_id)
        return [BOS_ID] + ids + [EOS_ID]

    def decode(self, ids: List[int]) -> str:
        raw: List[int] = []
        id_to_merge = {v: k for k, v in self.merge_to_id.items()}

        def expand(tok: int) -> List[int]:
            if tok < NUM_SPECIALS:
                return []  # special token, no byte content
            if tok < NUM_SPECIALS + BYTE_BASE:
                return [tok - NUM_SPECIALS]
            if tok not in id_to_merge:
                return []
            a, b = id_to_merge[tok]
            return expand(a) + expand(b)

        for tok in ids:
            raw.extend(expand(tok))
        try:
            return bytes(raw).decode("utf-8", errors="replace")
        except Exception:
            return ""


def _apply_merge(ids: List[int], a: int, b: int, merged_id: int) -> List[int]:
    out = []
    i = 0
    n = len(ids)
    while i < n:
        if i < n - 1 and ids[i] == a and ids[i + 1] == b:
            out.append(merged_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


def train_bpe(train_texts: List[str], vocab_size: int, train_manifest_hash: str) -> Tokenizer:
    """Deterministic byte-level BPE training.

    train_texts must already be filtered to TRAIN-role documents only,
    in a stable (doc_id-sorted) order -- the caller (shards.py /
    run_demo.py) is responsible for that filtering; this function does
    not know about roles, it only tokenizes what it is given.
    """
    corpus_ids: List[List[int]] = [[byte_val + NUM_SPECIALS for byte_val in _prep(t)] for t in train_texts]

    num_merges_available = vocab_size - NUM_SPECIALS - BYTE_BASE
    if num_merges_available <= 0:
        raise ValueError("vocab_size too small for byte base + specials")

    merges: List[Tuple[int, int]] = []
    merge_to_id: Dict[Tuple[int, int], int] = {}
    next_id = NUM_SPECIALS + BYTE_BASE  # merge ids continue upward from the byte range

    for _ in range(num_merges_available):
        pair_counts: Dict[Tuple[int, int], int] = {}
        for seq in corpus_ids:
            for i in range(len(seq) - 1):
                pair = (seq[i], seq[i + 1])
                pair_counts[pair] = pair_counts.get(pair, 0) + 1

        if not pair_counts:
            break

        best_count = max(pair_counts.values())
        # deterministic tie-break: lexicographically smallest pair among
        # those achieving the max count (config.TOKENIZER_MERGE_TIEBREAK)
        candidates = [p for p, c in pair_counts.items() if c == best_count]
        best_pair = min(candidates)

        if best_count < 2:
            break  # no benefit to merging singleton pairs

        merges.append(best_pair)
        merge_to_id[best_pair] = next_id
        corpus_ids = [_apply_merge(seq, best_pair[0], best_pair[1], next_id) for seq in corpus_ids]
        next_id += 1

    tok = Tokenizer(
        merges=merges,
        merge_to_id=merge_to_id,
        vocab_size=NUM_SPECIALS + BYTE_BASE + len(merges),
        train_manifest_hash=train_manifest_hash,
    )
    tok.freeze()
    return tok
