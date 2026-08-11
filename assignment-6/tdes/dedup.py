"""
tdes.dedup
==========
Two independent hashes per document, computed and recorded separately:

  - content_hash: sha256 over the RAW, unmodified UTF-8 bytes. This is
    the root of the immutability chain (manifests, shards). It must
    NEVER be computed over normalized text, or "immutable and hashed"
    stops meaning "the bytes we actually trained on".

  - shingle_set: a set of sha256(w-shingle) values computed over the
    NORMALIZED word sequence (tdes.normalize). Used only for near-
    duplicate detection via exact Jaccard similarity -- no MinHash, no
    randomness: at this corpus size exact set overlap is affordable and
    introduces no seeded-permutation state that would need to be frozen
    and replayed like everything else.

Two-stage dedup, run BEFORE role assignment and BEFORE tokenizer
training (see roles.py, tokenizer.py docstrings for why the ordering
matters):

  1. Exact dedup: documents with identical content_hash collapse to one
     canonical document (first by doc_id ascending). This catches the
     Sangraha-specific failure mode where the same underlying text
     surfaces under two different doc_ids via different tier pipelines
     (verified -> perplexity-filtered into unverified, or translated
     into synthetic).

  2. Near-dup quarantine: among the *surviving* documents, any pair with
     shingle Jaccard >= config.JACCARD_DUP_THRESHOLD has the later
     document (by doc_id) routed to the QUARANTINE role instead of
     whatever role its hash would have assigned. Quarantined documents
     are never consumed downstream.
"""

from __future__ import annotations
import dataclasses
from typing import Dict, List, Set, Tuple

from . import hashing, normalize, config


@dataclasses.dataclass
class DocRecord:
    doc_id: str
    tier: str
    type_: str | None
    text: str
    content_hash: str = ""
    shingle_set: frozenset = frozenset()

    def compute(self) -> None:
        self.content_hash = hashing.sha256_text(self.text)
        words = normalize.normalize_for_shingling(self.text)
        w = config.SHINGLE_WINDOW
        if len(words) < w:
            shingles = {" ".join(words)} if words else set()
        else:
            shingles = {" ".join(words[i:i + w]) for i in range(len(words) - w + 1)}
        self.shingle_set = frozenset(hashing.sha256_text(s) for s in shingles)


def jaccard(a: frozenset, b: frozenset) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


@dataclasses.dataclass
class DedupResult:
    survivors: List[DocRecord]
    quarantined: List[DocRecord]
    exact_duplicates_dropped: List[Tuple[str, str]]     # (dup_doc_id, canonical_doc_id)
    near_duplicates_quarantined: List[Tuple[str, str, float]]  # (doc_id, matched_against, jaccard)


def run_dedup(records: List[DocRecord]) -> DedupResult:
    for r in records:
        r.compute()

    # Stage 1: exact content-hash dedup, deterministic canonical choice
    # (lowest doc_id wins) so the outcome does not depend on input order.
    by_hash: Dict[str, List[DocRecord]] = {}
    for r in sorted(records, key=lambda x: x.doc_id):
        by_hash.setdefault(r.content_hash, []).append(r)

    survivors: List[DocRecord] = []
    exact_dropped: List[Tuple[str, str]] = []
    for h, group in by_hash.items():
        group_sorted = sorted(group, key=lambda x: x.doc_id)
        canonical = group_sorted[0]
        survivors.append(canonical)
        for dup in group_sorted[1:]:
            exact_dropped.append((dup.doc_id, canonical.doc_id))

    # Stage 2: near-dup quarantine via pairwise exact-Jaccard over
    # shingle sets. O(n^2) in survivor count -- fine at our corpus size
    # (a few thousand documents). A production system would bucket by
    # a cheap pre-filter (e.g. shared shingle count via an inverted
    # index) before falling back to full Jaccard; noted here, not
    # implemented, since it is a performance optimization, not a
    # correctness requirement at this scale.
    survivors_sorted = sorted(survivors, key=lambda x: x.doc_id)
    quarantined_ids: Set[str] = set()
    near_dup_events: List[Tuple[str, str, float]] = []

    for i in range(len(survivors_sorted)):
        a = survivors_sorted[i]
        if a.doc_id in quarantined_ids:
            continue
        for j in range(i + 1, len(survivors_sorted)):
            b = survivors_sorted[j]
            if b.doc_id in quarantined_ids:
                continue
            sim = jaccard(a.shingle_set, b.shingle_set)
            if sim >= config.JACCARD_DUP_THRESHOLD:
                # later doc_id (lexicographically larger) is quarantined
                quarantined_ids.add(b.doc_id)
                near_dup_events.append((b.doc_id, a.doc_id, sim))

    final_survivors = [r for r in survivors_sorted if r.doc_id not in quarantined_ids]
    quarantined_records = [r for r in survivors_sorted if r.doc_id in quarantined_ids]

    return DedupResult(
        survivors=final_survivors,
        quarantined=quarantined_records,
        exact_duplicates_dropped=exact_dropped,
        near_duplicates_quarantined=near_dup_events,
    )
