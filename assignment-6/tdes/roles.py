"""
tdes.roles
==========
Assigns each surviving (post-dedup) document exactly one role:

    TRAIN       loss-bearing training data
    VALIDATION  forward-only; MAY steer curriculum/mixture (see mixture.py)
    EVAL        sealed; never read until the final audit
    PROXY       OPUS target-direction reference only; never loss-bearing
    QUARANTINE  failed dedup; never consumed by anything

Assignment is a deterministic function of doc_id alone: sha256(doc_id)
mod ROLE_HASH_BUCKETS, bucketed against config.ROLE_SPLIT_BOUNDARIES.
This is:

  - Reproducible from the doc_id with no external state.
  - Independent of ingestion order.
  - Stable under corpus growth in the append-only sense (a new document's
    role never depends on which documents came before it) -- though NOT
    stable if ROLE_SPLIT_BOUNDARIES itself changes, which is exactly why
    those boundaries are pinned inside the tokenizer/manifest hash chain.

QUARANTINE overrides any hash-assigned role: a document is quarantined
because dedup said so, not because of where its hash falls.

The role assignment table itself becomes part of the corpus manifest
(role_table_hash), so a change to the assignment logic invalidates
every downstream artifact -- deliberately.
"""

from __future__ import annotations
import dataclasses
from typing import Dict, List

from . import hashing, config
from .dedup import DocRecord


ROLE_ORDER = ("TRAIN", "VALIDATION", "EVAL", "PROXY")  # QUARANTINE assigned separately


def _bucket(doc_id: str) -> int:
    h = hashing.sha256_text(doc_id)
    # take first 8 hex chars as an integer, mod bucket count -- deterministic,
    # uniform enough for our purposes, no external RNG involved.
    return int(h[:8], 16) % config.ROLE_HASH_BUCKETS


def assign_role(doc_id: str) -> str:
    b = _bucket(doc_id)
    prev_bound = 0
    for role in ROLE_ORDER:
        bound = config.ROLE_SPLIT_BOUNDARIES[role]
        if prev_bound <= b < bound:
            return role
        prev_bound = bound
    return "PROXY"  # unreachable given boundaries sum to ROLE_HASH_BUCKETS


@dataclasses.dataclass
class RoleAssignment:
    table: Dict[str, str]           # doc_id -> role
    counts: Dict[str, int]
    table_hash: str


def assign_roles(survivors: List[DocRecord], quarantined: List[DocRecord]) -> RoleAssignment:
    table: Dict[str, str] = {}
    for r in sorted(survivors, key=lambda x: x.doc_id):
        table[r.doc_id] = assign_role(r.doc_id)
    for r in sorted(quarantined, key=lambda x: x.doc_id):
        table[r.doc_id] = "QUARANTINE"

    counts: Dict[str, int] = {}
    for role in table.values():
        counts[role] = counts.get(role, 0) + 1

    table_hash = hashing.hash_object({"boundaries": config.ROLE_SPLIT_BOUNDARIES,
                                       "buckets": config.ROLE_HASH_BUCKETS,
                                       "table": table})
    return RoleAssignment(table=table, counts=counts, table_hash=table_hash)


def verify_partition(assignment: RoleAssignment, all_doc_ids: List[str]) -> None:
    """Invariant check used by tests/audit: the role table is a total,
    disjoint partition of every input doc_id -- no doc_id missing, none
    assigned more than one role (trivially true for a dict, but we also
    check nothing was silently dropped)."""
    missing = set(all_doc_ids) - set(assignment.table.keys())
    if missing:
        raise AssertionError(f"role assignment missing doc_ids: {sorted(missing)[:5]}...")
    valid_roles = set(ROLE_ORDER) | {"QUARANTINE"}
    bad = {r for r in assignment.table.values() if r not in valid_roles}
    if bad:
        raise AssertionError(f"invalid roles present: {bad}")
