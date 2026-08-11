"""
tdes.hashing
============
Two distinct hashing concerns, kept deliberately separate:

1. Content / integrity hashing (sha256 over canonical bytes) -- used for
   immutability, manifest chaining, ledger chaining, tokenizer freezing.

2. Deterministic key derivation for all "randomness" in the system
   (dropout masks, OPUS Boltzmann sampling, packing tie-breaks that need
   a random draw, rollback child-branch divergence). There is NO global
   RNG object anywhere in this codebase. Every stochastic draw is a pure
   function of (master_seed, branch_id, step, purpose, *extra), which is
   what makes resume and replay exact: a draw needs no history, only its
   own coordinates.

The `require_no_global_random` test in tests/test_no_global_rng.py greps
the source tree for `random.` / `os.urandom` / `secrets.` outside this
module to enforce that nothing else in the codebase reaches for ambient
randomness.
"""

from __future__ import annotations
import hashlib
import json
import struct
from typing import Any, Iterable, Sequence


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def canonical_json(obj: Any) -> bytes:
    """Deterministic JSON serialization: sorted keys, fixed separators,
    no whitespace ambiguity. This is the ONLY serialization used before
    hashing any structured object, so hashes are stable across runs and
    across machines."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def hash_object(obj: Any) -> str:
    return sha256_bytes(canonical_json(obj))


def chain_hash(prev_hash: str, record: Any) -> str:
    """Hash-chain link: H(prev_hash || canonical(record)). Used by both
    the manifest chain and the ledgers so that any record's hash commits
    to the entire history before it."""
    payload = (prev_hash or "GENESIS").encode("utf-8") + b"|" + canonical_json(record)
    return sha256_bytes(payload)


# ---------------------------------------------------------------------------
# Deterministic key derivation (replaces a global RNG stream entirely)
# ---------------------------------------------------------------------------

def derive_key(master_seed: int, branch_id: str, step: int, purpose: str, *extra: Any) -> int:
    """Derive a 64-bit unsigned integer key deterministically from explicit
    coordinates. No object here holds state -- calling this twice with the
    same arguments always returns the same key, regardless of what else
    has happened in the process. This is what makes a dropout mask at
    step 151 reproducible after a crash/resume without needing to
    serialize any RNG's internal state into the checkpoint."""
    parts = [str(master_seed), str(branch_id), str(step), str(purpose)] + [str(e) for e in extra]
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).digest()
    return struct.unpack(">Q", digest[:8])[0]


class KeyedStream:
    """A tiny deterministic pseudo-random stream derived from a single
    64-bit key via a counter-based construction (hash(key, counter)).
    This is NOT python's `random` module and holds no ambient state --
    it is fully specified by (key, counter), both of which are always
    known from context (never advanced implicitly across unrelated
    calls). Used for dropout masks and Boltzmann sampling.
    """

    __slots__ = ("_key",)

    def __init__(self, key: int):
        self._key = key

    def _word(self, counter: int) -> int:
        payload = struct.pack(">QQ", self._key, counter)
        digest = hashlib.sha256(payload).digest()
        return struct.unpack(">Q", digest[:8])[0]

    def uniform(self, counter: int) -> float:
        """Uniform float in [0, 1), deterministic in (key, counter)."""
        return self._word(counter) / (2**64)

    def uniforms(self, n: int) -> list:
        return [self.uniform(i) for i in range(n)]


def keyed_uniform(master_seed: int, branch_id: str, step: int, purpose: str, index: int, *extra: Any) -> float:
    """Convenience: one deterministic uniform draw identified fully by
    its coordinates, with no intermediate stream object needed."""
    key = derive_key(master_seed, branch_id, step, purpose, *extra)
    return KeyedStream(key).uniform(index)
