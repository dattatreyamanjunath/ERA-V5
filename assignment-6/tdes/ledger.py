"""
tdes.ledger
===========
Two append-only, hash-chained ledgers, persisted as JSONL:

  ConsumptionLedger: what was consumed, when, from where. Every batch
  goes through the write-ahead protocol:

      append("INTENT", {step, batch_id, batch_hash, lane, seq_ids, ...})
      <optimizer step happens>
      append("COMMIT", {step, batch_id})

  If the process crashes between those two appends, the batch is
  AMBIGUOUS: its INTENT is on disk but its COMMIT is not, so the model
  was never durably updated for it (see recovery.py). Resume discards
  that batch's in-memory effect (there is none, by construction -- see
  recovery.py docstring) and re-executes the SAME step number, which
  deterministically reproduces the SAME batch_hash. Comparing the
  orphaned INTENT's batch_hash to the new one is exactly what
  "[PASS] resume_next_batch_matched" checks.

  LearningLedger: per-batch loss, per-lane decomposition, and a sampled
  set of per-token losses, linked to ConsumptionLedger records by
  batch_id -- this is what lets audit.py prove "loss linked to source
  data" rather than merely asserting it.

Every record is chained: chain_hash = H(prev_chain_hash || record). Any
edit to any past record changes every subsequent chain_hash, which is
exactly the tamper-detection property audit.py's negative test relies
on. Ledgers are NEVER truncated (not even by rollback -- see
recovery.py: rollback appends a ROLLBACK event and forks; it never
rewrites history).
"""

from __future__ import annotations
import dataclasses
import json
import os
from typing import Any, Dict, List, Optional

from . import hashing


@dataclasses.dataclass
class LedgerRecord:
    index: int
    record_type: str
    payload: Dict[str, Any]
    prev_hash: str
    chain_hash: str


class Ledger:
    def __init__(self, path: str):
        self.path = path
        self.records: List[LedgerRecord] = []
        self.head_hash = "GENESIS"
        if os.path.exists(path):
            self._load()

    def _load(self) -> None:
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                self.records.append(LedgerRecord(**d))
        if self.records:
            self.head_hash = self.records[-1].chain_hash
        self.verify_chain()

    def append(self, record_type: str, payload: Dict[str, Any]) -> LedgerRecord:
        index = len(self.records)
        chain_payload = {"index": index, "record_type": record_type, "payload": payload}
        chain_hash = hashing.chain_hash(self.head_hash, chain_payload)
        rec = LedgerRecord(index=index, record_type=record_type, payload=payload,
                            prev_hash=self.head_hash, chain_hash=chain_hash)
        self.records.append(rec)
        self.head_hash = chain_hash
        self._append_to_disk(rec)
        return rec

    def _append_to_disk(self, rec: LedgerRecord) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(dataclasses.asdict(rec), ensure_ascii=False, sort_keys=True))
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())

    def verify_chain(self) -> bool:
        prev = "GENESIS"
        for rec in self.records:
            expected = hashing.chain_hash(prev, {"index": rec.index, "record_type": rec.record_type, "payload": rec.payload})
            if expected != rec.chain_hash or rec.prev_hash != prev:
                raise IntegrityError(f"ledger chain broken at index {rec.index} in {self.path}")
            prev = rec.chain_hash
        return True

    def find_unmatched_intents(self) -> List[LedgerRecord]:
        """Returns INTENT records for which no later COMMIT with the
        same (step, batch_id) exists -- the ambiguous-batch set that
        resume must discard and re-execute."""
        committed = set()
        for rec in self.records:
            if rec.record_type == "COMMIT":
                committed.add((rec.payload["step"], rec.payload["batch_id"]))
        unmatched = []
        for rec in self.records:
            if rec.record_type == "INTENT":
                key = (rec.payload["step"], rec.payload["batch_id"])
                if key not in committed:
                    unmatched.append(rec)
        return unmatched

    def last_committed_step(self) -> int:
        steps = [rec.payload["step"] for rec in self.records if rec.record_type == "COMMIT"]
        return max(steps) if steps else -1

    def records_in_step_range(self, lo: int, hi: int) -> List[LedgerRecord]:
        return [r for r in self.records
                if "step" in r.payload and lo <= r.payload["step"] <= hi]


class IntegrityError(Exception):
    pass
