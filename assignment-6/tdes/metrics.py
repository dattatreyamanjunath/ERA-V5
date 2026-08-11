"""
tdes.metrics
============
Two utilization numbers, reported together (not one headlined over the
other being hidden):

  non_pad_utilization      = non-pad tokens / total tokens
  loss_bearing_utilization = loss-bearing tokens / total tokens  (stricter,
                              the number that actually matters for
                              training efficiency)

Throughput is reported as tokens/sec in two forms: CONSUMED (every
token that occupied a sequence position, matching what packing actually
produced) and EFFECTIVE (only loss-bearing tokens -- "useful loss-
bearing tokens per second", the number the assignment explicitly asks
for). The gap between them is itself a useful efficiency signal and is
reported, not discarded.

Every number here is computed from the actual PackedSequence objects
and ledger records produced by the run -- nothing is a hardcoded
constant. See audit.py for how this is checked against evidence.json.
"""

from __future__ import annotations
import dataclasses
from typing import Dict, List

from .packing import PackedSequence


@dataclasses.dataclass
class UtilizationReport:
    lane: str
    n_sequences: int
    total_tokens: int
    non_pad_tokens: int
    loss_bearing_tokens: int
    non_pad_utilization: float
    loss_bearing_utilization: float


def compute_utilization(lane: str, sequences: List[PackedSequence]) -> UtilizationReport:
    total = sum(len(s.input_ids) for s in sequences)
    non_pad = sum(1 for s in sequences for b in s.attention_block_ids if b != -1)
    loss_bearing = sum(sum(s.loss_mask) for s in sequences)
    return UtilizationReport(
        lane=lane, n_sequences=len(sequences), total_tokens=total,
        non_pad_tokens=non_pad, loss_bearing_tokens=loss_bearing,
        non_pad_utilization=(non_pad / total) if total else 0.0,
        loss_bearing_utilization=(loss_bearing / total) if total else 0.0,
    )


@dataclasses.dataclass
class ThroughputReport:
    wall_seconds: float
    n_steps: int
    consumed_tokens: int
    effective_tokens: int
    consumed_tokens_per_sec: float
    effective_tokens_per_sec: float
    effective_over_consumed_ratio: float


def compute_throughput(wall_seconds: float, n_steps: int, consumed_tokens: int, effective_tokens: int) -> ThroughputReport:
    return ThroughputReport(
        wall_seconds=wall_seconds, n_steps=n_steps, consumed_tokens=consumed_tokens,
        effective_tokens=effective_tokens,
        consumed_tokens_per_sec=(consumed_tokens / wall_seconds) if wall_seconds > 0 else 0.0,
        effective_tokens_per_sec=(effective_tokens / wall_seconds) if wall_seconds > 0 else 0.0,
        effective_over_consumed_ratio=(effective_tokens / consumed_tokens) if consumed_tokens else 0.0,
    )
