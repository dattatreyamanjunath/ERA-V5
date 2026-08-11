"""
tdes.mixture
============
Deterministic quota (deficit) scheduling: at every draw, the lane whose
actual consumed share is furthest BELOW its planned target share is
chosen. Given our short demo run, stochastic sampling would make a
correct system look non-compliant just from ordinary sampling noise, so
we track exact deficits instead -- this is also what production mixture
schedulers actually do.

Curriculum: three fixed stages (config.CURRICULUM_STAGES), each
widening the active lane set. Advancement is validation-triggered (two
consecutive evaluations with improvement below
STAGE_ADVANCE_VAL_IMPROVEMENT_EPS) with a hard step-count fallback
(STAGE_ADVANCE_MAX_STEPS) so every stage provably appears within a short
demo run even if validation behaves unhelpfully.

Protected floor: verified lanes' combined share, measured over a
sliding window of the last config.FLOOR_WINDOW batches, must not fall
below config.PROTECTED_FLOOR_VERIFIED_SHARE. `floor_breach()` reports
this per draw; OPUS (opus.py) is the only place that may act on a
breach, via the floor-override path.

Rollback demotion (see recovery.py): halves the implicated lane's
weight within the CURRENT stage and renormalizes the remaining active
weights to sum to 1. This can itself push the lane under its floor --
that collision is intentional (see design notes) and is exactly the
kind of governance interaction the floor-override path exists to
resolve.
"""

from __future__ import annotations
import collections
import dataclasses
from typing import Deque, Dict, List, Optional, Tuple

from . import config


VERIFIED_LANES = {"verified-web", "verified-speech", "verified-pdf"}


@dataclasses.dataclass
class StageAdvanceEvent:
    step: int
    from_stage: str
    to_stage: str
    reason: str


@dataclasses.dataclass
class RollbackDemotionEvent:
    step: int
    lane: str
    old_weight: float
    new_weight: float
    stage: str


class MixtureScheduler:
    def __init__(self):
        self.stage_index = 0
        self.step_in_stage = 0
        self.total_step = 0
        self.lane_consumed_effective_tokens: Dict[str, int] = {l: 0 for l in config.LANES}
        self.lane_consumed_total_tokens: Dict[str, int] = {l: 0 for l in config.LANES}
        self.window: Deque[str] = collections.deque(maxlen=config.FLOOR_WINDOW)
        self.val_loss_history: List[float] = []
        self.stage_events: List[StageAdvanceEvent] = []
        self.demotion_events: List[RollbackDemotionEvent] = []
        # mutable per-stage weights, keyed by (stage_index) -> {lane: weight};
        # copied from config at first access, mutated in place by demotion
        self._stage_weights: Dict[int, Dict[str, float]] = {}

    # -- stage/weight access -------------------------------------------
    @property
    def stage(self) -> dict:
        return config.CURRICULUM_STAGES[self.stage_index]

    def active_weights(self) -> Dict[str, float]:
        if self.stage_index not in self._stage_weights:
            self._stage_weights[self.stage_index] = dict(self.stage["weights"])
        return self._stage_weights[self.stage_index]

    def active_lanes(self) -> Tuple[str, ...]:
        return self.stage["lanes"]

    # -- deficit-based lane selection ------------------------------------
    def next_lane(self) -> str:
        weights = self.active_weights()
        lanes = self.active_lanes()
        total_weight = sum(weights[l] for l in lanes)
        total_consumed = sum(self.lane_consumed_effective_tokens[l] for l in lanes)

        best_lane = None
        best_deficit = None
        for lane in sorted(lanes):  # deterministic tie-break: lane name ascending
            target_share = weights[lane] / total_weight if total_weight > 0 else 1.0 / len(lanes)
            deficit = target_share * total_consumed - self.lane_consumed_effective_tokens[lane]
            if best_deficit is None or deficit > best_deficit:
                best_deficit = deficit
                best_lane = lane
        return best_lane

    def lane_deficit(self, lane: str) -> float:
        weights = self.active_weights()
        lanes = self.active_lanes()
        if lane not in weights:
            return 0.0
        total_weight = sum(weights[l] for l in lanes)
        total_consumed = sum(self.lane_consumed_effective_tokens[l] for l in lanes)
        target_share = weights[lane] / total_weight if total_weight > 0 else 1.0 / len(lanes)
        return target_share * total_consumed - self.lane_consumed_effective_tokens[lane]

    def record_consumption(self, lane: str, effective_tokens: int, total_tokens: int) -> None:
        self.lane_consumed_effective_tokens[lane] += effective_tokens
        self.lane_consumed_total_tokens[lane] += total_tokens
        self.window.append(lane)
        self.total_step += 1
        self.step_in_stage += 1

    # -- protected floor ---------------------------------------------------
    def floor_breach(self) -> bool:
        if len(self.window) < min(config.FLOOR_WINDOW, 4):
            return False  # not enough history yet to judge meaningfully
        verified_count = sum(1 for l in self.window if l in VERIFIED_LANES)
        share = verified_count / len(self.window)
        return share < config.PROTECTED_FLOOR_VERIFIED_SHARE

    def planned_vs_actual(self) -> Dict[str, Dict[str, float]]:
        weights = self.active_weights()
        lanes = self.active_lanes()
        total_weight = sum(weights[l] for l in lanes)
        total_consumed = sum(self.lane_consumed_effective_tokens[l] for l in lanes)
        out = {}
        for lane in lanes:
            planned = weights[lane] / total_weight if total_weight > 0 else 1.0 / len(lanes)
            actual = (self.lane_consumed_effective_tokens[lane] / total_consumed) if total_consumed > 0 else 0.0
            out[lane] = {"planned_share": planned, "actual_share": actual}
        return out

    # -- stage advancement ---------------------------------------------
    def record_validation(self, val_loss: float) -> None:
        self.val_loss_history.append(val_loss)

    def maybe_advance_stage(self, step: int) -> Optional[StageAdvanceEvent]:
        if self.stage_index >= len(config.CURRICULUM_STAGES) - 1:
            return None

        reason = None
        if self.step_in_stage >= config.STAGE_ADVANCE_MAX_STEPS:
            reason = f"step_count_fallback (step_in_stage={self.step_in_stage} >= {config.STAGE_ADVANCE_MAX_STEPS})"
        elif len(self.val_loss_history) >= 2 and self.stage.get("min_steps", 0) <= self.step_in_stage:
            last2 = self.val_loss_history[-2:]
            improvement = last2[0] - last2[1]
            if improvement < config.STAGE_ADVANCE_VAL_IMPROVEMENT_EPS:
                reason = f"validation_plateau (improvement={improvement:.5f} < eps)"

        if reason is None:
            return None

        from_name = self.stage["name"]
        self.stage_index += 1
        self.step_in_stage = 0
        self.val_loss_history = []
        to_name = self.stage["name"]
        event = StageAdvanceEvent(step=step, from_stage=from_name, to_stage=to_name, reason=reason)
        self.stage_events.append(event)
        return event

    # -- rollback demotion -----------------------------------------------
    def demote_lane(self, lane: str, step: int) -> RollbackDemotionEvent:
        weights = self.active_weights()
        old = weights.get(lane, 0.0)
        new = old * config.ROLLBACK_WEIGHT_DEMOTION_FACTOR
        weights[lane] = new
        # renormalize active lane weights to sum to 1, so downstream
        # deficit math continues to interpret weights as shares
        total = sum(weights[l] for l in self.active_lanes())
        if total > 0:
            for l in self.active_lanes():
                weights[l] = weights[l] / total
        event = RollbackDemotionEvent(step=step, lane=lane, old_weight=old, new_weight=weights[lane],
                                       stage=self.stage["name"])
        self.demotion_events.append(event)
        return event

    # -- snapshot for checkpointing ---------------------------------------
    def snapshot(self) -> dict:
        return {
            "stage_index": self.stage_index,
            "step_in_stage": self.step_in_stage,
            "total_step": self.total_step,
            "lane_consumed_effective_tokens": dict(self.lane_consumed_effective_tokens),
            "lane_consumed_total_tokens": dict(self.lane_consumed_total_tokens),
            "window": list(self.window),
            "val_loss_history": list(self.val_loss_history),
            "stage_weights": {str(k): dict(v) for k, v in self._stage_weights.items()},
            "stage_events": [dataclasses.asdict(e) for e in self.stage_events],
            "demotion_events": [dataclasses.asdict(e) for e in self.demotion_events],
        }

    @staticmethod
    def restore(snap: dict) -> "MixtureScheduler":
        s = MixtureScheduler()
        s.stage_index = snap["stage_index"]
        s.step_in_stage = snap["step_in_stage"]
        s.total_step = snap["total_step"]
        s.lane_consumed_effective_tokens = dict(snap["lane_consumed_effective_tokens"])
        s.lane_consumed_total_tokens = dict(snap["lane_consumed_total_tokens"])
        s.window = collections.deque(snap["window"], maxlen=config.FLOOR_WINDOW)
        s.val_loss_history = list(snap["val_loss_history"])
        s._stage_weights = {int(k): dict(v) for k, v in snap["stage_weights"].items()}
        s.stage_events = [StageAdvanceEvent(**e) for e in snap.get("stage_events", [])]
        s.demotion_events = [RollbackDemotionEvent(**e) for e in snap.get("demotion_events", [])]
        return s
