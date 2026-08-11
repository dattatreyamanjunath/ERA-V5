"""
tdes.trainer
============
Orchestrates one training step end to end:

  scheduler.next_lane() [x BATCH_SIZE] -> pop admitted candidates
      -> WAL INTENT -> forward+backward+SGD -> learning ledger
      -> WAL COMMIT -> mixture.record_consumption -> (periodic) checkpoint

OPUS scoring is decoupled from batch consumption: a scoring round (every
config.OPUS_SCORE_EVERY_N_STEPS steps) advances each active lane's
candidate frontier through the OPUS DAG and appends ACCEPTed candidates
to that lane's admitted queue (Boltzmann-ordered for diversity). Batch
formation only ever draws from admitted queues -- if a lane's admitted
queue is empty when needed, a small fallback scoring round is run for
just that lane so the demo never stalls waiting for the next scheduled
round.

This module contains the ONLY code path that may call
firewall.admit_for_training with a batch destined for the optimizer --
see packing.pack_lane and opus.run_opus_dag, both of which also assert
the same gate independently (defense in depth, not redundancy for its
own sake: three independent call sites must all agree a shard is TRAIN
before any of its tokens influence a gradient).
"""

from __future__ import annotations
import dataclasses
from typing import Dict, List, Optional, Set

from . import config, hashing, firewall, model as model_mod, mixture, opus, ledger
from .packing import PackedSequence


@dataclasses.dataclass
class LaneCandidates:
    queue: List[PackedSequence]
    pointer: int = 0
    admitted: List[PackedSequence] = dataclasses.field(default_factory=list)
    defer_counts: Dict[str, int] = dataclasses.field(default_factory=dict)
    decisions: List[opus.OpusDecision] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class RunState:
    branch_id: str
    step: int
    model: model_mod.ModelState
    scheduler: mixture.MixtureScheduler
    lanes: Dict[str, LaneCandidates]
    proxy_seqs: List[PackedSequence]
    proxy_role: str
    val_seqs: List[PackedSequence]
    consumption_ledger: ledger.Ledger
    learning_ledger: ledger.Ledger
    rollback_count: int = 0
    last_val_loss: Optional[float] = None
    best_val_loss: Optional[float] = None
    rollback_fired: bool = False
    lane_role: str = "TRAIN"


def batch_hash(seq_ids: List[str]) -> str:
    return hashing.hash_object({"seq_ids": sorted(seq_ids)})


def _score_lane_round(state: RunState, lane: str, budget: int, proxy_sketch: List[float]) -> int:
    """Advances `lane`'s candidate frontier through the OPUS DAG,
    consuming up to `budget` candidates. Returns how many were
    actually processed (fewer than budget if the lane's queue is
    exhausted or the frontier hits a DEFER, which stops that lane for
    this round -- see module docstring). `proxy_sketch` is computed
    ONCE per scoring round by the caller (see maybe_run_scoring_round)
    and shared across all lanes scored in that round, since the model
    state -- and therefore the proxy direction -- does not change
    between lanes within a single round (only a completed batch commit
    changes it)."""
    lc = state.lanes[lane]
    processed = 0
    accepted_this_round = []
    while processed < budget and lc.pointer < len(lc.queue):
        cand = lc.queue[lc.pointer]
        prior_defers = lc.defer_counts.get(cand.seq_id, 0)
        floor_breach = state.scheduler.floor_breach() and (lane in mixture.VERIFIED_LANES)
        decision = opus.run_opus_dag(
            candidate=cand, candidate_role="TRAIN", proxy_direction_sketch=proxy_sketch,
            m=state.model, branch_id=state.branch_id, step=state.step, seq_index=lc.pointer,
            prior_defer_count=prior_defers, lane_deficit=state.scheduler.lane_deficit(lane),
            lane_floor_breach=floor_breach,
        )
        lc.decisions.append(decision)
        processed += 1

        if decision.classification == "ACCEPT":
            accepted_this_round.append(cand)
            lc.pointer += 1
        elif decision.classification == "REJECT":
            lc.pointer += 1
        else:  # DEFER
            lc.defer_counts[cand.seq_id] = decision.defer_count
            break  # stop this lane for the round; retry next round

    if accepted_this_round:
        scored = [(c.seq_id, next(d.score for d in reversed(lc.decisions) if d.candidate_id == c.seq_id))
                  for c in accepted_this_round]
        order = opus.boltzmann_order(scored, state.branch_id, state.step)
        by_id = {c.seq_id: c for c in accepted_this_round}
        lc.admitted.extend(by_id[sid] for sid in order)

    return processed


def _compute_proxy_sketch(state: RunState) -> List[float]:
    proxy_batch = state.proxy_seqs[:config.OPUS_PROXY_BATCH_SIZE]
    direction = opus.compute_proxy_direction(
        state.model, proxy_batch, [state.proxy_role] * len(proxy_batch), state.branch_id, state.step,
    )
    return opus.countsketch_project(direction, config.OPUS_SKETCH_DIM)


def _ensure_admitted(state: RunState, lane: str) -> bool:
    """Returns True if `lane` has at least one admitted candidate ready
    (running a small fallback scoring round if needed). Returns False
    only if the lane's candidate pool is fully exhausted."""
    lc = state.lanes[lane]
    if lc.admitted:
        return True
    if lc.pointer >= len(lc.queue):
        return False
    proxy_sketch = _compute_proxy_sketch(state)
    _score_lane_round(state, lane, max(4, config.OPUS_CANDIDATE_POOL // len(state.lanes)), proxy_sketch)
    return bool(lc.admitted)


def maybe_run_scoring_round(state: RunState) -> None:
    if state.step % config.OPUS_SCORE_EVERY_N_STEPS != 0:
        return
    active = sorted(state.scheduler.active_lanes())
    if not active:
        return
    proxy_sketch = _compute_proxy_sketch(state)
    per_lane_budget = max(1, config.OPUS_CANDIDATE_POOL // len(active))
    for lane in active:
        _score_lane_round(state, lane, per_lane_budget, proxy_sketch)


def form_batch(state: RunState) -> List[PackedSequence]:
    batch: List[PackedSequence] = []
    exhausted: Set[str] = set()
    while len(batch) < config.BATCH_SIZE:
        active = [l for l in state.scheduler.active_lanes() if l not in exhausted]
        if not active:
            break  # entire active lane set exhausted -- demo corpus too small; caller handles short batch
        # pick lane with the largest deficit among non-exhausted active lanes
        weights = state.scheduler.active_weights()
        best_lane, best_deficit = None, None
        for lane in sorted(active):
            d = state.scheduler.lane_deficit(lane)
            if best_deficit is None or d > best_deficit:
                best_deficit, best_lane = d, lane
        if not _ensure_admitted(state, best_lane):
            exhausted.add(best_lane)
            continue
        seq = state.lanes[best_lane].admitted.pop(0)
        batch.append(seq)
        eff = sum(seq.loss_mask)
        tot = len(seq.input_ids)
        state.scheduler.record_consumption(best_lane, eff, tot)
    return batch


class CrashSimulated(Exception):
    """Raised deliberately by run_demo.py's crash-injection hook to
    prove the WAL protocol: raised AFTER the INTENT record is durably
    on disk but BEFORE compute/COMMIT, so the batch is left ambiguous
    exactly as a real process crash would leave it."""
    pass


def run_training_step(state: RunState, write_ledger: bool = True, crash_hook=None) -> dict:
    """Runs exactly one training step: form batch, WAL intent, compute,
    WAL commit, update learning ledger. When write_ledger=False (used
    during checkpoint-catch-up replay after a crash -- see recovery.py),
    the model/scheduler are still advanced but no new ledger records are
    written, since the original records for that step already exist on
    disk from before the crash."""
    maybe_run_scoring_round(state)
    batch = form_batch(state)
    seq_ids = [s.seq_id for s in batch]
    bhash = batch_hash(seq_ids)

    intent_payload = {
        "step": state.step, "batch_id": f"batch-{state.branch_id}-{state.step}",
        "batch_hash": bhash, "seq_ids": seq_ids,
        "lanes": [s.lane for s in batch],
    }
    if write_ledger:
        state.consumption_ledger.append("INTENT", intent_payload)
        if crash_hook is not None and crash_hook(state.step):
            raise CrashSimulated(f"simulated crash after INTENT for step {state.step}")

    total_loss = 0.0
    total_loss_tokens = 0
    per_lane_loss: Dict[str, List[float]] = {}
    accum_grad = None
    per_seq_records = []

    for idx, seq in enumerate(batch):
        loss, n_loss, grads, per_tok = model_mod.forward_and_loss(
            state.model, seq.input_ids, seq.position_ids, seq.attention_block_ids, seq.loss_mask,
            state.branch_id, state.step, idx, training=True,
        )
        total_loss += loss * n_loss
        total_loss_tokens += n_loss
        per_lane_loss.setdefault(seq.lane, []).append(loss)
        per_seq_records.append({
            "seq_id": seq.seq_id, "lane": seq.lane, "loss": loss, "n_loss_tokens": n_loss,
            "doc_ids": seq.doc_ids_present,
        })
        if accum_grad is None:
            accum_grad = {"d_head": [row[:] for row in grads["d_head"]], "d_bhead": grads["d_bhead"][:]}
        else:
            for i, row in enumerate(grads["d_head"]):
                arow = accum_grad["d_head"][i]
                for j, v in enumerate(row):
                    arow[j] += v
            for j, v in enumerate(grads["d_bhead"]):
                accum_grad["d_bhead"][j] += v

    if batch and accum_grad is not None:
        model_mod.sgd_update(state.model, accum_grad, max(total_loss_tokens, 1), config.MODEL_LR)

    avg_loss = (total_loss / total_loss_tokens) if total_loss_tokens else 0.0

    commit_payload = {"step": state.step, "batch_id": intent_payload["batch_id"],
                       "n_sequences": len(batch), "n_loss_tokens": total_loss_tokens}
    if write_ledger:
        state.consumption_ledger.append("COMMIT", commit_payload)
        state.learning_ledger.append("BATCH_LOSS", {
            "step": state.step, "batch_id": intent_payload["batch_id"], "avg_loss": avg_loss,
            "per_lane_avg_loss": {l: sum(v) / len(v) for l, v in per_lane_loss.items()},
            "sequences": per_seq_records,
        })

    result = {"step": state.step, "avg_loss": avg_loss, "batch_hash": bhash,
              "n_sequences": len(batch), "n_loss_tokens": total_loss_tokens}
    state.step += 1
    return result


def run_validation(state: RunState) -> float:
    losses = []
    sample = state.val_seqs[:config.VALIDATION_SAMPLE_SIZE]
    for idx, seq in enumerate(sample):
        firewall.admit_for_validation("VALIDATION", seq.seq_id)
        loss, n_loss, _grads, _pt = model_mod.forward_and_loss(
            state.model, seq.input_ids, seq.position_ids, seq.attention_block_ids, seq.loss_mask,
            state.branch_id, state.step, -1000 - idx, training=False,
        )
        if n_loss > 0:
            losses.append(loss)
    val_loss = sum(losses) / len(losses) if losses else 0.0
    state.last_val_loss = val_loss
    state.scheduler.record_validation(val_loss)
    return val_loss


def apply_rollback(state: RunState, trigger_step: int, trigger_val_loss: float, best_val_loss: float,
                    write_ledger: bool) -> dict:
    """The deterministic policy mutation that makes a rollback meaningful
    (see recovery.py module docstring for the full rationale): demotes
    the weight of whichever currently-active lane has the largest actual
    consumed share, then records a ROLLBACK event. Called from
    run_step_and_govern so that a rollback triggered during the ORIGINAL
    run is deterministically re-triggered, identically, during
    redo/replay -- both paths compute the same trigger condition from
    the same state and therefore reach the same decision independently,
    rather than one having to replay a recorded decision from the other."""
    pva = state.scheduler.planned_vs_actual()
    implicated_lane = max(pva.keys(), key=lambda l: pva[l]["actual_share"])
    demotion = state.scheduler.demote_lane(implicated_lane, trigger_step)
    payload = {
        "step": trigger_step, "branch_id": state.branch_id, "trigger_val_loss": trigger_val_loss,
        "best_val_loss": best_val_loss, "margin": config.ROLLBACK_VAL_DEGRADE_MARGIN,
        "implicated_lane": implicated_lane, "old_weight": demotion.old_weight, "new_weight": demotion.new_weight,
    }
    if write_ledger:
        state.consumption_ledger.append("ROLLBACK", payload)
    state.rollback_count += 1
    return payload


def run_step_and_govern(state: RunState, write_ledger: bool = True, crash_hook=None) -> dict:
    """Single source of truth for 'advance the run by one step', used
    by BOTH the live training loop (run_demo.py) and every recovery
    path (redo_range / replay_interval). Bundles the training step with
    the validation + stage-advance + rollback-trigger governance that
    follows it in the original run -- this is deliberate: those
    governance decisions are themselves deterministic functions of
    state, so replaying them independently during redo must reach
    identical decisions to the original run, not merely reproduce their
    recorded outcome. Before this function existed, redo only replayed
    the training step and silently diverged from the original run
    whenever a validation-triggered stage advance or rollback had
    occurred -- exactly the class of bug tests/test_recovery.py and the
    replay demo are meant to catch.
    """
    result = run_training_step(state, write_ledger=write_ledger, crash_hook=crash_hook)

    if state.step % config.VALIDATION_EVERY_N_STEPS == 0:
        val_loss = run_validation(state)
        if state.best_val_loss is None or val_loss < state.best_val_loss:
            state.best_val_loss = val_loss
        elif (not state.rollback_fired) and val_loss > state.best_val_loss + config.ROLLBACK_VAL_DEGRADE_MARGIN:
            state.rollback_fired = True
            result["rollback"] = apply_rollback(state, state.step, val_loss, state.best_val_loss, write_ledger)
        adv = state.scheduler.maybe_advance_stage(state.step)
        if adv:
            result["stage_advance"] = adv
        result["val_loss"] = val_loss

    return result
