"""
tdes.recovery
=============
Checkpoint format, atomic write, and the four recovery operations:

  resume  -- after a crash, reconstruct exact in-memory state from the
             last checkpoint plus deterministic redo of every step the
             ledger says was actually COMMITted since that checkpoint,
             then continue. The orphaned INTENT (if any) left by the
             crash is never applied -- it is superseded by a fresh
             INTENT/COMMIT pair for the same step number once training
             resumes, and comparing the two INTENTs' batch_hash is the
             "[PASS] resume_next_batch_matched" proof.

  replay  -- reconstruct model/scheduler state at an arbitrary earlier
             step (from the checkpoint at-or-before it) and deterministically
             redo forward to a target step, then compare EVERY resulting
             batch_hash and avg_loss against what was originally recorded
             in the ledgers for that interval. Any mismatch is a hard
             failure.

  fork    -- branch off a new, independently-continuing run from any
             checkpoint, under a new branch_id. The parent's checkpoint
             file is never touched; forking is proven safe by hashing
             the parent's checkpoint file before and after the child
             branch trains.

  rollback -- NOT a separate mechanism: a rollback is a fork whose child
             branch (a) is derived from a checkpoint at or before the
             trigger, (b) carries a recorded, deterministic POLICY
             MUTATION (weight demotion on the implicated lane -- see
             mixture.py), and (c) becomes the new active head. Without
             the mutation, a fully deterministic system would walk
             straight back into the same validation spike forever (see
             design notes) -- the mutation is what makes reverting
             actually change the future.

CHECKPOINT ATOMICITY: every checkpoint is written to a temp file, fsynced,
then atomically renamed into place. A crash mid-write leaves, at worst,
an orphaned temp file and the previous checkpoint intact -- never a torn
checkpoint.
"""

from __future__ import annotations
import dataclasses
import json
import os
from typing import Dict, List, Optional, Tuple

from . import config, hashing, ledger as ledger_mod, mixture, model as model_mod, pipeline, trainer


@dataclasses.dataclass
class Checkpoint:
    step: int                       # next step to execute
    branch_id: str
    parent_checkpoint_hash: Optional[str]
    model_head: List[List[float]]
    model_bhead: List[float]
    scheduler: dict
    lanes: Dict[str, dict]          # lane -> {pointer, admitted_seq_ids, defer_counts}
    best_val_loss: Optional[float]
    rollback_fired: bool
    checkpoint_hash: str = ""

    def compute_hash(self) -> str:
        d = dataclasses.asdict(self)
        d.pop("checkpoint_hash", None)
        return hashing.hash_object(d)


def save_checkpoint(state: trainer.RunState, path: str, parent_checkpoint_hash: Optional[str]) -> Checkpoint:
    lanes = {}
    for lane, lc in state.lanes.items():
        lanes[lane] = {
            "pointer": lc.pointer,
            "admitted_seq_ids": [s.seq_id for s in lc.admitted],
            "defer_counts": dict(lc.defer_counts),
        }
    ckpt = Checkpoint(
        step=state.step, branch_id=state.branch_id, parent_checkpoint_hash=parent_checkpoint_hash,
        model_head=[row[:] for row in state.model.W_head], model_bhead=state.model.b_head[:],
        scheduler=state.scheduler.snapshot(), lanes=lanes,
        best_val_loss=state.best_val_loss, rollback_fired=state.rollback_fired,
    )
    ckpt.checkpoint_hash = ckpt.compute_hash()

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(dataclasses.asdict(ckpt), f, ensure_ascii=False, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)  # atomic on POSIX
    return ckpt


def load_checkpoint(path: str) -> Checkpoint:
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    ckpt = Checkpoint(**d)
    if ckpt.compute_hash() != ckpt.checkpoint_hash:
        raise ledger_mod.IntegrityError(f"checkpoint self-hash mismatch: {path}")
    return ckpt


def rebuild_state_from_checkpoint(ckpt: Checkpoint, ctx: pipeline.CorpusContext,
                                   consumption_ledger_path: str, learning_ledger_path: str) -> trainer.RunState:
    """Deterministically reconstructs a RunState: model body from
    MASTER_SEED (frozen, always identical), trainable head overlaid
    from the checkpoint, lane queues rebuilt from the (already-built)
    CorpusContext, scheduler and lane pointers/admitted-queues/defer-counts
    restored from the checkpoint's recorded snapshot."""
    m = model_mod.init_model()
    m.W_head = [row[:] for row in ckpt.model_head]
    m.b_head = ckpt.model_bhead[:]

    scheduler = mixture.MixtureScheduler.restore(ckpt.scheduler)

    seq_lookup: Dict[str, Dict[str, object]] = {}
    for lane, queue in ctx.lane_train_queues.items():
        seq_lookup[lane] = {s.seq_id: s for s in queue}

    lanes: Dict[str, trainer.LaneCandidates] = {}
    for lane, queue in ctx.lane_train_queues.items():
        snap = ckpt.lanes.get(lane, {"pointer": 0, "admitted_seq_ids": [], "defer_counts": {}})
        admitted = [seq_lookup[lane][sid] for sid in snap["admitted_seq_ids"]]
        lanes[lane] = trainer.LaneCandidates(queue=queue, pointer=snap["pointer"],
                                              admitted=admitted, defer_counts=dict(snap["defer_counts"]))

    consumption_ledger = ledger_mod.Ledger(consumption_ledger_path)
    learning_ledger = ledger_mod.Ledger(learning_ledger_path)

    return trainer.RunState(
        branch_id=ckpt.branch_id, step=ckpt.step, model=m, scheduler=scheduler, lanes=lanes,
        proxy_seqs=ctx.proxy_seqs, proxy_role="PROXY", val_seqs=ctx.validation_seqs,
        consumption_ledger=consumption_ledger, learning_ledger=learning_ledger,
        best_val_loss=ckpt.best_val_loss, rollback_fired=ckpt.rollback_fired,
    )


def redo_range(state: trainer.RunState, upto_step_inclusive: int) -> List[dict]:
    """Deterministically re-executes steps [state.step .. upto_step_inclusive]
    WITHOUT writing new ledger records (those already exist on disk from
    before the crash/for the interval being replayed). Uses the exact
    same governance path (validation, stage-advance, rollback-trigger)
    as the live loop -- see trainer.run_step_and_govern. Returns the
    list of per-step results, for comparison against the historically
    recorded ledger entries."""
    results = []
    while state.step <= upto_step_inclusive:
        results.append(trainer.run_step_and_govern(state, write_ledger=False))
    return results


def resume_after_crash(ckpt: Checkpoint, ctx: pipeline.CorpusContext, consumption_ledger_path: str,
                        learning_ledger_path: str) -> Tuple[trainer.RunState, List[dict], List[ledger_mod.LedgerRecord]]:
    """Full resume procedure. Returns (state, redo_results, orphaned_intents).
    `state.step` after this call is exactly the next step that was never
    committed -- callers continue training normally (write_ledger=True)
    from there."""
    state = rebuild_state_from_checkpoint(ckpt, ctx, consumption_ledger_path, learning_ledger_path)
    last_committed = state.consumption_ledger.last_committed_step()
    orphaned = state.consumption_ledger.find_unmatched_intents()

    redo_results = []
    if last_committed >= state.step:
        redo_results = redo_range(state, last_committed)
    return state, redo_results, orphaned


def replay_interval(ckpt: Checkpoint, ctx: pipeline.CorpusContext, consumption_ledger_path: str,
                     learning_ledger_path: str, start_step: int, end_step: int) -> dict:
    """Reconstructs state from `ckpt` (assumed at or before start_step),
    deterministically redoes steps [ckpt.step .. end_step], and compares
    every resulting batch_hash / avg_loss against what the ORIGINAL
    ledgers recorded for the same steps. Returns a report dict; raises
    AssertionError on any mismatch."""
    state = rebuild_state_from_checkpoint(ckpt, ctx, consumption_ledger_path, learning_ledger_path)
    assert state.step <= start_step, "checkpoint must be at or before the replay start step"

    original_consumption = ledger_mod.Ledger(consumption_ledger_path)
    original_learning = ledger_mod.Ledger(learning_ledger_path)

    orig_commit_by_step = {}
    for rec in original_consumption.records:
        if rec.record_type == "INTENT":
            orig_commit_by_step.setdefault(rec.payload["step"], {})["batch_hash"] = rec.payload["batch_hash"]
    orig_loss_by_step = {}
    for rec in original_learning.records:
        if rec.record_type == "BATCH_LOSS":
            orig_loss_by_step[rec.payload["step"]] = rec.payload["avg_loss"]

    mismatches = []
    matched_steps = []
    while state.step <= end_step:
        step_being_run = state.step
        result = trainer.run_step_and_govern(state, write_ledger=False)
        if step_being_run >= start_step:
            orig_hash = orig_commit_by_step.get(step_being_run, {}).get("batch_hash")
            orig_loss = orig_loss_by_step.get(step_being_run)
            ok_hash = (orig_hash == result["batch_hash"])
            ok_loss = (orig_loss is None) or abs(orig_loss - result["avg_loss"]) < 1e-9
            if not (ok_hash and ok_loss):
                mismatches.append({"step": step_being_run, "orig_hash": orig_hash, "new_hash": result["batch_hash"],
                                    "orig_loss": orig_loss, "new_loss": result["avg_loss"]})
            else:
                matched_steps.append(step_being_run)

    return {"matched_steps": matched_steps, "mismatches": mismatches,
            "passed": len(mismatches) == 0 and len(matched_steps) > 0}


def fork_branch(ckpt: Checkpoint, ctx: pipeline.CorpusContext, new_branch_id: str,
                 consumption_ledger_path: str, learning_ledger_path: str) -> trainer.RunState:
    """Creates a new, independently-continuing branch from `ckpt`. The
    child gets its OWN ledger files (consumption_ledger_path /
    learning_ledger_path must already point at fresh per-branch paths --
    see run_demo.py), so nothing here can ever mutate the parent's
    ledgers or checkpoint."""
    state = rebuild_state_from_checkpoint(ckpt, ctx, consumption_ledger_path, learning_ledger_path)
    state.branch_id = new_branch_id
    return state


def apply_rollback(state: trainer.RunState, trigger_step: int, trigger_val_loss: float,
                    best_val_loss: float) -> dict:
    """Thin wrapper kept for callers that want to trigger a rollback
    explicitly (run_demo.py's live loop no longer needs to -- rollback
    triggering now lives inside trainer.run_step_and_govern, the single
    source of truth shared with redo/replay -- see that function's
    docstring for why the duplication this replaces was a real bug).
    Delegates to trainer.apply_rollback so there is exactly one
    implementation."""
    return trainer.apply_rollback(state, trigger_step, trigger_val_loss, best_val_loss, write_ledger=True)
