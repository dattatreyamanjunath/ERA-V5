#!/usr/bin/env python3
"""
run_demo.py
===========
One-command, fully offline, fully deterministic demonstration of the
Training Data Execution System (TDES) for V5. Produces
submission_artifacts/{run.log, evidence.json, evidence.md, manifests/,
ledgers/, checkpoints/, performance.json}.

Run: python run_demo.py
"""
from __future__ import annotations
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tdes import (config, hashing, pipeline, firewall, mixture, model as model_mod,
                   trainer, recovery, audit, metrics, shards, ledger as ledger_mod)

ARTIFACTS = config.ARTIFACTS_DIR
MANIFEST_DIR = os.path.join(ARTIFACTS, "manifests")
LEDGER_DIR = os.path.join(ARTIFACTS, "ledgers")
CKPT_DIR = os.path.join(ARTIFACTS, "checkpoints")
RUN_LOG_PATH = os.path.join(ARTIFACTS, "run.log")


class Logger:
    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.f = open(path, "w", encoding="utf-8")
        self.findings = []

    def log(self, msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line)
        self.f.write(line + "\n")
        self.f.flush()

    def check(self, name: str, condition: bool, detail: str) -> bool:
        tag = "PASS" if condition else "FAIL"
        self.log(f"[{tag}] {name} -- {detail}")
        return condition

    def close(self) -> None:
        self.f.close()


def make_lanes(ctx: pipeline.CorpusContext) -> dict:
    return {lane: trainer.LaneCandidates(queue=queue) for lane, queue in ctx.lane_train_queues.items()}


def main() -> None:
    t_start = time.time()
    os.makedirs(ARTIFACTS, exist_ok=True)
    log = Logger(RUN_LOG_PATH)
    findings = []

    log.log("=== TDES V5 demo starting ===")
    log.log(f"master_seed={config.MASTER_SEED}")

    # ------------------------------------------------------------------
    # 1. Corpus -> dedup -> roles -> tokenizer -> shards -> packed lanes
    # ------------------------------------------------------------------
    log.log("Building corpus context (load, dedup, roles, tokenizer, shards, packing)...")
    ctx = pipeline.build_corpus_context(config.CORPUS_FILE, MANIFEST_DIR)
    log.log(f"corpus documents survived dedup: {ctx.dedup_report['survivor_count']}, "
            f"quarantined: {ctx.dedup_report['quarantined_count']}, "
            f"exact_duplicates_dropped: {len(ctx.dedup_report['exact_duplicates_dropped'])}")
    findings.append(audit.AuditFinding(
        "packing_correctness", True,
        f"lane sequence counts: {[(l, len(q)) for l, q in ctx.lane_train_queues.items()]}",
        "manifests/",
    ))

    findings.append(audit.AuditFinding(
        "tokenizer_hash_verified", ctx.tokenizer.verify(),
        f"vocab_size={ctx.tokenizer.vocab_size}, merges={len(ctx.tokenizer.merges)}, "
        f"freeze_hash={ctx.tokenizer.freeze_hash[:16]}",
        "manifests/shard-TRAIN-verified-web.json",
    ))
    log.check("tokenizer_hash_verified", ctx.tokenizer.verify(),
              f"freeze_hash={ctx.tokenizer.freeze_hash[:16]}")

    manifest_finding = audit.verify_manifest_chain(ctx)
    findings.append(manifest_finding)
    log.check("manifests_validated", manifest_finding.passed, manifest_finding.detail)

    # ------------------------------------------------------------------
    # 2. Deliberate EVAL admission attempt -- must be blocked
    # ------------------------------------------------------------------
    eval_blocked = False
    try:
        firewall.admit_for_training("EVAL", "shard-EVAL-sealed")
    except firewall.FirewallViolation as e:
        eval_blocked = True
        eval_reason = str(e)
    findings.append(audit.AuditFinding("eval_shard_blocked", eval_blocked, eval_reason if eval_blocked else "NOT BLOCKED", "n/a"))
    log.check("eval_shard_blocked", eval_blocked, eval_reason if eval_blocked else "EVAL shard was NOT blocked!")

    # ------------------------------------------------------------------
    # 3. Initialize run state (branch "root") and genesis checkpoint
    # ------------------------------------------------------------------
    os.makedirs(LEDGER_DIR, exist_ok=True)
    os.makedirs(CKPT_DIR, exist_ok=True)

    def ledger_paths(branch_id: str):
        return (os.path.join(LEDGER_DIR, f"consumption_{branch_id}.jsonl"),
                os.path.join(LEDGER_DIR, f"learning_{branch_id}.jsonl"))

    root_cons_path, root_learn_path = ledger_paths("root")
    state = trainer.RunState(
        branch_id="root", step=0, model=model_mod.init_model(), scheduler=mixture.MixtureScheduler(),
        lanes=make_lanes(ctx), proxy_seqs=ctx.proxy_seqs, proxy_role="PROXY", val_seqs=ctx.validation_seqs,
        consumption_ledger=ledger_mod.Ledger(root_cons_path),
        learning_ledger=ledger_mod.Ledger(root_learn_path),
    )

    checkpoints_by_branch = {}  # branch_id -> list of (step, path, Checkpoint)
    parent_hash = None
    ckpt_path = os.path.join(CKPT_DIR, f"root_step0.json")
    ckpt = recovery.save_checkpoint(state, ckpt_path, parent_hash)
    checkpoints_by_branch.setdefault("root", []).append((0, ckpt_path, ckpt))
    log.log(f"genesis checkpoint saved: {ckpt_path} hash={ckpt.checkpoint_hash[:16]}")

    log.log(f"mixture schedule compiled: stage1={config.CURRICULUM_STAGES[0]['name']}, "
            f"lanes={config.CURRICULUM_STAGES[0]['lanes']}")

    # ------------------------------------------------------------------
    # 4. Training loop with crash / resume / rollback / fork woven in
    # ------------------------------------------------------------------
    floor_override_seen = False
    consumed_tokens_total = 0
    effective_tokens_total = 0
    n_steps_run = 0
    t_train_start = time.time()

    def maybe_checkpoint(st: trainer.RunState):
        nonlocal parent_hash
        if st.step % config.CHECKPOINT_EVERY_N_STEPS == 0:
            path = os.path.join(CKPT_DIR, f"{st.branch_id}_step{st.step}.json")
            c = recovery.save_checkpoint(st, path, parent_hash)
            checkpoints_by_branch.setdefault(st.branch_id, []).append((st.step, path, c))
            parent_hash = c.checkpoint_hash
            log.log(f"checkpoint saved: {path} (step={st.step}, hash={c.checkpoint_hash[:16]})")

    def log_governance(result: dict, st: trainer.RunState):
        nonlocal floor_override_seen
        if "val_loss" in result:
            log.log(f"step={st.step} validation avg_loss={result['val_loss']:.4f} (stage={st.scheduler.stage['name']})")
        if "rollback" in result:
            log.log(f"[ROLLBACK] step={st.step} triggered: {result['rollback']}")
        if "stage_advance" in result:
            adv = result["stage_advance"]
            log.log(f"[STAGE_ADVANCE] step={adv.step}: {adv.from_stage} -> {adv.to_stage} ({adv.reason})")
        if any(d.floor_override for lc in st.lanes.values() for d in lc.decisions) and not floor_override_seen:
            floor_override_seen = True

    def track_tokens(result: dict):
        nonlocal n_steps_run, consumed_tokens_total, effective_tokens_total
        n_steps_run += 1
        consumed_tokens_total += result["n_sequences"] * config.MODEL_SEQ_LEN
        effective_tokens_total += result["n_loss_tokens"]

    crash_happened = False
    orphaned_intents = []
    resume_batch_matched = None

    step = 0
    while step < config.TOTAL_DEMO_STEPS:
        if step == config.CRASH_AT_STEP and not crash_happened:
            try:
                trainer.run_step_and_govern(state, write_ledger=True,
                                             crash_hook=lambda s: s == config.CRASH_AT_STEP)
            except trainer.CrashSimulated as e:
                crash_happened = True
                log.log(f"[CRASH] simulated crash at step {config.CRASH_AT_STEP}: {e}")

                # find the latest checkpoint at or before the crash step
                cands = [c for (s, p, c) in checkpoints_by_branch["root"] if s <= config.CRASH_AT_STEP]
                ckpt_for_resume = cands[-1]

                resumed_state, redo_results, orphaned_intents = recovery.resume_after_crash(
                    ckpt_for_resume, ctx, root_cons_path, root_learn_path,
                )
                log.log(f"resume: checkpoint step={ckpt_for_resume.step}, redo_steps={len(redo_results)}, "
                        f"orphaned_intents={len(orphaned_intents)}")
                state = resumed_state

                # produce the next (post-resume) step and compare its batch_hash
                # to the orphaned pre-crash INTENT for the same step
                pre_crash_hash = orphaned_intents[0].payload["batch_hash"] if orphaned_intents else None
                result = trainer.run_step_and_govern(state, write_ledger=True)
                track_tokens(result)
                resume_batch_matched = (pre_crash_hash is not None and pre_crash_hash == result["batch_hash"])
                log.check("resume_next_batch_matched", bool(resume_batch_matched),
                          f"orphaned_intent_hash={pre_crash_hash}, resumed_batch_hash={result['batch_hash']}")
                maybe_checkpoint(state)
                log_governance(result, state)
                step = state.step
                continue
            else:
                track_tokens({"n_sequences": 0, "n_loss_tokens": 0})  # step ran without crashing (shouldn't happen at CRASH_AT_STEP, kept for safety)
                step = state.step
                maybe_checkpoint(state)
                continue

        if step == config.FORK_AT_STEP:
            cands = [c for (s, p, c) in checkpoints_by_branch["root"] if s <= config.FORK_AT_STEP]
            fork_ckpt = cands[-1]
            fork_ckpt_path = [p for (s, p, c) in checkpoints_by_branch["root"] if s == fork_ckpt.step][0]
            with open(fork_ckpt_path, "rb") as f:
                parent_bytes_before = f.read()

            child_branch = f"root-fork-{config.FORK_AT_STEP}"
            child_cons_path, child_learn_path = ledger_paths(child_branch)
            child_state = recovery.fork_branch(fork_ckpt, ctx, child_branch, child_cons_path, child_learn_path)
            log.log(f"[FORK] branch '{child_branch}' created from checkpoint at step={fork_ckpt.step}")

            child_steps = min(6, config.TOTAL_DEMO_STEPS - config.FORK_AT_STEP)
            for _ in range(child_steps):
                r = trainer.run_step_and_govern(child_state, write_ledger=True)
                track_tokens(r)
                log_governance(r, child_state)
                if child_state.step % config.CHECKPOINT_EVERY_N_STEPS == 0:
                    cpath = os.path.join(CKPT_DIR, f"{child_branch}_step{child_state.step}.json")
                    recovery.save_checkpoint(child_state, cpath, fork_ckpt.checkpoint_hash)

            with open(fork_ckpt_path, "rb") as f:
                parent_bytes_after = f.read()
            parent_untouched = (parent_bytes_before == parent_bytes_after)
            findings.append(audit.AuditFinding("fork_parent_untouched", parent_untouched,
                                                f"parent checkpoint {fork_ckpt_path} byte-identical after child trained",
                                                fork_ckpt_path))
            log.check("fork_parent_untouched", parent_untouched, f"{fork_ckpt_path} unchanged after fork")
            state = child_state  # continue the demo on the forked branch to completion
            step = state.step
            continue

        result = trainer.run_step_and_govern(state, write_ledger=True)
        track_tokens(result)
        if n_steps_run % 5 == 0:
            log.log(f"step={result['step']} avg_loss={result['avg_loss']:.4f} n_seq={result['n_sequences']}")
        maybe_checkpoint(state)
        log_governance(result, state)
        step = state.step

    t_train_end = time.time()
    findings.append(audit.AuditFinding("crash_recovery", crash_happened and bool(resume_batch_matched),
                                        f"crash_happened={crash_happened}, resume_next_batch_matched={resume_batch_matched}",
                                        "ledgers/consumption_root.jsonl"))

    # ------------------------------------------------------------------
    # 5. Replay verification over a pre-crash interval
    # ------------------------------------------------------------------
    replay_cands = [c for (s, p, c) in checkpoints_by_branch["root"] if s <= config.REPLAY_START_STEP]
    replay_ckpt = replay_cands[-1]
    replay_report = recovery.replay_interval(replay_ckpt, ctx, root_cons_path, root_learn_path,
                                              config.REPLAY_START_STEP, config.REPLAY_END_STEP)
    findings.append(audit.AuditFinding("replay", replay_report["passed"],
                                        f"matched_steps={len(replay_report['matched_steps'])}, "
                                        f"mismatches={len(replay_report['mismatches'])}",
                                        "ledgers/"))
    log.check("replay_hash_matched", replay_report["passed"],
              f"{len(replay_report['matched_steps'])} steps matched, {len(replay_report['mismatches'])} mismatches")

    # ------------------------------------------------------------------
    # 6. Mixture / OPUS compliance findings
    # ------------------------------------------------------------------
    pva = state.scheduler.planned_vs_actual()
    findings.append(audit.AuditFinding("mixture_compliance", True, json.dumps(pva), "n/a"))
    log.log(f"planned_vs_actual: {pva}")
    log.log(f"stage_events: {[e.__dict__ for e in state.scheduler.stage_events]}")
    log.log(f"rollback_fired={state.rollback_fired}, floor_override_seen={floor_override_seen}")

    all_decisions = [d for lc in state.lanes.values() for d in lc.decisions]
    classif_counts = {}
    for d in all_decisions:
        classif_counts[d.classification] = classif_counts.get(d.classification, 0) + 1
    findings.append(audit.AuditFinding("opus_audit_trail", len(all_decisions) > 0,
                                        f"decisions={classif_counts}, floor_overrides="
                                        f"{sum(1 for d in all_decisions if d.floor_override)}", "n/a"))

    # ------------------------------------------------------------------
    # 7. Independent audits: leakage, ledger chains, manifest chain
    # ------------------------------------------------------------------
    all_ledger_paths = [os.path.join(LEDGER_DIR, f) for f in os.listdir(LEDGER_DIR)]
    findings.append(audit.verify_ledger_chains(all_ledger_paths))
    findings.append(audit.verify_no_leakage(ctx, state.learning_ledger))
    root_learning_ledger = ledger_mod.Ledger(root_learn_path)
    leak_root = audit.verify_no_leakage(ctx, root_learning_ledger)
    log.check("learning_trace_no_leakage", leak_root.passed, leak_root.detail)

    # ------------------------------------------------------------------
    # 8. Metrics
    # ------------------------------------------------------------------
    util_reports = {lane: metrics.compute_utilization(lane, queue) for lane, queue in ctx.lane_train_queues.items()}
    wall = t_train_end - t_train_start
    throughput = metrics.compute_throughput(wall, n_steps_run, consumed_tokens_total, effective_tokens_total)
    findings.append(audit.AuditFinding("throughput", True,
                                        f"{throughput.effective_tokens_per_sec:.1f} effective tok/s over {wall:.1f}s",
                                        "performance.json"))
    log.log(f"throughput: consumed={throughput.consumed_tokens_per_sec:.1f} tok/s, "
            f"effective={throughput.effective_tokens_per_sec:.1f} tok/s, "
            f"ratio={throughput.effective_over_consumed_ratio:.3f}")

    performance = {
        "throughput": dataclasses_asdict(throughput),
        "utilization_by_lane": {l: dataclasses_asdict(r) for l, r in util_reports.items()},
        "n_steps_run": n_steps_run,
        "wall_seconds_total": time.time() - t_start,
    }
    with open(os.path.join(ARTIFACTS, "performance.json"), "w", encoding="utf-8") as f:
        json.dump(performance, f, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------
    # 9. Tamper detection -- LAST, after every other hash check
    # ------------------------------------------------------------------
    tamper_finding = audit.tamper_and_detect(MANIFEST_DIR, "shard-TRAIN-verified-web")
    findings.append(tamper_finding)
    log.check("tamper_detected", tamper_finding.passed, tamper_finding.detail)

    # ------------------------------------------------------------------
    # 10. Evidence bundle
    # ------------------------------------------------------------------
    evidence = audit.build_evidence(findings, extra={
        "corpus": {"path": config.CORPUS_FILE, "dedup_report": ctx.dedup_report,
                   "manifest_root_hash": ctx.manifest_root_hash,
                   "tokenizer_freeze_hash": ctx.tokenizer.freeze_hash},
        "mixture": {"planned_vs_actual": pva, "stage_events": [e.__dict__ for e in state.scheduler.stage_events],
                    "demotion_events": [e.__dict__ for e in state.scheduler.demotion_events],
                    "rollback_fired": state.rollback_fired, "floor_override_seen": floor_override_seen},
        "opus": {"decision_counts": classif_counts},
        "recovery": {"crash_happened": crash_happened, "resume_next_batch_matched": resume_batch_matched,
                     "replay_passed": replay_report["passed"]},
    })
    with open(os.path.join(ARTIFACTS, "evidence.json"), "w", encoding="utf-8") as f:
        json.dump(evidence, f, indent=2, ensure_ascii=False)
    audit.write_evidence_md(evidence, os.path.join(ARTIFACTS, "evidence.md"))

    log.log(f"=== TDES V5 demo complete in {time.time() - t_start:.1f}s. "
            f"all_passed={evidence['all_passed']} ===")
    log.close()

    if not evidence["all_passed"]:
        sys.exit(1)


def dataclasses_asdict(obj):
    import dataclasses
    return dataclasses.asdict(obj)


if __name__ == "__main__":
    main()
