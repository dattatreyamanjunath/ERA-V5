"""
tdes.audit
==========
Post-hoc checks that re-derive their verdicts from the actual artifacts
on disk (ledgers, manifests, shard bytes) rather than trusting anything
the training loop asserted about itself in passing. This is what
distinguishes evidence from narration: every [PASS]/[FAIL] line in
run.log and every row of evidence.md is backed by a function here that
recomputes the claim from scratch.
"""

from __future__ import annotations
import dataclasses
import json
import os
from typing import Dict, List

from . import config, shards, ledger as ledger_mod, pipeline


@dataclasses.dataclass
class AuditFinding:
    name: str
    passed: bool
    detail: str
    evidence_path: str


def verify_no_leakage(ctx: pipeline.CorpusContext, learning_ledger: ledger_mod.Ledger) -> AuditFinding:
    """Re-derives, from the learning ledger's own recorded per-sequence
    doc_ids and the role table built during corpus construction, that
    every document that ever contributed a loss-bearing token was
    TRAIN-role. This does not trust the firewall's own claims -- it
    independently recomputes the check from the two artifacts."""
    bad = []
    for rec in learning_ledger.records:
        if rec.record_type != "BATCH_LOSS":
            continue
        for seq_rec in rec.payload["sequences"]:
            for doc_id in seq_rec["doc_ids"]:
                base_doc_id = doc_id.split("::frag")[0]
                role = ctx.role_table.get(base_doc_id)
                if role != "TRAIN":
                    bad.append({"step": rec.payload["step"], "doc_id": doc_id, "role": role})
    passed = len(bad) == 0
    detail = "no non-TRAIN document ever contributed a loss-bearing token" if passed else f"{len(bad)} violations: {bad[:5]}"
    return AuditFinding("learning_trace_no_leakage", passed, detail, "ledgers/learning_ledger.jsonl")


def verify_ledger_chains(paths: List[str]) -> AuditFinding:
    bad = []
    for p in paths:
        try:
            ledger_mod.Ledger(p).verify_chain()
        except ledger_mod.IntegrityError as e:
            bad.append(str(e))
    passed = len(bad) == 0
    detail = "all ledger hash chains verified" if passed else f"broken chains: {bad}"
    return AuditFinding("ledger_chain_integrity", passed, detail, ",".join(paths))


def verify_manifest_chain(ctx: pipeline.CorpusContext) -> AuditFinding:
    recomputed = shards.build_manifest_chain(ctx.shard_manifests)
    passed = recomputed == ctx.manifest_root_hash
    detail = f"root_hash={recomputed}" if passed else f"MISMATCH: recomputed={recomputed} stored={ctx.manifest_root_hash}"
    return AuditFinding("manifest_chain_integrity", passed, detail, "manifests/")


def tamper_and_detect(shard_dir: str, shard_id: str) -> AuditFinding:
    """Deliberately corrupts one byte of a shard's binary file, proves
    shards.load_shard raises IntegrityError, then restores the exact
    original bytes and proves load succeeds again. Run LAST in
    run_demo.py, after the rest of the audit, so it cannot poison any
    other hash check."""
    bin_path = os.path.join(shard_dir, f"{shard_id}.bin")
    with open(bin_path, "rb") as f:
        original = f.read()

    corrupted = bytearray(original)
    if not corrupted:
        return AuditFinding("tamper_detection", False, "shard binary is empty, cannot corrupt", bin_path)
    corrupted[0] ^= 0xFF
    with open(bin_path, "wb") as f:
        f.write(bytes(corrupted))

    detected = False
    try:
        shards.load_shard(shard_dir, shard_id)
    except shards.IntegrityError:
        detected = True
    finally:
        with open(bin_path, "wb") as f:
            f.write(original)

    restored_ok = False
    try:
        shards.load_shard(shard_dir, shard_id)
        restored_ok = True
    except shards.IntegrityError:
        restored_ok = False

    passed = detected and restored_ok
    detail = f"corruption_detected={detected}, restoration_verified={restored_ok}"
    return AuditFinding("tamper_detection", passed, detail, bin_path)


def build_evidence(findings: List[AuditFinding], extra: dict) -> dict:
    evidence = {
        "findings": [dataclasses.asdict(f) for f in findings],
        "all_passed": all(f.passed for f in findings),
        **extra,
    }
    return evidence


def write_evidence_md(evidence: dict, path: str) -> None:
    rows = []
    label_map = {
        "tokenizer_hash_verified": "Tokenizer integrity",
        "eval_shard_blocked": "Evaluation firewall",
        "packing_correctness": "Packing correctness",
        "mixture_compliance": "Mixture compliance",
        "opus_audit_trail": "OPUS audit trail",
        "crash_recovery": "Crash recovery",
        "replay": "Replay",
        "learning_trace_no_leakage": "Learning trace",
        "throughput": "Throughput",
        "ledger_chain_integrity": "Ledger integrity",
        "manifest_chain_integrity": "Manifest integrity",
        "tamper_detection": "Tamper detection",
    }
    lines = ["# Evidence Summary", "", "| Requirement | Result | Evidence |", "|---|---|---|"]
    for f in evidence["findings"]:
        label = label_map.get(f["name"], f["name"])
        result = "PASS" if f["passed"] else "FAIL"
        lines.append(f"| {label} | {result} | `{f['evidence_path']}` -- {f['detail']} |")
    lines.append("")
    lines.append(f"**Overall: {'PASS' if evidence['all_passed'] else 'FAIL'}**")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
