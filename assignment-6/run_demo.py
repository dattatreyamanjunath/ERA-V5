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
