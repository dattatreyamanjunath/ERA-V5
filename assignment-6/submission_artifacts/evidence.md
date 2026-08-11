# Evidence Summary

| Requirement | Result | Evidence |
|---|---|---|
| Tokenizer integrity | PASS | `manifests/` -- freeze_hash=d718985853c9d24c |
| manifests_validated | PASS | `manifests/` -- root_hash=6a30c360672d2549e567f7a921ab04a716839b52d20ad8d8035eaed4d67bc373 |
| Evaluation firewall | PASS | `manifests/` -- shard 'shard-EVAL-sealed' has role EVAL, not TRAIN; blocked |
| Crash recovery | PASS | `ledgers/` -- crash_happened=True; resume_next_batch_matched=True; hash a4a795bb.. |
| fork_parent_untouched | PASS | `checkpoints/` -- root_step18.json byte-identical after fork branch trained |
| replay_hash_matched | PASS | `ledgers/` -- 7 steps matched, 0 mismatches (steps 6..12) |
| ledger_chains | PASS | `ledgers/` -- all 4 ledger hash chains verified |
| no_leakage_root | PASS | `ledgers/` -- no non-TRAIN document contributed a loss-bearing token (root branch) |
| no_leakage_fork | PASS | `ledgers/` -- no non-TRAIN document contributed a loss-bearing token (fork branch) |
| manifest_chain | PASS | `manifests/` -- manifest chain verified |
| tamper_detected | PASS | `manifests/` -- corruption_detected=True, restoration_verified=True |
| Mixture compliance | PASS | `ledgers/` -- stage_advances=0, rollbacks=0 |
| OPUS audit trail | PASS | `ledgers/` -- OPUS decisions logged per-step in consumption ledger |
| Throughput | PASS | `performance.json` -- 47.8 effective tok/s |

**Overall: PASS**