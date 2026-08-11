"""
tdes.pipeline
=============
Builds everything upstream of training, in the strict order required by
the contamination guards discussed throughout the design:

  load corpus -> content-hash + dedup -> role assignment
  -> tokenizer training (TRAIN role only) -> shard writing (chained manifests)
  -> per-lane packing (TRAIN) / proxy packing (PROXY) / validation packing (VALIDATION)

This module is called TWICE in a normal run_demo.py execution: once at
startup, and once again inside recovery.rebuild_state_from_checkpoint
after a simulated crash. Both calls must produce byte-identical results
given the same corpus file and config -- that is the whole basis for
"resume reconstructs exact state" and is checked by
tests/test_pipeline_determinism.py.
"""

from __future__ import annotations
import dataclasses
import json
from typing import Dict, List, Tuple

from . import config, hashing, dedup, roles as roles_mod, tokenizer as tok_mod, shards, packing, firewall


@dataclasses.dataclass
class CorpusContext:
    tokenizer: tok_mod.Tokenizer
    role_table: Dict[str, str]
    shard_manifests: List[shards.ShardManifest]
    manifest_root_hash: str
    lane_train_queues: Dict[str, List[packing.PackedSequence]]
    proxy_seqs: List[packing.PackedSequence]
    validation_seqs: List[packing.PackedSequence]
    eval_doc_ids: List[str]              # sealed -- never packed, listed only for the firewall demo
    quarantine_doc_ids: List[str]
    dedup_report: dict


def _load_corpus(path: str) -> List[dedup.DocRecord]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            lane = _lane_for(d["tier"], d.get("type"))
            records.append(dedup.DocRecord(doc_id=d["doc_id"], tier=d["tier"], type_=d.get("type"), text=d["text"]))
    return records


def _lane_for(tier: str, type_: str | None) -> str | None:
    for lane, (t, ty) in config.LANE_SOURCE.items():
        if t == tier and (ty is None or ty == type_):
            return lane
    return None


def build_corpus_context(corpus_path: str, shard_out_dir: str) -> CorpusContext:
    raw_records = _load_corpus(corpus_path)
    # lane assignment lookup, keyed by doc_id, built directly from the raw file
    tier_type_by_doc: Dict[str, Tuple[str, str | None]] = {}
    with open(corpus_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            tier_type_by_doc[d["doc_id"]] = (d["tier"], d.get("type"))

    dedup_result = dedup.run_dedup(raw_records)
    # Role assignment is a total partition over the POST-DEDUP document set
    # (survivors + quarantined) -- exact duplicates that collapsed into a
    # canonical document during dedup are not separately role-assigned;
    # they are recorded in dedup_report.exact_duplicates_dropped instead.
    deduped_doc_ids = [r.doc_id for r in dedup_result.survivors] + [r.doc_id for r in dedup_result.quarantined]
    role_assignment = roles_mod.assign_roles(dedup_result.survivors, dedup_result.quarantined)
    roles_mod.verify_partition(role_assignment, deduped_doc_ids)

    survivors_by_id = {r.doc_id: r for r in dedup_result.survivors}

    train_docs = [r for r in dedup_result.survivors if role_assignment.table[r.doc_id] == "TRAIN"]
    val_docs = [r for r in dedup_result.survivors if role_assignment.table[r.doc_id] == "VALIDATION"]
    eval_docs = [r for r in dedup_result.survivors if role_assignment.table[r.doc_id] == "EVAL"]
    proxy_docs = [r for r in dedup_result.survivors if role_assignment.table[r.doc_id] == "PROXY"]

    train_docs_sorted = sorted(train_docs, key=lambda r: r.doc_id)
    train_manifest_hash = hashing.hash_object({"doc_ids": [r.doc_id for r in train_docs_sorted],
                                                "content_hashes": [r.content_hash for r in train_docs_sorted]})

    tokenizer = tok_mod.train_bpe([r.text for r in train_docs_sorted], config.TOKENIZER_VOCAB_SIZE, train_manifest_hash)

    # -- shard + pack every role's documents, grouped by lane -------------
    shard_manifests: List[shards.ShardManifest] = []
    lane_train_queues: Dict[str, List[packing.PackedSequence]] = {}
    prev_manifest_hash = "GENESIS"

    def encode_docs(doc_list):
        out = []
        for d in sorted(doc_list, key=lambda r: r.doc_id):
            ids = tokenizer.encode(d.text)
            out.append((d, ids))
        return out

    for lane in config.LANES:
        lane_train_docs = [d for d in train_docs_sorted if _lane_for(*tier_type_by_doc[d.doc_id]) == lane]
        encoded = encode_docs(lane_train_docs)
        if not encoded:
            lane_train_queues[lane] = []
            continue
        all_tokens = []
        offsets = {}
        cursor = 0
        for d, ids in encoded:
            offsets[d.doc_id] = (cursor, cursor + len(ids))
            all_tokens.extend(ids)
            cursor += len(ids)
        shard_id = f"shard-TRAIN-{lane}"
        manifest = shards.write_shard(shard_out_dir, shard_id, lane, "TRAIN", tokenizer.freeze_hash,
                                       all_tokens, offsets, prev_manifest_hash)
        shard_manifests.append(manifest)
        prev_manifest_hash = manifest.manifest_hash

        doc_tokens = [packing.DocTokens(doc_id=d.doc_id, role="TRAIN", lane=lane, tokens=ids) for d, ids in encoded]
        lane_train_queues[lane] = packing.pack_lane(lane, "TRAIN", doc_tokens, config.MODEL_SEQ_LEN)

    # PROXY: pack as a single pseudo-lane using the same packer (allow_split,
    # generous group size) purely to get PackedSequence objects; role stays PROXY.
    proxy_encoded = encode_docs(proxy_docs)
    proxy_seqs: List[packing.PackedSequence] = []
    if proxy_encoded:
        all_tokens, offsets, cursor = [], {}, 0
        for d, ids in proxy_encoded:
            offsets[d.doc_id] = (cursor, cursor + len(ids))
            all_tokens.extend(ids)
            cursor += len(ids)
        manifest = shards.write_shard(shard_out_dir, "shard-PROXY", "proxy", "PROXY", tokenizer.freeze_hash,
                                       all_tokens, offsets, prev_manifest_hash)
        shard_manifests.append(manifest)
        prev_manifest_hash = manifest.manifest_hash
        doc_tokens = [packing.DocTokens(doc_id=d.doc_id, role="PROXY", lane="proxy", tokens=ids) for d, ids in proxy_encoded]
        proxy_seqs = _pack_role_agnostic(doc_tokens, config.MODEL_SEQ_LEN)

    # VALIDATION: same treatment
    val_encoded = encode_docs(val_docs)
    validation_seqs: List[packing.PackedSequence] = []
    if val_encoded:
        all_tokens, offsets, cursor = [], {}, 0
        for d, ids in val_encoded:
            offsets[d.doc_id] = (cursor, cursor + len(ids))
            all_tokens.extend(ids)
            cursor += len(ids)
        manifest = shards.write_shard(shard_out_dir, "shard-VALIDATION", "validation", "VALIDATION",
                                       tokenizer.freeze_hash, all_tokens, offsets, prev_manifest_hash)
        shard_manifests.append(manifest)
        prev_manifest_hash = manifest.manifest_hash
        doc_tokens = [packing.DocTokens(doc_id=d.doc_id, role="VALIDATION", lane="validation", tokens=ids) for d, ids in val_encoded]
        validation_seqs = _pack_role_agnostic(doc_tokens, config.MODEL_SEQ_LEN)

    # EVAL: sealed. Content-hashed and manifested (so the firewall demo has
    # something concrete to attempt admitting) but NEVER packed/tokenized
    # into training-shaped sequences here.
    eval_docs_sorted = sorted(eval_docs, key=lambda r: r.doc_id)
    if eval_docs_sorted:
        eval_manifest_payload = {"doc_ids": [d.doc_id for d in eval_docs_sorted],
                                  "content_hashes": [d.content_hash for d in eval_docs_sorted]}
        eval_manifest = shards.ShardManifest(
            shard_id="shard-EVAL-sealed", lane="eval", role="EVAL", tokenizer_hash=tokenizer.freeze_hash,
            content_hash=hashing.hash_object(eval_manifest_payload), doc_count=len(eval_docs_sorted),
            token_count=0, offsets={}, created_at="frozen", parent_manifest_hash=prev_manifest_hash,
        )
        eval_manifest.manifest_hash = eval_manifest.compute_hash()
        shard_manifests.append(eval_manifest)
        prev_manifest_hash = eval_manifest.manifest_hash

    root_hash = shards.build_manifest_chain(shard_manifests)

    return CorpusContext(
        tokenizer=tokenizer, role_table=role_assignment.table, shard_manifests=shard_manifests,
        manifest_root_hash=root_hash, lane_train_queues=lane_train_queues, proxy_seqs=proxy_seqs,
        validation_seqs=validation_seqs, eval_doc_ids=[d.doc_id for d in eval_docs_sorted],
        quarantine_doc_ids=[d.doc_id for d in dedup_result.quarantined],
        dedup_report={
            "exact_duplicates_dropped": dedup_result.exact_duplicates_dropped,
            "near_duplicates_quarantined": dedup_result.near_duplicates_quarantined,
            "survivor_count": len(dedup_result.survivors),
            "quarantined_count": len(dedup_result.quarantined),
        },
    )


def _pack_role_agnostic(doc_tokens: List[packing.DocTokens], seq_len: int) -> List[packing.PackedSequence]:
    """Packs PROXY/VALIDATION documents using the same exact-DP/FFD
    packer as TRAIN lanes, but bypassing firewall.admit_for_training
    (packing.pack_lane calls that gate deliberately, since it is meant
    only for loss-bearing lanes). PROXY/VALIDATION sequences are never
    used for a gradient update outside their designated firewall-gated
    entry points (opus.compute_proxy_direction / trainer.run_validation)."""
    policy = config.PackingPolicy(allow_split=True, max_docs_per_sequence=6, group_size=14,
                                   min_fragment_len=16, pad_side="right", tie_break="doc_id_asc")
    docs_sorted = sorted(doc_tokens, key=lambda d: d.doc_id)
    fragments = []
    for d in docs_sorted:
        fragments.extend(packing.split_document(d, seq_len, policy.min_fragment_len, policy.allow_split))
    fragments.sort(key=lambda f: f.doc_id)
    from . import binpack
    groups = [fragments[i:i + policy.group_size] for i in range(0, len(fragments), policy.group_size)]
    out = []
    seq_counter = 0
    for g_idx, group in enumerate(groups):
        lengths = [len(f.tokens) for f in group]
        num_bins, assignment = binpack.exact_min_bins(lengths, seq_len) if len(group) <= config.DP_EXACT_MAX_GROUP \
            else binpack.first_fit_decreasing(lengths, seq_len)
        bins: Dict[int, List[int]] = {}
        for i, b in enumerate(assignment):
            bins.setdefault(b, []).append(i)
        for b_idx in sorted(bins.keys()):
            frags_in_bin = [group[i] for i in bins[b_idx]]
            seq = packing._materialize_sequence(doc_tokens[0].lane, g_idx, b_idx, frags_in_bin, seq_len, seq_counter)
            out.append(seq)
            seq_counter += 1
    return out
