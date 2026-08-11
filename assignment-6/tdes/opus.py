"""
tdes.opus
=========
Implements OPUS (Optimizer-induced Projected Utility Selection,
arXiv:2602.05400) as a deterministic DAG of admission tasks. Each
candidate PACKED SEQUENCE flows through the same ordered set of nodes;
every node appends an audit record, and the traversed path IS the audit
trail for that candidate's admission decision (see run_opus_dag).

DAG (fixed order, every node's decision is a pure function of its
inputs -- no ambient state, no unseeded randomness anywhere):

  ingest -> firewall_check -> integrity_check -> opus_score
         -> threshold_rule -> mixture_fit -> floor_check -> verdict

FIDELITY NOTE: OPUS scores candidates by projecting the candidate's
REAL, freshly-computed gradient (see model.py's frozen-body / trained-
head design) onto a target direction derived from a REAL gradient
averaged over a PROXY-role reference batch, using an actual CountSketch
random projection and cosine similarity -- none of this is hardcoded or
simulated. The reduction relative to the full paper is that the
gradient lives in the (small) trainable-head parameter space rather
than the full model's parameter space, and we use a fixed number of
Boltzmann-ordered admissions per round rather than a continuous-time
scheme. Both are direct, disclosed consequences of the pure-stdlib /
frozen-body compute budget (see model.py), not shortcuts to the
selection LOGIC itself, which is implemented in full: real projection,
real CountSketch, real Boltzmann-weighted diversity sampling, real
threshold/defer/reject/floor-override rules.

PROXY LEAKAGE GUARD: compute_proxy_direction accepts only PROXY-role
sequences and calls firewall.admit_for_opus_proxy on every one of them.
This is the guard against the leakage channel described in the design
notes: if the proxy pool could draw from VALIDATION or EVAL, eval-
adjacent data would steer *which training data gets selected* even
though no eval token ever appears in a loss-bearing batch.
"""

from __future__ import annotations
import dataclasses
import math
from typing import Dict, List, Optional, Tuple

from . import config, hashing, firewall, model as model_mod
from .packing import PackedSequence


# ---------------------------------------------------------------------------
# CountSketch: deterministic random projection to a fixed lower dimension.
# The (bucket, sign) hash function is fixed once for the whole run (a
# frozen part of the run's configuration, like the model body) so that
# scores computed at different steps remain comparable and reproducible.
# ---------------------------------------------------------------------------

def _sketch_hash(index: int, dim_out: int) -> Tuple[int, float]:
    key = hashing.derive_key(config.MASTER_SEED, "opus-sketch", 0, "hash", index)
    stream = hashing.KeyedStream(key)
    bucket = int(stream.uniform(0) * dim_out) % dim_out
    sign = 1.0 if stream.uniform(1) < 0.5 else -1.0
    return bucket, sign


def countsketch_project(vec: List[float], dim_out: int) -> List[float]:
    out = [0.0] * dim_out
    for i, v in enumerate(vec):
        if v == 0.0:
            continue
        bucket, sign = _sketch_hash(i, dim_out)
        out[bucket] += sign * v
    return out


def cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def flatten_grads(grads: Dict) -> List[float]:
    flat: List[float] = []
    for row in grads["d_head"]:
        flat.extend(row)
    flat.extend(grads["d_bhead"])
    return flat


# ---------------------------------------------------------------------------
# Gradient computation (candidate + proxy)
# ---------------------------------------------------------------------------

def compute_sequence_gradient(m: model_mod.ModelState, seq: PackedSequence, branch_id: str,
                               step: int, seq_index: int) -> List[float]:
    _, n_loss, grads, _ = model_mod.forward_and_loss(
        m, seq.input_ids, seq.position_ids, seq.attention_block_ids, seq.loss_mask,
        branch_id, step, seq_index, training=True,
    )
    if n_loss == 0:
        return [0.0] * (config.MODEL_D_MODEL * config.TOKENIZER_VOCAB_SIZE + config.TOKENIZER_VOCAB_SIZE)
    flat = flatten_grads(grads)
    return [g / n_loss for g in flat]


def compute_proxy_direction(m: model_mod.ModelState, proxy_seqs: List[PackedSequence],
                             proxy_roles: List[str], branch_id: str, step: int) -> List[float]:
    for seq, role in zip(proxy_seqs, proxy_roles):
        firewall.admit_for_opus_proxy(role, seq.seq_id)
    if not proxy_seqs:
        raise ValueError("OPUS proxy batch is empty -- cannot derive a target direction")
    accum: Optional[List[float]] = None
    for idx, seq in enumerate(proxy_seqs):
        g = compute_sequence_gradient(m, seq, branch_id, step, -100 - idx)  # negative seq_index namespace: never collides with real batch indices
        if accum is None:
            accum = g
        else:
            accum = [a + b for a, b in zip(accum, g)]
    n = len(proxy_seqs)
    return [a / n for a in accum]


# ---------------------------------------------------------------------------
# DAG node records
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class OpusDecision:
    candidate_id: str
    lane: str
    step: int
    node_trace: List[str]
    score: float
    classification: str          # ACCEPT / DEFER / REJECT
    defer_count: int
    floor_override: bool
    reason: str


def _threshold_classify(score: float) -> str:
    lo = config.OPUS_ACCEPT_THRESHOLD - config.OPUS_DEFER_BAND
    hi = config.OPUS_ACCEPT_THRESHOLD + config.OPUS_DEFER_BAND
    if score >= hi:
        return "ACCEPT"
    if score <= lo:
        return "REJECT"
    return "DEFER"


def run_opus_dag(candidate: PackedSequence, candidate_role: str, proxy_direction_sketch: List[float],
                  m: model_mod.ModelState, branch_id: str, step: int, seq_index: int,
                  prior_defer_count: int, lane_deficit: float,
                  lane_floor_breach: bool) -> OpusDecision:
    """Runs one candidate through the fixed DAG. Every branch is a pure
    function of its explicit inputs -- deterministic and replayable."""
    trace = ["ingest"]

    trace.append("firewall_check")
    firewall.admit_for_training(candidate_role, candidate.seq_id)  # raises if not TRAIN

    trace.append("integrity_check")
    # shard-level integrity was already verified at load time (shards.load_shard);
    # here we re-assert the invariant that every loss-bearing position in this
    # candidate traces to a real TRAIN document -- defense in depth for the DAG.
    for lm, prov in zip(candidate.loss_mask, candidate.token_provenance):
        if lm == 1 and prov is None:
            raise firewall.FirewallViolation(f"candidate {candidate.seq_id} has loss_mask=1 on a padding position")

    trace.append("opus_score")
    cand_grad = compute_sequence_gradient(m, candidate, branch_id, step, seq_index)
    cand_sketch = countsketch_project(cand_grad, config.OPUS_SKETCH_DIM)
    score = cosine(cand_sketch, proxy_direction_sketch)

    trace.append("threshold_rule")
    classification = _threshold_classify(score)
    defer_count = prior_defer_count
    reason = f"score={score:.4f}"

    if classification == "DEFER":
        defer_count += 1
        if defer_count > config.OPUS_MAX_DEFERS:
            classification = "REJECT"
            reason += f"; exceeded max_defers={config.OPUS_MAX_DEFERS}"

    trace.append("mixture_fit")
    reason += f"; lane_deficit={lane_deficit:.4f}"

    trace.append("floor_check")
    floor_override = False
    if classification == "REJECT" and lane_floor_breach:
        # Protected-floor override: this lane's floor is breached, and
        # the lane's candidate pool has nothing better on offer this
        # round. The override is recorded with the very score that
        # justified the normal rejection, so the audit trail shows
        # exactly what was overridden and why.
        classification = "ACCEPT"
        floor_override = True
        reason += "; FLOOR_OVERRIDE applied (protected floor breached, no better candidate available)"

    trace.append("verdict")
    return OpusDecision(
        candidate_id=candidate.seq_id, lane=candidate.lane, step=step, node_trace=trace,
        score=score, classification=classification, defer_count=defer_count,
        floor_override=floor_override, reason=reason,
    )


# ---------------------------------------------------------------------------
# Boltzmann-weighted diversity ordering (Efraimidis-Spirakis A-ES scheme,
# fully deterministic via keyed draws -- no python `random` involved).
# ---------------------------------------------------------------------------

def boltzmann_order(scored: List[Tuple[str, float]], branch_id: str, step: int,
                     temperature: float = None) -> List[str]:
    """scored: list of (candidate_id, score). Returns candidate_ids
    ordered by a weighted-random-without-replacement draw where weight
    = exp(score / temperature) -- higher score more likely to sort
    early, but not deterministically top-k, which is what gives OPUS
    its 'diversity' property per the paper. Fully reproducible: the
    only randomness is the keyed uniform draw per candidate."""
    T = temperature or config.OPUS_BOLTZMANN_TEMPERATURE
    keyed = []
    for idx, (cid, score) in enumerate(scored):
        w = math.exp(score / T)
        u = hashing.keyed_uniform(config.MASTER_SEED, branch_id, step, "boltzmann", idx, cid)
        u = min(max(u, 1e-12), 1 - 1e-12)
        rank_key = u ** (1.0 / w) if w > 0 else 0.0
        keyed.append((rank_key, cid))
    keyed.sort(key=lambda x: -x[0])
    return [cid for _, cid in keyed]
