"""
tdes.config
===========
Single source of truth for every policy parameter in the system.

Design rule: if a value influences what gets consumed, how it is packed,
scored, or replayed, it lives HERE and nowhere else, so that the whole
policy surface can be serialized and hashed as one object
(see hashing.config_hash()). Do not hardcode any of these constants
anywhere else in the codebase.
"""

from __future__ import annotations
import dataclasses
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# Master seed. All randomness in the system is a deterministic function of
# (MASTER_SEED, branch_id, step, purpose). There is no global RNG stream
# anywhere in this codebase -- see hashing.derive_key().
# ---------------------------------------------------------------------------
MASTER_SEED = 1337

# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------
TOKENIZER_VOCAB_SIZE = 512          # 256 byte base + specials + merges
TOKENIZER_SPECIALS = ["<PAD>", "<BOS>", "<EOS>", "<UNK>"]
TOKENIZER_NFC = True                 # Unicode NFC normalization before BPE
TOKENIZER_PRETOKENIZE = "byte"       # pre-tokenization rule: raw UTF-8 bytes
TOKENIZER_MERGE_TIEBREAK = "lexicographic_pair"

# ---------------------------------------------------------------------------
# Model (deliberately tiny -- pure stdlib forward/backward, no ML points)
# ---------------------------------------------------------------------------
MODEL_D_MODEL = 32
MODEL_N_LAYERS = 2
MODEL_N_HEADS = 2
MODEL_SEQ_LEN = 128
MODEL_DROPOUT_P = 0.1
MODEL_LR = 0.05

# ---------------------------------------------------------------------------
# Dedup / near-duplicate detection
# ---------------------------------------------------------------------------
SHINGLE_WINDOW = 5                   # w-shingle size, in content words
JACCARD_DUP_THRESHOLD = 0.8          # >= this => quarantine as near-duplicate
STOPWORDS_KN: Tuple[str, ...] = (
    # Small curated set of high-frequency Kannada function words / particles.
    # NOTE: this is a coarse approximation, not a linguistic ground truth --
    # documented explicitly in README as a limitation.
    "ಮತ್ತು", "ಅಥವಾ", "ಈ", "ಆ", "ಒಂದು", "ಅವರು", "ಅವನು", "ಅವಳು",
    "ಇದು", "ಅದು", "ಎಂದು", "ಆದರೆ", "ಹಾಗೂ", "ಗೆ", "ದ", "ನ",
)
# Reversed-suffix trie entries: common Kannada case/plural inflectional
# suffixes, approximate coverage only (documented limitation in README).
KANNADA_SUFFIXES: Tuple[str, ...] = (
    "ಗಳು", "ಗಳಲ್ಲಿ", "ದಲ್ಲಿ", "ನಲ್ಲಿ", "ಗೆ", "ನ್ನು", "ರಿಂದ",
    "ಕ್ಕೆ", "ವನ್ನು", "ದಿಂದ", "ಯಲ್ಲಿ", "ರ", "ದ", "ನ", "ವು", "ಗಳ",
)

# ---------------------------------------------------------------------------
# Role assignment (doc_id hash based, deterministic partition)
# ---------------------------------------------------------------------------
ROLE_SPLIT_BOUNDARIES = {
    # cumulative upper bounds on hash-bucket [0, 1000)
    "TRAIN": 700,
    "VALIDATION": 800,
    "EVAL": 900,
    "PROXY": 1000,
    # QUARANTINE is not hash-assigned; it is assigned by dedup outcome only.
}
ROLE_HASH_BUCKETS = 1000

# ---------------------------------------------------------------------------
# Lanes. Each lane pulls from one Sangraha config/type combination.
# ---------------------------------------------------------------------------
LANES: Tuple[str, ...] = (
    "verified-web",
    "verified-speech",
    "verified-pdf",
    "unverified",
    "synthetic",
)

LANE_SOURCE = {
    # (sangraha config, type-filter or None)
    "verified-web": ("verified", "web"),
    "verified-speech": ("verified", "speech"),
    "verified-pdf": ("verified", "pdf"),
    "unverified": ("unverified", None),
    "synthetic": ("synthetic", None),
}

# ---------------------------------------------------------------------------
# Packing policy per lane. Same algorithm (exact-DP-on-groups, FFD above
# group size) with lane-specific parameters.
# ---------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class PackingPolicy:
    allow_split: bool          # may a document be split across sequences?
    max_docs_per_sequence: int
    group_size: int            # exact-DP group size (bitmask DP, <= 20)
    min_fragment_len: int      # smallest allowed split fragment, in tokens
    pad_side: str              # "right" (fixed; left-pad provides no value
                                # under explicit masks -- see design notes)
    tie_break: str              # "doc_id_asc"


PACKING_POLICY: Dict[str, PackingPolicy] = {
    "verified-web":    PackingPolicy(allow_split=False, max_docs_per_sequence=8,  group_size=16, min_fragment_len=8,  pad_side="right", tie_break="doc_id_asc"),
    "verified-speech": PackingPolicy(allow_split=False, max_docs_per_sequence=12, group_size=18, min_fragment_len=8,  pad_side="right", tie_break="doc_id_asc"),
    "verified-pdf":    PackingPolicy(allow_split=True,  max_docs_per_sequence=4,  group_size=12, min_fragment_len=16, pad_side="right", tie_break="doc_id_asc"),
    "unverified":      PackingPolicy(allow_split=True,  max_docs_per_sequence=6,  group_size=14, min_fragment_len=16, pad_side="right", tie_break="doc_id_asc"),
    "synthetic":       PackingPolicy(allow_split=True,  max_docs_per_sequence=6,  group_size=14, min_fragment_len=16, pad_side="right", tie_break="doc_id_asc"),
}

DP_EXACT_MAX_GROUP = 18   # bitmask DP feasibility ceiling (2^18 states)

# ---------------------------------------------------------------------------
# Mixture / curriculum
# ---------------------------------------------------------------------------
CURRICULUM_STAGES: Tuple[dict, ...] = (
    {
        "name": "stage1_verified_only",
        "lanes": ("verified-web", "verified-speech", "verified-pdf"),
        "weights": {"verified-web": 0.45, "verified-speech": 0.25, "verified-pdf": 0.30},
        "min_steps": 8,
    },
    {
        "name": "stage2_plus_unverified",
        "lanes": ("verified-web", "verified-speech", "verified-pdf", "unverified"),
        "weights": {"verified-web": 0.35, "verified-speech": 0.20, "verified-pdf": 0.20, "unverified": 0.25},
        "min_steps": 8,
    },
    {
        "name": "stage3_plus_synthetic",
        "lanes": ("verified-web", "verified-speech", "verified-pdf", "unverified", "synthetic"),
        "weights": {"verified-web": 0.28, "verified-speech": 0.17, "verified-pdf": 0.15, "unverified": 0.20, "synthetic": 0.20},
        "min_steps": 8,
    },
)

# floor: verified lanes combined must never fall below this share within
# any sliding window of FLOOR_WINDOW batches.
PROTECTED_FLOOR_VERIFIED_SHARE = 0.30
FLOOR_WINDOW = 16

# stage advancement: validation-triggered with a step-count fallback so
# every stage provably appears within a short demo run.
STAGE_ADVANCE_MAX_STEPS = 8        # fallback: force advance after N steps
STAGE_ADVANCE_VAL_IMPROVEMENT_EPS = 0.01  # val loss improvement below this
                                            # for 2 consecutive evals => advance

# ---------------------------------------------------------------------------
# OPUS (Optimizer-induced Projected Utility Selection)
# arXiv:2602.05400 -- scores are the projection of a candidate's
# optimizer-shaped effective update onto a target direction derived from a
# stable, PROXY-role, in-distribution reference batch.
# ---------------------------------------------------------------------------
OPUS_CANDIDATE_POOL = 24            # candidates scored per selection round
OPUS_SCORE_EVERY_N_STEPS = 4
OPUS_SKETCH_DIM = 64                # CountSketch projection dimension
OPUS_BOLTZMANN_TEMPERATURE = 0.5
OPUS_ACCEPT_THRESHOLD = 0.0         # score >= this => accept
OPUS_DEFER_BAND = 0.15              # within threshold +/- band => defer
OPUS_MAX_DEFERS = 3                 # after this many defers => reject
OPUS_PROXY_BATCH_SIZE = 4

# ---------------------------------------------------------------------------
# Rollback (validation-triggered fork with deterministic policy mutation)
# ---------------------------------------------------------------------------
ROLLBACK_VAL_DEGRADE_MARGIN = 0.02    # val loss above best-so-far by this
                                        # margin triggers a rollback
ROLLBACK_WEIGHT_DEMOTION_FACTOR = 0.5  # implicated lane's weight is halved
                                        # in the child branch after rollback
#
# --- Rollback budget: PLACEHOLDER ONLY for this local/small-scale demo ---
# In a real, long-running training system this must be a hard resource
# guard: a maximum number of rollbacks per run, a minimum step-distance
# between consecutive rollbacks, and a floor on how far back a rollback
# may reach (e.g. never revert past the current curriculum stage's start,
# or past N checkpoints). Without such a budget a pathological validation
# signal could stall a multi-week run indefinitely by triggering repeated
# reverts. For THIS submission -- a short, local, fully-deterministic demo
# -- we do not need the budget to be enforced (we want the mechanism to be
# exercisable), so it is defined but not wired to abort the run.
ROLLBACK_MAX_PER_RUN = None            # None => unlimited (demo only!)
ROLLBACK_MIN_STEP_GAP = 0              # 0 => unlimited (demo only!)
ROLLBACK_MAX_LOOKBACK_STEPS = None     # None => unlimited (demo only!)
# Real-world equivalents, for documentation:
#   ROLLBACK_MAX_PER_RUN = 5
#   ROLLBACK_MIN_STEP_GAP = 500
#   ROLLBACK_MAX_LOOKBACK_STEPS = 2000

# ---------------------------------------------------------------------------
# Training loop / demo shape
# ---------------------------------------------------------------------------
BATCH_SIZE = 4
TOTAL_DEMO_STEPS = 30
CHECKPOINT_EVERY_N_STEPS = 6
VALIDATION_EVERY_N_STEPS = 3
VALIDATION_SAMPLE_SIZE = 8   # deterministic subsample of the VALIDATION pool used
                                # per validation call -- the full pool (~80+ sequences)
                                # at 0.6s/sequence forward would make each of the ~14
                                # validation calls in the demo cost nearly a minute by
                                # itself; a fixed, deterministic subsample keeps the
                                # signal (same subsample every time, no sampling noise
                                # between calls) while keeping runtime bounded.
CRASH_AT_STEP = 12          # deliberately simulated crash point
REPLAY_START_STEP = 6
REPLAY_END_STEP = 12
FORK_AT_STEP = 18

# ---------------------------------------------------------------------------
# Shards
# ---------------------------------------------------------------------------
SHARD_MAX_TOKENS = 64_000
TOKEN_DTYPE_CODE = "H"       # array('H') == unsigned short, 2 bytes; vocab<=512 fits
TOKEN_ENDIANNESS = "little"

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = "data"
CORPUS_FILE = "data/sangraha_kan_sample.jsonl"
ARTIFACTS_DIR = "submission_artifacts"
