"""
tdes.firewall
=============
The evaluation/validation firewall is not a filter that is applied and
might be forgotten -- it is the single mandatory gate every shard must
pass through before its tokens can be handed to the packer for a
loss-bearing batch. `admit_for_training` is the ONLY function in the
codebase that is allowed to return tokens destined for a loss-bearing
batch, and it FAILS CLOSED: an unrecognized or missing role raises,
rather than defaulting to "admit".

This single choke point is what makes "[PASS] eval_shard_blocked" mean
something structural rather than "we happened to remember to skip it
here". mixture.py, packing.py and trainer.py must all route through
this function -- there is no other way to obtain training tokens.
"""

from __future__ import annotations
from typing import List, Tuple

LOSS_BEARING_ROLE = "TRAIN"
KNOWN_ROLES = {"TRAIN", "VALIDATION", "EVAL", "PROXY", "QUARANTINE"}


class FirewallViolation(Exception):
    pass


def admit_for_training(shard_role: str, shard_id: str) -> None:
    """Raises FirewallViolation unless shard_role == TRAIN. Call this
    before ANY shard's tokens are handed to the packer for a loss-
    bearing batch. Deliberately has no default / fallback branch: an
    unknown role is a violation, not a pass-through."""
    if shard_role not in KNOWN_ROLES:
        raise FirewallViolation(
            f"shard {shard_id!r} carries unrecognized role {shard_role!r}; "
            f"fail-closed -- refusing to admit for training"
        )
    if shard_role != LOSS_BEARING_ROLE:
        raise FirewallViolation(
            f"shard {shard_id!r} has role {shard_role!r}, not TRAIN; "
            f"blocked from loss-bearing consumption"
        )


def admit_for_validation(shard_role: str, shard_id: str) -> None:
    """Validation forward passes may read VALIDATION-role shards only.
    Never used to produce a loss-bearing batch (see mixture.py /
    trainer.py -- validation losses never flow into the optimizer)."""
    if shard_role != "VALIDATION":
        raise FirewallViolation(
            f"shard {shard_id!r} has role {shard_role!r}, not VALIDATION; "
            f"blocked from validation read"
        )


def admit_for_opus_proxy(shard_role: str, shard_id: str) -> None:
    """OPUS's target direction must be derived from PROXY-role data only.
    This is the guard against proxy leakage described in opus.py: if the
    proxy pool could draw from VALIDATION or EVAL, then eval-adjacent
    data would steer *which training data gets selected*, contaminating
    the run through the gradient of the selection rule even though no
    eval token ever appears in a loss-bearing batch."""
    if shard_role != "PROXY":
        raise FirewallViolation(
            f"shard {shard_id!r} has role {shard_role!r}, not PROXY; "
            f"blocked from OPUS proxy use"
        )


def assert_role_known(role: str) -> None:
    if role not in KNOWN_ROLES:
        raise FirewallViolation(f"unrecognized role {role!r}; fail-closed")
