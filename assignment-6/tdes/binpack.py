"""
tdes.binpack
============
Minimum-bin-count packing. Two solvers, used together (see packing.py):

  exact_min_bins()   Bitmask "profile" DP over subsets. For n items this
                      explores 2^n states; feasible up to
                      config.DP_EXACT_MAX_GROUP (18) items, ~2^18 = 262k
                      states with O(n) transitions each. Provably
                      OPTIMAL: minimizes the true number of bins for the
                      group it is given.

  first_fit_decreasing()  O(n log n) heuristic used above the exact
                      threshold. Provably within 11/9*OPT + 6/9 bins
                      (Johnson et al.). tests/test_binpack.py asserts
                      this bound holds on generated instances by
                      comparing against exact_min_bins on the same data
                      at sizes small enough to solve exactly.

WHY NOT ALWAYS EXACT: bin packing (decide "do k bins suffice?") is
NP-complete by reduction from PARTITION, and no polynomial algorithm can
guarantee better than a 3/2 approximation unless P=NP (a 3/2-or-better
algorithm would distinguish 2 bins from 3, solving PARTITION exactly).
Exhaustive/DP search is therefore only tractable at small n -- which is
exactly why packing.py chunks each lane into fixed-size, doc_id-ordered
groups before calling exact_min_bins on each group. The grouping makes
the guarantee "optimal within group_size", not globally optimal, and
that qualifier is recorded in the packing report.
"""

from __future__ import annotations
from typing import List, Tuple


def exact_min_bins(lengths: List[int], capacity: int) -> Tuple[int, List[int]]:
    """Exact minimum-bin packing via bitmask profile DP.

    Returns (num_bins, assignment) where assignment[i] is the bin index
    (0-indexed, in the order bins were opened during DP reconstruction)
    that item i was placed into.

    Any single item longer than `capacity` makes packing infeasible --
    callers (packing.py) must split or truncate oversized documents
    BEFORE calling this function; that is not this function's job.
    """
    n = len(lengths)
    if n == 0:
        return 0, []
    for L in lengths:
        if L > capacity:
            raise ValueError(f"item length {L} exceeds capacity {capacity}; split/truncate before packing")

    full = 1 << n
    INF = (n + 1, n + 1)  # worst-possible (bins_used, neg_rem) sentinel
    # dp[mask] = (bins_used, -remaining_capacity_in_currently_open_bin)
    # Smaller tuple = better: fewer bins first, then MORE remaining
    # capacity (i.e. more negative -remaining) as tiebreak, since more
    # leftover space can only help future placements.
    dp = [INF] * full
    parent: List = [None] * full
    dp[0] = (0, 0)

    order = sorted(range(full), key=lambda m: bin(m).count("1"))
    for mask in order:
        cur = dp[mask]
        if cur == INF:
            continue
        bins_used, neg_rem = cur
        rem = -neg_rem
        for i in range(n):
            bit = 1 << i
            if mask & bit:
                continue
            nmask = mask | bit
            length_i = lengths[i]

            # Option A: place into the currently open bin, if any and if it fits.
            if bins_used > 0 and rem >= length_i:
                cand = (bins_used, -(rem - length_i))
                if cand < dp[nmask]:
                    dp[nmask] = cand
                    parent[nmask] = (mask, i, False)

            # Option B: open a fresh bin for this item.
            cand = (bins_used + 1, -(capacity - length_i))
            if cand < dp[nmask]:
                dp[nmask] = cand
                parent[nmask] = (mask, i, True)

    final_mask = full - 1
    bins_used_total, _ = dp[final_mask]
    if bins_used_total > n:
        raise RuntimeError("packing DP failed to reach a solution")

    assignment = [-1] * n
    mask = final_mask
    while mask != 0:
        prev_mask, i, _opened_new = parent[mask]
        # item i's bin index is always (bins_used at the AFTER state) - 1:
        # whether i opened a fresh bin or joined the currently-open one,
        # it ends up in whatever the last bin is once i is placed.
        bins_used_after = dp[mask][0]
        assignment[i] = bins_used_after - 1
        mask = prev_mask

    return bins_used_total, assignment


def first_fit_decreasing(lengths: List[int], capacity: int) -> Tuple[int, List[int]]:
    """FFD heuristic: sort items by length descending (ties broken by
    original index ascending, for determinism), place each into the
    first bin with room, else open a new bin.

    Approximation bound (Johnson, 1973 / Dosa 2007 refined form):
    FFD(I) <= (11/9)*OPT(I) + 6/9, for any instance I. Verified against
    exact_min_bins on small instances in tests/test_binpack.py.
    """
    n = len(lengths)
    order = sorted(range(n), key=lambda i: (-lengths[i], i))
    bin_remaining: List[int] = []
    assignment = [-1] * n
    for i in order:
        L = lengths[i]
        placed = False
        for b in range(len(bin_remaining)):
            if bin_remaining[b] >= L:
                bin_remaining[b] -= L
                assignment[i] = b
                placed = True
                break
        if not placed:
            bin_remaining.append(capacity - L)
            assignment[i] = len(bin_remaining) - 1
    return len(bin_remaining), assignment


def ffd_within_bound(ffd_bins: int, opt_bins: int) -> bool:
    """Checks FFD(I) <= (11/9)*OPT(I) + 6/9, the proven Johnson bound."""
    return ffd_bins <= (11.0 / 9.0) * opt_bins + 6.0 / 9.0 + 1e-9
