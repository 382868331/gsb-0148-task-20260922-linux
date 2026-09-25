"""Block/edge frequency profiles and frequency-aware spill costing.

Frequencies are non-negative integers supplied by the caller.  They model
*how often* each block executes (e.g. a hot loop body gets a large number, a
cold error path gets a small one).  Frequencies are purely a heuristic input:
they never change which allocations are legal or the meaning of a program,
only *which* variable is chosen when spilling is unavoidable.  When no
profile is supplied the allocator uses its original loop-depth weighting.

Three quantities live here:

* :func:`edge_frequencies` — per-CFG-edge execution counts estimated from the
  block profile with deterministic structural rules (block-local action
  weight stays exactly the input block frequency; edge actions use the
  estimated frequency of their edge);

* :func:`spill_costs` — the **estimated** weighted reload/store traffic that
  spilling each candidate would insert, counted from the spill-form plans of
  the *current* iteration (real use/def points plus per-edge phi uses).  This
  estimate only drives victim choice;

* :func:`memory_traffic` — the **actual** weighted number of load/store
  machine instructions present in the final lowered program.  Reported
  results come from this function, never from the heuristic estimate.
"""

from __future__ import annotations

from .liveness import predecessors
from .model import Program, ProgramFormatError, check_int


# --------------------------------------------------------------------------
# input validation
# --------------------------------------------------------------------------

def normalize_frequencies(program: Program, frequencies: object) -> tuple[int, ...] | None:
    """Validate a caller block-frequency vector.

    Accepted shapes: ``None`` (no profile -> legacy heuristic), a sequence of
    exactly ``len(blocks)`` non-negative ints (``bool`` rejected, like every
    other integer input).  Values are *not* capped: they are heuristic
    weights, and all costing below is integer-only, so large profiles simply
    produce large weights without any float rounding.
    """
    if frequencies is None:
        return None
    if isinstance(frequencies, (str, bytes)) or not isinstance(frequencies, (list, tuple)):
        raise ProgramFormatError(
            "frequencies: expected a list/tuple of non-negative ints with one "
            "entry per block, or None"
        )
    n = len(program.blocks)
    if len(frequencies) != n:
        raise ProgramFormatError(
            f"frequencies: expected {n} entries (one per block), "
            f"got {len(frequencies)}"
        )
    out: list[int] = []
    for i, value in enumerate(frequencies):
        out.append(check_int(value, f"frequencies[{i}]", minimum=0))
    return tuple(out)


# --------------------------------------------------------------------------
# edge frequency estimation
# --------------------------------------------------------------------------

def _back_edges(program: Program) -> set[tuple[int, int]]:
    """Back edges p -> s where s dominates p (natural-loop latch edges)."""
    blocks = program.blocks
    n = len(blocks)
    preds = predecessors(blocks)
    dom: list[set[int]] = [set(range(n)) for _ in range(n)]
    dom[0] = {0}
    changed = True
    while changed:
        changed = False
        for i in range(1, n):
            if not preds[i]:
                continue
            new = {i} | set.intersection(*(dom[p] for p in preds[i]))
            if new != dom[i]:
                dom[i] = new
                changed = True
    back: set[tuple[int, int]] = set()
    for p, b in enumerate(blocks):
        for s in b.succs:
            if s in dom[p]:
                back.add((p, s))
    return back


def edge_frequencies(program: Program, block_freq: tuple[int, ...]) -> dict[tuple[int, int], int]:
    """Estimate execution counts of the CFG edges from block frequencies.

    The block profile is the contract input and is honoured exactly for
    block-local actions.  Edge actions (phi copies and phi-related spill
    traffic) need an edge count, estimated with deterministic rules applied
    in a fixed order:

    1. a predecessor with one successor sends its whole block frequency;
    2. a successor with one predecessor is entered exactly as often as it
       executes, so the edge receives ``min(freq[pred], freq[succ])`` —
       min-clamped so a predecessor with frequency 0 always sends 0 (a rare
       cold block reached by a branch is therefore charged at its own low
       frequency, not at an even share of its hot predecessor; this is the
       rule that separates a cold error arm inside a hot loop);
    3. a back edge ``latch -> header`` carries
       ``max(0, freq[header] - sum(already known forward incoming edges))``
       (the header re-enters the loop roughly once per body execution minus
       the single entry edge);
    4. any edge still unset (a fork arm into a multi-predecessor join) shares
       the predecessor's not-yet-allocated frequency evenly with the other
       unset outgoing edges, remainder handed out in ascending successor
       order.

    Rules 1 and 2 are clamped rather than forced on flow-inconsistent hand-
    written profiles (rule 1 applied first); every value is non-negative, no
    floats are used, and the result is a heuristic estimate that never
    affects legality or program semantics.
    """
    blocks = program.blocks
    n = len(blocks)
    preds = predecessors(blocks)
    back = _back_edges(program)
    edge_f: dict[tuple[int, int], int] = {}

    # rule 1: unique successor
    for p, b in enumerate(blocks):
        if len(b.succs) == 1:
            edge_f[(p, b.succs[0])] = block_freq[p]

    # rule 2: unique predecessor (never overrides a rule-1 value), clamped
    # by the predecessor's own frequency so zero always flows as zero
    for s in range(n):
        if len(preds[s]) == 1:
            p = preds[s][0]
            edge_f.setdefault((p, s), min(block_freq[p], block_freq[s]))

    # rule 3: back edges from the header flow balance
    for (p, h) in sorted(back):
        incoming_known = sum(
            edge_f.get((q, h), 0) for q in preds[h] if (q, h) not in back
        )
        edge_f.setdefault((p, h), max(0, block_freq[h] - incoming_known))

    # rule 4: distribute each block's residual frequency over unset exits
    for p, b in enumerate(blocks):
        unset = [s for s in b.succs if (p, s) not in edge_f]
        if not unset:
            continue
        used = sum(edge_f.get((p, s), 0) for s in b.succs if (p, s) in edge_f)
        pot = max(0, block_freq[p] - used)
        share, remainder = divmod(pot, len(unset))
        for s in unset:
            edge_f[(p, s)] = share
        i = 0
        while remainder > 0:
            edge_f[(p, unset[i])] += 1
            remainder -= 1
            i += 1
    return edge_f


# --------------------------------------------------------------------------
# frequency-aware spill cost estimate (drives victim choice only)
# --------------------------------------------------------------------------

def _block_action_weight(block_freq: tuple[int, ...], bi: int) -> int:
    # instructions inside a block execute whenever the block executes
    return block_freq[bi]


def spill_costs(
    program: Program,
    vp,
    block_freq: tuple[int, ...],
    edge_freq: dict[tuple[int, int], int],
) -> dict[int, int]:
    """Estimated weighted reload+store count if a still-resident variable
    were spilled next, measured on the current spill form.

    The count sums the memory actions the spill-form builder actually creates
    for the candidate:

    * one **reload** per use at an instruction point (block frequency);
    * one **store** per def at an instruction point (block frequency);
    * one reload and/or store per phi edge action on the edges where the
      variable participates as a phi source / phi result (edge frequency);
    * a spilled *parameter* additionally stores nothing at entry (it is
      stack-passed, i.e. born in its slot), but every use still reloads.

    Only candidates currently present in the interference graph
    (``vp.real_nodes``) receive a cost; absent variables are already spilled.
    """
    blocks = program.blocks
    costs: dict[int, int] = {v: 0 for v in vp.real_nodes}

    def add(v: int, amount: int) -> None:
        if v in costs:
            costs[v] += amount

    # block-local use/def traffic
    for bi, b in enumerate(blocks):
        w = _block_action_weight(block_freq, bi)
        if w == 0:
            continue
        for ins in b.instrs:
            for u in ins.uses:
                add(u, w)          # reload before the instruction
            for d in ins.defs:
                add(d, w)          # store after the instruction

    # phi edge traffic estimated for *candidate* variables.  The current
    # spill-form edge plans only mention variables already in the spilled
    # set, so the candidate's prospective actions are derived from the phi
    # nodes directly:
    #   * a spilled phi RESULT d makes the edge copy end in slot d -> one
    #     weighted store on that predecessor edge;
    #   * a spilled phi SOURCE s makes the edge copy start from slot s ->
    #     one weighted reload on that edge.
    # (When s and d are the same variable it is a candidate here and cannot
    # be a no-op identity edge yet; the identity elision only applies once
    # the variable is actually spilled, at which point it leaves the
    # candidate set.)
    for bi, b in enumerate(blocks):
        if not b.phis:
            continue
        for phi in b.phis:
            for p, src in phi.sources:
                if src == phi.def_var:
                    # self-loop identity edge slot{v} <- slot{v} is elided in
                    # the spill form, so it carries no prospective traffic
                    continue
                w = edge_freq.get((p, bi), 0)
                if w == 0:
                    continue
                add(phi.def_var, w)   # store of the spilled phi result
                add(src, w)           # reload of the spilled phi source

    return costs


# --------------------------------------------------------------------------
# actual delivered memory traffic (the reported number, not the heuristic)
# --------------------------------------------------------------------------

def memory_traffic(
    program: Program,
    allocated_program,
    block_freq: tuple[int, ...] | None,
    edge_freq: dict[tuple[int, int], int] | None,
) -> dict:
    """Weighted counts of the load/store instructions actually emitted.

    Every lowered block carries ``origin`` (source block index) or, for a
    critical-edge split block, ``split_edge = (pred, succ)``.  Block-origin
    code is weighted by the source block frequency; a split block holds only
    that edge's actions and is weighted by the estimated edge frequency.

    Without a profile, structural loop nesting (10**depth) is used, matching
    the legacy weighting so old and new runs remain comparable.

    Returns a dict with raw and weighted counts for loads/stores/totals plus
    the per-block detail, so the headline number is auditable instruction by
    instruction.
    """
    if block_freq is None:
        block_freq, edge_freq = _legacy_weights(program)
    assert edge_freq is not None

    detail = []
    raw_loads = raw_stores = 0
    w_loads = w_stores = 0
    for ab in allocated_program.blocks:
        if ab.origin >= 0:
            w = block_freq[ab.origin]
            where = f"block[{ab.origin}]"
        else:
            w = edge_freq.get(ab.split_edge, 0)
            where = f"edge{ab.split_edge}"
        nl = sum(1 for mi in ab.instrs if mi.kind == "load")
        ns = sum(1 for mi in ab.instrs if mi.kind == "store")
        raw_loads += nl
        raw_stores += ns
        w_loads += nl * w
        w_stores += ns * w
        if nl or ns:
            detail.append({
                "site": where,
                "frequency": w,
                "loads": nl,
                "stores": ns,
                "weighted_loads": nl * w,
                "weighted_stores": ns * w,
            })
    return {
        "loads": raw_loads,
        "stores": raw_stores,
        "total": raw_loads + raw_stores,
        "weighted_loads": w_loads,
        "weighted_stores": w_stores,
        "weighted_total": w_loads + w_stores,
        "by_site": detail,
    }


def _legacy_weights(program: Program):
    """Loop-depth structural weights (10**depth), used when no profile given.

    Edge frequencies collapse to the successor block's body weight for
    non-split accounting; split edge blocks are weighted by the predecessor
    body weight (the edge actions would otherwise have lived at its tail).
    """
    # local import: allocator owns dominator/loop analysis
    from .allocator import _compute_loop_depths

    depth = _compute_loop_depths(program)
    block_freq = tuple(10 ** depth[i] for i in range(len(program.blocks)))
    edge_freq = {}
    for i, b in enumerate(program.blocks):
        for s in b.succs:
            edge_freq[(i, s)] = block_freq[i]
    return block_freq, edge_freq
