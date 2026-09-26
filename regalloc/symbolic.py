"""Independent symbolic executor: source CFG vs allocated CFG.

This module does **not** call the allocator core; it is a separate
interpreter.  It interprets the original :class:`~regalloc.model.Program` and
the allocator's :class:`~regalloc.lower.AllocatedProgram` in a symbolic value
domain and, on finitely many deterministically explored complete paths,
checks that

* both sides execute the same original instructions (same uid order);
* every instruction sees the same symbolic inputs and defines the same
  symbolic outputs;
* on every traversed CFG edge, each variable in the edge-sensitive live set
  arrives at its allocated location (home register, or its spill slot) with
  exactly the token the source program gives it on that edge;
* a call-clobbered register never carries a still-needed value: a clobber
  turns the register into a distinct ``CLOBBERED`` token and any later read
  fails.

Values are provenance tokens: parameters are distinct seeds and each
definition is ``("op", uid, input_tokens, def_index)``; a phi copies the
token carried by its edge source.  Equivalence is token equality, so a wrong
register, a missing reload or a stale value is detected rather than excused.

Paths are enumerated by bounded DFS (ascending successor order).  The bounds
are *coverage* limits — the result states whether they were reached; they are
not numeric tolerances.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .liveness import liveness_with_phis

CLOBBERED = ("__clobbered__",)
UNINITIALIZED = ("__uninitialized__",)


@dataclass
class EquivalenceResult:
    equivalent: bool
    paths_explored: int
    edges_checked: int
    ops_checked: int
    steps_used: int
    depth_bound: int
    step_budget: int
    bound_reached: bool
    failure: str | None = None

    def summary(self) -> str:
        status = "EQUIVALENT" if self.equivalent else f"NOT EQUIVALENT: {self.failure}"
        tail = " (coverage bound reached)" if self.bound_reached else ""
        return (
            f"{status}; paths={self.paths_explored} edges={self.edges_checked} "
            f"ops={self.ops_checked}{tail}"
        )


def _op_token(uid: int, inputs: tuple[Any, ...], def_index: int) -> tuple:
    return ("op", uid, inputs, def_index)


def equivalence_check(
    program,
    allocation,
    *,
    depth_bound: int = 80,
    step_budget: int = 20_000,
) -> EquivalenceResult:
    """Run the paired symbolic executor (module docstring lists the checks)."""
    ap = allocation.allocated_program
    colors = allocation.assignment
    spilled = set(allocation.slots)
    k = allocation.k
    report = liveness_with_phis(program)
    frame = getattr(allocation.allocated_program, "frame", None)

    n = len(program.blocks)
    orig_to_ab: dict[int, int] = {}
    for ai, ab in enumerate(ap.blocks):
        if ab.origin >= 0:
            orig_to_ab[ab.origin] = ai

    counters = {"edges": 0, "ops": 0}
    steps_used = 0
    paths = 0
    bound_reached = False

    def loc_for_var(v: int, regs: dict[int, Any], slots: dict[int, Any]):
        if v in spilled:
            return slots.get(slot_address(v), UNINITIALIZED)
        if v not in colors:
            return UNINITIALIZED
        return regs.get(colors[v], UNINITIALIZED)

    # When frame packing ran, memory is addressed by *byte offset*: two
    # logical slots sharing an offset genuinely alias, so a wrong overlap
    # decision manifests as a corrupted token on the real executed path.
    def slot_address(slot_id: int, mi_offset: int = -1):
        if frame is not None:
            off = mi_offset if mi_offset >= 0 else frame.slot_offsets[slot_id]
            return ("mem", off)
        return ("slot", slot_id)

    def run_path(edge_seq: list[int]) -> str | None:
        """Execute one paired path.  ``edge_seq`` is the list of successors
        taken from the entry; the path ends in an exit (no successors)."""
        env: dict[int, Any] = {v: ("param", v) for v in program.params}
        regs: dict[int, Any] = {r: UNINITIALIZED for r in range(k)}
        slots: dict[Any, Any] = {}
        # register-passed parameters start in their home register;
        # stack-passed (spilled) parameters are born in their spill slot
        for v in program.params:
            if v in spilled:
                slots[slot_address(v)] = ("param", v)
            else:
                regs[colors[v]] = ("param", v)

        cur = 0
        acur = orig_to_ab[0]

        for step, q in enumerate(edge_seq + [None]):
            sb = program.blocks[cur]
            ab = ap.blocks[acur]
            if ab.origin != cur:
                return f"allocated CFG desync at source block {cur}"

            op_indices = [i for i, mi in enumerate(ab.instrs) if mi.kind == "op"]
            if len(op_indices) != len(sb.instrs):
                return f"op count mismatch in block {cur}"
            last_op_idx = op_indices[-1] if op_indices else -1
            dst_ops = [ab.instrs[i] for i in op_indices]

            # ---- paired body: prefix through the last op ----
            src_i = 0
            for idx in range(last_op_idx + 1):
                mi = ab.instrs[idx]
                if mi.kind == "copy":
                    regs[mi.dst] = regs[mi.src]
                elif mi.kind == "load":
                    regs[mi.dst] = slots.get(slot_address(mi.slot, mi.offset), UNINITIALIZED)
                elif mi.kind == "store":
                    slots[slot_address(mi.slot, mi.offset)] = regs[mi.src]
                elif mi.kind == "op":
                    ins = sb.instrs[src_i]
                    if mi.uid != ins.uid:
                        return f"uid order mismatch in block {cur}"
                    src_inputs = tuple(env[u] for u in ins.uses)
                    dst_inputs = tuple(regs[r] for r in mi.reads)
                    if src_inputs != dst_inputs:
                        return (
                            f"input mismatch at op uid {ins.uid} (block {cur}): "
                            f"src={src_inputs} dst={dst_inputs}"
                        )
                    if any(t is CLOBBERED or t is UNINITIALIZED for t in dst_inputs):
                        return f"clobbered/uninitialized register read at op uid {ins.uid}"
                    # the call first destroys clobbered registers, then its
                    # outputs become valid (a result may be written into a
                    # register the call itself clobbered)
                    for r in mi.clobbers:
                        regs[r] = CLOBBERED
                    for di, d in enumerate(ins.defs):
                        tok = _op_token(ins.uid, src_inputs, di)
                        env[d] = tok
                        regs[mi.writes[di]] = tok
                    src_i += 1
                    counters["ops"] += 1
                else:  # pragma: no cover
                    return f"unknown machine instruction kind {mi.kind!r}"

            if q is None:
                if sb.succs:
                    return "path ended inside a non-exit block"
                return None

            if q not in sb.succs:
                return f"illegal edge {cur}->{q}"

            # source-side token of each edge-live variable, captured BEFORE
            # the edge code runs (phi application happens afterwards)
            qb = program.blocks[q]
            pre_edge_env = dict(env)

            # ---- edge transition on the allocated side ----
            # 1) tail of the current block (edge code when edge not split)
            for idx in range(last_op_idx + 1, len(ab.instrs)):
                mi = ab.instrs[idx]
                if mi.kind == "op":
                    return f"unexpected op after edge boundary in block {cur}"
                if mi.kind == "copy":
                    regs[mi.dst] = regs[mi.src]
                elif mi.kind == "load":
                    regs[mi.dst] = slots.get(slot_address(mi.slot, mi.offset), UNINITIALIZED)
                elif mi.kind == "store":
                    slots[slot_address(mi.slot, mi.offset)] = regs[mi.src]

            # 2) follow the successor chain (possibly through split blocks)
            t = acur
            target = orig_to_ab[q]
            while True:
                candidates = ab_chain_targets(ap, t, q)
                if len(candidates) != 1:
                    return f"allocated CFG edge {cur}->{q} not uniquely realisable"
                t = candidates[0]
                if t == target:
                    break
                cb = ap.blocks[t]
                if any(mi.kind == "op" for mi in cb.instrs):
                    return f"split block {cb.label} unexpectedly contains an op"
                for mi in cb.instrs:
                    if mi.kind == "copy":
                        regs[mi.dst] = regs[mi.src]
                    elif mi.kind == "load":
                        regs[mi.dst] = slots.get(slot_address(mi.slot, mi.offset), UNINITIALIZED)
                    elif mi.kind == "store":
                        slots[slot_address(mi.slot, mi.offset)] = regs[mi.src]
            acur = target

            # 3) edge-sensitive live values must be in their home locations.
            # Phi *sources* are consumed by the parallel copies that just ran
            # (their home register may legitimately be permuted away); they
            # are verified below via the phi-result tokens.  Everything else
            # live on this edge must still hold its pre-edge token.
            phi_srcs_here = {
                dict(phi.sources)[cur] for phi in qb.phis
            }
            for v in report.edge_live[(cur, q)]:
                if v in phi_srcs_here:
                    continue
                tok = pre_edge_env.get(v, UNINITIALIZED)
                got = loc_for_var(v, regs, slots)
                if got != tok:
                    return (
                        f"edge {cur}->{q}: v{v} arrived as {got!r}, expected {tok!r}"
                    )
            counters["edges"] += 1

            # 4) source-side phi application (all source tokens come from the
            # pre-edge snapshot); phi results must land in their home slot
            phi_pairs = [
                (phi.def_var, pre_edge_env[dict(phi.sources)[cur]])
                for phi in qb.phis
            ]
            for d, tok in phi_pairs:
                env[d] = tok
            for d, tok in phi_pairs:
                got = loc_for_var(d, regs, slots)
                if got != tok:
                    return f"edge {cur}->{q}: phi result v{d} = {got!r}, expected {tok!r}"
            cur = q
        return "path exceeded depth without reaching an exit"

    # ---- bounded DFS enumeration of complete paths (ascending succ order) --
    frames: list[tuple[int, int]] = [(0, 0)]
    prefix: list[int] = []
    budget = step_budget
    while frames:
        b, si = frames[-1]
        succs = list(program.blocks[b].succs)
        if not succs:
            err = run_path(list(prefix))
            if err:
                return EquivalenceResult(False, paths, counters["edges"],
                                         counters["ops"], steps_used,
                                         depth_bound, step_budget, bound_reached, err)
            paths += 1
            frames.pop()
            if prefix:
                prefix.pop()
            continue
        if len(prefix) >= depth_bound or budget <= 0:
            bound_reached = True
            frames.pop()
            if prefix:
                prefix.pop()
            continue
        if si < len(succs):
            frames[-1] = (b, si + 1)
            s = succs[si]
            prefix.append(s)
            budget -= 1
            steps_used += 1
            frames.append((s, 0))
        else:
            frames.pop()
            if prefix:
                prefix.pop()

    return EquivalenceResult(
        equivalent=True,
        paths_explored=paths,
        edges_checked=counters["edges"],
        ops_checked=counters["ops"],
        steps_used=steps_used,
        depth_bound=depth_bound,
        step_budget=step_budget,
        bound_reached=bound_reached,
    )


def ab_chain_targets(ap, start: int, wanted_origin: int) -> list[int]:
    """Successors of allocated block ``start`` whose split chain ends in the
    allocated block originating from source block ``wanted_origin``."""
    out: list[int] = []
    for cand in ap.blocks[start].succs:
        t = cand
        seen = set()
        while ap.blocks[t].origin < 0 and t not in seen:
            seen.add(t)
            t = ap.blocks[t].succs[0]
        if ap.blocks[t].origin == wanted_origin:
            out.append(cand)
    return out
