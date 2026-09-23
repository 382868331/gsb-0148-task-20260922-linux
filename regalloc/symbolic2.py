"""Independent bit-level symbolic executor: source CFG vs allocated CFG (v2).

This module does **not** call the allocator core.  It interprets the
original v2 :class:`~regalloc.model.Program` and the allocated
:class:`~regalloc.lower2.AllocatedProgram2` in a symbolic value domain and,
on finitely many deterministically explored complete paths (normal AND
exceptional exits), checks that

* both sides execute the same original instructions (same uid order);
* every instruction sees the same symbolic inputs and defines the same
  symbolic outputs — registers are modelled as **bit segments**, so
  writing a narrow view (e.g. 32-bit ``w0``) only replaces those bits and
  reading an aliasing wide view (``x0``) yields a composite token that
  compares unequal to any whole value: alias corruption is detected, not
  excused;
* on every traversed CFG edge (normal and exceptional), each variable in
  the edge-sensitive live set arrives at its allocated location (home view
  or stack slot) with exactly the token the source program gives it;
* on an exceptional transition the throwing instruction's defs do NOT
  materialise (its call clobbers do), instructions after it in the block
  are skipped, and handler phis are applied from the pre-edge snapshot;
* a call-clobbered view becomes a ``CLOBBERED`` token and any later read
  of those bits fails.

Values are provenance tokens: parameters are distinct seeds and each
definition is ``("op", uid, input_tokens, def_index)``.  Paths are
enumerated by bounded DFS (normal successors and exceptional successors in
declaration order); the bounds are *coverage* limits — the result states
whether they were reached.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .liveness import liveness_v2

CLOBBERED = ("__clobbered__",)
UNINITIALIZED = ("__uninitialized__",)


@dataclass
class EquivalenceResult2:
    equivalent: bool
    paths_explored: int
    edges_checked: int
    ops_checked: int
    normal_exits: int
    exc_exits: int
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
            f"ops={self.ops_checked} normal_exits={self.normal_exits} "
            f"exc_exits={self.exc_exits}{tail}"
        )


def _op_token(uid: int, inputs: tuple[Any, ...], def_index: int) -> tuple:
    return ("op", uid, inputs, def_index)


def _has_bad(token: Any) -> bool:
    if token is CLOBBERED or token is UNINITIALIZED:
        return True
    if isinstance(token, tuple) and token and token[0] == "bits":
        return any(_has_bad(t) for t in token[1])
    return False


class _Registers:
    """Bit-segment model of the physical register groups.

    ``segs[group]`` is a sorted list of ``[lo, hi, token]`` covering parts
    of the group; adjacent segments with equal tokens are merged so reads
    are canonical.
    """

    def __init__(self, machine):
        self.machine = machine
        self.segs: dict[int, list[list]] = {}

    def write(self, view_name: str, token: Any) -> None:
        v = self.machine.view(view_name)
        lo, hi = v.lo, v.hi
        out: list[list] = []
        for seg_lo, seg_hi, tok in self.segs.get(v.group, []):
            if seg_hi <= lo or seg_lo >= hi:
                out.append([seg_lo, seg_hi, tok])
                continue
            if seg_lo < lo:
                out.append([seg_lo, lo, tok])
            if seg_hi > hi:
                out.append([hi, seg_hi, tok])
        out.append([lo, hi, token])
        out.sort()
        # merge adjacent segments carrying equal tokens
        merged: list[list] = []
        for seg in out:
            if merged and merged[-1][1] == seg[0] and merged[-1][2] == seg[2]:
                merged[-1][1] = seg[1]
            else:
                merged.append(seg)
        self.segs[v.group] = merged

    def read(self, view_name: str) -> Any:
        v = self.machine.view(view_name)
        tokens: list[Any] = []
        pos = v.lo
        for seg_lo, seg_hi, tok in self.segs.get(v.group, []):
            if seg_hi <= v.lo or seg_lo >= v.hi:
                continue
            if seg_lo > pos:
                tokens.append(UNINITIALIZED)
            tokens.append(tok)
            pos = max(pos, seg_hi)
        if pos < v.hi:
            tokens.append(UNINITIALIZED)
        if not tokens:
            return UNINITIALIZED
        if len(tokens) == 1:
            return tokens[0]
        return ("bits", tuple(tokens))


def equivalence_check2(
    program,
    machine,
    result,
    *,
    depth_bound: int = 80,
    step_budget: int = 20_000,
) -> EquivalenceResult2:
    """Run the paired bit-level symbolic executor (see module docstring)."""
    ap = result.allocated_program
    assignment = result.assignment or {}
    slots = result.slots or {}
    spilled = set(slots)
    report = liveness_v2(program)

    n = len(program.blocks)
    orig_to_ab: dict[int, int] = {}
    for ai, ab in enumerate(ap.blocks):
        if ab.origin >= 0:
            orig_to_ab[ab.origin] = ai

    counters = {"edges": 0, "ops": 0, "normal": 0, "exc": 0}
    steps_used = 0
    paths = 0
    bound_reached = False

    def run_path(moves: list[tuple[str, int]]) -> str | None:
        env: dict[int, Any] = {v: ("param", v) for v in program.params}
        regs = _Registers(machine)
        mem: dict[int, Any] = {}
        for v in program.params:
            if v in spilled:
                mem[slots[v].offset] = ("param", v)
            elif v in assignment:
                regs.write(assignment[v], ("param", v))

        def loc_for_var(v: int) -> Any:
            if v in spilled:
                return mem.get(slots[v].offset, UNINITIALIZED)
            if v not in assignment:
                return UNINITIALIZED
            return regs.read(assignment[v])

        cur = 0
        acur = orig_to_ab[0]
        last_move_kind = "n"

        for move in list(moves) + [None]:
            sb = program.blocks[cur]
            ab = ap.blocks[acur]
            if ab.origin != cur:
                return f"allocated CFG desync at source block {cur}"

            exc_taken = move is not None and move[0] == "x"
            op_indices = [i for i, mi in enumerate(ab.instrs) if mi.kind == "op"]
            if len(op_indices) != len(sb.instrs):
                return f"op count mismatch in block {cur}"

            # ---- body ------------------------------------------------
            src_i = 0
            exc_jump = False
            stop_idx = len(ab.instrs)  # exclusive end of body execution
            for idx, mi in enumerate(ab.instrs):
                if mi.kind == "op":
                    ins = sb.instrs[src_i]
                    if mi.uid != ins.uid:
                        return f"uid order mismatch in block {cur}"
                    src_inputs = tuple(env[u] for u in ins.uses)
                    dst_inputs = tuple(regs.read(r) for r in mi.reads)
                    if src_inputs != dst_inputs:
                        return (
                            f"input mismatch at op uid {ins.uid} (block {cur}): "
                            f"src={src_inputs} dst={dst_inputs}"
                        )
                    if any(_has_bad(t) for t in dst_inputs):
                        return (
                            f"clobbered/uninitialized register bits read at op "
                            f"uid {ins.uid}"
                        )
                    for r in mi.clobbers:
                        regs.write(r, CLOBBERED)
                    if exc_taken and ins.may_throw:
                        # the throw: defs do NOT materialise; the rest of
                        # the block is normal-path-only
                        src_i += 1
                        counters["ops"] += 1
                        exc_jump = True
                        stop_idx = idx + 1
                        break
                    for di, d in enumerate(ins.defs):
                        tok = _op_token(ins.uid, src_inputs, di)
                        env[d] = tok
                        regs.write(mi.writes[di], tok)
                    src_i += 1
                    counters["ops"] += 1
                elif mi.kind == "copy":
                    regs.write(mi.dst, regs.read(mi.src))
                elif mi.kind == "load":
                    regs.write(mi.dst, mem.get(mi.slot, UNINITIALIZED))
                elif mi.kind == "store":
                    mem[mi.slot] = regs.read(mi.src)
                else:  # pragma: no cover
                    return f"unknown machine instruction kind {mi.kind!r}"

            if move is None:
                if sb.succs or sb.exc_succs:
                    return "path ended inside a non-exit block"
                if src_i != len(sb.instrs):
                    return f"block {cur}: not all instructions executed"
                counters["exc" if last_move_kind == "x" else "normal"] += 1
                return None

            kind, q = move
            if kind == "n":
                if q not in sb.succs:
                    return f"illegal normal edge {cur}->{q}"
            else:
                if q not in sb.exc_succs:
                    return f"illegal exceptional edge {cur}->{q}"
                if not exc_jump:
                    return f"exceptional edge {cur}->{q} but no may_throw op ran"
            last_move_kind = kind

            qb = program.blocks[q]
            pre_edge_env = dict(env)

            # ---- edge transition on the allocated side ----------------
            if kind == "n":
                # tail of the current block (edge code when not split)
                for idx in range(stop_idx, len(ab.instrs)):
                    mi = ab.instrs[idx]
                    if mi.kind == "op":
                        return f"unexpected op after edge boundary in block {cur}"
                    if mi.kind == "copy":
                        regs.write(mi.dst, regs.read(mi.src))
                    elif mi.kind == "load":
                        regs.write(mi.dst, mem.get(mi.slot, UNINITIALIZED))
                    elif mi.kind == "store":
                        mem[mi.slot] = regs.read(mi.src)
                sidx = sb.succs.index(q)
                target = ab.succs[sidx]
            else:
                sidx = sb.exc_succs.index(q)
                target = ab.exc_succs[sidx]

            # follow the split-block chain (copy/load/store only)
            t = target
            seen = set()
            while ap.blocks[t].origin < 0:
                if t in seen:
                    return f"split chain loops at block {ap.blocks[t].label}"
                seen.add(t)
                cb = ap.blocks[t]
                for mi in cb.instrs:
                    if mi.kind == "copy":
                        regs.write(mi.dst, regs.read(mi.src))
                    elif mi.kind == "load":
                        regs.write(mi.dst, mem.get(mi.slot, UNINITIALIZED))
                    elif mi.kind == "store":
                        mem[mi.slot] = regs.read(mi.src)
                    else:
                        return f"split block {cb.label} unexpectedly contains an op"
                if len(cb.succs) != 1:
                    return f"split block {cb.label} does not have exactly one succ"
                t = cb.succs[0]
            if ap.blocks[t].origin != q:
                return f"allocated edge {cur}->{q} ends at wrong block"
            acur = t

            # ---- edge-sensitive live values at their homes ------------
            phi_srcs_here = {dict(phi.sources)[cur] for phi in qb.phis}
            for v in report.edge_live[(cur, q)]:
                if v in phi_srcs_here:
                    continue
                tok = pre_edge_env.get(v, UNINITIALIZED)
                got = loc_for_var(v)
                if got != tok:
                    return (
                        f"edge {cur}->{q} ({kind}): v{v} arrived as {got!r}, "
                        f"expected {tok!r}"
                    )
            counters["edges"] += 1

            # ---- source-side phi application ---------------------------
            phi_pairs = [
                (phi.def_var, pre_edge_env[dict(phi.sources)[cur]])
                for phi in qb.phis
            ]
            for d, tok in phi_pairs:
                env[d] = tok
            for d, tok in phi_pairs:
                got = loc_for_var(d)
                if got != tok:
                    return (
                        f"edge {cur}->{q} ({kind}): phi result v{d} = {got!r}, "
                        f"expected {tok!r}"
                    )
            cur = q
        return "path exceeded depth without reaching an exit"

    # ---- bounded DFS enumeration of complete paths -----------------------
    def moves_of(b: int) -> list[tuple[str, int]]:
        blk = program.blocks[b]
        return [("x", q) for q in blk.exc_succs] + [("n", s) for s in blk.succs]

    frames: list[tuple[int, int]] = [(0, 0)]
    prefix: list[tuple[str, int]] = []
    budget = step_budget
    while frames:
        b, si = frames[-1]
        moves = moves_of(b)
        if not moves:
            err = run_path(list(prefix))
            if err:
                return EquivalenceResult2(
                    False, paths, counters["edges"], counters["ops"],
                    counters["normal"], counters["exc"], steps_used,
                    depth_bound, step_budget, bound_reached, err,
                )
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
        if si < len(moves):
            frames[-1] = (b, si + 1)
            prefix.append(moves[si])
            budget -= 1
            steps_used += 1
            frames.append((moves[si][1], 0))
        else:
            frames.pop()
            if prefix:
                prefix.pop()

    return EquivalenceResult2(
        equivalent=True,
        paths_explored=paths,
        edges_checked=counters["edges"],
        ops_checked=counters["ops"],
        normal_exits=counters["normal"],
        exc_exits=counters["exc"],
        steps_used=steps_used,
        depth_bound=depth_bound,
        step_budget=step_budget,
        bound_reached=bound_reached,
    )
