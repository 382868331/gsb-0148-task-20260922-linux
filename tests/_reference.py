"""Independent reference helpers used by the tests.

These deliberately do not import the allocator core's graph/planner; they
verify outputs from first principles:

* :func:`check_liveness_equations` validates a liveness report directly
  against the edge-sensitive dataflow equations and re-derives per-instruction
  sets by an independent backward scan;
* :func:`simulate_parallel` computes the intended parallel assignment and
  compares it with the effect of sequentially executing emitted machine
  instructions.
"""

from __future__ import annotations


def check_liveness_equations(program, report) -> list[str]:
    """Return violations of the liveness equations (empty == consistent)."""
    n = len(program.blocks)
    preds: list[list[int]] = [[] for _ in range(n)]
    for i, b in enumerate(program.blocks):
        for s in b.succs:
            preds[s].append(i)

    phi_defs = [{p.def_var for p in b.phis} for b in program.blocks]
    phi_src_edge: dict[tuple[int, int], set[int]] = {}
    for j, b in enumerate(program.blocks):
        for p in b.phis:
            for pred, src in p.sources:
                phi_src_edge.setdefault((pred, j), set()).add(src)

    errs: list[str] = []
    for i, b in enumerate(program.blocks):
        # upExp / body-killed via an independent forward scan
        killed: set[int] = set(phi_defs[i])
        up: set[int] = set()
        for ins in b.instrs:
            for u in ins.uses:
                if u not in killed:
                    up.add(u)
            killed.update(ins.defs)

        expected_out: set[int] = set()
        for s in b.succs:
            expected_out |= (set(report.live_in[s]) - phi_defs[s]) | phi_src_edge.get((i, s), set())
        expected_in = up | (expected_out - killed)

        if set(report.live_out[i]) != expected_out:
            errs.append(f"block {i} live-out mismatch")
        if set(report.live_in[i]) != expected_in:
            errs.append(f"block {i} live-in mismatch")

        for s in b.succs:
            want = (set(report.live_in[s]) - phi_defs[s]) | phi_src_edge.get((i, s), set())
            if set(report.edge_live[(i, s)]) != want:
                errs.append(f"edge {i}->{s} live mismatch")

        # independently re-derive per-instruction sets
        live = set(expected_out)
        for k in range(len(b.instrs) - 1, -1, -1):
            if set(report.instr_after[i][k]) != live:
                errs.append(f"block {i} instr {k} after mismatch")
            ins = b.instrs[k]
            live = (live - set(ins.defs)) | set(ins.uses)
            if set(report.instr_before[i][k]) != live:
                errs.append(f"block {i} instr {k} before mismatch")
    return errs


def simulate_parallel(actions, emitted_mis, *, k: int, slots, scratch_slot):
    """Compare an emitted sequential instruction list against the parallel
    assignment it claims to implement.

    ``actions``: list of ("copy", dst_reg, src_reg, -1) or
    ("store", -1, src_reg, slot).  Registers start holding their own identity
    token ``("R", r)``; slots start holding ``("M", slot)``.

    Returns a list of mismatches (empty == correct).
    """
    # intended final state computed as a true simultaneous assignment
    regs0 = {r: ("R", r) for r in range(k)}
    slots0 = {s: ("M", s) for s in slots}
    intended_regs = dict(regs0)
    intended_slots = dict(slots0)
    for kind, dst, src, slot in actions:
        if kind == "copy":
            intended_regs[dst] = regs0[src]
        elif kind == "store":
            intended_slots[slot] = regs0[src]

    # execute emitted instructions
    regs = dict(regs0)
    smem = dict(slots0)
    smem[scratch_slot] = ("M", scratch_slot)
    for mi in emitted_mis:
        if mi.kind == "copy":
            regs[mi.dst] = regs[mi.src]
        elif mi.kind == "load":
            regs[mi.dst] = smem[mi.slot]
        elif mi.kind == "store":
            smem[mi.slot] = regs[mi.src]
        else:
            return [f"unexpected {mi.kind} inside parallel sequence"]

    bad: list[str] = []
    for r in range(k):
        if regs[r] != intended_regs.get(r):
            bad.append(f"r{r} is {regs[r]} expected {intended_regs.get(r)}")
    for s, tok in intended_slots.items():
        if smem.get(s) != tok:
            bad.append(f"slot{s} is {smem.get(s)} expected {tok}")
    return bad
