"""Independent reference helpers for the v2 tests.

These never import the allocator, the lowerer or the liveness module.  They
provide:

* :func:`brute_feasibility` — enumerate EVERY complete register-view
  assignment of a small program's variables and test each one against the
  documented constraints (bit-overlap at every live point, call clobbers,
  early clobber), giving an independently computed feasibility answer for
  the deterministic small reference domain;
* :func:`simulate_parallel_bits` — compare an emitted sequential
  instruction list against the true parallel assignment using a standalone
  bit-segment register model (alias writes included).
"""

from __future__ import annotations


# --------------------------------------------------------------------------
# independent liveness (exceptional edges; same documented equations)
# --------------------------------------------------------------------------


def independent_liveness(program):
    blocks = program.blocks
    n = len(blocks)
    phi_defs = [set() for _ in blocks]
    phi_src = {}
    for i, b in enumerate(blocks):
        for phi in b.phis:
            phi_defs[i].add(phi.def_var)
            for p, s in phi.sources:
                phi_src.setdefault((p, i), set()).add(s)
    throw_at = []
    for b in blocks:
        t = None
        for k, ins in enumerate(b.instrs):
            if ins.may_throw:
                t = k
        throw_at.append(t)

    live_in = [set() for _ in blocks]

    def edge(p, s):
        return (live_in[s] - phi_defs[s]) | phi_src.get((p, s), set())

    changed = True
    while changed:
        changed = False
        for i in range(n):
            b = blocks[i]
            out_i = set()
            for s in b.succs:
                out_i |= edge(i, s)
            live = set(out_i)
            t = throw_at[i]
            rng = (
                range(len(b.instrs) - 1, -1, -1)
                if t is None
                else list(range(len(b.instrs) - 1, t, -1))
                + [-1]  # marker: inject exceptional requirements
                + list(range(t, -1, -1))
            )
            for k in rng:
                if k == -1:
                    for q in b.exc_succs:
                        live |= edge(i, q)
                    continue
                ins = b.instrs[k]
                live = (live - set(ins.defs)) | set(ins.uses)
            in_i = live - phi_defs[i]
            if in_i != live_in[i]:
                live_in[i] = in_i
                changed = True

    edge_live = {}
    for i, b in enumerate(blocks):
        for s in b.succs:
            edge_live[(i, s)] = edge(i, s)
        for q in b.exc_succs:
            edge_live[(i, q)] = edge(i, q)

    before = [[] for _ in blocks]
    after = [[] for _ in blocks]
    for i, b in enumerate(blocks):
        live = set()
        for s in b.succs:
            live |= edge_live[(i, s)]
        t = throw_at[i]
        bs = [set() for _ in b.instrs]
        as_ = [set() for _ in b.instrs]
        rng = (
            range(len(b.instrs) - 1, -1, -1)
            if t is None
            else list(range(len(b.instrs) - 1, t, -1))
            + [-1]
            + list(range(t, -1, -1))
        )
        for k in rng:
            if k == -1:
                for q in b.exc_succs:
                    live |= edge_live[(i, q)]
                continue
            ins = b.instrs[k]
            as_[k] = set(live)
            live = (live - set(ins.defs)) | set(ins.uses)
            bs[k] = set(live)
        before[i] = bs
        after[i] = as_
    return edge_live, before, after


# --------------------------------------------------------------------------
# brute-force feasibility over complete register-view assignments
# --------------------------------------------------------------------------


def _all_vars(program):
    out = set(program.params)
    for b in program.blocks:
        for ins in b.instrs:
            out.update(ins.defs)
        for phi in b.phis:
            out.add(phi.def_var)
    return out


def _overlap(machine, va, vb):
    a, b = machine.view(va), machine.view(vb)
    return a.group == b.group and a.lo < b.hi and b.lo < a.hi


def assignment_is_valid(program, machine, assign):
    """True iff a complete var->view assignment satisfies the constraints."""
    edge_live, before, after = independent_liveness(program)

    def check(where, live):
        items = [(v, assign[v]) for v in sorted(live) if v in assign]
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                if _overlap(machine, items[i][1], items[j][1]):
                    return False, f"{where}: {items[i][0]}/{items[j][0]} overlap"
        return True, ""

    for (p, q), live in sorted(edge_live.items()):
        ok, why = check(f"edge {p}->{q}", live)
        if not ok:
            return False, why
    for bi, b in enumerate(program.blocks):
        for ii, ins in enumerate(b.instrs):
            ok, why = check(f"b{bi} uid{ins.uid} before", before[bi][ii])
            if not ok:
                return False, why
            ok, why = check(f"b{bi} uid{ins.uid} after", after[bi][ii])
            if not ok:
                return False, why
            # The op physically writes every def view, dead or alive, so a
            # written view may not overlap a surviving value; an
            # early-clobber write also may not overlap any read view.
            survivors = after[bi][ii] - set(ins.defs)
            for d in ins.defs:
                for v in survivors:
                    if _overlap(machine, assign[d], assign[v]):
                        return False, f"b{bi} uid{ins.uid}: def v{d} clobbers v{v}"
            for d in ins.early_clobber:
                for u in ins.uses:
                    if _overlap(machine, assign[d], assign[u]):
                        return False, f"b{bi} uid{ins.uid}: early-clobber overlap"
            # call clobbers
            if ins.is_call:
                clob = set(machine.clobbered_by_call(ins.clobber_views))
            else:
                clob = set(ins.clobber_views)
            for v in survivors:
                if v not in assign:
                    continue
                for c in clob:
                    if _overlap(machine, assign[v], c):
                        return False, f"b{bi} uid{ins.uid}: v{v} survives clobber"
            # fixed pins
            for v, view in ins.fixed_views.items():
                if assign.get(v) != view:
                    return False, f"b{bi} uid{ins.uid}: v{v} not at {view}"
    return True, ""


def brute_feasibility(program, machine, var_classes):
    """Enumerate every view assignment; return (feasible, example, tried)."""
    variables = sorted(_all_vars(program))
    domains = [list(machine.classes[var_classes[v]]) for v in variables]
    tried = 0
    chosen = [None]

    def rec(i, acc):
        nonlocal tried
        if i == len(variables):
            tried += 1
            ok, _ = assignment_is_valid(program, machine, dict(acc))
            if ok:
                chosen[0] = dict(acc)
                return True
            return False
        for view in domains[i]:
            acc.append((variables[i], view))
            if rec(i + 1, acc):
                return True
            acc.pop()
        return False

    feas = rec(0, [])
    return feas, chosen[0], tried


# --------------------------------------------------------------------------
# standalone bit-segment simulation of emitted edge actions
# --------------------------------------------------------------------------


class _Bits:
    def __init__(self, machine):
        self.machine = machine
        self.segs = {}

    def write(self, name, tok):
        v = self.machine.view(name)
        out = []
        for lo, hi, t in self.segs.get(v.group, []):
            if hi <= v.lo or lo >= v.hi:
                out.append([lo, hi, t])
            else:
                if lo < v.lo:
                    out.append([lo, v.lo, t])
                if hi > v.hi:
                    out.append([v.hi, hi, t])
        out.append([v.lo, v.hi, tok])
        out.sort()
        merged = []
        for s in out:
            if merged and merged[-1][1] == s[0] and merged[-1][2] == s[2]:
                merged[-1][1] = s[1]
            else:
                merged.append(s)
        self.segs[v.group] = merged

    def read(self, name):
        v = self.machine.view(name)
        toks = []
        pos = v.lo
        for lo, hi, t in self.segs.get(v.group, []):
            if hi <= v.lo or lo >= v.hi:
                continue
            if lo > pos:
                toks.append(("UNINIT",))
            toks.append(t)
            pos = max(pos, hi)
        if pos < v.hi:
            toks.append(("UNINIT",))
        if not toks:
            return ("UNINIT",)
        return toks[0] if len(toks) == 1 else ("bits", tuple(toks))


def simulate_parallel_bits(actions, emitted_mis, machine, slot_offsets,
                          live_views=None):
    """Check emitted MI2s implement the intended parallel assignment.

    ``actions``: [("copy", dst_view, src_view)] or [("store", off, src_view)].
    Views start holding identity tokens ("V", name); slots ("M", off).

    Only views in ``live_views`` (default: views named by the actions) and
    slots in ``slot_offsets`` must finish with their intended content; a
    genuinely dead temporary view/scratch slot may be clobbered by the
    sequence.
    """
    if live_views is None:
        live_views = set()
        for a in actions:
            live_views.add(a[1])
            live_views.add(a[2])

    def fresh():
        regs = _Bits(machine)
        for vw in machine.views:
            regs.write(vw.name, ("V", vw.name))
        slots = {off: ("M", off) for off in slot_offsets}
        return regs, slots

    regs0, slots0 = fresh()
    intended_regs = _Bits(machine)
    for vw in machine.views:
        intended_regs.write(vw.name, ("V", vw.name))
    intended_slots = dict(slots0)
    for a in actions:
        if a[0] == "copy":
            intended_regs.write(a[1], regs0.read(a[2]))
        else:
            intended_slots[a[1]] = regs0.read(a[2])

    regs, slots = fresh()
    for mi in emitted_mis:
        if mi.kind == "copy":
            regs.write(mi.dst, regs.read(mi.src))
        elif mi.kind == "load":
            regs.write(mi.dst, slots.get(mi.slot, ("UNINIT",)))
        elif mi.kind == "store":
            slots[mi.slot] = regs.read(mi.src)
        else:
            return [f"unexpected {mi.kind} in edge sequence"]

    bad = []
    for name in sorted(live_views):
        if regs.read(name) != intended_regs.read(name):
            bad.append(
                f"view {name}: {regs.read(name)!r} != "
                f"{intended_regs.read(name)!r}"
            )
    for off, tok in intended_slots.items():
        if slots.get(off) != tok:
            bad.append(f"slot {off}: {slots.get(off)!r} != {tok!r}")
    return bad
