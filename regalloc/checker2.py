"""Independent verification for format-v2 allocations.

The checker never calls the allocator core.  It contains:

1. an **independent liveness derivation** for v2 programs (normal edges,
   exception edges, per-edge parallel copies), cross-checked against the
   report the allocator ships;
2. a **bit-level machine model**: every bank stores one symbolic token per
   bit position and every stack frame unit stores one token, so writes
   through narrow aliased views (e.g. ``ax`` over ``rax``) are tracked
   bit-for-bit;
3. a **per-edge symbolic transfer check** — EVERY CFG edge, not just one
   execution path: seed the edge's live values with distinct bit-vector
   symbols and all other bits/units with distinct background symbols, run
   the machine code that implements that edge, and assert that each
   parallel-copy/phi destination receives its source's bits, every
   live-through value keeps its bits, no unrelated/aliased location is read
   or overwritten, and call-clobbered views never carry a needed value;
4. a **slot overlap check**: frame units shared by two spilled values are
   allowed only when the two values are never simultaneously live;
5. a **paired bit-level symbolic executor** comparing the original
   three-address interpreter (opaque ``f(uses) -> defs``) against the
   allocated machine program on bounded DFS paths, covering BOTH normal
   exits and exception exits (on an exception transfer the excepting
   instruction's defs are treated as never written).

Failures carry an exact location (edge ``p->s``, exception instruction
index, machine instruction index).  The allocator runs the edge-complete
check itself and, on failure, raises rather than emitting an unverifiable
"complete" program; it cannot swallow edges it could not check.
"""

from __future__ import annotations

from dataclasses import dataclass, field

CLOBBERED = ("__clobbered__",)
BACKGROUND = ("__bg__",)


# --------------------------------------------------------------------------
# independent liveness (from first principles; does not import liveness2)
# --------------------------------------------------------------------------

def independent_liveness(program):
    """Return edge_live, instr_before/after, instr_except sets."""
    blocks = program.blocks
    n = len(blocks)

    phi_defs = [set() for _ in range(n)]
    phi_src: dict[tuple[int, int], set[int]] = {}
    for s, b in enumerate(blocks):
        for phi in b.phis:
            phi_defs[s].add(phi.def_var)
            for p, src in phi.sources:
                phi_src.setdefault((p, s), set()).add(src)
    edge_dst: dict[tuple[int, int], set[int]] = {}
    edge_src: dict[tuple[int, int], set[int]] = {}
    for p, b in enumerate(blocks):
        for ec in b.edge_copies:
            edge_dst.setdefault((p, ec.dst_block), set())
            edge_src.setdefault((p, ec.dst_block), set())
            for d, s in ec.pairs:
                edge_dst[(p, ec.dst_block)].add(d)
                edge_src[(p, ec.dst_block)].add(s)

    exc_set = [{t for ins in b.instrs for t in ins.except_to} for b in blocks]
    normal_succs = [[s for s in b.succs if s not in exc_set[i]]
                    for i, b in enumerate(blocks)]
    all_edge_dst = [set() for _ in range(n)]
    for (_, s), ds in edge_dst.items():
        all_edge_dst[s] |= ds

    up_exp, killed = [], []
    for i, b in enumerate(blocks):
        kb = set(phi_defs[i]) | all_edge_dst[i]
        ub: set[int] = set()
        for ins in b.instrs:
            ub.update(v for v in ins.uses if v not in kb)
            kb.update(ins.defs)
        up_exp.append(ub)
        killed.append(kb)

    def req(p, s, li):
        return ((set(li) - phi_defs[s] - edge_dst.get((p, s), set()))
                | phi_src.get((p, s), set()) | edge_src.get((p, s), set()))

    live_in = [set() for _ in range(n)]
    live_out = [set() for _ in range(n)]
    changed = True
    while changed:
        changed = False
        for i in range(n):
            out: set[int] = set()
            for s in normal_succs[i]:
                out |= req(i, s, live_in[s])
            ini = up_exp[i] | (out - killed[i])
            if out != live_out[i] or ini != live_in[i]:
                live_out[i], live_in[i] = out, ini
                changed = True

    edge_live = {}
    for i, b in enumerate(blocks):
        for s in b.succs:
            edge_live[(i, s)] = frozenset(req(i, s, live_in[s]))

    before, after, excp = [], [], []
    for bi, b in enumerate(blocks):
        live: set[int] = set()
        for s in normal_succs[bi]:
            live |= set(edge_live[(bi, s)])
        bs, af, ex = [], [], []
        for j in range(len(b.instrs) - 1, -1, -1):
            ins = b.instrs[j]
            af.append(frozenset(live))
            e: set[int] = set()
            for t in ins.except_to:
                e |= set(edge_live[(bi, t)])
            ex.append(frozenset(e))
            live = ((live - set(ins.defs)) | set(ins.uses) | e) - set()
            bs.append(frozenset(live))
        before.append(tuple(reversed(bs)))
        after.append(tuple(reversed(af)))
        excp.append(tuple(reversed(ex)))
    return {
        "edge_live": edge_live,
        "before": tuple(before),
        "after": tuple(after),
        "except": tuple(excp),
        "live_in": tuple(frozenset(s) for s in live_in),
    }


# --------------------------------------------------------------------------
# bit-level machine
# --------------------------------------------------------------------------

class BitMachine:
    def __init__(self, rt, frame_units: int):
        self.rt = rt
        self.bank_width = {}
        for v in rt.desc.views:
            self.bank_width[v.bank] = max(self.bank_width.get(v.bank, 0), v.lo + v.width)
        self.nb = rt.num_banks
        self.frame = frame_units
        self.reset()

    def reset(self):
        self.bits = [[BACKGROUND] * self.bank_width[b] for b in range(self.nb)]
        self.mem = [BACKGROUND] * self.frame
        # reads that observed a still-background bit (potential unset use)
        self.read_log: list[tuple] = []

    def write_unit(self, unit: int, token):
        if not 0 <= unit < len(self.mem):
            raise _BadMachineAccess(f"write outside frame: slot unit {unit} "
                                    f"(frame has {len(self.mem)} units)")
        self.mem[unit] = token

    def read_unit(self, unit: int):
        if not 0 <= unit < len(self.mem):
            raise _BadMachineAccess(f"read outside frame: slot unit {unit} "
                                    f"(frame has {len(self.mem)} units)")
        tok = self.mem[unit]
        if tok is BACKGROUND or tok is CLOBBERED:
            self.read_log.append((f"slot[{unit}]", (tok,)))
        return tok

    def write_view(self, view_name, tokens):
        if view_name not in self.rt._views:
            raise _BadMachineAccess(f"write to unknown view {view_name!r}")
        v = self.rt.view(view_name)
        if len(tokens) != v.width:
            raise _BadMachineAccess(
                f"write width {len(tokens)} != view {view_name} width {v.width}")
        for k, tok in enumerate(tokens):
            self.bits[v.bank][v.lo + k] = tok

    def read_view(self, view_name):
        if view_name not in self.rt._views:
            raise _BadMachineAccess(f"read from unknown view {view_name!r}")
        v = self.rt.view(view_name)
        out = tuple(self.bits[v.bank][v.lo + k] for k in range(v.width))
        if any(t is BACKGROUND or t is CLOBBERED for t in out):
            self.read_log.append((view_name, out))
        return out

    def clobber_view(self, view_name: str):
        if view_name not in self.rt._views:
            raise _BadMachineAccess(f"clobber unknown view {view_name!r}")
        v = self.rt.view(view_name)
        for k in range(v.width):
            self.bits[v.bank][v.lo + k] = CLOBBERED


class _BadMachineAccess(Exception):
    """Raised when emitted machine code touches an out-of-frame slot or an
    unknown view; converted into a verification failure."""

@dataclass
class EdgeFailure:
    edge: tuple[int, int]
    where: str
    detail: str
    exc_instr: int | None = None


@dataclass
class VerificationReport:
    ok: bool
    edges_checked: int
    edge_failures: list[EdgeFailure] = field(default_factory=list)
    slot_failures: list[str] = field(default_factory=list)
    liveness_mismatches: list[str] = field(default_factory=list)
    paths_explored: int = 0
    normal_exits: int = 0
    exception_exits: int = 0
    path_failures: list[str] = field(default_factory=list)
    bound_reached: bool = False
    steps_used: int = 0

    def failure_text(self) -> str:
        lines = []
        for f in self.edge_failures:
            loc = f"edge {f.edge[0]}->{f.edge[1]}"
            if f.exc_instr is not None:
                loc += f" (exception at instr {f.exc_instr})"
            lines.append(f"{loc} {f.where}: {f.detail}")
        lines.extend(self.slot_failures)
        lines.extend(self.liveness_mismatches)
        lines.extend(self.path_failures)
        return "; ".join(lines)


# --------------------------------------------------------------------------
# helpers: expected edge parallel assignments and home locations
# --------------------------------------------------------------------------

def expected_edge_assignments(program):
    """edge -> list of (dst_var, src_var) simultaneous copies (phi + explicit)."""
    out: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for s, b in enumerate(program.blocks):
        for phi in b.phis:
            for p, src in phi.sources:
                out.setdefault((p, s), []).append((phi.def_var, src))
    for p, b in enumerate(program.blocks):
        for ec in b.edge_copies:
            for d, s in ec.pairs:
                out.setdefault((p, ec.dst_block), []).append((d, s))
    return out


def var_class_map(program):
    rt = program.target.resolved()
    default = rt.default_class_name
    cls = {}
    for v in program.params:
        cls[v] = program.param_classes.get(v, default)
    for b in program.blocks:
        for ins in b.instrs:
            for v in ins.defs:
                cls[v] = ins.classes.get(v, default)
        for phi in b.phis:
            cls[phi.def_var] = default
        for ec in b.edge_copies:
            for d, _ in ec.pairs:
                cls.setdefault(d, default)
    return cls


def home_of(alloc2, v, cls):
    """('reg', view_tuple) or ('slot', (start,size))."""
    if v in alloc2.slots:
        start, size = alloc2.slots[v]
        return ("slot", (start, size))
    return ("reg", alloc2.assignment[v])


# --------------------------------------------------------------------------
# machine code belonging to each CFG edge
# --------------------------------------------------------------------------

def edge_machine_code(program, ap):
    """Return {(p,s): [(exc_instr_or_None, [MI2,...]), ...]}.

    Exception splits contribute one entry per excepting instruction; normal
    edge code either lives in a dedicated split block or (non-critical,
    no-exception block) in the tail of the predecessor's last segment.
    """
    out: dict[tuple[int, int], list] = {}
    for ab in ap.blocks:
        if ab.origin < 0 and ab.split_edge is not None:
            out.setdefault(ab.split_edge, []).append((ab.exc_instr, list(ab.instrs)))
    # v2 always isolates an edge's actions in dedicated split blocks, so the
    # only machine code that belongs to an edge is the code in those blocks.
    # Edges with no entry here have no edge actions (checked as an empty
    # transfer, which must leave every seeded live value untouched).
    return out


# --------------------------------------------------------------------------
# per-edge bit-level symbolic transfer check
# --------------------------------------------------------------------------

def _value_pieces(rt, class_name, bits):
    """Split a value bit vector into one piece per occupied frame unit.

    A frame unit is opaque (no sub-slot aliasing): one unit holds one
    word-sized piece; a size-1 value keeps its whole vector.
    """
    size = value_unit_count(rt, class_name)
    if size == 1:
        return [tuple(bits)]
    h = len(bits) // 2
    return [tuple(bits[:h]), tuple(bits[h:])]


def value_unit_count(rt, class_name):
    word = rt.class_info(rt.default_class_name).width
    w = rt.class_info(class_name).width
    return max(1, -(-w // word))


def _seed_edge(machine: BitMachine, alloc2, live_vars, cls, vmap):
    """Place distinct bit symbols for each live value at its home.

    Returns the set of banks that hold a seeded value (live value or edge
    source); other banks are free temporaries the edge code may overwrite.
    """
    rt = machine.rt
    seeded_banks: set[int] = set()
    for v in live_vars:
        bits = tuple((("V", v), k) for k in range(_class_width(rt, cls[v])))
        vmap[v] = bits
        kind, loc = home_of(alloc2, v, cls)
        if kind == "reg":
            for vn in loc:
                seeded_banks.add(rt.view(vn).bank)
            pieces = _split_pair_views(rt, cls[v], loc, bits)
            for view, tok in zip(loc, pieces):
                machine.write_view(view, tok)
        else:
            start, size = loc
            for k, piece in enumerate(_value_pieces(rt, cls[v], bits)):
                machine.write_unit(start + k, piece)
    return seeded_banks


def _class_width(rt, class_name):
    return rt.class_info(class_name).width


def _split_pair_views(rt, class_name, views, bits):
    """Split a value bit vector into per-view vectors (pair halves)."""
    if len(views) == 1:
        return [bits]
    half = len(bits) // 2
    # views are (lo, hi); lo gets the low half bits
    return [bits[:half], bits[half:]]


def check_all_edges(program, alloc2) -> list[EdgeFailure]:
    rt = program.target.resolved()
    live = independent_liveness(program)
    cls = var_class_map(program)
    assigns = expected_edge_assignments(program)
    code = edge_machine_code(program, alloc2.allocated_program)
    failures: list[EdgeFailure] = []

    for (p, s), live_set in sorted(live["edge_live"].items()):
        entries = code.get((p, s), [(None, [])])
        for exc_instr, mis in entries:
            failures.extend(
                _check_one_edge(program, alloc2, rt, cls, live, assigns,
                                (p, s), live_set, exc_instr, mis)
            )
    return failures


def _edge_dst_vars(assigns, edge):
    return {d for d, _ in assigns.get(edge, ())}


def _edge_survivors(live, assigns, edge):
    """Variables that must still exist AFTER the edge transfer at s:
    successor live-in minus the values born on this edge.  Pure copy/phi
    sources (they die on the edge) are not survivors."""
    p, s = edge
    born = _edge_dst_vars(assigns, edge)
    return {v for v in live["live_in"][s] if v not in born}


def _check_one_edge(program, alloc2, rt, cls, live, assigns, edge,
                    live_set, exc_instr, mis):
    p, s = edge
    machine = BitMachine(rt, max(alloc2.frame_slots, 1))
    vmap: dict[int, tuple] = {}
    seeded_banks = _seed_edge(machine, alloc2, live_set, cls, vmap)
    seeded_units = {
        u for v in live_set if v in alloc2.slots
        for u in range(alloc2.slots[v][0], alloc2.slots[v][0] + alloc2.slots[v][1])
    }
    # snapshot of every seeded bit/unit before the edge code runs
    _initial_bits = {(b, k): machine.bits[b][k] for b in seeded_banks
                     for k in range(machine.bank_width[b])}
    _initial_units = {u: machine.mem[u] for u in seeded_units}

    group = dict(assigns.get(edge, ()))
    # destinations that are born on this edge
    dst_set = set(group)
    # destination homes are pre-seeded with background; record expected bits
    expected: dict = {}
    for d, src in group.items():
        if src not in vmap:
            return [EdgeFailure(edge, "parallel copy",
                                f"source v{src} is not live on this edge", exc_instr)]
        expected[d] = vmap[src]

    def home_bits(kind, loc_views, width_units=None):
        if kind == "reg":
            pieces = []
            for view in loc_views:
                pieces.append(machine.read_view(view))
            return tuple(t for piece in pieces for t in piece)
        start, size = loc_views
        out = []
        for k in range(size):
            out.extend(machine.read_unit(start + k))
        return tuple(out)

    try:
        for idx, mi in enumerate(mis):
            where = f"machine instr {idx} ({mi.render()})"
            if mi.kind == "copy":
                toks = machine.read_view(mi.src)
                machine.write_view(mi.dst, toks)
            elif mi.kind == "load":
                toks = machine.read_unit(mi.slot + mi.off)
                machine.write_view(mi.dst, toks)
            elif mi.kind == "store":
                toks = machine.read_view(mi.src)
                machine.write_unit(mi.slot + mi.off, toks)
            else:
                return [EdgeFailure(edge, where, "unexpected op inside edge code", exc_instr)]
            if machine.read_log:
                view, toks = machine.read_log[-1]
                return [EdgeFailure(edge, where,
                                    f"read of unset/clobbered/aliased location {view}",
                                    exc_instr)]
    except Exception as exc:  # noqa: BLE001 - report as verification failure
        return [EdgeFailure(edge, "machine execution", repr(exc), exc_instr)]

    # every edge-copy/phi destination must hold its source value at home
    for d, want_bits in expected.items():
        kind, loc = home_of(alloc2, d, cls)
        got = home_bits(kind, loc if kind == "slot" else loc)
        if got != want_bits:
            return [EdgeFailure(edge, f"destination v{d}",
                                f"got bits {got[:4]}..., expected source bits {want_bits[:4]}...",
                                exc_instr)]

    # every live-through variable keeps its bits at home.  A pure copy/phi
    # source that dies on this edge is not a survivor: its home may be
    # overwritten by the parallel assignment.
    survivors = _edge_survivors(live, assigns, edge)
    src_set = set(group.values())
    for v in live_set:
        if v in dst_set:
            continue
        if v in src_set and v not in survivors:
            continue  # consumed and dying on this edge
        kind, loc = home_of(alloc2, v, cls)
        want = vmap[v]
        got = home_bits(kind, loc)
        if got != want:
            return [EdgeFailure(edge, f"live-through v{v}",
                                "bits were changed by the edge transfer", exc_instr)]

    # write protection applies only to banks/units that held a seeded live
    # value or edge source.  Completely free banks may serve as reload temps
    # and unseeded frame units may be transient; the private scratch slot is
    # dedicated to cycle breaking and is always exempt.  The ONLY seeded
    # bank whose bits may change is the home of a source that dies right
    # here (consumed by the parallel assignment, not live into the successor).
    allowed_bank_bits: set = set()
    for _, src in group.items():
        if src in live_set and src not in dst_set and src not in survivors:
            kind, loc = home_of(alloc2, src, cls)
            if kind == "reg":
                for vn in loc:
                    vv = rt.view(vn)
                    for k in range(vv.width):
                        allowed_bank_bits.add((vv.bank, vv.lo + k))
    for b in seeded_banks:
        for bit in range(machine.bank_width[b]):
            if machine.bits[b][bit] == _initial_bits.get((b, bit)):
                continue
            if (b, bit) in allowed_bank_bits:
                continue
            return [EdgeFailure(edge, f"bank {b} bit {bit}",
                                "edge code clobbered a live aliased view's bits",
                                exc_instr)]
    sc = alloc2.allocated_program.scratch_slot
    for u in seeded_units:
        if sc <= u < sc + alloc2.allocated_program.scratch_size:
            continue
        if machine.mem[u] != _initial_units.get(u):
            return [EdgeFailure(edge, f"frame unit {u}",
                                "edge code overwrote a live value's stack unit",
                                exc_instr)]
    return []


# --------------------------------------------------------------------------
# slot overlap check
# --------------------------------------------------------------------------

def check_slot_overlap(program, alloc2) -> list[str]:
    rt = program.target.resolved()
    cls = var_class_map(program)
    live = independent_liveness(program)
    spilled = set(alloc2.slots)
    failures: list[str] = []

    # collect simultaneous-live pairs from every program point
    colive: set[frozenset[int]] = set()

    def note(points):
        for grp in points:
            grp = [v for v in grp if v in spilled]
            for i in range(len(grp)):
                for j in range(i + 1, len(grp)):
                    colive.add(frozenset((grp[i], grp[j])))

    for bi in range(len(program.blocks)):
        note(live["before"][bi])
        note(live["after"][bi])
        note(live["except"][bi])
        for s in program.blocks[bi].succs:
            note([live["edge_live"][(bi, s)]])

    intervals: dict[int, tuple[int, int]] = {v: (a, a + size - 1)
                                             for v, (a, size) in alloc2.slots.items()}
    for a, b in ((x, y) for x in spilled for y in spilled if x < y):
        ia, ib = intervals[a], intervals[b]
        overlap = ia[0] <= ib[1] and ib[0] <= ia[1]
        if overlap and frozenset((a, b)) in colive:
            failures.append(
                f"slots of v{a} {alloc2.slots[a]} and v{b} {alloc2.slots[b]} overlap "
                f"while both are simultaneously live"
            )

    # scratch must be disjoint from every data slot
    sc = alloc2.allocated_program.scratch_slot
    for v, (start, size) in alloc2.slots.items():
        if start < sc + alloc2.allocated_program.scratch_size and sc < start + size:
            failures.append(f"scratch slot {sc} overlaps data slot of v{v}")

    # alignment of every slot start
    align = program.stack.align
    for v, (start, size) in alloc2.slots.items():
        req = min(align, size) if align > 1 else 1
        if start % req:
            failures.append(f"slot of v{v} starts at {start}, misaligned for size {size}/align {align}")
    return failures




# --------------------------------------------------------------------------
# paired bit-level execution vs the original three-address interpreter
# --------------------------------------------------------------------------

def _class_width(rt, name):
    return rt.class_info(name).width


def _pieces(width_per_view, bits):
    if len(width_per_view) == 1:
        return [bits]
    # pair: views are (lo, hi)
    h = len(bits) // 2
    return [bits[:h], bits[h:]]


def paired_execution(program, alloc2, *, depth_bound=120, step_budget=40_000):
    """Bounded DFS pairing the source three-address interpreter with the
    bit-level machine.  Every instruction can complete normally or transfer
    exceptionally through each of its ``except_to`` targets.

    Returns a ``VerificationReport``-like dict with failures, path counts,
    normal/exception terminal counts and bound flags.  Expected values are
    derived solely from the source program; the allocator core is never
    called to produce them.
    """
    rt = program.target.resolved()
    cls = var_class_map(program)
    ap = alloc2.allocated_program
    assigns = expected_edge_assignments(program)

    split_code: dict[tuple[int, int, object], list] = {}
    for ab in ap.blocks:
        if ab.origin < 0 and ab.split_edge is not None:
            split_code[(ab.split_edge, ab.exc_instr)] = list(ab.instrs)
    # normal-only post blocks for excepting instructions:
    #   mid_post[(b,j)] -> post MIs when more source instructions follow
    #   end_post[b]     -> post MIs on the block-end normal path
    mid_post: dict[tuple[int, int], list] = {}
    end_post: dict[int, list] = {}
    for ab in ap.blocks:
        if ab.origin < 0 and ab.split_edge is None and ab.normal_post_of is not None:
            if ab.normal_post_instr >= 0:
                mid_post[(ab.normal_post_of, ab.normal_post_instr)] = list(ab.instrs)
            else:
                end_post[ab.normal_post_of] = list(ab.instrs)

    stats = {"paths": 0, "normal": 0, "exception": 0, "steps": 0, "bound": False}
    failures: list[str] = []

    def step_mi(machine, mi):
        if mi.kind == "copy":
            machine.write_view(mi.dst, machine.read_view(mi.src))
        elif mi.kind == "load":
            machine.write_view(mi.dst, machine.read_unit(mi.slot + mi.off))
        elif mi.kind == "store":
            machine.write_unit(mi.slot + mi.off, machine.read_view(mi.src))
        else:
            raise AssertionError("op MI stepped as transfer")

    def groups_of(ab):
        """Split a segment's MI list into per-source-instruction groups:
        [(pre:[MI], op:MI, post:[MI]), ...] in program order."""
        groups = []
        cur_pre: list = []
        for mi in ab.instrs:
            if mi.kind == "op":
                groups.append([cur_pre, mi, []])
                cur_pre = []
            elif groups:
                groups[-1][2].append(mi)
            else:
                cur_pre.append(mi)
        if cur_pre:
            # pre-MIs before any op: attach to first group defensively
            if groups:
                groups[0][0] = cur_pre + groups[0][0]
        return groups

    def machine_snapshot(machine):
        return ([row[:] for row in machine.bits], machine.mem[:])

    def machine_restore(machine, snap):
        bits, mem = snap
        machine.bits = [row[:] for row in bits]
        machine.mem = mem[:]
        machine.read_log = []

    def write_value(machine, v, bits):
        kind, loc = home_of(alloc2, v, cls)
        if kind == "reg":
            views = loc
            widths = [rt.view(x).width for x in views]
            for view, piece in zip(views, _pieces(widths, bits)):
                machine.write_view(view, piece)
        else:
            start, size = loc
            for k, piece in enumerate(_value_pieces(rt, cls[v], bits)):
                machine.write_unit(start + k, piece)

    def read_value(machine, v):
        kind, loc = home_of(alloc2, v, cls)
        if kind == "reg":
            out = []
            for view in loc:
                out.extend(machine.read_view(view))
            return tuple(out)
        start, size = loc
        out = []
        for k in range(size):
            out.extend(machine.read_unit(start + k))
        return tuple(out)

    def verify_edge_homes(machine, env, p, s):
        for d, src in assigns.get((p, s), ()):
            want = env[src]
            got = read_value(machine, d)
            if got != want:
                return f"edge {p}->{s}: v{d} holds {got[:4]}..., expected source v{src} {want[:4]}..."
        dst_vars = {d for d, _ in assigns.get((p, s), ())}
        for v in env_dummy_edge_live(program, p, s):
            if v in dst_vars:
                continue
            if read_value(machine, v) != env[v]:
                return f"edge {p}->{s}: live-through v{v} changed"
        return None

    def env_dummy_edge_live(program, p, s):
        # computed from independent liveness inside caller; use a closure var
        return _EDGE_LIVE[(p, s)]

    def apply_edge_env(env, p, s):
        new_env = dict(env)
        for d, src in assigns.get((p, s), ()):
            new_env[d] = env[src]
        return new_env

    live = independent_liveness(program)
    _EDGE_LIVE = live["edge_live"]

    # per source block: one group per source instruction, derived from the
    # actual allocated segment blocks (in segment order), matching by uid.
    origin_blocks: dict[int, list] = {}
    for ab in ap.blocks:
        if ab.origin >= 0:
            origin_blocks.setdefault(ab.origin, []).append(ab)
    for lst in origin_blocks.values():
        lst.sort(key=lambda ab: ab.seg)

    def block_group_list(b):
        groups = []
        for ab in origin_blocks[b]:
            groups.extend(groups_of(ab))
        return groups

    def rec(b, env, machine, depth):
        if stats["steps"] > step_budget or depth < 0:
            stats["bound"] = True
            return
        blk = program.blocks[b]
        groups = block_group_list(b)
        if len(groups) != len(blk.instrs):
            failures.append(
                f"block {b}: allocated op count {len(groups)} != "
                f"source instrs {len(blk.instrs)}"
            )
            return
        for gi, ins in enumerate(blk.instrs):
            pre, opmi, post = groups[gi]
            if opmi.uid != ins.uid:
                failures.append(f"block {b}: uid order {opmi.uid} != {ins.uid}")
                return
            # exception branch(es): state as of BEFORE this instruction
            if ins.except_to:
                snap = machine_snapshot(machine)
                env_snap = dict(env)
            for mi in pre:
                step_mi(machine, mi)
                stats["steps"] += 1
            # input equivalence
            src_bits: list = []
            for u in ins.uses:
                if u not in env:
                    failures.append(f"uid {ins.uid}: source read of undefined v{u}")
                    return
                src_bits.extend(env[u])
            dst_bits: list = []
            for view in opmi.reads:
                dst_bits.extend(machine.read_view(view))
            if machine.read_log:
                failures.append(f"uid {ins.uid} block {b}: machine reads unset/clobbered bits")
                return
            if tuple(src_bits) != tuple(dst_bits):
                failures.append(
                    f"uid {ins.uid} block {b}: input bits differ: "
                    f"src {tuple(src_bits)[:4]}... dst {tuple(dst_bits)[:4]}..."
                )
                return
            # normal completion: clobbers then defs
            for view in opmi.clobbersv:
                machine.clobber_view(view)
            new_bits: dict[int, tuple] = {}
            bases = tuple(env[u][0][0] if env[u] else None for u in ins.uses)
            for di, d in enumerate(ins.defs):
                width = _class_width(rt, cls[d])
                base = ("op", ins.uid, bases, di)
                bits = tuple((base, kk) for kk in range(width))
                new_bits[d] = bits
                env[d] = bits
            # machine writes, sliced per def
            wi = 0
            for d in ins.defs:
                if d in alloc2.slots:
                    nviews = alloc2.slots[d][1]
                else:
                    nviews = len(alloc2.assignment[d])
                slice_views = opmi.writes[wi:wi + nviews]
                wi += nviews
                widths = [rt.view(x).width for x in slice_views]
                for view, piece in zip(slice_views, _pieces(widths, new_bits[d])):
                    machine.write_view(view, piece)
            for mi in post:
                step_mi(machine, mi)
                stats["steps"] += 1
            stats["steps"] += 1
            # exceptional continuations from the pre-instruction state
            for t in ins.except_to:
                m2 = BitMachine(rt, max(ap.frame_slots, 1))
                machine_restore(m2, snap)
                code = split_code.get(((b, t), gi))
                if code is None:
                    failures.append(f"missing exception split code for {b}->{t} at instr {gi}")
                    continue
                for mi in code:
                    step_mi(m2, mi)
                env2 = apply_edge_env(env_snap, b, t)
                err = _verify_edge(m2, env_snap, env2, b, t)
                if err:
                    failures.append(err)
                    continue
                stats["exception"] += 1
                rec(t, env2, m2, depth - 1)
            # normal path: an excepting instruction's result post-actions run
            # only here (they are absent on the exception transfers)
            for mi in mid_post.get((b, gi), ()):
                step_mi(machine, mi)
                stats["steps"] += 1
        # normal block-end successors
        exc_targets = {t for ins in blk.instrs for t in ins.except_to}
        normal = [s for s in blk.succs if s not in exc_targets]
        if not normal:
            stats["normal"] += 1
            stats["paths"] += 1
            return
        for s in normal:
            m2 = BitMachine(rt, max(ap.frame_slots, 1))
            machine_restore(m2, machine_snapshot(machine))
            # block-ending excepting instruction normal-only post actions
            for mi in end_post.get(b, ()):
                step_mi(m2, mi)
            code = split_code.get(((b, s), None))
            if code:
                for mi in code:
                    step_mi(m2, mi)
            env2 = apply_edge_env(env, b, s)
            err = _verify_edge(m2, env, env2, b, s)
            if err:
                failures.append(err)
                continue
            rec(s, env2, m2, depth - 1)

    def _verify_edge(machine, env_pre, env_post, p, s):
        for d, src in assigns.get((p, s), ()):
            if read_value(machine, d) != env_pre[src]:
                return f"edge {p}->{s}: v{d} != source v{src} bits"
        dst_vars = {d for d, _ in assigns.get((p, s), ())}
        src_dying = {src for _, src in assigns.get((p, s), ())}
        survivors = _edge_survivors(live, assigns, (p, s))
        for v in _EDGE_LIVE[(p, s)]:
            if v in dst_vars:
                continue
            if v in src_dying and v not in survivors:
                continue
            if read_value(machine, v) != env_pre[v]:
                return f"edge {p}->{s}: live-through v{v} corrupted"
        return None

    machine0 = BitMachine(rt, max(ap.frame_slots, 1))
    env0: dict[int, tuple] = {}
    for v in program.params:
        width = _class_width(rt, cls[v])
        bits = tuple((("param", v), kk) for kk in range(width))
        env0[v] = bits
        write_value(machine0, v, bits)
    try:
        rec(0, env0, machine0, depth_bound)
    except _BadMachineAccess as exc:
        failures.append(f"illegal machine memory access: {exc}")
    return {
        "failures": failures,
        "paths": stats["paths"],
        "normal_exits": stats["normal"],
        "exception_edges": stats["exception"],
        "bound_reached": stats["bound"],
        "steps": stats["steps"],
    }


# --------------------------------------------------------------------------
# top-level verification
# --------------------------------------------------------------------------

def verify_allocation(program, alloc2, *, depth_bound=120, step_budget=40_000) -> VerificationReport:
    """Run every independent check.  The edge and slot checks are complete
    over all edges/points; the paired execution is a bounded DFS."""
    report = VerificationReport(ok=False, edges_checked=0)

    # 1) shipped liveness report must match the independent derivation
    ind = independent_liveness(program)
    shipped = alloc2.liveness
    for ek in sorted(set(ind["edge_live"]) | set(shipped.edge_live)):
        if ind["edge_live"].get(ek, frozenset()) != shipped.edge_live.get(ek, frozenset()):
            report.liveness_mismatches.append(
                f"edge_live {ek}: {sorted(ind['edge_live'].get(ek, ()))} != "
                f"{sorted(shipped.edge_live.get(ek, ()))}"
            )
    for bi in range(len(program.blocks)):
        for j in range(len(program.blocks[bi].instrs)):
            if ind["before"][bi][j] != shipped.instr_before[bi][j]:
                report.liveness_mismatches.append(f"before[{bi}][{j}] mismatch")
            if ind["after"][bi][j] != shipped.instr_after[bi][j]:
                report.liveness_mismatches.append(f"after[{bi}][{j}] mismatch")
            if ind["except"][bi][j] != shipped.instr_except_live[bi][j]:
                report.liveness_mismatches.append(f"except[{bi}][{j}] mismatch")

    # 2) every edge, bit-level transfer (exhaustive over ALL edges)
    edge_failures = check_all_edges(program, alloc2)
    report.edge_failures = edge_failures
    report.edges_checked = len(ind["edge_live"])

    # 3) stack slot overlap / alignment
    report.slot_failures = check_slot_overlap(program, alloc2)

    # 4) bounded paired execution (normal + exception paths)
    pr = paired_execution(program, alloc2, depth_bound=depth_bound, step_budget=step_budget)
    report.paths_explored = pr["paths"]
    report.normal_exits = pr["normal_exits"]
    report.exception_exits = pr["exception_edges"]
    report.bound_reached = pr["bound_reached"]
    report.steps_used = pr["steps"]
    report.path_failures = pr["failures"]

    report.ok = not (report.edge_failures or report.slot_failures
                     or report.liveness_mismatches or report.path_failures)
    return report
