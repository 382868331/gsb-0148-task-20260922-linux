"""Independent-checker tests for v2.

Three kinds of coverage:

1. **complete small-domain enumeration** — for a family of tiny programs on
   a 4-view machine, a brute-force enumerator in ``_reference2`` decides
   feasibility by checking EVERY complete register-view assignment; the
   solver must agree (feasible -> allocated assignment that independently
   validates; register-infeasible-at-a-point -> infeasible status);
2. **mutation tests** — deliberately corrupted allocations must be caught
   by the verifier or by the bit-level symbolic executor;
3. **parallel-copy reference** — emitted edge sequences must match a true
   simultaneous assignment under a standalone bit-segment simulator.
"""

import unittest

from regalloc import allocate2, verify_allocation
from regalloc.allocator2 import AllocationResult
from regalloc.lower2 import sequential_parallel2
from regalloc.machine import Machine, RegView
from regalloc.model import Block, Instr, PhiNode, Program
from regalloc.samples2 import machine_torture, sample_v2_torture, machine_16
from regalloc.symbolic2 import equivalence_check2

from _reference2 import brute_feasibility, simulate_parallel_bits


# --------------------------------------------------------------------------
# a tiny machine: two groups, w0/w1 (32-bit) and x0/x1 (64-bit aliases)
# --------------------------------------------------------------------------


def tiny_machine(*, callee_saved=False):
    views = (
        RegView("w0", 0, 0, 32, "w", callee_saved=callee_saved),
        RegView("x0", 0, 0, 64, "x", callee_saved=callee_saved),
        RegView("w1", 1, 0, 32, "w", callee_saved=callee_saved),
        RegView("x1", 1, 0, 64, "x", callee_saved=callee_saved),
    )
    return Machine(
        views=views,
        aliases={"w0": ("x0",), "x0": ("w0",), "w1": ("x1",), "x1": ("w1",)},
        class_align={"w": 4, "x": 8},
    )


def _w(uid, uses, defs=(), **kw):
    return Instr(uid=uid, uses=uses, defs=defs, **kw)


def _straight(instrs, params, classes):
    return Program(
        blocks=(Block(label="e", instrs=tuple(instrs), succs=()),),
        params=tuple(params),
        var_classes=dict(classes),
    )


def _complete_enumeration_cases():
    W, X = "w", "x"
    cases = []
    # feasible: one def, one use
    cases.append(("one", _straight(
        [_w(0, (), (1,)), _w(1, (1,), (2,)), _w(2, (2,), ())],
        (), {1: W, 2: W})))
    # feasible: two simultaneous w values, two w views
    cases.append(("pair", _straight(
        [_w(0, (), (1,)), _w(1, (), (2,)), _w(2, (1, 2), (3,)), _w(3, (3,), ())],
        (), {1: W, 2: W, 3: W})))
    # infeasible at the read point: three simultaneous w uses, two w views
    cases.append(("triple-read", _straight(
        [_w(0, (0, 1, 2), ())],
        (0, 1, 2), {0: W, 1: W, 2: W})))
    # feasible with mixed classes: w and x values can share a group only
    # via different groups; w + x coexist here (groups 0 and 1 exist)
    cases.append(("mixed", _straight(
        [_w(0, (), (1,)), _w(1, (), (2,)), _w(2, (1, 2), (3,)),
         _w(3, (3,), ())],
        (), {1: W, 2: X, 3: W})))
    # infeasible: w use and w def pinned to the SAME view, early clobber
    cases.append(("early-pin", _straight(
        [_w(0, (0,), (1,), fixed_views={0: "w0", 1: "w0"},
           early_clobber=(1,))],
        (0,), {0: W, 1: W})))
    return cases


class TestCompleteSmallDomain(unittest.TestCase):
    """Brute-force enumerates the whole domain; the solver must agree."""

    def test_feasibility_agrees_with_enumeration(self):
        m = tiny_machine()
        for name, prog in _complete_enumeration_cases():
            with self.subTest(case=name):
                feas, example, tried = brute_feasibility(
                    prog, m, prog.var_classes
                )
                self.assertGreater(tried, 0)
                r = allocate2(prog, m)
                if not feas:
                    self.assertEqual(
                        r.status, "infeasible",
                        f"{name}: brute force says infeasible, solver says "
                        f"{r.status}",
                    )
                else:
                    self.assertEqual(
                        r.status, "allocated",
                        f"{name}: brute force found an assignment "
                        f"({example}), solver says {r.status}: {r.message}",
                    )
                    # no spills needed in these tiny register-feasible cases
                    self.assertFalse(
                        r.slots, f"{name}: unexpected spills {sorted(r.slots)}"
                    )
                    rep = verify_allocation(prog, m, r)
                    self.assertTrue(rep.ok, (name, rep.failures))


class TestParallelCopyReference(unittest.TestCase):
    def test_simple_alias_widths(self):
        m = machine_16()
        actions = [["copy", "w0", "w1"], ["copy", "w1", "x0"]]
        mis, breaks = sequential_parallel2(actions, m, set(), None)
        bad = simulate_parallel_bits(actions, mis, m, set())
        self.assertEqual(bad, [])

    def test_cycle_with_free_temp_uses_temp(self):
        m = machine_16()
        actions = [["copy", "w0", "w1"], ["copy", "w1", "w0"]]
        mis, breaks = sequential_parallel2(actions, m, set(), None)
        self.assertIn("temp", breaks)
        bad = simulate_parallel_bits(actions, mis, m, set())
        self.assertEqual(bad, [])

    def test_cycle_with_no_free_temp_uses_slot(self):
        # every group's low 32 bits is occupied by a live value, so no
        # temporary view can be borrowed
        m = machine_16()
        occupied = {f"w{g}" for g in range(8)}
        actions = [["copy", "w0", "w1"], ["copy", "w1", "w0"]]
        from regalloc.allocator2 import Slot
        scratch = Slot(offset=64, size=8, align=8)
        mis, breaks = sequential_parallel2(actions, m, occupied, scratch)
        self.assertEqual(breaks, ("slot",))
        # the scratch slot is scratch storage (clobberable by the sequence),
        # so it is not part of the reference's expected slots
        bad = simulate_parallel_bits(actions, mis, m, set())
        self.assertEqual(bad, [])
        # the scratch store must not target a live view's register... it
        # targets memory, and the saved view w0 is itself in the cycle
        kinds = [mi.kind for mi in mis]
        self.assertIn("store", kinds)
        self.assertIn("load", kinds)


class TestVerifierMutation(unittest.TestCase):
    def setUp(self):
        self.prog = sample_v2_torture()
        self.m = machine_torture()
        self.r = allocate2(self.prog, self.m)
        self.assertEqual(self.r.status, "allocated")

    def _copy_result(self):
        return AllocationResult(
            status=self.r.status,
            program=self.r.program,
            machine=self.r.machine,
            assignment=dict(self.r.assignment),
            slots=dict(self.r.slots),
            scratch_slot=self.r.scratch_slot,
            frame_size=self.r.frame_size,
            allocated_program=self.r.allocated_program,
            liveness=self.r.liveness,
            iterations=self.r.iterations,
            history=self.r.history,
            message=self.r.message,
        )

    def test_good_result_passes(self):
        rep = verify_allocation(self.prog, self.m, self.r)
        self.assertTrue(rep.ok, rep.failures)

    def _colive_pair(self, spilled: bool = False):
        """Two vars live together at some point (spilled pair or reg pair)."""
        rep = self.r.liveness
        points = list(rep.edge_live.values())
        for bi in range(len(self.prog.blocks)):
            points.extend(rep.instr_before[bi])
            points.extend(rep.instr_after[bi])
        for live in points:
            vs = sorted(live)
            for i in range(len(vs)):
                for j in range(i + 1, len(vs)):
                    a, b = vs[i], vs[j]
                    if spilled:
                        if a in self.r.slots and b in self.r.slots:
                            return a, b
                    else:
                        if a in self.r.assignment and b in self.r.assignment:
                            va, vb = self.r.assignment[a], self.r.assignment[b]
                            if self.m.views_overlap(va, vb) is False and \
                                    self.m.view(va).cls == self.m.view(vb).cls:
                                return a, b
        return None

    def test_swapped_homes_caught(self):
        pair = self._colive_pair()
        self.assertIsNotNone(pair, "expected a co-live register pair")
        a, b = pair
        r2 = self._copy_result()
        # force both co-live values into the SAME view -> they share bits
        shared = r2.assignment[a]
        r2.assignment[b] = shared
        rep = verify_allocation(self.prog, self.m, r2)
        self.assertFalse(rep.ok)
        self.assertTrue(any("share register bits" in f for f in rep.failures))

    def test_overlapping_slot_offsets_caught(self):
        pair = self._colive_pair(spilled=True)
        self.assertIsNotNone(pair, "expected a co-live spilled pair")
        a, b = pair
        r2 = self._copy_result()
        r2.slots[a] = r2.slots[b]
        rep = verify_allocation(self.prog, self.m, r2)
        self.assertFalse(rep.ok)
        self.assertTrue(any("share stack bytes" in f for f in rep.failures))

    def test_misaligned_slot_caught(self):
        from regalloc.allocator2 import Slot
        r2 = self._copy_result()
        v = next(iter(r2.slots))
        s = r2.slots[v]
        r2.slots[v] = Slot(offset=s.offset + 1, size=s.size, align=s.align)
        rep = verify_allocation(self.prog, self.m, r2)
        self.assertFalse(rep.ok)
        self.assertTrue(any("alignment" in f for f in rep.failures))

    def test_early_clobber_corruption_caught(self):
        r2 = self._copy_result()
        # rewrite uid 12's early-clobber write view to overlap a read view
        for ab in r2.allocated_program.blocks:
            for mi in ab.instrs:
                if mi.kind == "op" and mi.uid == 12:
                    if mi.reads and mi.writes:
                        # use the first read's view as the write view
                        object.__setattr__(mi, "writes", (mi.reads[0],) + mi.writes[1:])
        rep = verify_allocation(self.prog, self.m, r2)
        self.assertFalse(rep.ok)
        self.assertTrue(any("early-clobber" in f for f in rep.failures))

    def test_dropped_exceptional_edge_actions_caught(self):
        r2 = self._copy_result()
        # delete the instructions of the exceptional split block; the
        # symbolic executor then observes the handler phi value never
        # arriving (the verifier handles structural edge realizability;
        # bit-level arrival is the executor's job)
        removed = False
        for ab in r2.allocated_program.blocks:
            if ab.split_kind == "exc" and ab.instrs:
                ab.instrs = []
                removed = True
        self.assertTrue(removed)
        eq = equivalence_check2(
            self.prog, self.m, r2, depth_bound=30, step_budget=20_000
        )
        self.assertFalse(eq.equivalent)
        self.assertTrue(eq.exc_exits >= 0)  # result object well-formed

    def test_status_flag_not_trusted(self):
        # a "budget" status carrying a program is a lie the checker rejects
        r2 = self._copy_result()
        object.__setattr__(r2, "status", "budget_exhausted")
        rep = verify_allocation(self.prog, self.m, r2)
        self.assertFalse(rep.ok)
        self.assertTrue(any("unallocated state" in f for f in rep.failures))


class TestSymbolicMutation(unittest.TestCase):
    def setUp(self):
        self.prog = sample_v2_torture()
        self.m = machine_torture()
        self.r = allocate2(self.prog, self.m)

    def test_good_result_equivalent_normal_and_exceptional(self):
        eq = equivalence_check2(
            self.prog, self.m, self.r, depth_bound=30, step_budget=20_000
        )
        self.assertTrue(eq.equivalent, eq.failure)
        self.assertGreater(eq.normal_exits, 0)
        self.assertGreater(eq.exc_exits, 0)

    def test_wrong_read_view_caught(self):
        for ab in self.r.allocated_program.blocks:
            for mi in ab.instrs:
                if mi.kind == "op" and mi.reads:
                    # flip the read to a different, non-aliasing view
                    others = [
                        v.name for v in self.m.views
                        if v.name != mi.reads[0]
                        and not self.m.views_overlap(v.name, mi.reads[0])
                    ]
                    object.__setattr__(
                        mi, "reads", (others[0],) + tuple(mi.reads[1:])
                    )
                    eq = equivalence_check2(
                        self.prog, self.m, self.r, depth_bound=30,
                        step_budget=20_000,
                    )
                    self.assertFalse(eq.equivalent)
                    return
        self.fail("no mutable op found")

    def test_alias_width_write_corruption_caught(self):
        # dedicated small case: a phi swap of w-class values on the loop
        # back edge forces w-copies.  Rewriting a copy source to the
        # aliasing 64-bit x-view yields a bit-composite token (upper bits
        # uninitialised), which the bit-segment executor must reject.
        b0 = Block(
            label="b0",
            instrs=(Instr(uid=0, uses=(), defs=(1,)),
                    Instr(uid=1, uses=(), defs=(2,))),
            succs=(1,),
        )
        b1 = Block(
            label="loop",
            instrs=(Instr(uid=2, uses=(3, 4), defs=(5,)),
                    Instr(uid=3, uses=(5,), defs=(), note="sink")),
            succs=(1, 2),
            phis=(
                PhiNode(def_var=3, sources=((0, 1), (1, 4))),
                PhiNode(def_var=4, sources=((0, 2), (1, 3))),
            ),
        )
        b2 = Block(
            label="exit",
            instrs=(Instr(uid=4, uses=(3, 4), defs=()),),
            succs=(),
        )
        prog = Program(
            blocks=(b0, b1, b2),
            var_classes={1: "w", 2: "w", 3: "w", 4: "w", 5: "w"},
        )

        def _m():
            views = (
                RegView("w0", 0, 0, 32, "w"),
                RegView("x0", 0, 0, 64, "x"),
                RegView("w1", 1, 0, 32, "w"),
                RegView("x1", 1, 0, 64, "x"),
            )
            return Machine(
                views=views,
                aliases={"w0": ("x0",), "x0": ("w0",),
                         "w1": ("x1",), "x1": ("w1",)},
                class_align={"w": 4, "x": 8},
            )

        m = _m()
        r = allocate2(prog, m)
        self.assertEqual(r.status, "allocated", r.message)
        good = equivalence_check2(prog, m, r, depth_bound=20, step_budget=5000)
        self.assertTrue(good.equivalent, good.failure)

        changed = False
        for ab in r.allocated_program.blocks:
            for mi in ab.instrs:
                if mi.kind == "copy" and mi.src.startswith("w"):
                    alias = "x" + mi.src[1:]
                    object.__setattr__(mi, "src", alias)
                    changed = True
        self.assertTrue(changed)
        bad = equivalence_check2(prog, m, r, depth_bound=20, step_budget=5000)
        self.assertFalse(bad.equivalent)

    def test_drop_exceptional_edge_copy_caught(self):
        changed = False
        for ab in self.r.allocated_program.blocks:
            if ab.split_kind == "exc":
                keeps = [mi for mi in ab.instrs if mi.kind != "copy"]
                if len(keeps) != len(ab.instrs):
                    ab.instrs = keeps
                    changed = True
        self.assertTrue(changed)
        eq = equivalence_check2(
            self.prog, self.m, self.r, depth_bound=30, step_budget=20_000
        )
        self.assertFalse(eq.equivalent)

    def test_clobbered_register_read_caught(self):
        # uid 12 reads v6, which survives the call in a callee-saved view;
        # redirect that read to w0, a view the call (uid 11) destroys
        changed = False
        for ab in self.r.allocated_program.blocks:
            for mi in ab.instrs:
                if mi.kind == "op" and mi.uid == 12 and mi.reads:
                    object.__setattr__(mi, "reads", ("w0",) + tuple(mi.reads[1:]))
                    changed = True
        self.assertTrue(changed)
        eq = equivalence_check2(
            self.prog, self.m, self.r, depth_bound=30, step_budget=20_000
        )
        self.assertFalse(eq.equivalent)
