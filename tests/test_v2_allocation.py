"""End-to-end v2 allocation tests: early clobber, wide values across calls,
copy cycles with no free temporary, slot reuse/alignment, and the combined
torture CFG.

Every allocation is checked by the independent verifier and by the paired
bit-level symbolic executor; neither calls the allocator.
"""

import unittest

from regalloc import allocate2, verify_allocation
from regalloc.allocator2 import COLORING_BUDGET
from regalloc.compat import allocate_legacy_via_v2
from regalloc.machine import Machine, RegView
from regalloc.model import Block, Instr, PhiNode, Program
from regalloc.samples import sample_register_pressure, sample_call_surviving
from regalloc.samples2 import (
    machine_16,
    machine_reuse,
    machine_torture,
    sample_slot_reuse,
    sample_v2_torture,
    sample_wide_across_call,
)
from regalloc.symbolic2 import equivalence_check2


def _check(prog, m, result, *, depth=40):
    rep = verify_allocation(prog, m, result)
    assert rep.ok, rep.failures
    eq = equivalence_check2(prog, m, result, depth_bound=depth, step_budget=20_000)
    assert eq.equivalent, eq.failure
    return rep, eq


class TestCombinedTorture(unittest.TestCase):
    """The one CFG that uses ALL v2 capabilities at once."""

    def test_allocates_and_passes_independent_checks(self):
        prog = sample_v2_torture()
        m = machine_torture()
        r = allocate2(prog, m)
        self.assertEqual(r.status, "allocated", r.message)
        rep, eq = _check(prog, m, r)
        self.assertGreater(eq.normal_exits, 0)
        self.assertGreater(eq.exc_exits, 0)
        self.assertGreater(eq.edges_checked, 0)

    def test_copy_cycle_broken_by_slot_when_no_free_temp(self):
        # the back edge 1->1 carries a swap of the only two s-class values
        # while x-views and w-views occupy the other register groups; no
        # free temporary view exists, so the cycle must use the scratch
        # STACK slot (never a live alias)
        prog = sample_v2_torture()
        r = allocate2(prog, machine_torture())
        self.assertEqual(r.allocated_program.edge_breaks.get((1, 1)), ("slot",))

    def test_early_clobber_def_avoids_use_views(self):
        prog = sample_v2_torture()
        m = machine_torture()
        r = allocate2(prog, m)
        # uid 12 is the early-clobber instruction
        op = next(
            mi
            for b in r.allocated_program.blocks
            for mi in b.instrs
            if mi.kind == "op" and mi.uid == 12
        )
        self.assertTrue(op.writes)
        for w in op.writes:
            wv = m.view(w)
            for rd in op.reads:
                rv = m.view(rd)
                self.assertFalse(
                    wv.group == rv.group and wv.lo < rv.hi and rv.lo < wv.hi,
                    f"early-clobber write {w} overlaps read {rd}",
                )

    def test_wide_value_survives_call_in_callee_saved_view(self):
        prog = sample_v2_torture()
        m = machine_torture()
        r = allocate2(prog, m)
        # v3 is an x-class value live across the may_throw call uid 11
        self.assertEqual(r.assignment[3][0], "x")
        self.assertTrue(m.view(r.assignment[3]).callee_saved)

    def test_genuine_spills_with_aligned_slots(self):
        prog = sample_v2_torture()
        r = allocate2(prog, machine_torture())
        self.assertTrue(r.slots)
        for v, slot in r.slots.items():
            self.assertEqual(slot.offset % slot.align, 0)

    def test_deterministic(self):
        prog = sample_v2_torture()
        m = machine_torture()
        a = allocate2(prog, m)
        b = allocate2(prog, m)
        self.assertEqual(a.assignment, b.assignment)
        self.assertEqual(
            {v: s.offset for v, s in a.slots.items()},
            {v: s.offset for v, s in b.slots.items()},
        )
        self.assertEqual(a.iterations, b.iterations)


class TestWideAcrossCall(unittest.TestCase):
    def test_x_value_not_placed_in_caller_saved_group(self):
        prog = sample_wide_across_call()
        m = machine_torture()
        r = allocate2(prog, m)
        self.assertEqual(r.status, "allocated", r.message)
        view = r.assignment[1]
        self.assertTrue(m.view(view).callee_saved)
        self.assertNotIn(view, {"x0"})
        _check(prog, m, r)

    def test_wide_write_then_narrow_read_alias_is_tracked(self):
        # a 32-bit write into w1 must be seen by a later x1 read as a
        # composite token, not the same value (verifier accepts, the
        # symbolic executor models this bit-level semantics)
        prog = sample_wide_across_call()
        m = machine_torture()
        r = allocate2(prog, m)
        _, eq = _check(prog, m, r)
        self.assertEqual(eq.ops_checked, 3)


class TestSlotReuse(unittest.TestCase):
    def setUp(self):
        self.prog = sample_slot_reuse()
        self.m = machine_reuse()
        self.r = allocate2(self.prog, self.m)
        self.assertEqual(self.r.status, "allocated", self.r.message)

    def test_spilling_actually_happens(self):
        self.assertTrue(self.r.slots)

    def test_disjoint_live_ranges_reuse_offsets(self):
        offs: dict[int, list[int]] = {}
        for v, slot in self.r.slots.items():
            offs.setdefault(slot.offset, []).append(v)
        reused = {o: vs for o, vs in offs.items() if len(vs) > 1}
        self.assertTrue(
            reused,
            f"expected at least one reused offset, got {offs}",
        )

    def test_alignments_honoured(self):
        for v, slot in self.r.slots.items():
            self.assertEqual(slot.offset % slot.align, 0)

    def test_independent_checks(self):
        _check(self.prog, self.m, self.r)


class TestFreeTempPreferred(unittest.TestCase):
    """When a genuinely free temporary view exists, cycles use it, not a
    slot: register bit segments that can hold the cycle's width and
    overlap nothing live/pending."""

    def test_temp_break_on_a_wide_machine(self):
        # a loop back edge carries a 2-cycle swap of two phis; with four
        # w-views and only two values live on that edge, a genuinely free
        # temporary view exists, so the cycle is broken by a register
        # (never memory, never a live alias)
        views = tuple(RegView(f"w{g}", g, 0, 32, "w") for g in range(4))
        m = Machine(views=views, aliases={}, class_align={"w": 4})
        b0 = Block(
            label="b0",
            instrs=(Instr(uid=0, uses=(), defs=(1,)),
                    Instr(uid=1, uses=(), defs=(2,))),
            succs=(1,),
        )
        b1 = Block(
            label="loop",
            instrs=(
                Instr(uid=2, uses=(3, 4), defs=(5,)),
                Instr(uid=3, uses=(5,), defs=(), note="sink"),
            ),
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
        r = allocate2(prog, m)
        self.assertEqual(r.status, "allocated", r.message)
        _check(prog, m, r)
        self.assertEqual(r.allocated_program.edge_breaks.get((1, 1)), ("temp",))


class TestOutcomeStatus(unittest.TestCase):
    def test_budget_returns_explicit_unallocated_state(self):
        prog = sample_v2_torture()
        r = allocate2(prog, machine_torture(), coloring_budget=1)
        self.assertEqual(r.status, "budget_exhausted")
        self.assertIsNone(r.allocated_program)
        self.assertIsNone(r.assignment)
        self.assertIsNone(r.slots)
        rep = verify_allocation(prog, machine_torture(), r)
        self.assertTrue(rep.ok)  # honestly empty, not a fake machine program

    def test_proven_infeasible_returns_explicit_status(self):
        # four w-operands read at once with only three w-views
        prog = Program(
            blocks=(Block(
                label="e",
                instrs=(Instr(uid=0, uses=(0, 1, 2, 3), defs=()),),
            ),),
            params=(0, 1, 2, 3),
            var_classes={i: "w" for i in range(4)},
        )
        r = allocate2(prog, machine_torture())
        self.assertEqual(r.status, "infeasible")
        self.assertIsNone(r.allocated_program)
        self.assertIn("proven infeasible", r.message)

    def test_aliasing_pins_infeasible(self):
        prog = Program(
            blocks=(Block(
                label="e",
                instrs=(Instr(
                    uid=0, uses=(0, 1), defs=(),
                    fixed_views={0: "w0", 1: "x0"},
                ),),
            ),),
            params=(0, 1),
            var_classes={0: "w", 1: "x"},
        )
        r = allocate2(prog, machine_torture())
        self.assertEqual(r.status, "infeasible")
        self.assertIn("overlapping views", r.message)


class TestLegacyCompat(unittest.TestCase):
    def test_legacy_program_through_v2_pipeline(self):
        for prog in (sample_register_pressure(), sample_call_surviving()):
            r = allocate_legacy_via_v2(prog, 3)
            self.assertEqual(r.status, "allocated", r.message)
            rep = verify_allocation(r.program, r.machine, r)
            self.assertTrue(rep.ok)
            eq = equivalence_check2(
                r.program, r.machine, r, depth_bound=20, step_budget=5000
            )
            self.assertTrue(eq.equivalent, eq.failure)
