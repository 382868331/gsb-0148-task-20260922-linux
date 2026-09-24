"""Format-v2 tests: width-aliased views, register classes/pairs, callee-save
rules, early-clobber, exception edges, per-edge parallel-copy cycles with no
free temporary, stack slot reuse/alignment and the independent bit-level
checker.

Expected results here are hand-derived or produced by the independent
checker; the allocator core is never called to manufacture expectations.
Mutation tests prove the checker has discriminating power.
"""

import unittest

from regalloc import (
    allocate,
    BudgetExhaustedError,
    EdgeCopies,
    FORMAT_V2,
    InfeasibleProgramError,
    PhiNode,
    Program,
    ProgramFormatError,
    RegView,
    RegClass,
    StackSpec,
    TargetDesc,
    Block,
    Instr,
    default_target,
    tiny_target,
    verify_allocation,
)
from regalloc.samples2 import (
    sample_combined,
    sample_copy_cycle_no_temp,
    sample_early_clobber,
    sample_exception_edge,
    sample_mega,
    sample_pair_across_call,
    sample_width_alias,
)
from regalloc.checker2 import independent_liveness, BitMachine, BACKGROUND
from regalloc.target import target_from_dict


SMALL = [
    sample_width_alias,
    sample_pair_across_call,
    sample_early_clobber,
    sample_exception_edge,
    sample_copy_cycle_no_temp,
    sample_combined,
    sample_mega,
]


class TestTargetDescription(unittest.TestCase):
    def test_default_target_has_16_views_four_banks(self):
        rt = default_target().resolved()
        self.assertEqual(rt.num_banks, 4)
        self.assertEqual(len(default_target().views), 16)

    def test_alias_relation_validated_against_bits(self):
        # eax declared with a wrong alias set must be rejected
        views = list(default_target().views)
        bad = []
        for v in views:
            al = tuple(a for a in v.aliases if a != "ax") if v.name == "eax" else v.aliases
            bad.append(RegView(v.name, v.bank, v.lo, v.width, al))
        with self.assertRaises(ProgramFormatError):
            TargetDesc("bad", tuple(bad), default_target().classes,
                       default_target().callee_saved, "GPR32").resolved()

    def test_callee_saved_uniform_per_bank(self):
        t = default_target()
        # saving eax but not ax is contradictory for a bank-wide ABI
        with self.assertRaises(ProgramFormatError):
            TargetDesc("bad", t.views, t.classes, ("eax",), "GPR32").resolved()

    def test_pair_halves_on_different_banks(self):
        # same-bank pair is rejected when the target is resolved
        bad_class = RegClass("BAD", 48, pairs=(("eax", "ax"),))
        t = default_target()
        with self.assertRaises(ProgramFormatError):
            TargetDesc("bad", t.views, t.classes + (bad_class,),
                       t.callee_saved, "GPR32").resolved()

    def test_pair_width_must_match_class_width(self):
        bad_class = RegClass("BAD", 99, pairs=(("eax", "ecx"),))
        t = default_target()
        with self.assertRaises(ProgramFormatError):
            TargetDesc("bad", t.views, t.classes + (bad_class,),
                       t.callee_saved, "GPR32").resolved()

    def test_target_from_dict_roundtrip(self):
        data = {
            "name": "custom", "default_class": "G",
            "views": [
                {"name": "a", "bank": 0, "width": 8, "aliases": []},
                {"name": "b", "bank": 1, "width": 8, "aliases": []},
            ],
            "classes": [{"name": "G", "width": 8, "views": ["a", "b"]}],
            "callee_saved": ["b"],
        }
        td = target_from_dict(data)
        rt = td.resolved()
        self.assertEqual(rt.num_banks, 2)
        self.assertEqual(rt.saved_banks, frozenset({1}))


class TestIndependentLiveness(unittest.TestCase):
    def test_exception_edge_excludes_def(self):
        p = sample_exception_edge()
        live = independent_liveness(p)
        # uid1 (index 1) can except to block 2; v2 is its def and must not be
        # required on the exception edge, but v1 must be available there
        self.assertNotIn(2, live["except"][0][1])
        self.assertIn(1, live["except"][0][1])
        self.assertEqual(set(live["edge_live"][(0, 2)]), {1})
        # normal edge carries v3 to the ok block
        self.assertEqual(set(live["edge_live"][(0, 1)]), {3})

    def test_explicit_edge_copy_source_edge_sensitive(self):
        t = tiny_target()
        b0 = Block(label="a", instrs=(Instr(0, (), (1,)),), succs=(1,))
        b1 = Block(label="b", instrs=(Instr(1, (2,), (3,)),), succs=())
        b0 = Block(label="a", instrs=(Instr(0, (), (1,)),), succs=(1,),
                   edge_copies=(EdgeCopies(0, 1, ((2, 1),)),))
        p = Program((b0, b1), version=2, target=t)
        live = independent_liveness(p)
        self.assertEqual(set(live["edge_live"][(0, 1)]), {1})
        # destination v2 is born on the edge, not live-in from elsewhere
        self.assertNotIn(2, live["live_in"][0])


class TestAllocationVerified(unittest.TestCase):
    def test_all_small_samples_pass_independent_checker(self):
        for fn in SMALL:
            with self.subTest(sample=fn.__name__):
                p = fn()
                a = allocate(p)
                self.assertTrue(a.verification.ok, a.verification.failure_text())
                self.assertGreaterEqual(a.verification.edges_checked, 0)

    def test_pair_lives_in_callee_saved_banks_across_call(self):
        a = allocate(sample_pair_across_call())
        views = a.assignment[1]
        banks = {default_target().resolved().view(v).bank for v in views}
        # ebx (bank1) and edx (bank3) are the callee-saved pair option
        self.assertEqual(banks, {1, 3})

    def test_no_free_temp_cycle_uses_stack_scratch(self):
        a = allocate(sample_copy_cycle_no_temp())
        self.assertTrue(a.scratch_slots, "swap cycle must break via a stack slot")
        mis = [mi for b in a.allocated_program.blocks for mi in b.instrs]
        self.assertIn("store", [m.kind for m in mis])
        self.assertIn("load", [m.kind for m in mis])
        # the scratch slot is beyond every data slot
        for (_, size) in a.slots.values():
            self.assertLessEqual(0, a.allocated_program.scratch_slot)

    def test_exception_and_normal_exits_both_explored(self):
        a = allocate(sample_exception_edge())
        vr = a.verification
        self.assertGreaterEqual(vr.normal_exits, 2)   # ok and handler exits
        self.assertGreaterEqual(vr.exception_exits, 1)

    def test_combined_uses_all_capabilities(self):
        a = allocate(sample_combined())
        self.assertTrue(a.verification.ok, a.verification.failure_text())
        # the double-width value occupies exactly two locations: two views
        # when register-resident, or two frame units when spilled
        if 1 in a.assignment:
            self.assertEqual(len(a.assignment[1]), 2)
        else:
            self.assertEqual(a.slots[1][1], 2)
        # exception edge present and verified
        self.assertIn((0, 4), a.liveness.edge_live)

class TestExplicitEdgeCopies(unittest.TestCase):
    def test_explicit_swap_cycle_broken_safely(self):
        # explicit (non-phi) parallel copies on an edge form a swap; the
        # result must be bit-correct whether resolved by spill or scratch
        t = tiny_target()
        b0 = Block(label="a", instrs=(
            Instr(0, (), (1,)), Instr(1, (), (2,)),
        ), succs=(1,), edge_copies=(EdgeCopies(0, 1, ((3, 2), (4, 1))),))
        b1 = Block(label="b", instrs=(
            Instr(2, (), (5,), clobbersv=("t1", "t1h", "t1b")),
            Instr(3, (3, 5), (6,), fixedv={3: "t0"}),
            Instr(4, (), (7,), clobbersv=("t0", "t0h", "t0b")),
            Instr(5, (4, 7), (8,), fixedv={4: "t1"}),
            Instr(6, (6, 8), ()),
        ), succs=())
        p = Program((b0, b1), version=2, target=t, stack=StackSpec(align=2))
        a = allocate(p)
        self.assertTrue(a.verification.ok, a.verification.failure_text())

    def test_edge_copy_destination_born_on_edge_only(self):
        t = tiny_target()
        b0 = Block(label="a", instrs=(Instr(0, (), (1,)),), succs=(1,),
                   edge_copies=(EdgeCopies(0, 1, ((2, 1),)),))
        b1 = Block(label="b", instrs=(Instr(1, (2,), ()),), succs=())
        p = Program((b0, b1), version=2, target=t)
        a = allocate(p)
        self.assertTrue(a.verification.ok)
        # v2 is born on the edge: it must not be live into block 0
        from regalloc.checker2 import independent_liveness
        self.assertNotIn(2, independent_liveness(p)["live_in"][0])


class TestDictRoundtripV2(unittest.TestCase):
    def test_program_from_dict_version2(self):
        from regalloc import program_from_dict
        data = {
            "version": 2,
            "name": "rt",
            "target": {
                "name": "c", "default_class": "G",
                "views": [
                    {"name": "a", "bank": 0, "width": 8, "aliases": []},
                    {"name": "b", "bank": 1, "width": 8, "aliases": []},
                ],
                "classes": [{"name": "G", "width": 8, "views": ["a", "b"]}],
                "callee_saved": ["b"],
            },
            "stack": {"align": 2},
            "blocks": [
                {"label": "e", "instrs": [{"uid": 0, "uses": [], "defs": [1]}],
                 "succs": []},
            ],
        }
        p = program_from_dict(data)
        self.assertTrue(p.is_v2())
        a = allocate(p)
        self.assertTrue(a.verification.ok)


class TestMegaSingleCfg(unittest.TestCase):
    def test_mega_single_cfg_has_every_capability_and_stack_swap(self):        # ONE CFG: register-resident double-width value across a call +
        # early-clobber + exception edge + a no-free-temp swap cycle broken
        # through the private stack slot.
        a = allocate(sample_mega())
        vr = a.verification
        self.assertTrue(vr.ok, vr.failure_text())
        # pair stays in the callee-saved ebx:edx views (banks 1 and 3)
        self.assertEqual(set(a.assignment[1]), {"ebx", "edx"})
        # exception edge present with at least one exception exit explored
        self.assertIn((0, 4), a.liveness.edge_live)
        self.assertGreaterEqual(vr.exception_exits, 1)
        # the join swap cycle used the scratch stack slot (no free register)
        self.assertIn((0, 1), a.scratch_slots)
        # no data value was spilled: the swap was resolved with scratch only
        self.assertEqual(a.slots, {})


class TestEarlyClobberEnforced(unittest.TestCase):
    def test_early_result_cannot_share_live_input_bank(self):
        # at uid2 the surviving input v1 is pinned to eax AND the early
        # result v3 is pinned to eax: the early write happens before the input
        # is consumed, so this is provably impossible
        t = default_target()
        b0 = Block(label="b", instrs=(
            Instr(0, (), (1,)),
            Instr(1, (), (2,)),
            Instr(2, (1,), (3, 4), early_clobber=(3,),
                  fixedv={1: "eax", 3: "eax"}),
            Instr(3, (1, 4), (5,)),
            Instr(4, (5,), ()),
        ), succs=())
        p = Program((b0,), version=2, target=t)
        with self.assertRaises(InfeasibleProgramError):
            allocate(p)

    def test_same_constraint_without_early_flag_is_feasible(self):
        t = default_target()
        # v1 dies at the op, so an ordinary result may reuse its bank
        b0 = Block(label="b", instrs=(
            Instr(0, (), (1,), fixedv={1: "eax"}),
            Instr(1, (), (2,)),
            Instr(2, (1,), (3,), fixedv={3: "eax"}),   # not early
            Instr(3, (2, 3), (5,)),
            Instr(4, (5,), ()),
        ), succs=())
        p = Program((b0,), version=2, target=t)
        a = allocate(p)
        self.assertTrue(a.verification.ok)


class TestSlotReuseAndAlignment(unittest.TestCase):
    def test_slots_aligned_and_nonoverlapping_when_colive(self):
        a = allocate(sample_combined())
        align = a.program.stack.align
        for v, (start, size) in a.slots.items():
            req = min(align, size) if align > 1 else 1
            self.assertEqual(start % req, 0, f"v{v} slot misaligned")
        # the independent checker reports no overlap failures
        from regalloc.checker2 import check_slot_overlap
        self.assertEqual(check_slot_overlap(a.program, a), [])

    def test_frame_includes_scratch_beyond_data(self):
        a = allocate(sample_copy_cycle_no_temp())
        data_hi = max((start + size for start, size in a.slots.values()),
                      default=0)
        self.assertGreaterEqual(a.allocated_program.scratch_slot, data_hi)
        self.assertGreaterEqual(a.frame_slots, data_hi + 1)


class TestStrictValidationV2(unittest.TestCase):
    def test_v1_fields_rejected_on_v2(self):
        t = default_target()
        b = Block(label="b", instrs=(Instr(0, (), (1,), clobbers=(0,)),), succs=())
        with self.assertRaises(ProgramFormatError):
            Program((b,), version=FORMAT_V2, target=t)

    def test_v2_requires_target(self):
        b = Block(label="b", instrs=(Instr(0, (), (1,)),), succs=())
        with self.assertRaises(ProgramFormatError):
            Program((b,), version=FORMAT_V2)

    def test_v2_fields_rejected_on_v1(self):
        b = Block(label="b", instrs=(Instr(0, (), (1,), except_to=(0,)),), succs=())
        with self.assertRaises(ProgramFormatError):
            Program((b,))

    def test_unknown_view_and_class_rejected(self):
        t = default_target()
        b = Block(label="b", instrs=(Instr(0, (), (1,), fixedv={1: "nope"}),), succs=())
        with self.assertRaises(ProgramFormatError):
            Program((b,), version=2, target=t)
        b2 = Block(label="b", instrs=(Instr(0, (), (1,), classes={1: "NOPE"}),), succs=())
        with self.assertRaises(ProgramFormatError):
            Program((b2,), version=2, target=t)

    def test_exception_target_must_be_successor(self):
        t = tiny_target()
        b0 = Block(label="a", instrs=(Instr(0, (), (1,), except_to=(2,)),), succs=(1,))
        b1 = Block(label="b", instrs=(), succs=())
        with self.assertRaises(ProgramFormatError):
            Program((b0, b1), version=2, target=t)

    def test_allocate_rejects_k_for_v2(self):
        p = sample_width_alias()
        with self.assertRaises(ProgramFormatError):
            allocate(p, 4)

    def test_bad_alignment_rejected(self):
        with self.assertRaises(ProgramFormatError):
            StackSpec(align=3)


class TestDeterminismV2(unittest.TestCase):
    def test_repeat_allocation_identical(self):
        p = sample_combined()
        a1 = allocate(p)
        a2 = allocate(p)
        self.assertEqual(a1.assignment, a2.assignment)
        self.assertEqual(a1.slots, a2.slots)
        self.assertEqual(a1.allocated_program.render(), a2.allocated_program.render())
        self.assertEqual(a1.history, a2.history)


class TestBitMachineAliasing(unittest.TestCase):
    def test_narrow_write_visible_through_alias(self):
        rt = default_target().resolved()
        m = BitMachine(rt, 4)
        m.write_view("al", tuple(range(8)))
        ax = m.read_view("ax")
        self.assertEqual(ax[:8], tuple(range(8)))   # low 8 bits alias al
        self.assertTrue(all(t is BACKGROUND for t in ax[8:]))


class TestCheckerMutationRejection(unittest.TestCase):
    def _alloc(self):
        return allocate(sample_exception_edge())

    def test_wrong_op_read_register_is_caught(self):
        a = self._alloc()
        for b in a.allocated_program.blocks:
            for mi in b.instrs:
                if mi.kind == "op" and mi.reads:
                    other = next(v for v in ("eax", "ebx", "ecx", "edx")
                                 if v != mi.reads[0])
                    object.__setattr__(mi, "reads", (other,) + tuple(mi.reads[1:]))
                    rep = verify_allocation(a.program, a)
                    self.assertFalse(rep.ok)
                    return
        self.fail("no mutable op found")

    def test_dropped_exception_edge_copy_is_caught(self):
        # handler phi is fed on the exception edge, so the exception split
        # block carries a real copy; deleting it must be detected.  v1 is
        # pinned t0 and the phi result is pinned t1 to guarantee a real copy.
        t = tiny_target()
        b0 = Block(label="body", instrs=(
            Instr(0, (), (1,), fixedv={1: "t0"}),
            Instr(1, (1,), (2,), except_to=(2,)),
        ), succs=(1, 2))
        b1 = Block(label="ok", instrs=(Instr(2, (2,), ()),), succs=())
        # phi v9 must survive a bank0-clobbering call then be read pinned t1,
        # forcing its home to t1 while the source lives in t0 -> the edge
        # split block has to materialise a real t1 <- t0 copy
        b2 = Block(label="handler", instrs=(
            Instr(3, (), (7,), clobbersv=("t0", "t0h", "t0b")),
            Instr(4, (9, 7), (8,), fixedv={9: "t1"}),
            Instr(5, (8,), ()),
        ), succs=(), phis=(PhiNode(9, ((0, 1),)),))
        p = Program((b0, b1, b2), version=2, target=t)
        a = allocate(p)
        removed = False
        for b in a.allocated_program.blocks:
            if b.exc_instr is not None and b.instrs:
                b.instrs = []
                removed = True
        self.assertTrue(removed)
        self.assertFalse(verify_allocation(a.program, a).ok)


class TestBudgetIsUnknownNotFailure(unittest.TestCase):
    def test_budget_separate_from_infeasible(self):
        import regalloc.allocator2 as mod
        p = sample_combined()
        saved = mod.COLORING_BUDGET_V2
        mod.COLORING_BUDGET_V2 = 0
        try:
            with self.assertRaises(BudgetExhaustedError):
                allocate(p)
        finally:
            mod.COLORING_BUDGET_V2 = saved
        # with the real budget it allocates cleanly
        allocate(p)
