"""Post-spill stack-slot layout and edge-sensitive memory liveness tests."""

import unittest

from regalloc import allocate, layout_stack_slots
from regalloc.model import Block, Instr, PhiNode, Program, ProgramFormatError
from regalloc.samples import (
    sample_branch_stack_reuse,
    sample_call_surviving,
    sample_register_pressure,
)
from regalloc.symbolic import equivalence_check


class TestDefaultCompatibility(unittest.TestCase):
    def test_switch_does_not_change_register_allocation(self):
        prog = sample_register_pressure()
        old = allocate(prog, 2)
        packed = allocate(prog, 2, pack_stack=True)
        self.assertEqual(old.assignment, packed.assignment)
        self.assertEqual(old.slots, packed.slots)
        self.assertEqual(old.history, packed.history)
        self.assertEqual(packed.stack_layout.frame_size, 48)

    def test_default_has_no_stack_layout(self):
        alloc = allocate(sample_register_pressure(), 2)
        self.assertIsNone(alloc.stack_layout)
        for block in alloc.allocated_program.blocks:
            for mi in block.instrs:
                self.assertEqual(mi.offset, -1)


class TestBranchExample(unittest.TestCase):
    def setUp(self):
        self.prog = sample_branch_stack_reuse()
        self.alloc = allocate(self.prog, 2, pack_stack=True)
        self.layout = self.alloc.stack_layout

    def test_exclusive_branch_slots_alias(self):
        self.assertEqual(self.alloc.slots, {0: 0, 3: 3, 6: 6})
        self.assertEqual(self.layout.offsets[3], self.layout.offsets[6])
        self.assertNotIn((3, 6), self.layout.interferences)

    def test_merge_survivor_cannot_alias_branch_values(self):
        self.assertIn((0, 3), self.layout.interferences)
        self.assertIn((0, 6), self.layout.interferences)
        self.assertNotEqual(self.layout.offsets[0], self.layout.offsets[3])

    def test_enumerated_minimum_frame_is_16(self):
        self.assertEqual(self.layout.frame_size, 16)
        self.assertEqual(self.layout.frame_alignment, 8)
        self.assertTrue(self.layout.optimal)
        self.assertFalse(self.layout.budget_exhausted)
        # Exhaustive search examined complete placements and found no 8-byte
        # frame: one survivor interval plus one mutually-exclusive interval.
        self.assertGreater(self.layout.placements_checked, 1)

    def test_phi_edges_are_separate_from_block_body_memory_events(self):
        edge_names = [name for name, _edge in self.layout.synthetic_blocks]
        self.assertIn("edge:1->3", edge_names)
        self.assertIn("edge:2->3", edge_names)
        body_accesses: dict[int, set[int]] = {}
        for access in self.layout.accesses:
            if access.edge is None:
                body_accesses.setdefault(access.block, set()).add(access.variable)
        self.assertEqual(body_accesses[1], {3})
        self.assertEqual(body_accesses[2], {6})
        # The survivor is carried on both predecessor edge nodes and must
        # not be treated as an ordinary in-block temporary.
        node_live = dict(zip(
            [name for name, _edge in self.layout.synthetic_blocks],
            self.layout.live_out,
        ))
        self.assertIn(0, node_live["edge:1->3"])
        self.assertIn(0, node_live["edge:2->3"])

    def test_every_load_and_store_reports_final_offset(self):
        reported = 0
        for access in self.layout.accesses:
            mi = self.alloc.allocated_program.blocks[access.block].instrs[access.index]
            self.assertEqual(mi.kind, access.kind)
            self.assertEqual(mi.slot, access.variable)
            self.assertEqual(mi.offset, access.offset)
            self.assertEqual(access.offset, self.layout.offsets[access.variable])
            reported += 1
        self.assertGreater(reported, 0)

    def test_symbolic_execution_after_real_memory_accesses(self):
        result = equivalence_check(self.prog, self.alloc)
        self.assertTrue(result.equivalent, msg=result.failure)
        self.assertEqual(result.paths_explored, 2)

    def test_phi_spill_memory_actions_are_modeled_on_predecessor_edge(self):
        # Two phi results meet crossing parameter(s) at K=2, forcing one phi
        # result to memory. Its edge store is a predecessor action, not an
        # ordinary in-block store.
        entry = Block("entry", (), (1, 2))
        left = Block(
            "left",
            tuple(Instr(i, (), (v,)) for i, v in zip(range(1, 3), (2, 3))),
            (3,),
        )
        right = Block(
            "right",
            tuple(Instr(i, (), (v,)) for i, v in zip(range(3, 5), (4, 5))),
            (3,),
        )
        merge = Block(
            "merge",
            (
                Instr(5, (6, 0), (8,)),
                Instr(6, (8, 7), (9,)),
                Instr(7, (9, 1), ()),
            ),
            (),
            (
                PhiNode(6, ((1, 2), (2, 4))),
                PhiNode(7, ((1, 3), (2, 5))),
            ),
        )
        prog = Program((entry, left, right, merge), params=(0, 1))
        alloc = allocate(prog, 2, pack_stack=True)
        edge_vars = {
            access.edge: access.variable
            for access in alloc.stack_layout.accesses
            if access.edge is not None
        }
        self.assertTrue(edge_vars)
        self.assertTrue(all(edge in ((1, 3), (2, 3)) for edge in edge_vars))
        self.assertTrue(equivalence_check(prog, alloc).equivalent)


class TestSizesAndAlignments(unittest.TestCase):
    def test_spilled_value_live_across_call_keeps_memory_lifetime(self):
        prog = sample_call_surviving()
        alloc = allocate(prog, 2, pack_stack=True)
        self.assertEqual(alloc.slots, {0: 0, 1: 1})
        v1_accesses = [a for a in alloc.stack_layout.accesses if a.variable == 1]
        self.assertEqual({a.kind for a in v1_accesses}, {"store", "load"})
        self.assertEqual({a.offset for a in v1_accesses}, {alloc.stack_layout.offsets[1]})
        self.assertTrue(equivalence_check(prog, alloc).equivalent)

    def test_disjoint_values_share_overlapping_bytes(self):
        prog = sample_branch_stack_reuse()
        alloc = allocate(prog, 2, pack_stack=True, slot_size=4, slot_alignment=4)
        self.assertEqual(alloc.stack_layout.offsets[3], alloc.stack_layout.offsets[6])
        self.assertNotEqual(alloc.stack_layout.offsets[0], alloc.stack_layout.offsets[3])
        self.assertEqual(alloc.stack_layout.frame_size, 8)
        self.assertTrue(alloc.stack_layout.optimal)

    def test_overlapping_byte_intervals_are_separated(self):
        prog = sample_register_pressure()
        alloc = allocate(prog, 2)
        sizes = {v: 8 for v in alloc.slots}
        aligns = {v: 8 for v in alloc.slots}
        layout = layout_stack_slots(alloc, sizes=sizes, alignments=aligns)
        offsets = layout.offsets
        for a, b in layout.interferences:
            if a in offsets and b in offsets:
                self.assertFalse(
                    offsets[a] < offsets[b] + sizes[b]
                    and offsets[b] < offsets[a] + sizes[a]
                )
        self.assertEqual(layout.frame_size, 48)

    def test_mixed_caller_supplied_size_and_alignment(self):
        prog = sample_register_pressure()
        alloc = allocate(prog, 2)
        sizes = {v: (4 if v % 2 else 8) for v in alloc.slots}
        aligns = {v: (4 if v % 2 else 8) for v in alloc.slots}
        layout = layout_stack_slots(
            alloc,
            sizes=sizes,
            alignments=aligns,
            frame_alignment=16,
        )
        self.assertEqual(layout.frame_alignment, 16)
        self.assertEqual(layout.frame_size % 16, 0)
        for var, offset in layout.offsets.items():
            self.assertEqual(offset % layout.alignments[var], 0)

    def test_bad_alignment_rejected(self):
        alloc = allocate(sample_register_pressure(), 2)
        with self.assertRaises(ProgramFormatError):
            layout_stack_slots(alloc, sizes=8, alignments=6)

    def test_zero_budget_returns_legal_nonoptimal_layout(self):
        alloc = allocate(sample_register_pressure(), 2)
        layout = layout_stack_slots(alloc, sizes=8, alignments=8, budget=0)
        self.assertTrue(layout.budget_exhausted)
        self.assertFalse(layout.optimal)
        for a, b in layout.interferences:
            self.assertFalse(
                layout.offsets[a] < layout.offsets[b] + 8
                and layout.offsets[b] < layout.offsets[a] + 8
            )
        self.assertGreaterEqual(layout.frame_size, 8)

if __name__ == "__main__":
    unittest.main()
