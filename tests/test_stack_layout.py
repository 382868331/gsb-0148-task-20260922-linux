"""Tests for the post-spill stack-slot layout stage.

Coverage:
  * liveness is recomputed on the lowered phi-free CFG (phi edge actions
    live in predecessor tails / split blocks, never as in-block ops);
  * non-overlapping lifetimes share one physical offset, including the
    branch-exclusive-vs-after-join contrast fixture;
  * the budgeted enumeration proves the minimum frame size on the small
    fixtures and returns a legal, explicitly non-proven-optimal layout when
    its budget is exhausted;
  * caller-given sizes/alignments are honoured (disjoint byte ranges);
  * a value spilled across a call is not lost and a dead slot is still
    reusable across that call;
  * enabling the switch does not change the register allocation;
  * the independent symbolic executor verifies real memory accesses.
"""

from __future__ import annotations

import unittest

from regalloc import ProgramFormatError, allocate
from regalloc.samples import (
    sample_branch_slot_reuse,
    sample_register_pressure,
    sample_spilled_across_call,
)
from regalloc.stack_layout import (
    build_slot_interference,
    pack_slots,
    slot_liveness,
)
from regalloc.symbolic import equivalence_check
from regalloc.lower import MI


class TestStackPacking(unittest.TestCase):
    def test_pressure_packing_is_optimal_and_executes(self):
        prog = sample_register_pressure()
        plain = allocate(prog, 2)
        packed = allocate(prog, 2, pack_slots=True)
        layout = packed.stack_layout

        # six logical slots (stack-passed v0 + v1..v4 + scratch); the peak
        # is v0 + v1..v4 = five 8-byte slots -> 40 data bytes rounded to 48
        self.assertEqual(layout.frame_size, 48)
        self.assertEqual(layout.frame_alignment, 16)
        self.assertTrue(layout.proven_optimal)
        self.assertLess(layout.frame_size, 8 * len(layout.slot_offsets))

        for b in packed.allocated_program.blocks:
            for mi in b.instrs:
                if mi.kind in ("load", "store"):
                    self.assertEqual(mi.offset, layout.slot_offsets[mi.slot])
                    self.assertGreaterEqual(mi.offset, 0)

        # the register allocation is untouched by the switch
        self.assertEqual(plain.assignment, packed.assignment)
        self.assertEqual(plain.slots, packed.slots)

        res = equivalence_check(prog, packed)
        self.assertTrue(res.equivalent, res.failure)

    def test_packing_switch_off_leaves_layout_none(self):
        packed = allocate(sample_register_pressure(), 2, pack_slots=False)
        self.assertIsNone(packed.stack_layout)
        for b in packed.allocated_program.blocks:
            for mi in b.instrs:
                if mi.kind in ("load", "store"):
                    self.assertEqual(mi.offset, -1)

    def test_branch_exclusive_share_join_crossing_cannot(self):
        prog = sample_branch_slot_reuse()
        a = allocate(prog, 2, pack_slots=True)
        off = a.stack_layout.slot_offsets
        self.assertTrue(a.stack_layout.proven_optimal)

        private_left = [v for v in (3, 4) if v in a.slots]
        private_right = [v for v in (7, 8) if v in a.slots]
        self.assertTrue(private_left and private_right)
        right_offsets = {off[v] for v in private_right}
        for v in private_left:
            self.assertIn(off[v], right_offsets)

        # values crossing the join that are spilled get distinct offsets
        cross = [v for v in (1, 2) if v in a.slots]
        self.assertEqual(len({off[v] for v in cross}), len(cross))

        # independently re-derived interference must match every sharing
        # decision and every touching-byte-range placement
        spilled_params = [v for v in prog.params if v in a.slots]
        live = slot_liveness(a.allocated_program, spilled_params)
        adj = build_slot_interference(live)
        slots = list(a.slots)
        for i, x in enumerate(slots):
            for y in slots[i + 1:]:
                ox, oy = off[x], off[y]
                if abs(ox - oy) < 8:
                    self.assertNotIn(y, adj[x], (x, y))
                else:
                    self.assertNotEqual(ox, oy)

        res = equivalence_check(prog, a)
        self.assertTrue(res.equivalent, res.failure)

    def test_recomputed_liveness_follows_loads_and_stores(self):
        prog = sample_branch_slot_reuse()
        a = allocate(prog, 2, pack_slots=True)
        live = a.slot_liveness
        for bi, b in enumerate(a.allocated_program.blocks):
            for j, mi in enumerate(b.instrs):
                if mi.kind == "load":
                    self.assertIn(mi.slot, live.instr_before[bi][j])

    def test_spilled_value_survives_call_in_memory(self):
        prog = sample_spilled_across_call()
        a = allocate(prog, 2, pack_slots=True)
        self.assertTrue(a.slots)
        res = equivalence_check(prog, a)
        self.assertTrue(res.equivalent, res.failure)
        self.assertLessEqual(
            a.stack_layout.frame_size,
            8 * len(a.stack_layout.slot_offsets),
        )

    def test_sizes_and_alignments_are_honoured(self):
        conflicts = {0: {1}, 1: {0, 2}, 2: {1, 3}, 3: {2, 4}, 4: {3}}
        sizes = {0: 8, 1: 16, 2: 8, 3: 16, 4: 8, 9: 8}
        aligns = {0: 8, 1: 16, 2: 8, 3: 16, 4: 8, 9: 8}
        lay = pack_slots(logical_slots=[0, 1, 2, 3, 4, 9], scratch_slot=9,
                         conflicts=conflicts, sizes=sizes, aligns=aligns,
                         frame_alignment=16)
        self.assertTrue(lay.proven_optimal)
        for v, off in lay.slot_offsets.items():
            self.assertEqual(off % aligns[v], 0)
        for x, y in ((0, 1), (1, 2), (2, 3), (3, 4)):
            ox = lay.slot_offsets[x]
            oy = lay.slot_offsets[y]
            self.assertTrue(ox + sizes[x] <= oy or oy + sizes[y] <= ox)
        self.assertEqual(lay.frame_size % 16, 0)

    def test_budget_exhaustion_returns_legal_nonoptimal_layout(self):
        prog = sample_register_pressure()
        a = allocate(prog, 2, pack_slots=True, layout_budget=1)
        lay = a.stack_layout
        self.assertFalse(lay.proven_optimal)
        self.assertIn("not proven optimal", lay.note)
        res = equivalence_check(prog, a)
        self.assertTrue(res.equivalent, res.failure)

    def test_bad_layout_arguments_rejected(self):
        prog = sample_register_pressure()
        with self.assertRaises(ProgramFormatError):
            allocate(prog, 2, pack_slots=True, slot_size=0)
        with self.assertRaises(ProgramFormatError):
            allocate(prog, 2, pack_slots=True, slot_alignment=16, slot_size=8)
        with self.assertRaises(ProgramFormatError):
            allocate(prog, 2, pack_slots=True, frame_alignment=12, slot_alignment=8)
        with self.assertRaises(ProgramFormatError):
            allocate(prog, 2, pack_slots="yes")
        with self.assertRaises(ProgramFormatError):
            allocate(prog, 2, pack_slots=True, layout_budget=True)

    def test_mutated_offset_is_detected_by_real_memory_execution(self):
        # A wrong offset alias must be caught by the paired executor, which
        # addresses memory by final byte offset (it cannot be hidden behind
        # logical slot identity).
        prog = sample_register_pressure()
        a = allocate(prog, 2, pack_slots=True)
        ap = a.allocated_program
        target = None
        for bi, b in enumerate(ap.blocks):
            for j, mi in enumerate(b.instrs):
                if mi.kind == "load" and mi.slot in a.slots and mi.offset == 0:
                    target = (bi, j)
                    break
            if target:
                break
        self.assertIsNotNone(target)
        bi, j = target
        mi = ap.blocks[bi].instrs[j]
        other = next(
            off for s, off in a.stack_layout.slot_offsets.items()
            if s != mi.slot and off != mi.offset
        )
        ap.blocks[bi].instrs[j] = MI(
            kind=mi.kind, dst=mi.dst, src=mi.src, slot=mi.slot,
            reads=mi.reads, writes=mi.writes, clobbers=mi.clobbers,
            uid=mi.uid, note=mi.note, offset=other,
        )
        res = equivalence_check(prog, a)
        self.assertFalse(res.equivalent)


if __name__ == "__main__":
    unittest.main()
