"""Extended verification: the non-toy main-scale example.

Run separately from the fast regression suite::

    python -m unittest discover -s tests_ext -v
"""

import os
import sys
import time
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))

from regalloc import allocate2, verify_allocation
from regalloc.samples2 import generate_scale_program, machine_16
from regalloc.symbolic2 import equivalence_check2


class TestScaleProgram(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prog = generate_scale_program()
        cls.machine = machine_16()
        cls.n_instr = sum(len(b.instrs) for b in cls.prog.blocks)
        t0 = time.perf_counter()
        cls.result = allocate2(cls.prog, cls.machine)
        cls.alloc_seconds = time.perf_counter() - t0

    def test_scale_bounds(self):
        self.assertGreaterEqual(len(self.prog.blocks), 100)
        self.assertGreaterEqual(self.n_instr, 1000)
        self.assertEqual(len(self.machine.views), 16)

    def test_non_degenerate_features(self):
        self.assertTrue(any(ins.is_call for b in self.prog.blocks
                            for ins in b.instrs))
        self.assertTrue(any(ins.may_throw for b in self.prog.blocks
                            for ins in b.instrs))
        self.assertTrue(any(ins.early_clobber for b in self.prog.blocks
                            for ins in b.instrs))
        self.assertTrue(any(b.exc_succs for b in self.prog.blocks))
        self.assertTrue(any(b.phis for b in self.prog.blocks))
        # more than one handler, more than one loop back edge
        self.assertGreaterEqual(
            sum(1 for b in self.prog.blocks if b.exc_succs), 2
        )

    def test_allocates_within_practical_time(self):
        self.assertEqual(self.result.status, "allocated", self.result.message)
        self.assertLess(self.alloc_seconds, 30.0)
        self.assertTrue(self.result.slots)  # genuine spilling expected

    def test_independent_verifier_passes(self):
        rep = verify_allocation(self.prog, self.machine, self.result)
        self.assertTrue(rep.ok, rep.failures[:5])
        self.assertGreater(rep.edges_checked, 100)
        self.assertGreater(rep.points_checked, 1000)

    def test_bit_level_executor_normal_and_exceptional(self):
        eq = equivalence_check2(
            self.prog, self.machine, self.result,
            depth_bound=90, step_budget=25_000,
        )
        self.assertTrue(eq.equivalent, eq.failure)
        self.assertGreater(eq.ops_checked, 1_000_000)
        self.assertGreater(eq.normal_exits, 0)
        self.assertGreater(eq.exc_exits, 0)

    def test_deterministic_repeat(self):
        r2 = allocate2(self.prog, self.machine)
        self.assertEqual(r2.assignment, self.result.assignment)
        self.assertEqual(r2.slots, self.result.slots)
        self.assertEqual(r2.iterations, self.result.iterations)

    def test_slot_alignments_and_frame(self):
        for v, slot in self.result.slots.items():
            self.assertEqual(slot.offset % slot.align, 0)
        self.assertEqual(
            self.result.frame_size, self.result.scratch_slot.end
        )
        # scratch must not overlap any variable slot
        sc = self.result.scratch_slot
        for v, slot in self.result.slots.items():
            self.assertFalse(
                slot.offset < sc.end and sc.offset < slot.end,
                f"scratch overlaps slot of v{v}",
            )


if __name__ == "__main__":
    unittest.main()
