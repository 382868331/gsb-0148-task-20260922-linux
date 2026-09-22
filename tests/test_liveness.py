"""Liveness tests with hand-computed expected sets.

Expected sets here were derived from the equations on paper, not by calling
the allocator.  Edge-sensitivity of phi sources is the key point.
"""

import unittest

from regalloc.liveness import liveness_plain, liveness_with_phis, predecessors
from regalloc.model import Block, Instr, Program
from regalloc.samples import sample_critical_phi, sample_loop_call


class TestPredecessors(unittest.TestCase):
    def test_basic(self):
        p = sample_critical_phi()
        self.assertEqual(predecessors(p.blocks), [[], [0], [1], [0, 1, 2]])


class TestStraightLineLiveness(unittest.TestCase):
    def test_simple_chain(self):
        # v0 param; v1 = f(v0); v2 = g(v1); use v2
        prog = Program(
            blocks=(Block(
                label="e",
                instrs=(
                    Instr(uid=0, uses=(0,), defs=(1,)),
                    Instr(uid=1, uses=(1,), defs=(2,)),
                    Instr(uid=2, uses=(2,), defs=()),
                ),
            ),),
            params=(0,),
        )
        rep = liveness_with_phis(prog)
        self.assertEqual(set(rep.live_in[0]), {0})
        self.assertEqual(set(rep.live_out[0]), set())
        # before each instruction
        self.assertEqual(set(rep.instr_before[0][0]), {0})
        self.assertEqual(set(rep.instr_before[0][1]), {1})
        self.assertEqual(set(rep.instr_before[0][2]), {2})
        self.assertEqual(set(rep.instr_after[0][0]), {1})
        self.assertEqual(set(rep.instr_after[0][1]), {2})
        self.assertEqual(set(rep.instr_after[0][2]), set())


class TestCriticalPhiLiveness(unittest.TestCase):
    def setUp(self):
        self.prog = sample_critical_phi()
        self.rep = liveness_with_phis(self.prog)

    def test_livein_merge(self):
        # phi def v4 is defined *at* block entry, so live-in (the phi-free
        # boundary value) is empty; v4 appears live before the first body
        # instruction instead.
        self.assertEqual(set(self.rep.live_in[3]), set())
        self.assertEqual(set(self.rep.instr_before[3][0]), {4})

    def test_edge_sensitive_phi_sources(self):
        # v1 lives only on edge 0->3, v2 only on 1->3, v3 only on 2->3
        self.assertEqual(set(self.rep.edge_live[(0, 3)]), {1})
        self.assertEqual(set(self.rep.edge_live[(1, 3)]), {2})
        self.assertEqual(set(self.rep.edge_live[(2, 3)]), {3})
        # the other outgoing edges do not carry the phi sources
        self.assertEqual(set(self.rep.edge_live[(0, 1)]), {1})
        self.assertEqual(set(self.rep.edge_live[(1, 2)]), {2})

    def test_phi_source_not_live_on_other_incoming_edge(self):
        # v1 must not appear on edge 1->3 or 2->3
        self.assertNotIn(1, self.rep.edge_live[(1, 3)])
        self.assertNotIn(1, self.rep.edge_live[(2, 3)])

    def test_liveout_of_predecessors(self):
        # live-out of b0 must include the union of its edge live sets
        self.assertEqual(set(self.rep.live_out[0]), {1})


class TestLoopPhiLiveness(unittest.TestCase):
    def setUp(self):
        self.prog = sample_loop_call()
        self.rep = liveness_with_phis(self.prog)

    def test_loop_phi_carried_value(self):
        # b1 defines phi v2. Edge entry->loop carries v1; edge loop->loop
        # carries v5 (plus anything live); loop body keeps v2 across call.
        self.assertEqual(set(self.rep.edge_live[(0, 1)]), {1})
        self.assertIn(5, self.rep.edge_live[(1, 1)])
        # v2 itself is the phi def, not a source on the self-edge
        self.assertNotIn(2, self.rep.edge_live[(1, 1)])

    def test_loop_exit_edge(self):
        # edge loop->exit feeds v2 to the result op
        self.assertEqual(set(self.rep.edge_live[(1, 2)]), {2})

    def test_livein_loop_block_is_phi_def(self):
        # the carried phi def v2 is killed at block entry so live-in is
        # empty; v2 is live at the first body instruction instead.
        self.assertEqual(set(self.rep.live_in[1]), set())
        self.assertEqual(set(self.rep.instr_before[1][0]), {2})


class TestPlainLivenessOnCopyIRShape(unittest.TestCase):
    def test_copy_does_not_extend_source(self):
        # a copy-like op reading v0 defining v1: v0 dies, v1 continues
        prog = Program(
            blocks=(Block(
                label="e",
                instrs=(
                    Instr(uid=0, uses=(0,), defs=(1,)),
                    Instr(uid=1, uses=(1,), defs=()),
                ),
            ),),
            params=(0,),
        )
        rep = liveness_plain(prog)
        self.assertEqual(set(rep.instr_after[0][0]), {1})


class TestFixpointDeterminism(unittest.TestCase):
    def test_repeated_runs_equal(self):
        prog = sample_loop_call()
        a = liveness_with_phis(prog)
        b = liveness_with_phis(prog)
        self.assertEqual(a.as_jsonable(), b.as_jsonable())
