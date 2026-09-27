"""Rematerialisation (recompute-pure-value) tests.

Covers:
* a marked cheap pure value is recomputed at its use point instead of being
  written to memory (it lands in ``Allocation.rematerialized``, not slots);
* the output strictly separates memory-spilled values from recomputed ones;
* recomputation on phi-source edges (including a critical-edge split);
* a marker that is structurally valid but intrinsically needs more than K
  simultaneous registers at a use point is declined (the allocator spills
  instead) — recomputation never invents an unallocatable temporary;
* non-pure markers (clobbers, multi-result, fixed registers, bad uid/cost)
  are rejected with ProgramFormatError;
* the independent paired symbolic executor still proves equivalence.
"""

import unittest

from regalloc import allocate, ProgramFormatError
from regalloc.allocator import analyze_remat
from regalloc.model import Block, Instr, PhiNode, Program
from regalloc.samples import (
    sample_remat_phi,
    sample_remat_pressure,
    sample_register_pressure,
)
from regalloc.symbolic import equivalence_check


class TestRematSelection(unittest.TestCase):
    def test_constant_is_rematerialised_not_spilled(self):
        prog = sample_remat_pressure()
        plain = allocate(prog, 2)
        self.assertIn(1, plain.slots)  # without a marker v1 is spilled

        alloc = allocate(prog, 2, remat={0: 1})
        self.assertNotIn(1, alloc.slots)
        self.assertIn(1, alloc.rematerialized)
        self.assertEqual(alloc.rematerialized[1], 1.0)
        # every variable is assigned / spilled / rematerialised exactly once
        all_vars = set(range(prog.var_count()))
        self.assertEqual(
            set(alloc.assignment) | set(alloc.slots) | set(alloc.rematerialized),
            all_vars,
        )
        self.assertTrue(
            set(alloc.rematerialized).isdisjoint(alloc.slots)
        )
        # the defining op uid0 is absent from the body (elided) and a
        # rematerialisation appears immediately before its late use uid8
        mis = [mi for b in alloc.allocated_program.blocks for mi in b.instrs]
        self.assertEqual([mi.uid for mi in mis if mi.kind == "op" and mi.uid == 0], [])
        remats = [mi for mi in mis if mi.kind == "remat"]
        self.assertTrue(remats)
        self.assertTrue(all(mi.uid == 0 for mi in remats))
        res = equivalence_check(prog, alloc)
        self.assertTrue(res.equivalent, msg=res.failure)

    def test_history_records_spill_and_remat_sets(self):
        prog = sample_remat_pressure()
        alloc = allocate(prog, 2, remat={0: 1})
        joined = "\n".join("|".join(h) for h in alloc.history)
        self.assertIn("remat:1", joined)
        self.assertIn("spill:", joined)
        # later entries never lose an earlier decision (monotonic)
        remat_seen: set[int] = set()
        for h in alloc.history:
            cur = {int(x) for x in h[1].split(":", 1)[1].split(",") if x}
            self.assertTrue(remat_seen <= cur)
            remat_seen = cur

    def test_no_pressure_means_no_remat(self):
        prog = sample_remat_pressure()
        alloc = allocate(prog, 8, remat={0: 1})
        self.assertEqual(alloc.slots, {})
        self.assertEqual(alloc.rematerialized, {})

    def test_deterministic(self):
        prog = sample_remat_pressure()
        a = allocate(prog, 2, remat={0: 1})
        b = allocate(prog, 2, remat={0: 1})
        self.assertEqual(a.rematerialized, b.rematerialized)
        self.assertEqual(a.allocated_program.render(), b.allocated_program.render())

    def test_float_cost_accepted_and_recorded(self):
        alloc = allocate(sample_remat_pressure(), 2, remat={0: 0.25})
        self.assertEqual(alloc.rematerialized[1], 0.25)


class TestRematPhi(unittest.TestCase):
    def test_remat_on_phi_edges_including_critical_split(self):
        prog = sample_remat_phi()
        plain = allocate(prog, 2)
        self.assertIn(1, plain.slots)
        alloc = allocate(prog, 2, remat={0: 1})
        self.assertIn(1, alloc.rematerialized)
        self.assertNotIn(1, alloc.slots)
        # recomputations live on all three predecessor edge actions,
        # including the critical-edge split block 0->3
        split = [b for b in alloc.allocated_program.blocks if b.split_edge == (0, 3)]
        self.assertEqual(len(split), 1)
        self.assertTrue(any(mi.kind == "remat" for mi in split[0].instrs))
        res = equivalence_check(prog, alloc)
        self.assertTrue(res.equivalent, msg=res.failure)


class TestRematSafety(unittest.TestCase):
    def test_remat_declined_when_use_point_exceeds_k(self):
        # v3 = pure(v2); at its only use (uid4 reads v2,v3) v9 is also live,
        # so recomputing v3 would need 3 simultaneous registers at K=2:
        # the marker must be declined and v3 spilled instead.
        b0 = Block(label="e", instrs=(Instr(0, uses=(0,), defs=(1,)),), succs=(1,))
        b1 = Block(label="l", instrs=(
            Instr(1, uses=(2,), defs=(3,)),
            Instr(2, uses=(), defs=(8,)),
            Instr(3, uses=(8,), defs=(9,)),
            Instr(4, uses=(3, 2), defs=(10,)),
            Instr(5, uses=(9, 10), defs=(5,)),
        ), succs=(1, 2), phis=(PhiNode(2, ((0, 1), (1, 5))),))
        b2 = Block(label="x", instrs=(Instr(6, uses=(2, 5), defs=(6,)),
                                      Instr(7, uses=(6,), defs=())), succs=())
        prog = Program(blocks=(b0, b1, b2), params=(0,))
        info = analyze_remat(prog, {1: 1})
        # structurally legal but the allocator must not pick it
        self.assertIn(3, info.eligible)
        alloc = allocate(prog, 2, remat={1: 1})
        self.assertEqual(alloc.rematerialized, {})
        self.assertTrue(alloc.slots)
        res = equivalence_check(prog, alloc)
        self.assertTrue(res.equivalent, msg=res.failure)

    def test_dead_operand_at_use_marks_ineligible(self):
        # v4 = pure(v3), but v3 dies before v4's only use and cannot itself
        # be rematerialised (its def is a call): the marker is legal but not
        # eligible, so it is never chosen.
        b = Block(label="e", instrs=(
            Instr(0, uses=(), defs=(3,), clobbers=(0,)),
            Instr(1, uses=(3,), defs=(4,)),
            Instr(2, uses=(4,), defs=(5,)),
            Instr(3, uses=(5,), defs=()),
        ), succs=())
        prog = Program(blocks=(b,))
        info = analyze_remat(prog, {1: 1})
        self.assertNotIn(4, info.eligible)
        alloc = allocate(prog, 2, remat={1: 1})
        self.assertNotIn(4, alloc.rematerialized)


class TestRematValidation(unittest.TestCase):
    def _prog(self):
        b = Block(label="e", instrs=(Instr(0, uses=(), defs=(1,)),), succs=())
        return Program(blocks=(b,))

    def test_bad_mapping_type(self):
        with self.assertRaises(ProgramFormatError):
            allocate(self._prog(), 2, remat=[(0, 1)])

    def test_unknown_uid(self):
        with self.assertRaises(ProgramFormatError) as cm:
            allocate(self._prog(), 2, remat={99: 1})
        self.assertIn("remat[99]", str(cm.exception))

    def test_bool_uid_rejected(self):
        with self.assertRaises(ProgramFormatError):
            allocate(self._prog(), 2, remat={True: 1})

    def test_cost_must_be_positive_finite(self):
        for bad in (0, -1, float("nan"), float("inf"), True, "x"):
            with self.assertRaises(ProgramFormatError):
                allocate(self._prog(), 2, remat={0: bad})

    def test_clobbering_instruction_is_not_pure(self):
        b = Block(label="e", instrs=(Instr(0, uses=(), defs=(1,), clobbers=(0,)),),
                  succs=())
        with self.assertRaises(ProgramFormatError) as cm:
            allocate(Program(blocks=(b,)), 2, remat={0: 1})
        self.assertIn("not pure", str(cm.exception))

    def test_multi_result_rejected(self):
        b = Block(label="e", instrs=(Instr(0, uses=(), defs=(1, 2),),), succs=())
        with self.assertRaises(ProgramFormatError) as cm:
            allocate(Program(blocks=(b,)), 2, remat={0: 1})
        self.assertIn("single pure result", str(cm.exception))

    def test_fixed_register_rejected(self):
        b = Block(label="e", instrs=(Instr(0, uses=(), defs=(1,), fixed={1: 0}),),
                  succs=())
        with self.assertRaises(ProgramFormatError):
            allocate(Program(blocks=(b,)), 2, remat={0: 1})

    def test_parameter_and_phi_result_rejected(self):
        bp = Block(label="e", instrs=(Instr(0, uses=(0,), defs=()),), succs=())
        with self.assertRaises(ProgramFormatError):
            allocate(Program(blocks=(bp,), params=(0,)), 2, remat={0: 1})
        # a single-result instruction whose result variable is also a phi
        # result is rejected (a rematerialised value must have one pure
        # static definition):
        b0 = Block(label="b0", instrs=(Instr(0, uses=(), defs=(1,)),), succs=(1,))
        b1 = Block(label="b1",
                   instrs=(Instr(1, uses=(), defs=(2,)),),
                   succs=(),
                   phis=(PhiNode(2, ((0, 1),)),))
        with self.assertRaises(ProgramFormatError):
            allocate(Program(blocks=(b0, b1)), 2, remat={1: 1})


class TestRematAllSamplesEquivalent(unittest.TestCase):
    def test_remat_branch_backtracks_to_spill(self):
        # A loop where eagerly rematerialising a chain (v3/v4/v5) drives the
        # short reload temps into an uncolorable graph, while plain spilling
        # succeeds.  The search must backtrack the remat decision and spill;
        # it must never raise and the result must be equivalent.
        b0 = Block(label="e", instrs=(Instr(0, (0,), (1,)),), succs=(1,))
        b1 = Block(label="l", instrs=(
            Instr(1, (2,), (3,)), Instr(2, (3,), (4,)), Instr(3, (3, 4), (5,)),
            Instr(4, (3, 4), (6,)), Instr(5, (4, 3), (7,)), Instr(6, (7, 6), (8,)),
            Instr(7, (2, 8), (10,)),
        ), succs=(1, 2), phis=(PhiNode(2, ((0, 1), (1, 10))),))
        b2 = Block(label="x", instrs=(Instr(8, (2,), (11,)),
                                      Instr(9, (11,), ())), succs=())
        prog = Program(blocks=(b0, b1, b2), params=(0,))
        markers = {ins.uid: 1 for b in prog.blocks for ins in b.instrs
                   if len(ins.defs) == 1 and not ins.clobbers and not ins.fixed}
        alloc = allocate(prog, 2, remat=markers)
        self.assertTrue(alloc.slots)
        res = equivalence_check(prog, alloc)
        self.assertTrue(res.equivalent, msg=res.failure)
        # and the backtracking search is deterministic
        again = allocate(prog, 2, remat=markers)
        self.assertEqual(again.slots, alloc.slots)
        self.assertEqual(again.allocated_program.render(),
                         alloc.allocated_program.render())

    def test_all_programs_equivalent_with_and_without_markers(self):
        for prog, markers in (
            (sample_remat_pressure(), {0: 1}),
            (sample_remat_phi(), {0: 1}),
            (sample_register_pressure(), {}),
        ):
            for k in (2, 3, 4):
                alloc = allocate(prog, k, remat=markers)
                res = equivalence_check(prog, alloc, depth_bound=40, step_budget=8000)
                self.assertTrue(res.equivalent, msg=f"{prog.name} k={k}: {res.failure}")
