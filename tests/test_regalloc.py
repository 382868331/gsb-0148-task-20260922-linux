"""Core allocation tests.

Expected behavior is established by the independent verifier in
``tests/verifier.py`` (its own liveness + symbolic token machine), not by
calling the allocator's internals.
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from regalloc import (  # noqa: E402
    AllocationBudgetExhausted,
    Block,
    Instruction,
    InvalidArgumentError,
    InvalidProgramError,
    Phi,
    Program,
    UnsatisfiableProgram,
    allocate,
)
from regalloc import examples  # noqa: E402
from tests.verifier import (  # noqa: E402
    independent_liveness,
    reaching_tokens,
    verify_allocation,
)
from tests.proggen import generate_program  # noqa: E402


class _Verified:
    def assertVerified(self, program, K):
        result = allocate(program, K)
        verify_allocation(program, result)  # raises on any discrepancy
        return result


class TestCriticalEdgePhi(_Verified, unittest.TestCase):
    def test_diamond_with_critical_edge_split(self):
        prog = examples.critical_edge_phi_program()
        for K in (2, 3, 4):
            with self.subTest(K=K):
                res = self.assertVerified(prog, K)
                ids = [mb.id for mb in res.program.blocks]
                # branch -> join is critical: a split block must exist.
                self.assertIn("..split.branch->join", ids)

    def test_phi_tokens_union(self):
        prog = examples.critical_edge_phi_program()
        R = reaching_tokens(prog)
        self.assertEqual(R["m"], frozenset({"c", "v1"}))

    def test_loop_backedge_phi(self):
        prog = examples.loop_phi_program()
        for K in (2, 3, 4):
            with self.subTest(K=K):
                res = self.assertVerified(prog, K)
                R = reaching_tokens(prog)
                self.assertEqual(R["i"], frozenset({"z0", "i2"}))
                # The loop body -> head edge is not critical (body has one
                # successor), so no split block is needed for it.
                ids = {mb.id for mb in res.program.blocks}
                self.assertNotIn("..split.body->head", ids)


class TestCallClobber(_Verified, unittest.TestCase):
    def test_live_across_call_kept_out_of_clobbered(self):
        prog = examples.call_pressure_program()
        res = self.assertVerified(prog, 3)
        # t1, t2, b survive the call; r0/r1 are clobbered -> any assigned
        # survivor must use r2.
        call_index = 2
        _lin, _lout, points, _edges, _ = independent_liveness(prog)
        before, after = points[("entry", call_index)]
        for v in sorted(set(before) & set(after)):
            reg = res.assignment[v]
            if reg is not None:
                self.assertNotIn(reg, (0, 1), f"{v} survives call but placed in clobbered r{reg}")

    def test_call_forces_spill_at_K2(self):
        prog = examples.call_pressure_program()
        res = self.assertVerified(prog, 2)
        self.assertTrue(res.stack_slots)
        self.assertGreaterEqual(res.iterations, 2)
        spilled = [v for v, r in res.assignment.items() if r is None]
        self.assertTrue(spilled)
        # slot indices assigned by sorted variable id order
        self.assertEqual([res.stack_slots[v] for v in sorted(res.stack_slots)],
                         list(range(len(res.stack_slots))))


class TestFixedRegisters(_Verified, unittest.TestCase):
    def test_fixed_use_and_def_honored(self):
        prog = examples.fixed_register_program()
        res = self.assertVerified(prog, 2)
        emitted = {mi.opcode: mi for mb in res.program.blocks for mi in mb.instrs}
        sys_i = emitted["syscall"]
        self.assertEqual(sys_i.reads[0].i, 1)
        self.assertEqual(sys_i.writes[0].i, 0)

    def test_fixed_register_collision_is_unsat(self):
        prog = examples.unsat_fixed_collision_program()
        with self.assertRaises(UnsatisfiableProgram):
            allocate(prog, 2)

    def test_fixed_register_into_live_clobber_is_unsat(self):
        prog = examples.unsat_fixed_clobber_program()
        with self.assertRaises(UnsatisfiableProgram):
            allocate(prog, 2)

    def test_point_pressure_over_K_is_unsat(self):
        # One instruction needs three distinct operands at once with K=2:
        # reloads live in registers at the use, so the point clique cannot be
        # broken by spilling -> genuinely impossible, not a budget give-up.
        prog = Program(
            entry="entry",
            params=("a", "b", "c"),
            blocks=(
                Block(
                    "entry",
                    instrs=(Instruction("tri", ("a", "b", "c"), ("x",)),
                            Instruction("ret", ("x",), ())),
                    succs=(),
                ),
            ),
        )
        with self.assertRaises(UnsatisfiableProgram):
            allocate(prog, 2)
        # K=3 is enough and must verify.
        self.assertVerified(prog, 3)

    def test_unsat_is_distinct_from_budget(self):
        prog = examples.unsat_fixed_collision_program()
        with self.assertRaises(UnsatisfiableProgram):
            try:
                allocate(prog, 2)
            except AllocationBudgetExhausted:  # pragma: no cover - wrong class
                self.fail("budget-exhausted raised for a provably impossible program")


class TestRegisterPressure(_Verified, unittest.TestCase):
    def test_high_pressure_spills(self):
        prog = examples.high_pressure_program(7)
        res = self.assertVerified(prog, 2)
        self.assertTrue(res.stack_slots)

    def test_all_K_bounds(self):
        prog = examples.high_pressure_program(7)
        for K in range(2, 9):
            with self.subTest(K=K):
                res = self.assertVerified(prog, K)
                # 7 simultaneous values need at least 7-K spills.
                self.assertGreaterEqual(len(res.stack_slots), max(0, 7 - K))

    def test_enough_registers_needs_no_spill(self):
        prog = examples.high_pressure_program(3)
        res = self.assertVerified(prog, 8)
        self.assertEqual(res.stack_slots, {})
        self.assertEqual(res.iterations, 1)


class TestAssignmentInvariants(_Verified, unittest.TestCase):
    def test_no_two_live_values_share_register(self):
        # The verifier already asserts this; double-check directly across all
        # example programs using the INDEPENDENT live sets.
        for factory in (
            examples.critical_edge_phi_program,
            examples.call_pressure_program,
            examples.fixed_register_program,
            examples.loop_phi_program,
            examples.high_pressure_program,
        ):
            prog = factory()
            for K in (2, 4, 8):
                res = allocate(prog, K)
                _li, _lo, points, edges, _ = independent_liveness(prog)
                live_sets = list(edges.values())
                for before, after in points.values():
                    live_sets.append(before)
                    live_sets.append(after)
                for live in live_sets:
                    placed = {}
                    for v in live:
                        r = res.assignment.get(v)
                        if r is None:
                            continue
                        self.assertNotIn(r, placed)
                        placed[r] = v

    def test_determinism(self):
        prog = examples.call_pressure_program()
        r1 = allocate(prog, 2)
        r2 = allocate(prog, 2)
        self.assertEqual(r1.assignment, r2.assignment)
        self.assertEqual(r1.stack_slots, r2.stack_slots)
        code1 = [(mb.id, tuple((m.opcode, m.reads, m.writes) for m in mb.instrs))
                 for mb in r1.program.blocks]
        code2 = [(mb.id, tuple((m.opcode, m.reads, m.writes) for m in mb.instrs))
                 for mb in r2.program.blocks]
        self.assertEqual(code1, code2)

    def test_input_program_not_mutated(self):
        prog = examples.critical_edge_phi_program()
        before = repr(prog)
        allocate(prog, 3)
        self.assertEqual(repr(prog), before)


class TestValidation(unittest.TestCase):
    def test_K_rejects_bool_float_out_of_range(self):
        prog = examples.critical_edge_phi_program()
        for bad in (True, False, 1, 9, 0, 2.0, "3", None):
            with self.subTest(bad=bad):
                with self.assertRaises(InvalidArgumentError):
                    allocate(prog, bad)

    def test_nan_infinity_rejected(self):
        from regalloc import check_finite_number

        for bad in (float("nan"), float("inf"), float("-inf")):
            with self.assertRaises(InvalidArgumentError):
                check_finite_number(bad, "x")
        # bool is not a number argument
        with self.assertRaises(InvalidArgumentError):
            check_finite_number(True, "x")
        self.assertEqual(check_finite_number(3, "x"), 3.0)

    def test_register_out_of_range(self):
        prog = Program(
            entry="e",
            blocks=(
                Block(
                    "e",
                    instrs=(Instruction("c", ("a",), ("x",), fixed_use={"a": 4}),),
                    succs=(),
                ),
            ),
            params=("a",),
        )
        with self.assertRaises(InvalidProgramError) as cm:
            allocate(prog, 2)
        self.assertIn("fixed_use", str(cm.exception))
        self.assertIsNotNone(cm.exception.location)

    def test_unknown_variable(self):
        prog = Program(
            entry="e",
            blocks=(Block("e", instrs=(Instruction("c", ("ghost",), ()),), succs=()),),
        )
        with self.assertRaises(InvalidProgramError):
            allocate(prog, 2)

    def test_duplicate_definition(self):
        prog = Program(
            entry="e",
            blocks=(
                Block(
                    "e",
                    instrs=(
                        Instruction("a", (), ("x",)),
                        Instruction("b", (), ("x",)),
                    ),
                    succs=(),
                ),
            ),
        )
        with self.assertRaises(InvalidProgramError):
            allocate(prog, 2)

    def test_phi_predecessor_mismatch(self):
        prog = Program(
            entry="e",
            blocks=(
                Block("e", instrs=(Instruction("g", (), ("c",)),), succs=("m",)),
                Block(
                    "m",
                    phis=(Phi("x", {"e": "c", "rogue": "c"}),),
                    instrs=(Instruction("r", ("x",), ()),),
                    succs=(),
                ),
            ),
        )
        with self.assertRaises(InvalidProgramError):
            allocate(prog, 2)

    def test_use_before_def_rejected(self):
        prog = Program(
            entry="e",
            blocks=(
                Block("e", instrs=(Instruction("u", ("x",), ()), Instruction("d", (), ("x",))), succs=()),
            ),
        )
        with self.assertRaises(InvalidProgramError):
            allocate(prog, 2)

    def test_undef_on_one_path_rejected(self):
        # value defined on only one branch, used after join without phi
        prog = Program(
            entry="e",
            blocks=(
                Block("e", instrs=(Instruction("t", (), ("c",)),), succs=("a", "b")),
                Block("a", instrs=(Instruction("d", (), ("x",)),), succs=("j")),
                Block("b", instrs=tuple(), succs=("j",)),
                Block("j", instrs=(Instruction("u", ("x",), ()),), succs=()),
            ),
        )
        with self.assertRaises(InvalidProgramError):
            allocate(prog, 2)

    def test_entry_with_predecessor_rejected(self):
        prog = Program(
            entry="a",
            blocks=(
                Block("a", succs=("b",)),
                Block("b", instrs=(Instruction("r", (), ()),), succs=("a",)),
            ),
        )
        with self.assertRaises(InvalidProgramError):
            allocate(prog, 2)

    def test_unreachable_block_rejected(self):
        prog = Program(
            entry="a",
            blocks=(
                Block("a", instrs=(Instruction("r", (), ()),), succs=()),
                Block("orphan", instrs=tuple(), succs=()),
            ),
        )
        with self.assertRaises(InvalidProgramError):
            allocate(prog, 2)

    def test_too_many_blocks(self):
        # A reachable 41-block chain (limit is 40).
        bs = [Block("b0", instrs=tuple(), succs=("b1",))]
        for i in range(1, 40):
            bs.append(Block(f"b{i}", instrs=tuple(),
                            succs=(f"b{i + 1}",)))
        bs.append(Block("b40", instrs=(Instruction("r", (), ()),), succs=()))
        big = Program(entry="b0", blocks=tuple(bs))
        with self.assertRaises(InvalidProgramError):
            allocate(big, 2)

    def test_forty_blocks_accepted(self):
        bs = [Block("b0", instrs=tuple(), succs=("b1",))]
        for i in range(1, 39):
            bs.append(Block(f"b{i}", instrs=tuple(), succs=(f"b{i + 1}",)))
        bs.append(Block("b39", instrs=(Instruction("r", (), ()),), succs=()))
        ok = Program(entry="b0", blocks=tuple(bs))
        verify_allocation(ok, allocate(ok, 2))

    def test_reserved_opcode_rejected(self):
        for op in ("mov", "load", "store", "param"):
            prog = Program(
                entry="e",
                blocks=(Block("e",
                              instrs=(Instruction(op, (), ("x",)),),
                              succs=()),),
            )
            with self.subTest(op=op):
                with self.assertRaises(InvalidProgramError):
                    allocate(prog, 2)

    def test_reserved_prefix_rejected(self):
        prog = Program(
            entry="e",
            blocks=(Block("e", instrs=(Instruction("d", (), ("..x",)),), succs=()),),
        )
        with self.assertRaises(InvalidProgramError):
            allocate(prog, 2)


class TestEdgeSensitiveLiveness(unittest.TestCase):
    def test_phi_source_live_only_on_bound_edge(self):
        prog = examples.critical_edge_phi_program()
        res = allocate(prog, 3)
        edge = res.live.edge_live
        self.assertIn("c", edge[("branch", "join")])
        self.assertIn("v1", edge[("left", "join")])
        # c must not be marked live on the unrelated branch->left... no edge;
        # and v1 must not leak onto branch->join.
        self.assertNotIn("v1", edge[("branch", "join")])
        self.assertNotIn("c", edge[("left", "join")])

    def test_phi_dst_not_in_livein(self):
        prog = examples.loop_phi_program()
        res = allocate(prog, 3)
        self.assertNotIn("i", res.live.live_in["head"])
        self.assertNotIn("acc", res.live.live_in["head"])


class TestGeneratedPrograms(_Verified, unittest.TestCase):
    """End-to-end property checks over deterministic generated SSA CFGs."""

    SEEDS = range(60)
    KS = (2, 3, 5, 8)

    def test_generated_programs_verify(self):
        from regalloc import allocator as alloc_mod

        for seed in self.SEEDS:
            prog = generate_program(seed)
            self.assertLessEqual(len(prog.blocks), 40, f"seed {seed} exceeds block cap")
            for K in self.KS:
                with self.subTest(seed=seed, K=K):
                    res = self.assertVerified(prog, K)
                    # Slot indices are contiguous and assigned by var id order.
                    slots = res.stack_slots
                    self.assertEqual(
                        [slots[v] for v in sorted(slots)],
                        list(range(len(slots))),
                    )
                    # Every spilled var is absent from registers and vice
                    # versa; a variable mapped to None without a slot must be
                    # provably dead (checked independently by the verifier).
                    lin, lout, pts, eds, _ = independent_liveness(prog)
                    ever_live = set()
                    for s in lin.values(): ever_live |= set(s)
                    for s in lout.values(): ever_live |= set(s)
                    for a, b in pts.values():
                        ever_live |= set(a) | set(b)
                    for s in eds.values(): ever_live |= set(s)
                    for v, r in res.assignment.items():
                        self.assertEqual(r is None, v in slots or v not in ever_live)
                    # iterations is honest and bounded.
                    self.assertGreaterEqual(res.iterations, 1)
                    self.assertLessEqual(res.iterations, alloc_mod.MAX_ITERATIONS)

    def test_generated_determinism(self):
        prog = generate_program(7)
        a = allocate(prog, 2)
        b = allocate(prog, 2)
        self.assertEqual(a.assignment, b.assignment)
        self.assertEqual(a.stack_slots, b.stack_slots)


class TestErrorSemantics(unittest.TestCase):
    def test_budget_exhausted_distinct_from_unsat(self):
        # A register-short program would eventually succeed by spilling; cap
        # the search budget to force AllocationBudgetExhausted and verify it
        # is NOT reported as UnsatisfiableProgram.
        from regalloc import allocator as alloc_mod

        prog = examples.high_pressure_program(7)
        old = alloc_mod.MAX_ITERATIONS
        alloc_mod.MAX_ITERATIONS = 1
        try:
            with self.assertRaises(AllocationBudgetExhausted):
                allocate(prog, 2)
        finally:
            alloc_mod.MAX_ITERATIONS = old

    def test_no_partial_change_on_failure(self):
        prog = examples.unsat_fixed_collision_program()
        snapshot = (prog, repr(prog))
        with self.assertRaises(UnsatisfiableProgram):
            allocate(prog, 2)
        # Same object, unchanged content.
        self.assertIs(prog, snapshot[0])
        self.assertEqual(repr(prog), snapshot[1])

    def test_wrong_program_type(self):
        with self.assertRaises(InvalidArgumentError):
            allocate(("not", "a", "program"), 3)

    def test_empty_blocks_rejected(self):
        with self.assertRaises(InvalidProgramError):
            allocate(Program(entry="e", blocks=()), 3)

    def test_unknown_entry(self):
        prog = Program(
            entry="nope",
            blocks=(Block("e", instrs=(Instruction("r", (), ()),), succs=()),),
        )
        with self.assertRaises(InvalidProgramError):
            allocate(prog, 3)


    def test_dead_value_needs_neither_register_nor_slot(self):
        prog = Program(
            entry="e",
            params=("u", "deadp"),
            blocks=(
                Block(
                    "e",
                    instrs=(
                        Instruction("d", (), ("alive",)),
                        Instruction("d2", (), ("deadv",)),
                        Instruction("r", ("u", "alive"), ("out",)),
                        Instruction("ret", ("out",), ()),
                    ),
                    succs=(),
                ),
            ),
        )
        from tests.verifier import verify_allocation
        res = allocate(prog, 2)
        verify_allocation(prog, res)
        self.assertIsNone(res.assignment["deadv"])
        self.assertNotIn("deadv", res.stack_slots)
        self.assertIsNone(res.assignment["deadp"])
        self.assertNotIn("deadp", res.stack_slots)


class TestParallelCopySequentialization(unittest.TestCase):
    """The simultaneous-copy lowering is simulated independently."""

    def _simulate(self, pairs, K, colors):
        from regalloc.allocator import _sequentialize_par
        from regalloc.allocator import EInstr

        ei = EInstr(
            "par",
            tuple(s for s, _ in pairs),
            tuple(d for _, d in pairs),
        )
        seq, _ = _sequentialize_par(ei, colors, 99)
        # Independent machine: distinct initial value per register, plus a
        # scratch memory cell; execute the emitted sequence strictly.
        regs = {i: f"R{i}" for i in range(K)}
        mem = {}
        want = {colors[d]: f"R{colors[s]}" for s, d in pairs
                if colors[s] != colors[d]}
        for mi in seq:
            if mi.opcode == "mov":
                regs[mi.writes[0].i] = regs[mi.reads[0].i]
            elif mi.opcode == "store":
                mem[mi.writes[0].i] = regs[mi.reads[0].i]
            elif mi.opcode == "load":
                regs[mi.writes[0].i] = mem[mi.reads[0].i]
            else:  # pragma: no cover
                self.fail(f"unexpected {mi.opcode}")
        for reg, val in want.items():
            self.assertEqual(regs[reg], val, f"r{reg} got {regs[reg]} expected {val}")

    def test_swap_cycle(self):
        # simultaneous (b=a, a=b) = swap
        self._simulate([("a", "b"), ("b", "a")], 2, {"a": 0, "b": 1})

    def test_three_cycle(self):
        self._simulate([("a", "b"), ("b", "c"), ("c", "a")], 3,
                       {"a": 0, "b": 1, "c": 2})

    def test_fanin_fanout_and_chains(self):
        # One value copied to two destinations, plus a chain and an isolated
        # move; destination registers stay distinct.
        pairs = [("a", "b"), ("b", "c"), ("a", "d"), ("x", "y"), ("p", "q")]
        colors = {"a": 0, "b": 1, "c": 2, "d": 3, "x": 4, "y": 5, "p": 6, "q": 7}
        self._simulate(pairs, 8, colors)

    def test_identity_moves_emitted_as_nothing(self):
        from regalloc.allocator import _sequentialize_par
        from regalloc.allocator import EInstr

        ei = EInstr("par", ("a",), ("a",))
        seq, scratch = _sequentialize_par(ei, {"a": 0}, 0)
        self.assertEqual(seq, [])
        self.assertFalse(scratch)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
