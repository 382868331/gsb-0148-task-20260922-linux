"""Strict input validation and error semantics.

Covers: bool rejected as int, NaN/Infinity rejected, K boundary 2..8,
locatable error paths, and the proven-infeasible vs format distinction.
"""

import unittest

from regalloc import allocate
from regalloc.model import (
    Block,
    Instr,
    PhiNode,
    Program,
    ProgramFormatError,
    check_finite_float,
    check_int,
    program_from_dict,
)
from regalloc.samples import (
    sample_fixed_collision,
    sample_register_pressure,
)


def _tiny():
    return Program(
        blocks=(Block(label="e", instrs=(Instr(uid=0, uses=(0,), defs=(1,)),), succs=()),),
        params=(0,),
    )


class TestScalarChecks(unittest.TestCase):
    def test_int_accepts_plain_int(self):
        self.assertEqual(check_int(3, "x", minimum=0), 3)

    def test_int_rejects_bool(self):
        for bad in (True, False):
            with self.assertRaises(ProgramFormatError):
                check_int(bad, "x")

    def test_int_rejects_float(self):
        for bad in (1.0, 2.9):
            with self.assertRaises(ProgramFormatError):
                check_int(bad, "x")

    def test_int_rejects_nan_inf_as_int(self):
        with self.assertRaises(ProgramFormatError):
            check_int(float("nan"), "x")
        with self.assertRaises(ProgramFormatError):
            check_int(float("inf"), "x")

    def test_finite_float_rejects_nonfinite(self):
        self.assertEqual(check_finite_float(2, "x"), 2.0)
        for bad in (float("nan"), float("inf"), float("-inf")):
            with self.assertRaises(ProgramFormatError):
                check_finite_float(bad, "x")

    def test_finite_float_rejects_bool(self):
        with self.assertRaises(ProgramFormatError):
            check_finite_float(True, "x")

    def test_int_minimum(self):
        with self.assertRaises(ProgramFormatError):
            check_int(-1, "x", minimum=0)


class TestKValidation(unittest.TestCase):
    def setUp(self):
        self.p = _tiny()

    def test_boundaries_accepted(self):
        allocate(self.p, 2)
        allocate(self.p, 8)

    def test_out_of_range(self):
        for bad in (1, 9, 0, -2):
            with self.assertRaises(ProgramFormatError):
                allocate(self.p, bad)

    def test_bool_rejected(self):
        with self.assertRaises(ProgramFormatError):
            allocate(self.p, True)

    def test_nan_inf_rejected(self):
        for bad in (float("nan"), float("inf"), float("-inf"), 2.0):
            with self.assertRaises(ProgramFormatError):
                allocate(self.p, bad)

    def test_bad_program_type(self):
        with self.assertRaises(ProgramFormatError):
            allocate({"blocks": []}, 3)


class TestStructureValidation(unittest.TestCase):
    def test_use_def_overlap(self):
        with self.assertRaises(ProgramFormatError):
            Instr(uid=0, uses=(1,), defs=(1,))

    def test_fixed_var_must_be_operand(self):
        with self.assertRaises(ProgramFormatError) as cm:
            Instr(uid=0, uses=(1,), defs=(), fixed={2: 0})
        self.assertIn("fixed", str(cm.exception))

    def test_register_checked_against_k(self):
        prog = Program(
            blocks=(Block(
                label="e",
                instrs=(Instr(uid=0, uses=(1,), defs=(), fixed={1: 3}),),
            ),),
            params=(1,),
        )
        with self.assertRaises(ProgramFormatError) as cm:
            allocate(prog, 3)  # registers are r0,r1,r2; r3 absent
        self.assertIn("fixed", str(cm.exception))

    def test_clobber_register_checked_against_k(self):
        prog = Program(
            blocks=(Block(
                label="e",
                instrs=(Instr(uid=0, uses=(), defs=(1,), clobbers=(2,)),),
            ),),
        )
        with self.assertRaises(ProgramFormatError):
            allocate(prog, 2)

    def test_too_many_blocks(self):
        blocks = []
        # a chain of 41 is reachable; block i -> i+1
        for i in range(41):
            succ = (i + 1,) if i < 40 else ()
            blocks.append(Block(label=f"b{i}", instrs=(), succs=succ))
        with self.assertRaises(ProgramFormatError) as cm:
            Program(blocks=tuple(blocks))
        self.assertIn("40", str(cm.exception))

    def test_unreachable_block(self):
        b0 = Block(label="b0", instrs=(), succs=())
        b1 = Block(label="b1", instrs=(), succs=())
        with self.assertRaises(ProgramFormatError) as cm:
            Program(blocks=(b0, b1))
        self.assertIn("unreachable", str(cm.exception))

    def test_bad_successor_index(self):
        with self.assertRaises(ProgramFormatError):
            Program(blocks=(Block(label="b0", instrs=(), succs=(5,)),))

    def test_unknown_variable(self):
        with self.assertRaises(ProgramFormatError) as cm:
            Program(
                blocks=(Block(label="b0", instrs=(Instr(uid=0, uses=(9,), defs=()),),),)
            )
        self.assertIn("never defined", str(cm.exception))

    def test_phi_missing_predecessor_source(self):
        b0 = Block(label="b0", instrs=(Instr(uid=0, uses=(), defs=(1,)),), succs=(1,))
        b1 = Block(
            label="b1",
            instrs=(Instr(uid=1, uses=(2,), defs=()),),
            succs=(),
            phis=(PhiNode(def_var=2, sources=()),),  # missing source from b0
        )
        with self.assertRaises(ProgramFormatError) as cm:
            Program(blocks=(b0, b1))
        self.assertIn("missing sources", str(cm.exception))

    def test_phi_unknown_predecessor(self):
        b0 = Block(label="b0", instrs=(Instr(uid=0, uses=(), defs=(1,)),), succs=(1,))
        b1 = Block(
            label="b1",
            instrs=(Instr(uid=1, uses=(2,), defs=()),),
            phis=(PhiNode(def_var=2, sources=((0, 1), (7, 1))),),
        )
        with self.assertRaises(ProgramFormatError) as cm:
            Program(blocks=(b0, b1))
        self.assertIn("non-predecessor", str(cm.exception))

    def test_entry_block_phi_rejected(self):
        with self.assertRaises(ProgramFormatError):
            Program(
                blocks=(Block(
                    label="b0",
                    instrs=(),
                    phis=(PhiNode(def_var=1, sources=()),),
                ),)
            )

    def test_duplicate_uid(self):
        with self.assertRaises(ProgramFormatError) as cm:
            Program(blocks=(Block(
                label="b0",
                instrs=(Instr(uid=0, uses=(), defs=(1,)),
                        Instr(uid=0, uses=(1,), defs=(2,))),
            ),))
        self.assertIn("duplicate instruction uid", str(cm.exception))

    def test_error_path_is_locatable_from_dict(self):
        data = {
            "blocks": [
                {"label": "b0", "instrs": [{"uid": "nope"}], "succs": []}
            ]
        }
        with self.assertRaises(ProgramFormatError) as cm:
            program_from_dict(data)
        msg = str(cm.exception)
        self.assertIn("blocks[0].instrs[0]", msg)


class TestProvenInfeasibility(unittest.TestCase):
    def test_fixed_register_collision_is_proven(self):
        from regalloc import InfeasibleProgramError
        with self.assertRaises(InfeasibleProgramError) as cm:
            allocate(sample_fixed_collision(), 3)
        self.assertIn("proven infeasible", str(cm.exception))

    def test_too_many_operands_is_proven(self):
        from regalloc import InfeasibleProgramError
        prog = Program(
            blocks=(Block(
                label="e",
                instrs=(Instr(uid=0, uses=(0, 1, 2, 3), defs=(4,)),),
            ),),
            params=(0, 1, 2, 3),
        )
        with self.assertRaises(InfeasibleProgramError) as cm:
            allocate(prog, 3)
        self.assertIn("proven infeasible", str(cm.exception))

    def test_pressure_program_is_feasible_with_spilling(self):
        a = allocate(sample_register_pressure(), 2)
        self.assertTrue(a.slots)  # spill genuinely occurred
