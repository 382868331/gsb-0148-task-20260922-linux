"""Strict input validation tests for the v2 format.

Covers: feature gating (v2 fields need var_classes), exceptional-edge
well-formedness, class/view cross-checks, JSON dict construction and the
v1/v2 entry-point separation.
"""

import unittest

from regalloc import (
    Machine,
    RegView,
    allocate,
    allocate2,
    program_from_dict,
    ProgramFormatError,
)
from regalloc.samples2 import machine_torture
from regalloc.model import Block, Instr, PhiNode, Program


class TestV2FeatureGating(unittest.TestCase):
    def test_early_clobber_requires_v2(self):
        with self.assertRaises(ProgramFormatError):
            Program(
                blocks=(Block(label="e", instrs=(
                    Instr(uid=0, uses=(0,), defs=(1,), early_clobber=(1,)),
                ),),),
                params=(0,),
            )

    def test_exc_succs_requires_v2(self):
        b0 = Block(label="b0", instrs=(), succs=(), exc_succs=(1,))
        b1 = Block(label="b1", instrs=())
        with self.assertRaises(ProgramFormatError):
            Program(blocks=(b0, b1))

    def test_legacy_fixed_rejected_in_v2(self):
        with self.assertRaises(ProgramFormatError):
            Program(
                blocks=(Block(label="e", instrs=(
                    Instr(uid=0, uses=(0,), defs=(1,), fixed={0: 0}),
                ),),),
                params=(0,),
                var_classes={0: "w", 1: "w"},
            )

    def test_legacy_clobbers_rejected_in_v2(self):
        with self.assertRaises(ProgramFormatError):
            Program(
                blocks=(Block(label="e", instrs=(
                    Instr(uid=0, uses=(0,), defs=(1,), clobbers=(0,)),
                ),),),
                params=(0,),
                var_classes={0: "w", 1: "w"},
            )

    def test_mixing_fixed_styles_rejected(self):
        with self.assertRaises(ProgramFormatError):
            Instr(
                uid=0, uses=(0,), defs=(1,),
                fixed={0: 0}, fixed_views={0: "w0"},
            )

    def test_early_clobber_must_be_a_def(self):
        with self.assertRaises(ProgramFormatError):
            Instr(uid=0, uses=(0,), defs=(1,), early_clobber=(0,))


class TestExceptionalEdges(unittest.TestCase):
    VC = {1: "w", 2: "w"}

    def test_normal_and_exc_successor_disjoint(self):
        with self.assertRaises(ProgramFormatError):
            Block(label="b0", instrs=(), succs=(1,), exc_succs=(1,))

    def test_exc_succs_needs_one_thrower(self):
        # declared exceptional successors but no may_throw instruction
        b0 = Block(label="b0", instrs=(), succs=(), exc_succs=(1,))
        b1 = Block(label="b1", instrs=())
        with self.assertRaises(ProgramFormatError):
            Program(blocks=(b0, b1), var_classes=self.VC)

    def test_thrower_without_exc_succs_rejected(self):
        b0 = Block(
            label="b0",
            instrs=(Instr(uid=0, uses=(), defs=(1,), may_throw=True),),
        )
        with self.assertRaises(ProgramFormatError):
            Program(blocks=(b0,), var_classes=self.VC)

    def test_two_throwers_rejected(self):
        b0 = Block(
            label="b0",
            instrs=(
                Instr(uid=0, uses=(), defs=(1,), may_throw=True),
                Instr(uid=1, uses=(1,), defs=(2,), may_throw=True),
            ),
            succs=(),
            exc_succs=(1,),
        )
        b1 = Block(label="h", instrs=(Instr(uid=2, uses=(1,), defs=()),))
        with self.assertRaises(ProgramFormatError):
            Program(blocks=(b0, b1), var_classes=self.VC)

    def test_exceptional_reachability(self):
        # handler not reachable on any edge -> unreachable block
        b0 = Block(
            label="b0",
            instrs=(Instr(uid=0, uses=(), defs=(1,), may_throw=True),),
            succs=(),
        )
        b1 = Block(label="h", instrs=())
        with self.assertRaises(ProgramFormatError):
            Program(blocks=(b0, b1), var_classes=self.VC)


class TestClassCrossChecks(unittest.TestCase):
    def test_unknown_class(self):
        prog = Program(
            blocks=(Block(label="e", instrs=(
                Instr(uid=0, uses=(), defs=(1,)),
            ),),),
            var_classes={1: "nope"},
        )
        with self.assertRaises(ProgramFormatError) as cm:
            allocate2(prog, machine_torture())
        self.assertIn("does not exist in machine", str(cm.exception))

    def test_class_for_unknown_variable_rejected(self):
        with self.assertRaises(ProgramFormatError):
            Program(
                blocks=(Block(label="e", instrs=(
                    Instr(uid=0, uses=(), defs=(1,)),
                ),),),
                var_classes={9: "w"},
            )

    def test_fixed_view_wrong_class(self):
        prog = Program(
            blocks=(Block(label="e", instrs=(
                Instr(uid=0, uses=(0,), defs=(), fixed_views={0: "x0"}),
            ),),),
            params=(0,),
            var_classes={0: "w"},
        )
        with self.assertRaises(ProgramFormatError) as cm:
            allocate2(prog, machine_torture())
        self.assertIn("class", str(cm.exception))

    def test_phi_classes_must_match(self):
        b0 = Block(label="b0", instrs=(Instr(uid=0, uses=(), defs=(1,)),),
                   succs=(1,))
        b1 = Block(
            label="m",
            instrs=(Instr(uid=1, uses=(2,), defs=()),),
            phis=(PhiNode(def_var=2, sources=((0, 1),)),),
        )
        # v1 is x-class, phi v2 is w-class
        prog = Program(blocks=(b0, b1), var_classes={1: "x", 2: "w"})
        with self.assertRaises(ProgramFormatError) as cm:
            allocate2(prog, machine_torture())
        self.assertIn("phi", str(cm.exception))

    def test_unknown_fixed_view(self):
        prog = Program(
            blocks=(Block(label="e", instrs=(
                Instr(uid=0, uses=(0,), defs=(), fixed_views={0: "nope"}),
            ),),),
            params=(0,),
            var_classes={0: "w"},
        )
        with self.assertRaises(ProgramFormatError):
            allocate2(prog, machine_torture())


class TestEntryPoints(unittest.TestCase):
    def test_allocate_rejects_v2_program(self):
        from regalloc.samples2 import sample_v2_torture
        with self.assertRaises(ProgramFormatError):
            allocate(sample_v2_torture(), 3)

    def test_allocate2_rejects_legacy_program(self):
        from regalloc.samples import sample_register_pressure
        with self.assertRaises(ProgramFormatError):
            allocate2(sample_register_pressure(), machine_torture())

    def test_allocate2_rejects_bad_types(self):
        with self.assertRaises(ProgramFormatError):
            allocate2("not a program", machine_torture())
        from regalloc.samples2 import sample_v2_torture
        with self.assertRaises(ProgramFormatError):
            allocate2(sample_v2_torture(), "not a machine")


class TestDictConstruction(unittest.TestCase):
    def test_v2_roundtrip(self):
        data = {
            "name": "t",
            "params": [1],
            "var_classes": {"1": "w"},
            "blocks": [
                {"label": "b0", "instrs": [
                    {"uid": 0, "uses": [1], "defs": [],
                     "is_call": True, "clobber_views": ["w2"],
                     "may_throw": True}],
                 "succs": [], "exc_succs": [1]},
                {"label": "h", "instrs": [
                    {"uid": 1, "uses": [1], "defs": []}]},
            ],
        }
        prog = program_from_dict(data)
        self.assertIsNotNone(prog.var_classes)
        self.assertTrue(prog.blocks[0].instrs[0].may_throw)
        self.assertEqual(prog.blocks[0].exc_succs, (1,))

    def test_v1_dict_unchanged(self):
        prog = program_from_dict({"blocks": [
            {"label": "b0", "instrs": [{"uid": 0, "uses": [], "defs": []}]}
        ]})
        self.assertIsNone(prog.var_classes)
        self.assertEqual(prog.blocks[0].exc_succs, ())
