"""Machine model validation tests (views, aliases, classes, callee-save)."""

import unittest

from regalloc import Machine, RegView, flat_machine, machine_from_dict
from regalloc.model import ProgramFormatError
from regalloc.samples2 import machine_16, machine_torture


def _two_view_machine():
    views = (
        RegView("w0", 0, 0, 32, "w"),
        RegView("x0", 0, 0, 64, "x"),
    )
    return Machine(
        views=views,
        aliases={"w0": ("x0",), "x0": ("w0",)},
        class_align={"w": 4, "x": 8},
    )


class TestViewValidation(unittest.TestCase):
    def test_empty_range_rejected(self):
        with self.assertRaises(ProgramFormatError):
            RegView("bad", 0, 32, 32, "w")

    def test_non_byte_width_rejected(self):
        with self.assertRaises(ProgramFormatError):
            RegView("bad", 0, 0, 7, "w")

    def test_bool_rejected_everywhere(self):
        with self.assertRaises(ProgramFormatError):
            RegView("bad", True, 0, 32, "w")
        with self.assertRaises(ProgramFormatError):
            RegView("bad", 0, 0, 32, "w", callee_saved=1)

    def test_duplicate_names_rejected(self):
        with self.assertRaises(ProgramFormatError):
            Machine(
                views=(RegView("a", 0, 0, 32, "w"), RegView("a", 1, 0, 32, "w")),
                aliases={},
                class_align={"w": 4},
            )


class TestAliasContract(unittest.TestCase):
    def test_alias_must_be_symmetric(self):
        with self.assertRaises(ProgramFormatError) as cm:
            Machine(
                views=(RegView("w0", 0, 0, 32, "w"), RegView("x0", 0, 0, 64, "x")),
                aliases={"w0": ("x0",)},  # x0 does not declare w0 back
                class_align={"w": 4, "x": 8},
            )
        self.assertIn("symmetric", str(cm.exception))

    def test_alias_must_match_bit_overlap(self):
        # w0 and x0 overlap, but declaring NO aliases is a lie -> rejected
        with self.assertRaises(ProgramFormatError) as cm:
            Machine(
                views=(RegView("w0", 0, 0, 32, "w"), RegView("x0", 0, 0, 64, "x")),
                aliases={},
                class_align={"w": 4, "x": 8},
            )
        self.assertIn("bit-overlap", str(cm.exception))

    def test_alias_declared_without_overlap_rejected(self):
        with self.assertRaises(ProgramFormatError):
            Machine(
                views=(RegView("a", 0, 0, 32, "w"), RegView("b", 1, 0, 32, "w")),
                aliases={"a": ("b",), "b": ("a",)},  # different groups!
                class_align={"w": 4},
            )

    def test_self_alias_rejected(self):
        with self.assertRaises(ProgramFormatError):
            Machine(
                views=(RegView("a", 0, 0, 32, "w"),),
                aliases={"a": ("a",)},
                class_align={"w": 4},
            )

    def test_overlap_relation_not_name_inequality(self):
        m = _two_view_machine()
        self.assertTrue(m.views_overlap("w0", "x0"))   # different names, shared bits
        self.assertTrue(m.views_overlap("w0", "w0"))
        m2 = machine_16()
        self.assertFalse(m2.views_overlap("w0", "w1"))  # different groups
        self.assertTrue(m2.views_overlap("w3", "x3"))


class TestClassContract(unittest.TestCase):
    def test_class_width_must_be_uniform(self):
        with self.assertRaises(ProgramFormatError) as cm:
            Machine(
                views=(RegView("a", 0, 0, 32, "w"), RegView("b", 1, 0, 64, "w")),
                aliases={},
                class_align={"w": 4},
            )
        self.assertIn("mixes widths", str(cm.exception))

    def test_missing_class_align_rejected(self):
        with self.assertRaises(ProgramFormatError):
            Machine(
                views=(RegView("a", 0, 0, 32, "w"),),
                aliases={},
                class_align={},
            )


class TestCalleeSavedContract(unittest.TestCase):
    def test_callee_saved_must_be_alias_closed(self):
        with self.assertRaises(ProgramFormatError) as cm:
            Machine(
                views=(
                    RegView("w0", 0, 0, 32, "w", callee_saved=True),
                    RegView("x0", 0, 0, 64, "x", callee_saved=False),
                ),
                aliases={"w0": ("x0",), "x0": ("w0",)},
                class_align={"w": 4, "x": 8},
            )
        self.assertIn("alias-closed", str(cm.exception))

    def test_call_clobbers_only_caller_saved(self):
        m = machine_torture()
        clob = m.clobbered_by_call(())
        self.assertEqual(set(clob), {"w0", "x0"})
        clob2 = m.clobbered_by_call(("s0",))
        self.assertEqual(set(clob2), {"w0", "x0", "s0"})


class TestAdapters(unittest.TestCase):
    def test_flat_machine(self):
        m = flat_machine(4)
        self.assertEqual(len(m.views), 4)
        self.assertEqual(m.classes, {"gpr": ("r0", "r1", "r2", "r3")})
        for v in m.views:
            self.assertEqual(m.aliases[v.name], ())

    def test_machine_from_dict_roundtrip(self):
        m = machine_from_dict({
            "views": [
                {"name": "w0", "group": 0, "lo": 0, "hi": 32, "cls": "w",
                 "callee_saved": True},
                {"name": "x0", "group": 0, "lo": 0, "hi": 64, "cls": "x",
                 "callee_saved": True},
            ],
            "aliases": {"w0": ["x0"], "x0": ["w0"]},
            "class_align": {"w": 4, "x": 8},
        })
        self.assertTrue(m.views_overlap("w0", "x0"))
        self.assertTrue(m.view("x0").callee_saved)

    def test_default_class_is_narrowest_then_name(self):
        m = machine_torture()
        # w and s are both 32-bit; "s" < "w" lexicographically
        self.assertEqual(m.default_class, "s")
