"""Target descriptions (format version 2): width-aliased register views,
register classes, register pairs and callee-save rules.

A *view* is a named bit window onto a physical register *bank*.  For example
the classic x86-style layout has, per bank, a 64-bit view, a 32-bit view, a
16-bit view and an 8-bit view; ``eax`` and ``ax`` are *different view names*
but their bit windows overlap, so they alias.  Alias sets are given
**explicitly** in the input; the declaration also carries the bit layout
(bank, lo bit, width), and the constructor checks that the explicit alias
relation is exactly the bit-overlap relation — the two declarations must
agree, neither is silently derived from the other.

Register *classes* say which views an operand of that class may occupy.  A
class is either::

    * single-width: options are individual views (one bank each);
    * pair: options are explicit (lo_view, hi_view) pairs occupying two
      different banks; the value width is the pair width.

``callee_saved`` lists the views whose bits are preserved across a call;
every other view is caller-saved and a call must list it in ``clobbersv``
when values must not survive in it.

Nothing here is random; everything is explicit and validated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .model import ProgramFormatError, _err, check_int


@dataclass(frozen=True)
class RegView:
    """A named bit window onto one physical register bank.

    ``aliases`` is the explicit set of OTHER view names whose bit windows
    overlap this one (symmetry is checked by :class:`TargetDesc`).
    ``lo`` is the bit offset inside the bank (bit 0 = least significant).
    """

    name: str
    bank: int
    lo: int
    width: int
    aliases: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise _err("view.name", "view name must be a non-empty string")
        check_int(self.bank, f"views[{self.name}].bank", minimum=0)
        check_int(self.lo, f"views[{self.name}].lo", minimum=0)
        check_int(self.width, f"views[{self.name}].width", minimum=1)
        if not isinstance(self.aliases, (list, tuple)):
            raise _err(f"views[{self.name}].aliases", "expected list/tuple")
        al = tuple(self.aliases)
        for a in al:
            if not isinstance(a, str) or not a:
                raise _err(f"views[{self.name}].aliases", "alias names must be non-empty strings")
        if self.name in al:
            raise _err(f"views[{self.name}].aliases", "a view cannot alias itself")
        object.__setattr__(self, "aliases", al)


@dataclass(frozen=True)
class RegClass:
    """An operand register class.

    * ``views``  — views a single-width operand may occupy (declaration
      order is the deterministic allocation preference order);
    * ``pairs`` — explicit ``(lo_view, hi_view)`` options for a double-width
      operand.  When non-empty the class is a *pair class*: an operand
      occupies both banks, holding a ``width``-bit value (lo = low half).
    """

    name: str
    width: int
    views: tuple[str, ...] = ()
    pairs: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise _err("class.name", "class name must be a non-empty string")
        check_int(self.width, f"classes[{self.name}].width", minimum=1)
        if not isinstance(self.views, (list, tuple)):
            raise _err(f"classes[{self.name}].views", "expected list/tuple")
        if not isinstance(self.pairs, (list, tuple)):
            raise _err(f"classes[{self.name}].pairs", "expected list/tuple")
        views = tuple(self.views)
        pairs: list[tuple[str, str]] = []
        for v in views:
            if not isinstance(v, str) or not v:
                raise _err(f"classes[{self.name}].views", "view names must be non-empty strings")
        for i, pr in enumerate(self.pairs):
            if not isinstance(pr, (list, tuple)) or len(pr) != 2:
                raise _err(f"classes[{self.name}].pairs[{i}]", "expected [lo_view, hi_view]")
            lo, hi = pr
            if not isinstance(lo, str) or not isinstance(hi, str) or not lo or not hi:
                raise _err(f"classes[{self.name}].pairs[{i}]", "view names must be non-empty strings")
            if lo == hi:
                raise _err(f"classes[{self.name}].pairs[{i}]", "pair halves must be different views")
            pairs.append((lo, hi))
        if views and pairs:
            raise _err(f"classes[{self.name}]", "a class is either single-width (views) or a pair class, not both")
        if not views and not pairs:
            raise _err(f"classes[{self.name}]", "a class must name views or pairs")
        object.__setattr__(self, "views", views)
        object.__setattr__(self, "pairs", tuple(pairs))

    @property
    def is_pair(self) -> bool:
        return bool(self.pairs)


@dataclass(frozen=True)
class ClassInfo:
    """Resolved per-class information used by the allocator."""

    name: str
    width: int                       # bits
    is_pair: bool
    # each option is a frozenset of banks; option order is declaration order
    options: tuple[frozenset[int], ...]
    # pair option key (sorted banks tuple) -> (lo view, hi view)
    pair_views: dict[tuple[int, int], tuple[str, str]] = field(default_factory=dict)
    # bank -> the class view that exactly covers the class width on that bank
    view_on_bank: dict[int, str] = field(default_factory=dict)


@dataclass(frozen=True)
class TargetDesc:
    """A complete physical register target description."""

    name: str
    views: tuple[RegView, ...]
    classes: tuple[RegClass, ...]
    callee_saved: tuple[str, ...] = ()
    default_class: str = ""

    # ---- resolved caches built once ----
    def resolved(self) -> "_ResolvedTarget":
        return _ResolvedTarget(self)


@dataclass(frozen=True)
class _ResolvedTarget:
    desc: TargetDesc

    def __post_init__(self) -> None:
        d = self.desc
        if not isinstance(d.name, str) or not d.name:
            raise _err("target.name", "target name must be a non-empty string")
        if not d.views:
            raise _err("target.views", "a target must declare at least one view")
        if not d.classes:
            raise _err("target.classes", "a target must declare at least one class")

        views: dict[str, RegView] = {}
        banks: set[int] = set()
        for v in d.views:
            if v.name in views:
                raise _err("target.views", f"duplicate view name {v.name!r}")
            views[v.name] = v
            banks.add(v.bank)
        for b in banks:
            if b < 0 or b != len([x for x in banks if x < b]):
                raise _err("target.views", f"bank ids must be dense 0..B-1, got {sorted(banks)}")

        # explicit alias relation must be symmetric ...
        for v in d.views:
            for a in v.aliases:
                if a not in views:
                    raise _err(f"views[{v.name}].aliases", f"unknown view {a!r}")
                if v.name not in views[a].aliases:
                    raise _err(
                        f"views[{v.name}].aliases",
                        f"alias {a!r} is not symmetric (declare it on both views)",
                    )

        # ... and exactly equal to the bit-overlap relation
        def overlaps(x: RegView, y: RegView) -> bool:
            if x.bank != y.bank:
                return False
            return x.lo < y.lo + y.width and y.lo < x.lo + x.width

        for x in d.views:
            want = {y.name for y in d.views if y is not x and overlaps(x, y)}
            got = set(x.aliases)
            if want != got:
                missing = sorted(want - got)
                extra = sorted(got - want)
                msg = []
                if missing:
                    msg.append(f"missing aliases {missing}")
                if extra:
                    msg.append(f"non-overlapping aliases {extra}")
                raise _err(
                    f"views[{x.name}].aliases",
                    "explicit alias set disagrees with bit layout: " + "; ".join(msg),
                )

        classes: dict[str, RegClass] = {}
        infos: dict[str, ClassInfo] = {}
        for c in d.classes:
            if c.name in classes:
                raise _err("target.classes", f"duplicate class name {c.name!r}")
            classes[c.name] = c
            options: list[frozenset[int]] = []
            pair_views: dict[tuple[int, int], tuple[str, str]] = {}
            view_on_bank: dict[int, str] = {}
            if c.is_pair:
                for lo_n, hi_n in c.pairs:
                    if lo_n not in views:
                        raise _err(f"classes[{c.name}].pairs", f"unknown view {lo_n!r}")
                    if hi_n not in views:
                        raise _err(f"classes[{c.name}].pairs", f"unknown view {hi_n!r}")
                    lo, hi = views[lo_n], views[hi_n]
                    if lo.bank == hi.bank:
                        raise _err(
                            f"classes[{c.name}].pairs",
                            f"pair {lo_n!r}/{hi_n!r} halves must be on different banks",
                        )
                    if lo.width + hi.width != c.width:
                        raise _err(
                            f"classes[{c.name}].pairs",
                            f"pair {lo_n!r}+{hi_n!r} width {lo.width + hi.width} "
                            f"!= class width {c.width}",
                        )
                    key = tuple(sorted((lo.bank, hi.bank)))
                    if frozenset(key) in options:
                        raise _err(f"classes[{c.name}].pairs", "duplicate bank pair option")
                    options.append(frozenset(key))
                    pair_views[key] = (lo_n, hi_n)
            else:
                for vn in c.views:
                    if vn not in views:
                        raise _err(f"classes[{c.name}].views", f"unknown view {vn!r}")
                    vv = views[vn]
                    if vv.width != c.width:
                        raise _err(
                            f"classes[{c.name}].views",
                            f"view {vn!r} width {vv.width} != class {c.name!r} width {c.width}",
                        )
                    if vv.bank in view_on_bank:
                        raise _err(
                            f"classes[{c.name}].views",
                            f"two views of class {c.name!r} on bank {vv.bank}",
                        )
                    view_on_bank[vv.bank] = vn
                    options.append(frozenset({vv.bank}))
            infos[c.name] = ClassInfo(
                name=c.name, width=c.width, is_pair=c.is_pair,
                options=tuple(options), pair_views=pair_views,
                view_on_bank=view_on_bank,
            )

        if not d.default_class:
            raise _err("target.default_class", "a default class name is required")
        if d.default_class not in classes:
            raise _err("target.default_class", f"unknown class {d.default_class!r}")

        saved: set[str] = set()
        for vn in d.callee_saved:
            if not isinstance(vn, str) or vn not in views:
                raise _err("target.callee_saved", f"unknown view {vn!r}")
            if vn in saved:
                raise _err("target.callee_saved", f"duplicate view {vn!r}")
            saved.add(vn)
        # callee-saved status is a bank-wide ABI property here: every view of
        # a bank must agree (a caller cannot preserve only "eax" but not "ax")
        saved_banks = {views[v].bank for v in saved}
        for v in d.views:
            is_saved = v.name in saved
            if (v.bank in saved_banks) != is_saved:
                raise _err(
                    "target.callee_saved",
                    f"view {v.name!r}: callee-saved status must be uniform per bank "
                    f"(bank {v.bank} has both saved and caller-saved views)",
                )

        object.__setattr__(self, "_views", views)
        object.__setattr__(self, "_banks", sorted(banks))
        object.__setattr__(self, "_classes", classes)
        object.__setattr__(self, "_infos", infos)
        object.__setattr__(self, "_saved_banks", frozenset(saved_banks))

    # ---- accessors ----
    @property
    def banks(self) -> tuple[int, ...]:
        return tuple(self._banks)

    @property
    def num_banks(self) -> int:
        return len(self._banks)

    def view(self, name: str) -> RegView:
        return self._views[name]

    def has_view(self, name: str) -> bool:
        return name in self._views

    def class_info(self, name: str) -> ClassInfo:
        return self._infos[name]

    def has_class(self, name: str) -> bool:
        return name in self._infos

    @property
    def default_class_name(self) -> str:
        return self.desc.default_class

    @property
    def saved_banks(self) -> frozenset[int]:
        return self._saved_banks

    def saved_view_names(self) -> frozenset[str]:
        return frozenset(self.desc.callee_saved)

    def home_view(self, class_name: str, banks: frozenset[int]) -> tuple[str, ...]:
        """The concrete view(s) covering an operand of ``class_name`` placed
        on ``banks``.  Returns one view (single class) or (lo, hi) (pair)."""
        info = self.class_info(class_name)
        if info.is_pair:
            key = tuple(sorted(banks))
            return info.pair_views[key]
        (b,) = tuple(banks)
        return (info.view_on_bank[b],)

    def free_temp_view(self, class_name: str, occupied: frozenset[int]) -> tuple[str, ...] | None:
        """First declared allocation option of the class whose banks are all
        outside ``occupied`` (i.e. no aliased bit position is live)."""
        info = self.class_info(class_name)
        for opt in info.options:
            if opt.isdisjoint(occupied):
                return self.home_view(class_name, opt)
        return None


# --------------------------------------------------------------------------
# built-in targets
# --------------------------------------------------------------------------

def _x86_like_views() -> tuple[RegView, ...]:
    # 4 banks A B C D; each with 64/32/16/8 bit windows -> 16 views.
    prefix = {0: ("rax", "eax", "ax", "al"),
              1: ("rbx", "ebx", "bx", "bl"),
              2: ("rcx", "ecx", "cx", "cl"),
              3: ("rdx", "edx", "dx", "dl")}
    widths = (64, 32, 16, 8)
    views: list[RegView] = []
    for b, names in prefix.items():
        bank_names = [n for _, ns in prefix.items() for n in ns]
        for name, w in zip(names, widths):
            aliases = tuple(n for n in bank_names
                            if any(n == x for x in names) and n != name)
            views.append(RegView(name=name, bank=b, lo=0, width=w, aliases=aliases))
    return tuple(views)


def default_target() -> TargetDesc:
    """The built-in 16-view target (4 banks x 64/32/16/8 views).

    * banks: A=rax (caller-saved), B=rbx (callee-saved),
             C=rcx (caller-saved), D=rdx (callee-saved);
    * classes: GPR8/GPR16/GPR32 (default)/GPR64 single-width, plus PAIR64,
      a double-width class with two declared pair options, eax:ecx and
      ebx:edx (only the latter can survive a call that clobbers rax,rcx).
    """
    views = _x86_like_views()
    classes = (
        RegClass("GPR8", 8, ("al", "bl", "cl", "dl")),
        RegClass("GPR16", 16, ("ax", "bx", "cx", "dx")),
        RegClass("GPR32", 32, ("eax", "ebx", "ecx", "edx")),
        RegClass("GPR64", 64, ("rax", "rbx", "rcx", "rdx")),
        RegClass("PAIR64", 64, pairs=(("eax", "ecx"), ("ebx", "edx"))),
    )
    callee = ("rbx", "ebx", "bx", "bl", "rdx", "edx", "dx", "dl")
    return TargetDesc(
        name="x86like-16",
        views=views,
        classes=classes,
        callee_saved=callee,
        default_class="GPR32",
    )


def tiny_target() -> TargetDesc:
    """A 2-bank / 6-view target with no spare register under pressure.

    Banks T0 (caller-saved) and T1 (callee-saved); each exposes 32/16/8 bit
    views.  With both banks occupied, a parallel-copy cycle has *no* free
    temporary register and must be broken via a stack slot.
    """
    spec = {0: ("t0", "t0h", "t0b"), 1: ("t1", "t1h", "t1b")}
    widths = (32, 16, 8)
    all_names = [n for ns in spec.values() for n in ns]
    views: list[RegView] = []
    for b, names in spec.items():
        for name, w in zip(names, widths):
            aliases = tuple(n for n in all_names if n in names and n != name)
            views.append(RegView(name=name, bank=b, lo=0, width=w, aliases=aliases))
    classes = (
        RegClass("T8", 8, ("t0b", "t1b")),
        RegClass("T16", 16, ("t0h", "t1h")),
        RegClass("T32", 32, ("t0", "t1")),
        # the two banks together form one double-width value; it is the only
        # pair option, so under any pressure it cannot be moved to a spare
        # register pair (there is none)
        RegClass("TPAIR", 64, pairs=(("t0", "t1"),)),
    )
    return TargetDesc(
        name="tiny-6",
        views=tuple(views),
        classes=classes,
        callee_saved=("t1", "t1h", "t1b"),
        default_class="T32",
    )


def target_from_dict(data: Any) -> TargetDesc:
    """Build a :class:`TargetDesc` from plain dicts/lists.

    Shape::

        {"name": str, "default_class": str, "callee_saved": [str],
         "views": [{"name": str, "bank": int, "lo": int, "width": int,
                    "aliases": [str]}],
         "classes": [{"name": str, "width": int,
                      "views": [str] | "pairs": [[lo, hi], ...]}]}
    """
    if not isinstance(data, dict):
        raise _err("target", "expected a mapping/object")
    views: list[RegView] = []
    for i, vd in enumerate(data.get("views", ())):
        if not isinstance(vd, dict):
            raise _err(f"target.views[{i}]", "expected an object")
        try:
            views.append(RegView(
                name=vd["name"], bank=vd["bank"], lo=vd.get("lo", 0),
                width=vd["width"], aliases=tuple(vd.get("aliases", ())),
            ))
        except KeyError as exc:
            raise _err(f"target.views[{i}]", f"missing field {exc}") from None
    classes: list[RegClass] = []
    for i, cd in enumerate(data.get("classes", ())):
        if not isinstance(cd, dict):
            raise _err(f"target.classes[{i}]", "expected an object")
        try:
            classes.append(RegClass(
                name=cd["name"], width=cd["width"],
                views=tuple(cd.get("views", ())),
                pairs=tuple(tuple(p) for p in cd.get("pairs", ())),
            ))
        except KeyError as exc:
            raise _err(f"target.classes[{i}]", f"missing field {exc}") from None
    td = TargetDesc(
        name=str(data.get("name", "custom")),
        views=tuple(views),
        classes=tuple(classes),
        callee_saved=tuple(data.get("callee_saved", ())),
        default_class=str(data.get("default_class", "")),
    )
    td.resolved()  # force full validation now
    return td
