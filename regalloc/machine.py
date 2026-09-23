"""Physical register file with width views, aliases and register classes.

A :class:`Machine` is the v2 hardware contract:

* **views** — named bit ranges of physical register groups.  Two views of
  the same group whose bit ranges overlap are *aliases* (e.g. a 32-bit
  ``w0`` covering bits 0..31 and a 64-bit ``x0`` covering bits 0..63 of
  group 0).  Two values may only coexist in views that share **no** bits;
  comparing view *names* is never enough.
* **aliases** — given explicitly by the caller and validated to be exactly
  the bit-overlap relation (symmetric, irreflexive).  Nothing is inferred
  silently and nothing is silently corrected.
* **classes** — every view belongs to a register class (``cls``); all views
  of a class have the same bit width.  A program variable is allocated a
  view from its own class only.
* **callee-saved** — a call (``Instr.is_call``) destroys every view that is
  not callee-saved.  The callee-saved set must be alias-closed: preserving
  64 bits while destroying a contained 32-bit half is contradictory and
  rejected.
* **stack slots** — a spilled variable of class ``c`` occupies
  ``width(c)/8`` bytes at an offset aligned to ``class_align[c]``.

Everything is deterministic: iteration order is the declaration order of
``views``; ties are broken by view name.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .model import ProgramFormatError, check_int


def _err(path: str, msg: str) -> ProgramFormatError:
    return ProgramFormatError(f"{path}: {msg}" if path else msg)


@dataclass(frozen=True)
class RegView:
    """One named bit range ``[lo, hi)`` of a physical register group."""

    name: str
    group: int
    lo: int
    hi: int
    cls: str
    callee_saved: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise _err("view.name", "view name must be a non-empty string")
        check_int(self.group, f"view[{self.name}].group", minimum=0)
        check_int(self.lo, f"view[{self.name}].lo", minimum=0)
        check_int(self.hi, f"view[{self.name}].hi", minimum=0)
        if self.hi <= self.lo:
            raise _err(
                f"view[{self.name}]", f"empty bit range [{self.lo}, {self.hi})"
            )
        if (self.hi - self.lo) % 8 != 0:
            raise _err(
                f"view[{self.name}]",
                "view width must be a whole number of bytes",
            )
        if not isinstance(self.cls, str) or not self.cls:
            raise _err(f"view[{self.name}].cls", "class must be a non-empty string")
        if not isinstance(self.callee_saved, bool):
            raise _err(f"view[{self.name}].callee_saved", "expected bool")

    @property
    def width(self) -> int:
        """Width in bits."""
        return self.hi - self.lo

    @property
    def size_bytes(self) -> int:
        return (self.hi - self.lo) // 8


@dataclass(frozen=True)
class Machine:
    """A register file: views + explicit alias sets + class alignments.

    ``aliases`` maps each view name to the tuple of *other* views that share
    at least one bit with it.  The mapping must be symmetric and must match
    the bit-overlap relation computed from the declared ranges exactly.
    """

    views: tuple[RegView, ...] = ()
    aliases: dict[str, tuple[str, ...]] = field(default_factory=dict)
    class_align: dict[str, int] = field(default_factory=dict)
    stack_align: int = 8
    name: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.views, (list, tuple)) or not self.views:
            raise _err("machine.views", "expected a non-empty list/tuple of RegView")
        views = tuple(self.views)
        for i, v in enumerate(views):
            if not isinstance(v, RegView):
                raise _err(f"machine.views[{i}]", "expected RegView")
        names = [v.name for v in views]
        if len(set(names)) != len(names):
            raise _err("machine.views", "duplicate view names")
        by_name = {v.name: v for v in views}

        check_int(self.stack_align, "machine.stack_align", minimum=1)
        if not isinstance(self.name, str):
            raise _err("machine.name", "name must be a string")

        # ---- classes: uniform width inside a class ----------------------
        classes: dict[str, list[str]] = {}
        class_width: dict[str, int] = {}
        for v in views:
            if v.cls in class_width and class_width[v.cls] != v.width:
                raise _err(
                    "machine.views",
                    f"class {v.cls!r} mixes widths "
                    f"{class_width[v.cls]} and {v.width}",
                )
            class_width[v.cls] = v.width
            classes.setdefault(v.cls, []).append(v.name)

        if not isinstance(self.class_align, dict):
            raise _err("machine.class_align", "expected a dict {class: align}")
        norm_align: dict[str, int] = {}
        for cls, al in self.class_align.items():
            if not isinstance(cls, str) or not cls:
                raise _err("machine.class_align", "class names must be strings")
            norm_align[cls] = check_int(al, f"machine.class_align[{cls}]", minimum=1)
        missing = sorted(set(classes) - set(norm_align))
        if missing:
            raise _err(
                "machine.class_align",
                f"missing alignment for class(es): {missing}",
            )

        # ---- explicit alias sets, validated against bit overlap ---------
        if not isinstance(self.aliases, dict):
            raise _err("machine.aliases", "expected a dict {view: [view, ...]}")
        norm_alias: dict[str, tuple[str, ...]] = {}
        for key, vals in self.aliases.items():
            if key not in by_name:
                raise _err("machine.aliases", f"unknown view {key!r}")
            if not isinstance(vals, (list, tuple)):
                raise _err(f"machine.aliases[{key}]", "expected a list/tuple")
            seen: list[str] = []
            for u in vals:
                if u not in by_name:
                    raise _err(f"machine.aliases[{key}]", f"unknown view {u!r}")
                if u == key:
                    raise _err(f"machine.aliases[{key}]", "a view cannot alias itself")
                if u in seen:
                    raise _err(f"machine.aliases[{key}]", f"duplicate alias {u!r}")
                seen.append(u)
            norm_alias[key] = tuple(seen)
        for v in views:
            norm_alias.setdefault(v.name, ())
        # symmetry
        for a, vals in norm_alias.items():
            for b in vals:
                if a not in norm_alias[b]:
                    raise _err(
                        "machine.aliases",
                        f"alias relation not symmetric: {a!r} <-> {b!r}",
                    )
        # must equal the bit-overlap relation exactly
        for a in views:
            want = {
                b.name
                for b in views
                if b.name != a.name
                and b.group == a.group
                and b.lo < a.hi
                and a.lo < b.hi
            }
            got = set(norm_alias[a.name])
            if want != got:
                raise _err(
                    "machine.aliases",
                    f"declared aliases of {a.name!r} {sorted(got)} do not match "
                    f"the bit-overlap relation {sorted(want)}",
                )

        # ---- callee-saved must be alias-closed --------------------------
        for v in views:
            if v.callee_saved:
                for u in norm_alias[v.name]:
                    if not by_name[u].callee_saved:
                        raise _err(
                            "machine.views",
                            f"callee-saved view {v.name!r} aliases caller-saved "
                            f"view {u!r}; the callee-saved set must be alias-closed",
                        )

        closure = {
            v.name: frozenset({v.name, *norm_alias[v.name]}) for v in views
        }
        default_class = min(
            sorted(classes), key=lambda c: (class_width[c], c)
        )
        object.__setattr__(self, "views", views)
        object.__setattr__(self, "aliases", norm_alias)
        object.__setattr__(self, "class_align", norm_align)
        object.__setattr__(self, "classes", {c: tuple(vs) for c, vs in classes.items()})
        object.__setattr__(self, "class_width", dict(class_width))
        object.__setattr__(self, "view_map", by_name)
        object.__setattr__(self, "alias_closure", closure)
        object.__setattr__(self, "default_class", default_class)

    # ------------------------------------------------------------------
    def view(self, name: str) -> RegView:
        return self.view_map[name]

    def views_overlap(self, a: str, b: str) -> bool:
        """True iff views ``a`` and ``b`` share at least one bit."""
        return b in self.alias_closure[a]

    def clobbered_by_call(self, extra: tuple[str, ...] = ()) -> tuple[str, ...]:
        """Views destroyed by a call, in machine order.

        Every non-callee-saved view plus any explicitly listed extras.
        """
        out = [v.name for v in self.views if not v.callee_saved]
        for e in extra:
            if e not in out:
                out.append(e)
        return tuple(out)

    def class_of(self, var: int, var_classes: dict[int, str] | None) -> str:
        """Register class of a program variable (documented default)."""
        if var_classes and var in var_classes:
            return var_classes[var]
        return self.default_class


def flat_machine(k: int) -> Machine:
    """Compatibility adapter: a machine equivalent to legacy ``K`` registers.

    Builds ``k`` independent 32-bit views ``r0..r{k-1}`` (one group each,
    class ``"gpr"``, no aliases, nothing callee-saved).  Together with
    ``var_classes`` mapping every variable to ``"gpr"`` this reproduces the
    legacy v1 setting inside the v2 pipeline (see ``regalloc.compat``).
    """
    check_int(k, "K", minimum=1)
    views = tuple(
        RegView(name=f"r{i}", group=i, lo=0, hi=32, cls="gpr") for i in range(k)
    )
    return Machine(
        views=views,
        aliases={},
        class_align={"gpr": 4},
        stack_align=4,
        name=f"flat-{k}",
    )


def machine_from_dict(data: Any) -> Machine:
    """Build a :class:`Machine` from plain dicts/lists (JSON shape)."""
    if not isinstance(data, dict):
        raise _err("machine", "expected a mapping/object")
    views = []
    for i, vd in enumerate(data.get("views", [])):
        vp = f"machine.views[{i}]"
        if not isinstance(vd, dict):
            raise _err(vp, "expected an object")
        if not all(key in vd for key in ("name", "group", "lo", "hi", "cls")):
            raise _err(vp, "missing one of name/group/lo/hi/cls")
        try:
            views.append(
                RegView(
                    name=vd["name"],
                    group=vd["group"],
                    lo=vd["lo"],
                    hi=vd["hi"],
                    cls=vd["cls"],
                    callee_saved=vd.get("callee_saved", False),
                )
            )
        except ProgramFormatError as exc:
            raise ProgramFormatError(f"{vp}: {exc}") from None
    return Machine(
        views=tuple(views),
        aliases={k: tuple(v) for k, v in data.get("aliases", {}).items()},
        class_align=dict(data.get("class_align", {})),
        stack_align=data.get("stack_align", 8),
        name=str(data.get("name", "")),
    )
