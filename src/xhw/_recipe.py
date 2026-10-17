# Copyright 2026 Daniel Camarena
# SPDX-License-Identifier: Apache-2.0

"""Spatial recipes of object tracking: footprint, minimum area, recipe steps and the recipe.

A recipe is an ordered list of steps over a boolean field (time, y, x) that starts as the
``exceedance`` or ``event_membership`` of an XMask. The order is part of the Spec because the
filters do not commute (declared numerical rules). Steps: ``persist`` (per-cell persistence,
the T of Hobday applied to the working field), ``open``/``close`` (binary morphology),
``segment`` (connected patches of each day with a minimum area) and ``track`` (links patches
of consecutive days by overlap). The default recipe is ``exceedance -> segment -> persist``
with 4-neighbour connectivity, a minimum area of 5 % of the domain and the persistence of the
mask's definition.
"""

from __future__ import annotations

from typing import Any, ClassVar, Literal

from ._errors import ValidationError
from ._schema import (
    SpecBase,
    check_choice,
    is_int,
    is_number,
    require,
)
from ._specs import _spec

_LAYERS = ("exceedance", "event_membership")
_OPS = ("persist", "open", "close", "segment", "track")
_JOIN_RULES = ("qualify_then_join", "join_then_qualify")
_AREA_KINDS = ("none", "percentile", "domain_fraction", "absolute_km", "absolute_km2",
               "absolute_km3")
# Scientific contract and invariant.
LINK_MEASURES = ("overlap_smaller", "iou", "source_fraction", "target_fraction")
DOMINANCE_MEASURES = LINK_MEASURES + ("shared",)
TIE_RULES = ("lowest_id", "largest_area")
BRANCH_RULES = ("inherit", "terminate_and_create")
#: Optional track rules; None selects the object tracking behaviour (the first value).
_TRACK_RULES = {"link_measure": LINK_MEASURES, "dominance_measure": DOMINANCE_MEASURES,
                "tie_rule": TIE_RULES, "branch_rule": BRANCH_RULES}


@_spec
class Footprint(SpecBase):
    """Neighbourhood of a cell on the index grid (x on a line; y and x on a surface; z, y and x
    in a volume), so it applies to structured grids of any shape: regular, curvilinear or
    sigma.

    ``kind="connectivity"``: ``connectivity`` 1 (faces: 2 neighbours on a line, 4 on a
    surface, 6 in a volume), 2 (faces and edges: 2 / 8 / 18) or 3 (all: 2 / 8 / 26; the rank
    of ``scipy.ndimage.generate_binary_structure``). ``kind="radius"``: every cell whose
    centre is within ``radius_km`` (great-circle distance, computed per cell, so it is not a
    fixed number of cells; surfaces only). ``kind="offsets"``: an explicit set of ``(dx,)``
    offsets on a line, ``(dy, dx)`` on a surface or ``(dz, dy, dx)`` in a volume, closed under
    negation. The centre is implicit.

    ``vertical_gap_m`` (volumes, connectivity only): physical vertical connectivity.
    Two cells of one column (and, for ranks 2 and 3, of neighbouring columns) are vertical
    neighbours when the layers *between* them add up to at most ``vertical_gap_m`` metres of
    thickness, in both columns: adjacent layers always touch (gap 0), and a gap of thin layers
    is bridged while a thick one is not, whatever the number of levels. ``0`` or ``None`` is the
    index connectivity.
    """

    _minor_added: ClassVar[dict[str, Any]] = {"vertical_gap_m": None}

    kind: Literal["connectivity", "radius", "offsets"]
    connectivity: int | None = None
    radius_km: float | None = None
    offsets: tuple[tuple[int, ...], ...] | None = None
    vertical_gap_m: float | None = None

    def __post_init__(self) -> None:
        check_choice("footprint.kind", self.kind, ("connectivity", "radius", "offsets"))
        chosen = {"connectivity": self.connectivity, "radius": self.radius_km,
                  "offsets": self.offsets}
        for name, value in chosen.items():
            require((value is not None) == (name == self.kind),
                    f"footprint kind {self.kind!r} needs exactly its own field ({name} is "
                    f"{'missing' if name == self.kind else 'not allowed'})")
        if self.kind == "connectivity":
            require(is_int(self.connectivity) and self.connectivity in (1, 2, 3),
                    "connectivity must be 1, 2 or 3")
        elif self.kind == "radius":
            require(is_number(self.radius_km) and self.radius_km > 0, "radius_km must be > 0")
        else:
            require(isinstance(self.offsets, tuple) and len(self.offsets) > 0,
                    "offsets must be a non-empty tuple of (dx,), (dy, dx) or (dz, dy, dx)")
            size = len(self.offsets[0])
            for item in self.offsets:
                require(isinstance(item, tuple) and len(item) == size and size in (1, 2, 3)
                        and all(map(is_int, item)),
                        "offsets entries must be integer tuples of the same length: (dx,), "
                        "(dy, dx) or (dz, dy, dx)")
                require(any(item), "the centre is implicit; do not list it")
            require(len(set(self.offsets)) == len(self.offsets), "offsets must be unique")
            listed = set(self.offsets)
            require(all(tuple(-v for v in item) in listed for item in self.offsets),
                    "offsets must be closed under negation (a symmetric neighbourhood)")
        if self.vertical_gap_m is not None:
            require(self.kind == "connectivity",
                    "vertical_gap_m is only used with kind='connectivity'")
            require(is_number(self.vertical_gap_m) and self.vertical_gap_m >= 0,
                    "vertical_gap_m must be a number >= 0 (metres)")

    @classmethod
    def connected(cls, rank: int = 1, vertical_gap_m: float | None = None) -> Footprint:
        return cls(kind="connectivity", connectivity=rank,
                   vertical_gap_m=None if vertical_gap_m is None else float(vertical_gap_m))

    @property
    def physical_vertical(self) -> bool:
        """True when the vertical neighbours depend on the layer thickness (physical vertical
        connectivity).
        """
        return bool(self.vertical_gap_m)

    @classmethod
    def disc(cls, radius_km: float) -> Footprint:
        return cls(kind="radius", radius_km=float(radius_km))

    @classmethod
    def explicit(cls, offsets) -> Footprint:
        return cls(kind="offsets", offsets=tuple(tuple(int(v) for v in item)
                                                 for item in offsets))


@_spec
class MinArea(SpecBase):
    """Minimum area (length of a line, volume of a volume grid) of a daily patch.

    ``percentile``: discard the ``value`` % smallest patches of the distribution of daily
    patch areas (retrospective: all patches of the analysed period; monitoring: only the
    patches of the baseline reference period, so it never uses later days).
    ``domain_fraction``: discard patches smaller than ``value`` % of the area (volume) of the
    grid's domain, that is, of the cells of its ``domain_mask`` (the ocean for an MHW or MCS
    grid, the land for an AHW grid). ``absolute_km``: discard patches shorter than ``value``
    km (lines). ``absolute_km2``: discard patches smaller than ``value`` km2 (surfaces).
    ``absolute_km3``: discard patches smaller than ``value`` km3 (volumes). ``none``: keep
    all. ``floor_km2`` (percentile only) is an absolute lower bound: the cut is ``max(floor,
    percentile)``.
    """

    kind: Literal["none", "percentile", "domain_fraction", "absolute_km", "absolute_km2",
                  "absolute_km3"]
    value: float | None = None
    floor_km2: float | None = None

    def __post_init__(self) -> None:
        check_choice("min_area.kind", self.kind, _AREA_KINDS)
        if self.kind == "none":
            require(self.value is None, "min_area 'none' takes no value")
        else:
            require(is_number(self.value), f"min_area {self.kind!r} needs a numeric value")
        if self.kind == "percentile":
            require(0.0 <= self.value < 100.0, "percentile value must be in [0, 100)")
        elif self.kind == "domain_fraction":
            require(0.0 < self.value <= 100.0, "domain_fraction value must be in (0, 100] (%)")
        elif self.kind in ("absolute_km", "absolute_km2", "absolute_km3"):
            require(self.value > 0.0, f"{self.kind} value must be > 0")
        if self.floor_km2 is not None:
            require(self.kind == "percentile", "floor_km2 is only used with kind='percentile'")
            require(is_number(self.floor_km2) and self.floor_km2 > 0, "floor_km2 must be > 0")

    @classmethod
    def default(cls) -> MinArea:
        """5 % of the domain area."""
        return cls(kind="domain_fraction", value=5.0)

    @classmethod
    def none(cls) -> MinArea:
        return cls(kind="none")


@_spec
class RecipeStep(SpecBase):
    """One step of a recipe. Fields not used by `op` must be None.

    - ``persist``: ``min_duration_days``, ``max_gap_days``, ``join_rule``; None inherits the
      value of the mask's DefinitionSpec.
    - ``open`` / ``close``: ``footprint``.
    - ``segment``: ``footprint`` (connectivity of the patches) and ``min_area``.
    - ``track``: ``min_overlap`` (threshold of ``link_measure``, 0 = any shared cell),
      ``max_gap_days`` (days an object may disappear and reappear as the same object) and
      ``min_duration_days`` (of the tracked object). Four optional rules are available
      (declared numerical rules); ``None`` is the object tracking behaviour, and passing that
      value explicitly is stored as ``None`` so the recipe keeps its identifier:

    - ``link_measure``: what ``min_overlap`` thresholds. ``overlap_smaller`` (shared / area of
      the smaller object; None), ``iou`` (shared / union), ``source_fraction`` (shared / area
      of the earlier object) or ``target_fraction`` (shared / area of the later object).
    - ``dominance_measure``: in ``monitoring``, which child inherits on a split, which parent
      survives a merge and which candidate bridges a gap (both modes): the largest value of
      one of the measures above (None: ``overlap_smaller``) or of ``shared`` (the shared
      area).
    - ``tie_rule``: equal dominance. ``lowest_id`` (None: the object first in raster order,
      and on a merge the lowest track first) or ``largest_area`` (the larger object first,
      then ``lowest_id``).
    - ``branch_rule``: ``inherit`` (None: the dominant child or parent keeps the identity) or
      ``terminate_and_create`` (no child inherits on a split and a merge opens a new identity;
      a 1 -> 1 continuation keeps it). Monitoring only.
    """

    _nested: ClassVar[dict[str, type]] = {"footprint": Footprint, "min_area": MinArea}
    _minor_added: ClassVar[dict[str, Any]] = {name: None for name in _TRACK_RULES}

    op: Literal["persist", "open", "close", "segment", "track"]
    footprint: Footprint | None = None
    min_area: MinArea | None = None
    min_duration_days: int | None = None
    max_gap_days: int | None = None
    join_rule: Literal["qualify_then_join", "join_then_qualify"] | None = None
    min_overlap: float | None = None
    link_measure: Literal["overlap_smaller", "iou", "source_fraction",
                          "target_fraction"] | None = None
    dominance_measure: Literal["overlap_smaller", "iou", "source_fraction", "target_fraction",
                               "shared"] | None = None
    tie_rule: Literal["lowest_id", "largest_area"] | None = None
    branch_rule: Literal["inherit", "terminate_and_create"] | None = None

    _USED: ClassVar[dict[str, tuple[str, ...]]] = {
        "persist": ("min_duration_days", "max_gap_days", "join_rule"),
        "open": ("footprint",),
        "close": ("footprint",),
        "segment": ("footprint", "min_area"),
        "track": ("min_overlap", "max_gap_days", "min_duration_days", *_TRACK_RULES),
    }
    _REQUIRED: ClassVar[dict[str, tuple[str, ...]]] = {
        "persist": (), "open": ("footprint",), "close": ("footprint",),
        "segment": ("footprint", "min_area"),
        "track": ("min_overlap", "max_gap_days", "min_duration_days"),
    }

    def __post_init__(self) -> None:
        check_choice("op", self.op, _OPS)
        for name, values in _TRACK_RULES.items():
            value = getattr(self, name)
            if value is not None:
                check_choice(name, value, values)
                # the object tracking default is stored as None
                if value == values[0]:
                    object.__setattr__(self, name, None)
        fields = ("footprint", "min_area", "min_duration_days", "max_gap_days", "join_rule",
                  "min_overlap", *_TRACK_RULES)
        for name in fields:
            value = getattr(self, name)
            if name not in self._USED[self.op]:
                require(value is None, f"step {self.op!r} does not use {name}")
            elif name in self._REQUIRED[self.op]:
                require(value is not None, f"step {self.op!r} needs {name}")
        if self.footprint is not None:
            require(isinstance(self.footprint, Footprint), "footprint must be Footprint")
        if self.min_area is not None:
            require(isinstance(self.min_area, MinArea), "min_area must be MinArea")
        if self.min_duration_days is not None:
            require(is_int(self.min_duration_days) and self.min_duration_days >= 1,
                    "min_duration_days must be an integer >= 1")
        if self.max_gap_days is not None:
            require(is_int(self.max_gap_days) and self.max_gap_days >= 0,
                    "max_gap_days must be an integer >= 0")
        if self.join_rule is not None:
            check_choice("join_rule", self.join_rule, _JOIN_RULES)
        if self.min_overlap is not None:
            require(is_number(self.min_overlap) and 0.0 <= self.min_overlap <= 1.0,
                    "min_overlap must be in [0, 1]")

    # -- constructors (the Spec itself stays explicit) -----------------------------------

    @classmethod
    def persist(cls, *, min_duration_days: int | None = None, max_gap_days: int | None = None,
                join_rule: str | None = None) -> RecipeStep:
        return cls(op="persist", min_duration_days=min_duration_days, max_gap_days=max_gap_days,
                   join_rule=join_rule)

    @classmethod
    def morph(cls, op: str, footprint: Footprint) -> RecipeStep:
        require(op in ("open", "close"), "morph op must be 'open' or 'close'")
        return cls(op=op, footprint=footprint)

    @classmethod
    def segment(cls, *, footprint: Footprint | None = None,
                min_area: MinArea | None = None) -> RecipeStep:
        return cls(op="segment", footprint=footprint or Footprint.connected(1),
                   min_area=min_area or MinArea.default())

    @classmethod
    def track(cls, *, min_overlap: float = 0.0, max_gap_days: int = 0,
              min_duration_days: int = 1, link_measure: str | None = None,
              dominance_measure: str | None = None, tie_rule: str | None = None,
              branch_rule: str | None = None) -> RecipeStep:
        return cls(op="track", min_overlap=min_overlap, max_gap_days=max_gap_days,
                   min_duration_days=min_duration_days, link_measure=link_measure,
                   dominance_measure=dominance_measure, tie_rule=tie_rule,
                   branch_rule=branch_rule)

    def rule(self, name: str) -> str:
        """Effective value of a track rule (``link_measure``, ``dominance_measure``, ``tie_rule``
        or ``branch_rule``): the stored value or the object tracking default.
        """
        require(name in _TRACK_RULES, f"unknown track rule {name!r}")
        value = getattr(self, name)
        return _TRACK_RULES[name][0] if value is None else value


@_spec
class SpatialRecipe(SpecBase):
    """Input layer and ordered steps. Use `SpatialRecipe.default` for the default."""

    _nested: ClassVar[dict[str, type]] = {"steps": RecipeStep}

    input_layer: Literal["exceedance", "event_membership"]
    steps: tuple[RecipeStep, ...]

    def __post_init__(self) -> None:
        check_choice("input_layer", self.input_layer, _LAYERS)
        require(isinstance(self.steps, tuple) and len(self.steps) > 0,
                "a recipe needs at least one step")
        require(all(isinstance(s, RecipeStep) for s in self.steps), "steps must be RecipeStep")
        ops = [s.op for s in self.steps]
        require(ops.count("track") <= 1, "a recipe can hold at most one track step")
        if "track" in ops:
            require(ops[-1] == "track", "the track step must be the last step")
            require(len(ops) >= 2 and ops[-2] == "segment",
                    "the track step must directly follow a segment step")
        if not any(op in ("segment", "persist", "open", "close") for op in ops):
            raise ValidationError("a recipe needs a segment, persist, open or close step")

    @classmethod
    def default(cls) -> SpatialRecipe:
        """``exceedance -> segment -> persist``: Hobday's idea (a spatial patch, then
        persistence) carried to space-time. Segment: 4 neighbours, minimum area 5 % of the
        domain (the ocean for an MHW or MCS grid, the land for an AHW grid: the grid's
        ``domain_mask``). Persist: the minimum duration and gap of the mask's DefinitionSpec,
        applied per cell to the cells that survive the area filter.
        """
        return cls(input_layer="exceedance",
                   steps=(RecipeStep.segment(), RecipeStep.persist()))

    @property
    def has_segment(self) -> bool:
        return any(s.op == "segment" for s in self.steps)

    @property
    def has_track(self) -> bool:
        return self.steps[-1].op == "track"
