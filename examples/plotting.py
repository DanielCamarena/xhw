# Copyright 2026 Daniel Camarena
# SPDX-License-Identifier: Apache-2.0

"""Matplotlib recipes for xHW products, used by the tutorials and the plotting guide.

xHW itself has no plotting API: its products are xarray datasets. These functions are
example code, not part of the installed package. Copy them, or run from ``examples/``::

    import plotting
    plotting.frequency_figure({"MHW": mask}, Path("figures"))

Each function receives xHW products (XMask, ObjectField, LineageGraph), draws a figure
and saves it as ``<output_dir>/<name>.png``. See docs/PLOTTING.md for every figure,
the call that produced it and how to read it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

import xhw

__all__ = [
    "connectivity_figure",
    "frequency_figure",
    "genealogy_figure",
    "largest_event",
    "map_panel",
    "mask_figure",
    "metrics_figure",
    "pyplot",
    "save_figure",
    "sequence_figure",
    "series_figure",
]


def pyplot():
    """Import Matplotlib's pyplot with the tutorial style; raise a clear error if absent."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("Figures need Matplotlib: install examples/requirements.txt.") from exc
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )
    return plt


def save_figure(fig, output_dir: Path, name: str):
    """Apply concise date ticks to date axes and save ``<output_dir>/<name>.png`` (150 dpi)."""
    from matplotlib.dates import AutoDateLocator, ConciseDateFormatter

    for ax in fig.axes:
        if ax.get_xlabel() in ("Date", "Local date (offset:-05:00)") or ax.get_xlabel().startswith(
            "Date;"
        ):
            locator = AutoDateLocator(minticks=3, maxticks=5)
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(ConciseDateFormatter(locator))
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{name}.png", dpi=150, bbox_inches="tight")
    return fig


def series_figure(
    raw: xr.Dataset,
    mask,
    *,
    variable: str,
    selection: dict,
    period: tuple[str, str],
    output_dir: Path,
    name: str = "temperature",
):
    """One cell's daily series: temperature, reference, threshold and the XMask flags.

    `raw` is the dataset before interpolation (missing days stay blank), `mask` an XMask,
    `selection` an ``isel`` mapping that picks one cell (e.g. ``{"lat": 0, "lon": 0}``)
    and `period` a ``(start, end)`` date pair. Red dots are exceedances, diamonds are
    interpolated (``filled``) values; the lower panel shows exceedance, membership and
    joined-gap days. Units are those of the mask's temperature.
    """
    plt = pyplot()
    point = mask.dataset.sel(time=slice(*period)).isel(selection)
    observed = raw[variable].sel(time=slice(*period)).isel(selection)
    fig, (ax, flags) = plt.subplots(
        2,
        1,
        figsize=(10, 5),
        sharex=True,
        gridspec_kw={"height_ratios": (3, 1)},
        layout="constrained",
    )
    times = point.time.values
    ax.plot(times, observed, "o-", color="#28364a", label="Observed temperature", markersize=3)
    ax.plot(times, point.reference, color="#167c80", label="Reference")
    ax.plot(times, point.threshold, "--", color="#c58220", label="Threshold")
    ax.scatter(
        times[point.exceedance.values],
        point.temperature.values[point.exceedance.values],
        color="#d94b48",
        s=18,
        label="Exceedance",
        zorder=4,
    )
    if "filled" in point:
        use = point.filled.values
        ax.scatter(
            times[use],
            point.temperature.values[use],
            marker="D",
            color="#7954a1",
            label="Filled (not used in baseline)",
            zorder=5,
        )
    ax.set(
        ylabel=f"Temperature ({point.temperature.attrs['units']})",
        title=f"{mask.definition.phenomenon.upper()}: P{mask.definition.percentile:g}, "
        f"minimum {mask.definition.min_duration_days} days, gaps ≤ "
        f"{mask.definition.max_gap_days} days",
    )
    ax.legend(loc="best", fontsize=8, ncol=2)
    for row, (var, color) in enumerate(
        (("exceedance", "#d94b48"), ("event_membership", "#167c80"), ("joined_gap", "#c58220"))
    ):
        flags.scatter(
            times, np.full(len(times), row), c=np.where(point[var], color, "#e2e6ea"), s=26
        )
    flags.set(
        yticks=(0, 1, 2),
        yticklabels=("Exceedance", "Membership", "Joined gap"),
        ylim=(-0.6, 2.6),
        xlabel="Date; grey marks false, blank temperature marks missing",
    )
    return save_figure(fig, output_dir, name)


def select_days(data, dates, *, time_dim="time"):
    """Select unique daily records by calendar date, preserving their stored clock.

    A scalar drops the time dimension; a sequence retains it in the requested order.
    Missing or ambiguous dates are rejected instead of choosing a nearest observation.
    """
    scalar = isinstance(dates, (str, np.datetime64))
    requested = [dates] if scalar else list(dates)
    stored = data[time_dim].values.astype("datetime64[D]")
    positions = []
    for date in requested:
        matches = np.flatnonzero(stored == np.datetime64(date, "D"))
        if not len(matches):
            raise ValueError(f"date {date!r} is absent from the product")
        if len(matches) != 1:
            raise ValueError(f"date {date!r} has multiple records; supply daily products")
        positions.append(int(matches[0]))
    return data.isel({time_dim: positions[0] if scalar else positions})


def _surface(da: xr.DataArray, grid=None) -> xr.DataArray:
    if grid is not None and grid.support == "volume":
        return da.isel({grid.spatial_dims[0]: 0}, drop=True)
    return da.isel(depth=0, drop=True) if "depth" in da.dims and da.ndim == 3 else da


def map_panel(ax, da: xr.DataArray, *, title: str, categorical: bool = False,
              grid=None, **kwargs):
    """Draw a 2-D surface using declared coordinates, or honestly labelled index axes.

    ``grid`` selects auxiliary/curvilinear coordinates and the first retained volume
    level. Without a grid, conventional lat/lon names are used when present.
    Explicit x/y plotting keywords take precedence. Singleton spatial axes are retained.
    """
    da = _surface(da, grid)
    if da.ndim != 2:
        raise ValueError("map_panel needs two spatial dimensions; select a surface first")
    lat = (
        grid.coords.lat if grid else next((n for n in ("lat", "latitude") if n in da.coords), None)
    )
    lon = (
        grid.coords.lon if grid else next((n for n in ("lon", "longitude") if n in da.coords), None)
    )
    geographic = lat in da.coords and lon in da.coords if lat and lon else False
    x = kwargs.pop("x", lon if geographic else da.dims[-1])
    y = kwargs.pop("y", lat if geographic else da.dims[-2])
    kwargs.setdefault("cmap", "tab20" if categorical else "viridis")
    da.plot.pcolormesh(ax=ax, x=x, y=y, **kwargs)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Longitude (degrees east)" if x == lon else f"{x} (index)")
    ax.set_ylabel("Latitude (degrees north)" if y == lat else f"{y} (index)")
    return ax


def sequence_figure(objects, dates: list[str], output_dir: Path, name: str = "objects"):
    """Daily patch labels (top) and retrospective track IDs (bottom) for selected `dates`.

    `objects` is an ObjectField built with a track step. Each patch is annotated with
    its number: daily labels restart every day, while a track ID follows overlapping
    patches across days.
    """
    if not dates:
        raise ValueError("sequence_figure needs at least one date")
    ds = objects.dataset
    if "track_id" not in ds:
        raise ValueError("sequence_figure needs objects built with a track step")
    for date in dates:
        select_days(ds, date, time_dim=objects.grid.time_dim)
    plt = pyplot()
    fig, axes = plt.subplots(
        2, len(dates), figsize=(3 * len(dates), 5.5), squeeze=False, layout="constrained"
    )
    ds = objects.dataset
    # One shared range per representation, so identical colors denote identical numbers.
    for col, date in enumerate(dates):
        for row, var in enumerate(("label", "track_id")):
            da = select_days(ds[var], date, time_dim=objects.grid.time_dim)
            map_panel(
                axes[row, col],
                da.where(da > 0),
                categorical=True,
                grid=objects.grid,
                title=f"{date}\n{'Daily label' if row == 0 else 'Retrospective track ID'}",
                vmin=1,
                vmax=max(2, int(ds[var].max())),
                add_colorbar=False,
            )
            surface = _surface(da, objects.grid)
            lat_name, lon_name = objects.grid.coords.lat, objects.grid.coords.lon
            lat, lon = xr.broadcast(surface[lat_name], surface[lon_name])
            lat, lon = lat.transpose(*surface.dims), lon.transpose(*surface.dims)
            for identifier in np.unique(surface.values):
                if identifier > 0:
                    rows, cols = np.nonzero(surface.values == identifier)
                    axes[row, col].text(
                        float(lon.values[rows, cols].mean()),
                        float(lat.values[rows, cols].mean()),
                        str(int(identifier)),
                        ha="center",
                        va="center",
                        fontsize=7,
                        bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "pad": 1},
                    )
    fig.suptitle("Daily labels restart each day; tracking links patches across days", fontsize=12)
    return save_figure(fig, output_dir, name)


def mask_figure(mask, processed, date: str, output_dir: Path):
    """Original ``event_membership`` versus the spatially processed mask on one `date`.

    `processed` is the ObjectField of a morphology recipe; its ``added`` and ``removed``
    layers show what the recipe changed. The original evidence is not modified.
    """
    plt = pyplot()
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.5), layout="constrained")
    layers = (
        (mask.dataset["event_membership"], "Original temporal evidence"),
        (processed.dataset["processed_mask"], "Spatially processed mask"),
        (processed.dataset["added"], "Added cells"),
        (processed.dataset["removed"], "Removed cells"),
    )
    for ax, (da, title) in zip(axes, layers, strict=True):
        map_panel(
            ax,
            select_days(da, date, time_dim=mask.grid.time_dim).astype(int),
            title=title,
            grid=mask.grid,
            vmin=0,
            vmax=1,
            add_colorbar=False,
        )
    fig.suptitle(f"{date}: binary closing; yellow = true, purple = false", fontsize=12)
    return save_figure(fig, output_dir, "morphology")


def connectivity_figure(mask, date: str, output_dir: Path):
    """Segment one `date` of `mask` with rank-1 and rank-2 connectivity, side by side.

    Rank 1 joins face neighbours only; rank 2 also joins diagonal neighbours.
    """
    plt = pyplot()
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5), layout="constrained")
    for rank, ax in zip((1, 2), axes, strict=True):
        recipe = xhw.SpatialRecipe(
            input_layer="event_membership",
            steps=(
                xhw.RecipeStep.segment(
                    footprint=xhw.Footprint.connected(rank), min_area=xhw.MinArea.none()
                ),
            ),
        )
        patches = xhw.build_objects(mask, recipe=recipe)
        patches.check_against(mask)
        day = select_days(patches.dataset.label, date, time_dim=mask.grid.time_dim)
        map_panel(
            ax,
            day.where(day > 0),
            title=f"Rank {rank}: {int(day.max())} patches",
            grid=mask.grid,
            categorical=True,
            add_colorbar=False,
        )
    fig.suptitle(f"{date}: face neighbours versus faces + diagonals")
    return save_figure(fig, output_dir, "connectivity")


def genealogy_figure(graph, event: int, output_dir: Path, name: str = "genealogy"):
    """Branch diagram of one spacetime `event` of a LineageGraph.

    Each bar is a branch (a chain of one-to-one links); arrows are branch links, so two
    outgoing arrows mark a split and two incoming ones a merge. At most 18 branches are
    drawn, and the title says so when more exist. Arrows show overlap, not causation.
    """
    plt = pyplot()
    nodes = graph.members(spacetime_event=event)
    # Branch DAG is more readable than hundreds of daily nodes for a real event.
    branch_ids = np.unique(graph.dataset["object_branch"].values[nodes])
    shown = branch_ids[:18]
    ds = graph.dataset
    fig, ax = plt.subplots(figsize=(11, max(3.5, len(shown) * 0.27)), layout="constrained")
    y = {int(b): i for i, b in enumerate(shown)}
    starts, ends = {}, {}
    for b in shown:
        idx = int(np.flatnonzero(ds.branch_number.values == b)[0])
        starts[int(b)] = ds.branch_start.values[idx]
        ends[int(b)] = ds.branch_end.values[idx]
        ax.plot(
            [starts[int(b)], ends[int(b)]],
            [y[int(b)]] * 2,
            "o-",
            linewidth=4,
            markersize=4,
            color="#167c80",
        )
        ax.annotate(
            f"B{b}",
            (starts[int(b)], y[int(b)]),
            xytext=(0, 6),
            textcoords="offset points",
            fontsize=8,
        )
    for a, b in zip(ds.branch_link_parent.values, ds.branch_link_child.values, strict=True):
        if int(a) in y and int(b) in y:
            ax.annotate(
                "",
                xy=(starts[int(b)], y[int(b)]),
                xytext=(ends[int(a)], y[int(a)]),
                arrowprops={"arrowstyle": "->", "color": "#c58220", "lw": 1.3},
            )
    ax.set(
        yticks=[],
        xlabel="Date",
        title=f"Spacetime event {event}: branch links "
        "(two outgoing arrows = split; two incoming = merge)",
    )
    if len(shown) < len(branch_ids):
        ax.set_title(ax.get_title() + f"\nFirst {len(shown)} of {len(branch_ids)} branches only")
    return save_figure(fig, output_dir, name)


def largest_event(graph) -> int:
    """Number of the spacetime event with the largest size-time (``event_area_days``)."""
    ds = graph.dataset
    idx = int(np.argmax(ds["event_area_days"].values))
    return int(ds.event_number.values[idx])


def metrics_figure(graph, objects, event: int, output_dir: Path):
    """Occupied size, signed intensity and occupied days per cell for one spacetime `event`.

    Uses ``graph.series`` and ``graph.maps(objects, ...)``. Sizes are shown in km² for
    surfaces and km³ for volumes. Intensity panels need a graph built with
    ``field=mask.intensity()``. The map counts occupied days, not an annual frequency.
    """
    plt = pyplot()
    series = graph.series(spacetime_event=event)
    maps = graph.maps(objects, spacetime_event=event)
    volume = series.area.attrs["units"] == "m3"
    divisor, unit = (1e9, "km³") if volume else (1e6, "km²")
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.7), layout="constrained")
    axes[0].plot(series.time, series.area / divisor, color="#167c80")
    axes[0].set(ylabel=f"Occupied size ({unit})", xlabel="Date", title=f"Event {event}")
    if "intensity_mean" in series:
        axes[1].plot(series.time, series.intensity_mean, color="#d94b48", label="Weighted mean")
        axes[1].plot(series.time, series.intensity_max, "--", color="#c58220", label="Maximum")
        axes[1].legend(fontsize=8)
    axes[1].set(ylabel="Signed intensity relative to reference (K)", xlabel="Date")
    map_panel(
        axes[2],
        maps.frequency.where(objects.dataset[objects.grid.domain_mask]),
        grid=objects.grid,
        title="Occupied days per cell" + (" (first retained level)" if volume else ""),
        cbar_kwargs={"label": "Days"},
    )
    return save_figure(fig, output_dir, "metrics")


def frequency_figure(masks: dict, output_dir: Path, name: str = "frequency"):
    """Percentage of valid days in an event, per cell, for each mask in `masks`.

    `masks` maps a panel title to an XMask. Numerator and denominator are restricted to
    ``valid`` cell-days, so invalid support never appears as a zero frequency.
    """
    plt = pyplot()
    fig, axes = plt.subplots(1, len(masks), figsize=(4 * len(masks), 4), layout="constrained")
    for ax, (label, mask) in zip(np.atleast_1d(axes), masks.items(), strict=True):
        valid = mask.dataset.valid.sum("time")
        occupied_valid = mask.dataset.event_membership & mask.dataset.valid
        pct = 100 * occupied_valid.sum("time") / valid.where(valid > 0)
        map_panel(ax, pct, grid=mask.grid, title=label, cbar_kwargs={"label": "% of valid days"})
    return save_figure(fig, output_dir, name)
