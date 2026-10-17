# Copyright 2026 Daniel Camarena
# SPDX-License-Identifier: Apache-2.0

"""Optional, local teaching helpers: animations and exploration of computed products.

Nothing here runs a baseline, detector, segmenter or tracker. Imported separately from
the installed xhw API. Requirements: examples/requirements-interactive.txt.
"""

from pathlib import Path

import numpy as np
import xarray as xr
from plotting import largest_event, map_panel, pyplot, select_days


def _plotly():
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError as exc:
        raise ImportError("Install examples/requirements-interactive.txt for exploration") from exc
    return go, make_subplots


def event_window(graph, event=None, max_days=60):
    """At most max_days consecutive dates around the selected event's size peak."""
    if max_days < 1:
        raise ValueError("max_days must be positive")
    event = largest_event(graph) if event is None else event
    ds = graph.dataset
    i = int(np.flatnonzero(ds.event_number.values == event)[0])
    first = ds.event_start.values[i].astype("datetime64[D]")
    last = ds.event_end.values[i].astype("datetime64[D]")
    peak = ds.event_peak_area_day.values[i].astype("datetime64[D]")
    start = max(first, peak - np.timedelta64(max_days // 2, "D"))
    end = min(last, start + np.timedelta64(max_days - 1, "D"))
    start = max(first, end - np.timedelta64(max_days - 1, "D"))
    return np.arange(start, end + np.timedelta64(1, "D")).astype("datetime64[ns]")


def _geography(da, grid):
    lat, lon = xr.broadcast(da[grid.coords.lat], da[grid.coords.lon])
    return lon.transpose(*da.dims).values, lat.transpose(*da.dims).values


def _event_labels(objects, graph, event, date, branch=None):
    ids = (
        graph.members(branch=branch) if branch is not None else graph.members(spacetime_event=event)
    )
    ds = graph.dataset
    ids = ids[ds.object_time.values[ids].astype("datetime64[D]") == np.datetime64(date, "D")]
    labels = select_days(objects.dataset.label, date, time_dim=objects.grid.time_dim)
    return labels.where(labels.isin(ds.object_label.values[ids]), 0)


def exploration_figure(
    mask,
    objects=None,
    graph=None,
    *,
    date=None,
    layer="temperature",
    depth_index=0,
    event=None,
    branch=None,
    cell=(0, 0),
):
    """Linked map, cell evidence, event size/intensity and branch diagram (no recomputation).

    A graph is optional: temporal-only products still expose maps and cell evidence.
    Graph selectors apply to patch/track displays; original thermal evidence stays visible.
    """
    go, make_subplots = _plotly()
    g, ds = mask.grid, mask.dataset
    if len(g.spatial_dims) < 2:
        raise ValueError("The explorer needs a surface or volume; use xarray for line products")
    date = str(ds[g.time_dim].values[-1])[:10] if date is None else str(date)[:10]
    event = largest_event(graph) if event is None and graph is not None else event
    if layer in ("label", "track_id", "processed_mask", "added", "removed"):
        if objects is None:
            raise ValueError("This layer requires an ObjectField")
        da = select_days(objects.dataset[layer], date, time_dim=g.time_dim)
        if graph is not None and layer in ("label", "track_id"):
            use = _event_labels(objects, graph, event, date, branch)
            da = da.where(use > 0)
    elif layer == "intensity":
        da = select_days(mask.intensity(), date, time_dim=g.time_dim)
    else:
        da = select_days(ds[layer], date, time_dim=g.time_dim)
    if "component" in da.dims:
        da = da.isel(component=0)
    selection = {g.spatial_dims[-2]: cell[0], g.spatial_dims[-1]: cell[1]}
    if g.support == "volume":
        da = da.isel({g.spatial_dims[0]: depth_index})
        selection[g.spatial_dims[0]] = depth_index
    da = da.transpose(*g.spatial_dims[-2:])
    x, y = _geography(da, g)
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            f"{date}: {layer}",
            f"Cell {cell}: temperature and thresholds",
            "Event size and intensity",
            "Branch overlap links",
        ),
        specs=[[{}, {}], [{"secondary_y": True}, {}]],
    )
    categorical = layer in ("label", "track_id")
    units = da.attrs.get("units", "1")
    if da[g.coords.lat].ndim == da[g.coords.lon].ndim == 1:
        fig.add_trace(
            go.Heatmap(
                x=x[0],
                y=y[:, 0],
                z=da.values.astype(float),
                colorscale="Turbo" if categorical else "Viridis",
                colorbar={"title": units, "len": 0.4, "y": 0.8},
            ),
            row=1,
            col=1,
        )
    else:
        fig.add_trace(
            go.Scattergl(
                x=x.ravel(),
                y=y.ravel(),
                mode="markers",
                marker={
                    "color": da.values.astype(float).ravel(),
                    "colorscale": "Viridis",
                    "showscale": True,
                },
                name="Retained cell centres",
            ),
            row=1,
            col=1,
        )
    point = ds.isel(selection)
    if "component" in point.dims:
        point = point.isel(component=0)
    selected_day = np.datetime64(date, "D")
    first = str(selected_day - np.timedelta64(30, "D"))
    last = str(selected_day + np.timedelta64(29, "D"))
    point = point.sel({g.time_dim: slice(first, last)})
    for var in ("temperature", "reference", "threshold"):
        fig.add_trace(
            go.Scatter(x=point[g.time_dim].values, y=point[var].values, name=var, mode="lines"),
            row=1,
            col=2,
        )
    for var, symbol in [
        ("exceedance", "circle"),
        ("event_membership", "diamond"),
        ("joined_gap", "x"),
    ]:
        on = point[var].values.astype(bool)
        fig.add_trace(
            go.Scatter(
                x=point[g.time_dim].values[on],
                y=point.temperature.values[on],
                name=var,
                mode="markers",
                marker={"symbol": symbol},
            ),
            row=1,
            col=2,
        )
    if graph is not None:
        kw = {"branch": branch} if branch is not None else {"spacetime_event": event}
        series = graph.series(**kw)
        fig.add_trace(
            go.Scatter(x=series.time, y=series.area, name="Size"), row=2, col=1, secondary_y=False
        )
        if "intensity_mean" in series:
            fig.add_trace(
                go.Scatter(x=series.time, y=series.intensity_mean, name="Signed mean intensity"),
                row=2,
                col=1,
                secondary_y=True,
            )
        fig.update_yaxes(title_text=series.area.attrs["units"], row=2, col=1, secondary_y=False)
        if "intensity_mean" in series:
            fig.update_yaxes(
                title_text=series.intensity_mean.attrs.get("units", ""),
                row=2,
                col=1,
                secondary_y=True,
            )
        nodes = graph.members(spacetime_event=event)
        branches = np.unique(graph.dataset.object_branch.values[nodes])
        shown = branches[:18]
        lookup = {int(b): i for i, b in enumerate(shown)}
        gd = graph.dataset
        for b in shown:
            i = int(np.flatnonzero(gd.branch_number.values == b)[0])
            fig.add_trace(
                go.Scatter(
                    x=[gd.branch_start.values[i], gd.branch_end.values[i]],
                    y=[lookup[int(b)]] * 2,
                    mode="lines+markers",
                    name=f"Branch {b}",
                    showlegend=False,
                ),
                row=2,
                col=2,
            )
        for parent, child in zip(gd.branch_link_parent.values, gd.branch_link_child.values):
            if int(parent) in lookup and int(child) in lookup:
                a, b = int(parent) - 1, int(child) - 1
                fig.add_trace(
                    go.Scatter(
                        x=[gd.branch_end.values[a], gd.branch_start.values[b]],
                        y=[lookup[int(parent)], lookup[int(child)]],
                        mode="lines",
                        line={"color": "#c58220"},
                        showlegend=False,
                    ),
                    row=2,
                    col=2,
                )
        fig.update_yaxes(title_text=f"First {len(shown)} of {len(branches)} branches", row=2, col=2)
    fig.update_xaxes(title_text="Longitude (degrees east)", row=1, col=1)
    fig.update_yaxes(title_text="Latitude (degrees north)", row=1, col=1)
    fig.update_yaxes(title_text=point.temperature.attrs.get("units", ""), row=1, col=2)
    fig.add_shape(
        type="line",
        x0=date,
        x1=date,
        y0=0,
        y1=1,
        xref="x2",
        yref="y2 domain",
        line={"color": "#555", "dash": "dot"},
    )
    fig.update_layout(
        height=750,
        title=f"{mask.definition.phenomenon.upper()}: computed products",
        uirevision="xhw-explorer",
        legend={"orientation": "h"},
    )
    return fig


def explorer(cases):
    """Widget controls for a mapping of names to (mask, objects, graph) computed products.

    Use (mask, None, None) for temporal-only cases. Returns the widget without displaying
    it. Date/depth/cell sliders update on release, avoiding redraws on every mouse move.
    """
    try:
        import ipywidgets as w
        from IPython.display import clear_output, display
    except ImportError as exc:
        raise ImportError("Install examples/requirements-interactive.txt in the kernel") from exc
    if not cases:
        raise ValueError("Supply at least one computed case")
    product = w.Dropdown(options=list(cases), description="Product")
    date = w.SelectionSlider(options=[""], description="Date", continuous_update=False)
    layer = w.Dropdown(description="Layer")
    depth = w.IntSlider(description="Depth index", continuous_update=False)
    event = w.BoundedIntText(description="Event", min=1)
    branch = w.Dropdown(description="Branch")
    row = w.IntSlider(description="Cell row", continuous_update=False)
    col = w.IntSlider(description="Cell column", continuous_update=False)
    output = w.Output()
    busy = False

    def configure(*_):
        nonlocal busy
        busy = True
        mask, objects, graph = cases[product.value]
        g, ds = mask.grid, mask.dataset
        date.options = [str(t)[:10] for t in ds[g.time_dim].values]
        event.disabled = graph is None
        event.max = int(graph.dataset.sizes["spacetime_event"]) if graph is not None else 1
        event.value = largest_event(graph) if graph is not None else 1
        date.value = (
            str(event_window(graph, event.value)[0])[:10] if graph is not None else date.options[-1]
        )
        names = [
            "temperature",
            "reference",
            "threshold",
            "intensity",
            "exceedance",
            "event_membership",
            "joined_gap",
            "valid",
        ]
        if objects is not None:
            names += [
                n
                for n in ("label", "track_id", "processed_mask", "added", "removed")
                if n in objects.dataset
            ]
        layer.options = names
        layer.value = names[0]
        depth.max = ds.sizes[g.spatial_dims[0]] - 1 if g.support == "volume" else 0
        depth.disabled = g.support != "volume"
        row.max, col.max = (ds.sizes[d] - 1 for d in g.spatial_dims[-2:])
        row.value, col.value = row.max // 2, col.max // 2
        busy = False
        configure_branches()

    def configure_branches(*_):
        nonlocal busy
        if busy:
            return
        busy = True
        graph = cases[product.value][2]
        ids = (
            np.unique(
                graph.dataset.object_branch.values[graph.members(spacetime_event=event.value)]
            )
            if graph is not None
            else []
        )
        branch.options = [("All", None)] + [(str(int(b)), int(b)) for b in ids]
        branch.value = None
        busy = False
        render()

    def render(*_):
        if busy:
            return
        with output:
            clear_output(wait=True)
            mask, objects, graph = cases[product.value]
            display(
                exploration_figure(
                    mask,
                    objects,
                    graph,
                    date=date.value,
                    layer=layer.value,
                    depth_index=depth.value,
                    event=event.value if graph else None,
                    branch=branch.value,
                    cell=(row.value, col.value),
                )
            )

    product.observe(configure, names="value")
    event.observe(configure_branches, names="value")
    for control in (date, layer, depth, branch, row, col):
        control.observe(render, names="value")
    configure()
    return w.VBox(
        [w.HBox([product, layer]), date, w.HBox([event, branch, depth]), w.HBox([row, col]), output]
    )


def objects_gif(objects, output_dir: Path, *, dates, name="evolution", fps=3):
    """Animate patch and track IDs using a fixed range, extent and actual date per frame."""
    from matplotlib.animation import FuncAnimation, PillowWriter
    from plotting import _surface

    dates = list(dates)
    if not dates:
        raise ValueError("Supply at least one animation date")
    plt = pyplot()
    ds, g = objects.dataset, objects.grid
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), layout="constrained")
    limits = {
        name: max(2, int(select_days(ds[name], dates, time_dim=g.time_dim).max()))
        for name in ("label", "track_id")
    }

    def frame(i):
        for ax, var in zip(axes, ("label", "track_id")):
            ax.clear()
            da = select_days(ds[var], dates[i], time_dim=g.time_dim)
            map_panel(
                ax,
                da.where(da > 0),
                grid=g,
                categorical=True,
                title=f"{str(dates[i])[:10]}: {var}",
                vmin=1,
                vmax=limits[var],
                add_colorbar=False,
            )
            surface = _surface(da, g)
            lon, lat = _geography(surface, g)
            for identifier in np.unique(surface.values):
                if identifier > 0:
                    on = surface.values == identifier
                    ax.text(
                        float(lon[on].mean()),
                        float(lat[on].mean()),
                        str(int(identifier)),
                        ha="center",
                        va="center",
                        fontsize=8,
                        bbox={"facecolor": "white", "alpha": 0.75, "pad": 1},
                    )
        fig.suptitle("Daily patches and tracking: overlap does not establish transport")
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{name}.gif"
    FuncAnimation(fig, frame, frames=len(dates), interval=1000 / fps).save(
        path, writer=PillowWriter(fps=fps), dpi=100
    )
    plt.close(fig)
    return path


def volume_figure(mask, objects, graph, *, date, event=None, max_points=50000):
    """3-D retained voxel centres in lon/lat/depth; no reconstructed surface or interpolation."""
    go, _ = _plotly()
    g = objects.grid
    if g.support != "volume":
        raise ValueError("A physical volume grid is required")
    event = largest_event(graph) if event is None else event
    labels = _event_labels(objects, graph, event, date)
    depth, lat, lon = xr.broadcast(
        labels[g.coords.depth], labels[g.coords.lat], labels[g.coords.lon]
    )
    use = labels.values > 0
    if int(use.sum()) > max_points:
        raise ValueError("More than 50000 retained cells; explicitly select a smaller event/domain")
    values = select_days(mask.intensity(), date, time_dim=g.time_dim).transpose(*labels.dims).values
    arrays = [v.transpose(*labels.dims).values[use] for v in (lon, lat, depth)]
    fig = go.Figure(
        go.Scatter3d(
            x=arrays[0],
            y=arrays[1],
            z=arrays[2],
            mode="markers",
            marker={
                "size": 4,
                "color": values[use],
                "colorscale": "Viridis",
                "colorbar": {
                    "title": f"Signed intensity ({mask.intensity().attrs.get('units', '')})"
                },
            },
            text=[f"Event {event}; {str(date)[:10]}"] * int(use.sum()),
        )
    )
    fig.update_layout(
        title=f"{str(date)[:10]}: retained voxel centres, event {event}",
        scene={
            "xaxis_title": "Longitude (degrees east)",
            "yaxis_title": "Latitude (degrees north)",
            "zaxis": {"title": "Depth (m, positive down)", "autorange": "reversed"},
            "aspectmode": "cube",
        },
        annotations=[
            {
                "text": "Axes use different units; visual aspect is not physical. "
                "No detail reconstructed between retained cells.",
                "xref": "paper",
                "yref": "paper",
                "x": 0,
                "y": 0,
                "showarrow": False,
            }
        ],
        height=650,
    )
    return fig


def volume_html(mask, objects, graph, output_dir: Path, *, event=None):
    """Offline HTML with up to 60 daily frames around the size peak of one event."""
    go, _ = _plotly()
    event = largest_event(graph) if event is None else event
    dates = event_window(graph, event)
    figures = [volume_figure(mask, objects, graph, date=str(d)[:10], event=event) for d in dates]
    fig = figures[0]
    finite = select_days(mask.intensity(), dates, time_dim=mask.grid.time_dim).values
    lo, hi = float(np.nanmin(finite)), float(np.nanmax(finite))
    for item in figures:
        item.data[0].marker.update(cmin=lo, cmax=hi)
    g, ds = mask.grid, mask.dataset
    fig.update_layout(
        scene={
            "xaxis": {"range": [float(ds[g.coords.lon].min()), float(ds[g.coords.lon].max())]},
            "yaxis": {"range": [float(ds[g.coords.lat].min()), float(ds[g.coords.lat].max())]},
            "zaxis": {
                "range": [float(ds[g.coords.depth].max()), float(ds[g.coords.depth].min())],
                "autorange": False,
            },
        }
    )
    fig.frames = [
        go.Frame(data=item.data, name=str(date)[:10], layout={"title": item.layout.title})
        for date, item in zip(dates, figures)
    ]
    fig.update_layout(
        sliders=[
            {
                "steps": [
                    {
                        "label": str(d)[:10],
                        "method": "animate",
                        "args": [
                            [str(d)[:10]],
                            {"mode": "immediate", "frame": {"duration": 0, "redraw": True}},
                        ],
                    }
                    for d in dates
                ]
            }
        ],
        updatemenus=[
            {
                "type": "buttons",
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [None, {"frame": {"duration": 300, "redraw": True}}],
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [[None], {"mode": "immediate", "frame": {"duration": 0}}],
                    },
                ],
            }
        ],
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "volume.html"
    fig.write_html(path, include_plotlyjs=True, auto_play=False)
    return path


def volume_preview(mask, objects, graph, output_dir: Path, *, event=None):
    """A small static counterpart of the retained-centre scene, readable on GitHub."""
    from plotting import save_figure

    event = largest_event(graph) if event is None else event
    dates = event_window(graph, event)
    date = dates[len(dates) // 2]
    labels = _event_labels(objects, graph, event, date)
    g = objects.grid
    z, y, x = xr.broadcast(labels[g.coords.depth], labels[g.coords.lat], labels[g.coords.lon])
    use = labels.values > 0
    arrays = [v.transpose(*labels.dims).values[use] for v in (x, y, z)]
    values = (
        select_days(mask.intensity(), date, time_dim=g.time_dim).transpose(*labels.dims).values[use]
    )
    plt = pyplot()
    fig = plt.figure(figsize=(8, 6), layout="constrained")
    ax = fig.add_subplot(111, projection="3d")
    points = ax.scatter(*arrays, c=values, cmap="viridis", s=12)
    ax.set(
        xlabel="Longitude (degrees east)",
        ylabel="Latitude (degrees north)",
        zlabel="Depth (m, positive down)",
        title=f"{str(date)[:10]}: retained cell centres, event {event}\n"
        "Visual aspect uses different units; no interpolation",
    )
    ax.invert_zaxis()
    fig.colorbar(
        points,
        ax=ax,
        shrink=0.6,
        label=f"Signed intensity ({mask.intensity().attrs.get('units', '')})",
    )
    save_figure(fig, output_dir, "volume_preview")
    plt.close(fig)
    return output_dir / "volume_preview.png"


def vertical_gif(mask, objects, graph, output_dir: Path, *, event=None, longitude_index=None):
    """Fixed-longitude retained-level section over the same event window as the 3-D HTML."""
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    g = mask.grid
    z, y, x = g.spatial_dims
    ds = mask.dataset
    index = ds.sizes[x] // 2 if longitude_index is None else longitude_index
    dates = event_window(graph, event)
    data = (
        select_days(ds.temperature, dates, time_dim=g.time_dim)
        .isel({x: index})
        .transpose(g.time_dim, z, y)
    )
    lo, hi = float(data.min()), float(data.max())
    plt = pyplot()
    fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
    fig.colorbar(
        ScalarMappable(norm=Normalize(lo, hi), cmap="viridis"),
        ax=ax,
        label=f"Temperature ({data.attrs.get('units', '')})",
    )

    def frame(i):
        ax.clear()
        data.isel({g.time_dim: i}).plot.pcolormesh(
            ax=ax,
            x=g.coords.lat,
            y=g.coords.depth,
            vmin=lo,
            vmax=hi,
            yincrease=False,
            add_colorbar=False,
        )
        ax.set(
            title=f"{str(dates[i])[:10]}: temperature, retained levels (no interpolation)",
            xlabel="Latitude (degrees north)",
            ylabel="Depth (m, positive down)",
        )
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "vertical_evolution.gif"
    FuncAnimation(fig, frame, frames=len(dates)).save(path, writer=PillowWriter(fps=3), dpi=100)
    plt.close(fig)
    return path
