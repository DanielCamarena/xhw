# Copyright 2026 Daniel Camarena
# SPDX-License-Identifier: Apache-2.0

"""Small, explicit data and specification helpers shared by scripts and notebooks.

No downloader runs on import. All arrays used by these tutorials are loaded eagerly.
The registered files remain outside the distribution; source manifests ship beside us.
"""

from __future__ import annotations

import hashlib
from dataclasses import replace

import numpy as np
import xarray as xr

import xhw

RADIUS_M = 6_371_000.0


def add_area(ds: xr.Dataset, lat: str, lon: str, step: float) -> xr.Dataset:
    radians = np.deg2rad(ds[lat].values)
    half = np.deg2rad(step / 2)
    area = RADIUS_M**2 * np.deg2rad(step) * np.abs(np.sin(radians + half) - np.sin(radians - half))
    ds["cell_area"] = ((lat, lon), np.repeat(area[:, None], ds.sizes[lon], axis=1), {"units": "m2"})
    return ds


def surface_grid(
    *,
    lat: str = "lat",
    lon: str = "lon",
    domain: str = "ocean",
    domain_mask: str = "domain_mask",
    product: str = "synthetic",
    resolution: str = "1 degree",
) -> xhw.GridSpec:
    return xhw.GridSpec(
        support="surface",
        topology="structured",
        time_dim="time",
        spatial_dims=(lat, lon),
        coords=xhw.CoordRoles(lat=lat, lon=lon),
        calendar="standard",
        frequency="1D",
        time_axes="single",
        periodic_x=False,
        cell_measure=xhw.CellMeasure(kind="area", var="cell_area", units="m2"),
        vertical=None,
        domain=domain,
        domain_mask=domain_mask,
        domain_fraction=None,
        native=xhw.NativeGrid(product=product, resolution=resolution, is_native=True),
    )


def marine_definition(
    *, cold: bool = False, variable: str = "sea_surface_temperature"
) -> xhw.DefinitionSpec:
    return xhw.DefinitionSpec(
        profile="schlegel2021" if cold else "hobday2016",
        phenomenon="mcs" if cold else "mhw",
        variable=variable,
        combination="single",
        daily_statistic="native",
        day_definition="native",
        tail="lower" if cold else "upper",
        comparator="<" if cold else ">",
        threshold_kind="percentile",
        percentile=10.0 if cold else 90.0,
        absolute_threshold=None,
        min_duration_days=5,
        max_gap_days=2,
        join_rule="qualify_then_join",
        missing_policy="as_non_exceedance",
        season_months=None,
        category_scheme="schlegel2021" if cold else "hobday2018",
    )


def fixed_baseline(
    source: xhw.SourceRef,
    start: str = "1991-01-01",
    end: str = "2020-12-31",
    *,
    atmospheric: bool = False,
) -> xhw.BaselineSpec:
    return xhw.BaselineSpec(
        method="fixed_climatology",
        computed_on="temperature",
        reference_period=(start, end),
        source=source,
        doy_window_days=15 if atmospheric else 11,
        smoothing_window_days=None if atmospheric else 31,
        percentile_estimator="hyndman_fan_8" if atmospheric else "hyndman_fan_7",
        leap_day_policy="hobday_366_interp",
        min_samples=360 if atmospheric else 264,
        fit_mode="retrospective",
        available_until=None,
        in_base_correction="none",
    )


def synthetic_inputs():
    """A hand-constructed 15-day event: connected -> split for 5 days -> merge.

    Reference is a constant 20 degC in 1973-2002, so P90 and mean are both 20.
    Two five-cell bars remain hot throughout. Their three-cell connector is hot
    on days 1-5 and 11-15 only. Consequently its long gap cannot be temporally joined.
    A separate cell illustrates a joined cool day and another a filled missing day.
    """
    time = np.arange(np.datetime64("1973-01-01"), np.datetime64("2003-02-01"))
    lat, lon = np.arange(-12.0, -5.0), np.arange(-82.0, -73.0)
    temperature = np.full((len(time), len(lat), len(lon)), 20.0)
    first = int(np.flatnonzero(time == np.datetime64("2003-01-01"))[0])
    temperature[first : first + 15, 2, 2:7] = 26.0
    temperature[first : first + 15, 4, 2:7] = 26.0
    temperature[first : first + 5, 3, 3:6] = 26.0
    temperature[first + 10 : first + 15, 3, 3:6] = 26.0
    temperature[first : first + 13, 0, 0] = 25.0
    temperature[first + 6, 0, 0] = 20.0  # one cool day between qualifying runs
    temperature[first : first + 13, 0, 8] = 25.0
    temperature[first + 6, 0, 8] = np.nan  # interior interpolation, with a separate flag
    # Diagonally touching cells: two rank-1 patches, one rank-2 patch.
    temperature[first + 5 : first + 10, 6, 7] = 25.0
    temperature[first + 5 : first + 10, 5, 8] = 25.0
    ds = xr.Dataset(
        {
            "sst": (("time", "lat", "lon"), temperature, {"units": "degC"}),
            "domain_mask": (("lat", "lon"), np.ones((7, 9), bool)),
        },
        coords={"time": time.astype("datetime64[ns]"), "lat": lat, "lon": lon},
    )
    ds = add_area(ds, "lat", "lon", 1.0)
    source = xhw.SourceRef(
        dataset_id="synthetic:split-merge-v1",
        variable="sst",
        accessed="2026-10-10",
        product_status="final",
        checksum="sha256:" + hashlib.sha256(temperature.tobytes()).hexdigest(),
    )
    return (
        ds,
        surface_grid(),
        fixed_baseline(source, "1973-01-01", "2002-12-31"),
        replace(marine_definition(), category_scheme=None),
    )


# --- A parametric reconstruction of the Peru SST field ------------------------------------
# Constants fitted once to OISST v2.1 (12S-4S, 82W-77W, 1991-2025): a smooth quadratic surface in
# (lat + 8, lon + 79.5) for the mean and for each seasonal harmonic, a coastline, and the
# statistics of the anomaly. No daily value of OISST is stored: everything random comes from a
# seeded generator, so the field is similar in structure and never a copy.
_PERU_SURFACE = {  # coefficients of 1, y, x, y^2, x^2, y*x with y = lat + 8, x = lon + 79.5
    "mean": (19.7428, -0.6472, -1.3668, -0.0269, -0.175, -0.1821),
    "cos1": (0.8451, -0.1601, -0.3181, 0.0008, -0.0725, -0.0641),
    "sin1": (2.0754, -0.304, -0.5551, -0.0153, -0.0648, -0.0325),
    "cos2": (0.0983, -0.0509, 0.0541, -0.0175, -0.003, -0.036),
    "sin2": (0.6354, 0.0533, 0.0824, -0.0021, -0.0008, 0.0065),
}
_PERU_COAST = (-84.071, -0.5841)  # easternmost ocean longitude = a + b * latitude
_PERU_TREND = -0.022  # degC per year before the warm pulses; the fitted OISST trend is -0.0093
_PERU_SLOW = (0.992, 0.70)  # domain-wide anomaly, slow part: AR(1) coefficient per day, sd (degC)
_PERU_FAST = (0.94, 0.60)  # domain-wide anomaly, fast part
_PERU_LOCAL_PHI, _PERU_LOCAL_SD, _PERU_LOCAL_SIGMA = 0.938, 0.68, 2.15  # weather; sigma in cells
# Warm pulses (centre, peak degC, width in days) near the dates of the strong events of the period.
_PERU_PULSES = (
    ("1992-03-01", 2.6, 70),
    ("1997-09-01", 5.4, 130),
    ("2015-10-01", 3.8, 110),
    ("2017-03-01", 3.2, 40),
    ("2023-04-15", 4.8, 90),
)


def synthetic_peru(seed: int = 7) -> xr.Dataset:
    """Daily SST on the 32 x 20 Peru grid, 1991-2025: a fast, seeded reconstruction of OISST.

    Same grid, coast, seasonal cycle, trend and anomaly statistics as the registered OISST case,
    with a domain-wide red-noise anomaly, five warm pulses and spatially correlated weather
    noise. It is statistically similar, not a copy of any observation, and is rounded to
    hundredths of a degree like OISST. Generating it takes about a second.
    """
    from scipy.ndimage import gaussian_filter
    from scipy.signal import lfilter

    rng = np.random.default_rng(seed)
    time = np.arange(np.datetime64("1991-01-01"), np.datetime64("2026-01-01"))
    lat = np.arange(-11.875, -4.0, 0.25)
    lon = np.arange(-81.875, -77.0, 0.25)
    la, lo = np.meshgrid(lat + 8.0, lon + 79.5, indexing="ij")
    basis = np.stack([np.ones_like(la), la, lo, la**2, lo**2, la * lo])
    surface = {k: np.tensordot(np.array(c), basis, axes=1) for k, c in _PERU_SURFACE.items()}
    days = (time - time[0]).astype(float)
    phase = 2 * np.pi * days / 365.25
    seasonal = (
        surface["mean"][None]
        + _PERU_TREND * days[:, None, None] / 365.25
        + surface["cos1"][None] * np.cos(phase)[:, None, None]
        + surface["sin1"][None] * np.sin(phase)[:, None, None]
        + surface["cos2"][None] * np.cos(2 * phase)[:, None, None]
        + surface["sin2"][None] * np.sin(2 * phase)[:, None, None]
    )
    common = np.zeros(time.size)
    for phi, sd in (_PERU_SLOW, _PERU_FAST):
        part = lfilter([np.sqrt(1 - phi**2)], [1, -phi], rng.standard_normal(time.size))
        common += sd * part / part.std()
    for centre, peak, width in _PERU_PULSES:
        offset = (time - np.datetime64(centre)).astype(float)
        common += peak * np.exp(-0.5 * (offset / width) ** 2)
    weather = gaussian_filter(
        rng.standard_normal((time.size, lat.size, lon.size)),
        sigma=(0, _PERU_LOCAL_SIGMA, _PERU_LOCAL_SIGMA),
        mode="nearest",
    )
    weather = lfilter([np.sqrt(1 - _PERU_LOCAL_PHI**2)], [1, -_PERU_LOCAL_PHI], weather, axis=0)
    weather *= _PERU_LOCAL_SD / weather.std()
    sst = np.round(seasonal + common[:, None, None] + weather, 2)
    ocean = lon[None, :] <= _PERU_COAST[0] + _PERU_COAST[1] * lat[:, None] + 0.125
    sst = np.where(ocean[None], sst, np.nan)
    ds = xr.Dataset(
        {
            "sst": (("time", "lat", "lon"), sst, {"units": "degC"}),
            "domain_mask": (("lat", "lon"), ocean),
        },
        coords={"time": time.astype("datetime64[ns]"), "lat": lat, "lon": lon},
    )
    return add_area(ds, "lat", "lon", 0.25)


def synthetic_peru_inputs(seed: int = 7):
    """`synthetic_peru` with its grid, fixed 1991-2020 baseline and marine MHW definition."""
    ds = synthetic_peru(seed)
    source = xhw.SourceRef(
        dataset_id=f"synthetic:peru-reconstruction-seed{seed}",
        variable="sst",
        accessed="2026-10-11",
        product_status="final",
        checksum="sha256:" + hashlib.sha256(ds["sst"].values.tobytes()).hexdigest(),
    )
    grid = surface_grid(product="synthetic reconstruction", resolution="0.25 degree")
    return ds, grid, fixed_baseline(source), marine_definition()
