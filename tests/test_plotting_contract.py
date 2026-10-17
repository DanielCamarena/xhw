# Copyright 2026 Daniel Camarena
# SPDX-License-Identifier: Apache-2.0

"""Coordinate correctness and single-frame regressions for local example helpers."""

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pytest
import xarray as xr
from _lineage_fixtures import objects
from _object_fixtures import make_mask

import xhw

matplotlib.use('Agg')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'examples'))
from plotting import map_panel, pyplot, sequence_figure  # noqa: E402

pytestmark = pytest.mark.filterwarnings('ignore::xhw.XHWDataQualityWarning')


def test_auxiliary_coordinates_use_geographic_values():
    da = xr.DataArray(np.ones((2, 2)), dims=('y', 'x'),
                      coords={'lat': ('y', [-10, -9]), 'lon': ('x', [-80, -79])})
    plt = pyplot()
    fig, ax = plt.subplots()
    map_panel(ax, da, title='Auxiliary coordinates')
    assert ax.get_xlim() == pytest.approx((-80.5, -78.5))
    assert ax.get_ylim() == pytest.approx((-10.5, -8.5))
    assert ax.get_xlabel() == 'Longitude (degrees east)'
    plt.close(fig)


def test_index_axes_are_not_labelled_as_degrees():
    plt = pyplot()
    fig, ax = plt.subplots()
    map_panel(ax, xr.DataArray(np.ones((2, 2)), dims=('y', 'x')), title='Indices')
    assert ax.get_xlabel() == 'x (index)'
    assert ax.get_ylabel() == 'y (index)'
    plt.close(fig)


@pytest.mark.parametrize('shape', [(2, 2), (1, 1)])
@pytest.mark.parametrize('clock_hours', [0, 12])
def test_sequence_accepts_one_date_and_singleton_spatial_axes(tmp_path, shape, clock_hours):
    exc = np.ones((2, *shape), bool)
    mask = make_mask(exc, unit_measure=True)
    mask = xhw.XMask.from_dataset(mask.to_dataset().assign_coords(
        time=mask.dataset.time + np.timedelta64(clock_hours, 'h')))
    obj = objects(exc, mask=mask)
    fig = sequence_figure(obj, ['2000-01-01'], tmp_path)
    assert len(fig.axes) == 2
    assert (tmp_path/'objects.png').is_file()
    pyplot().close(fig)
    with pytest.raises(ValueError, match='at least one'):
        sequence_figure(obj, [], tmp_path)
    with pytest.raises(ValueError, match='absent'):
        sequence_figure(obj, ['1999-01-01'], tmp_path)


def test_daily_selection_preserves_clock_and_rejects_ambiguous_dates():
    from plotting import select_days

    data = xr.DataArray([10, 20], dims='time', coords={
        'time': np.array(['2000-01-01T12:00', '2000-01-02T12:00'], dtype='datetime64[ns]')})
    selected = select_days(data, '2000-01-02')
    assert int(selected) == 20
    assert selected.time.values == np.datetime64('2000-01-02T12:00')
    assert list(select_days(data, ['2000-01-02', '2000-01-01']).values) == [20, 10]
    duplicate = data.assign_coords(time=[np.datetime64('2000-01-01T00:00'),
                                       np.datetime64('2000-01-01T12:00')])
    with pytest.raises(ValueError, match='multiple records'):
        select_days(duplicate, '2000-01-01')
