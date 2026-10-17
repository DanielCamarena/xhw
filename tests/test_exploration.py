# Copyright 2026 Daniel Camarena
# SPDX-License-Identifier: Apache-2.0

"""Optional examples operate only on computed products and preserve frame identity."""

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pytest
from _lineage_fixtures import objects
from _object_fixtures import make_mask, make_volume_mask

import xhw

matplotlib.use('Agg')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'examples'))
pytestmark = pytest.mark.filterwarnings('ignore::xhw.XHWDataQualityWarning')


@pytest.mark.parametrize('clock_hours', [0, 12])
def test_exploration_and_controls_never_recompute_scientific_products(monkeypatch, clock_hours):
    from exploration import exploration_figure, explorer

    exc = np.ones((3, 2, 2), bool)
    mask = make_mask(exc, unit_measure=True)
    mask = xhw.XMask.from_dataset(mask.to_dataset().assign_coords(
        time=mask.dataset.time + np.timedelta64(clock_hours, 'h')))
    obj = objects(exc, mask=mask)
    graph = xhw.build_lineage(obj, field=mask.intensity())

    def forbidden(*args, **kwargs):
        raise AssertionError('visual controls reran the scientific processing')

    for name in ('fit_baseline', 'detect', 'build_objects', 'build_lineage'):
        monkeypatch.setattr(xhw, name, forbidden)
    fig = exploration_figure(mask, obj, graph, date='2000-01-02', layer='label')
    assert fig.layout.xaxis.title.text == 'Longitude (degrees east)'
    widget = explorer({'Tracked': (mask, obj, graph), 'Temporal': (mask, None, None)})
    widget.children[1].value = '2000-01-03'
    widget.children[0].children[1].value = 'valid'
    widget.children[0].children[0].value = 'Temporal'
    assert widget.children[2].children[0].disabled
    widget.close()


def test_gif_has_requested_dates_and_fixed_frame_count(tmp_path):
    from exploration import objects_gif
    from PIL import Image

    exc = np.array([[[1, 0], [0, 0]], [[1, 1], [0, 0]], [[0, 1], [0, 1]]], bool)
    obj = objects(exc, mask=make_mask(exc, unit_measure=True))
    path = objects_gif(obj, tmp_path, dates=obj.dataset.time.values)
    with Image.open(path) as gif:
        assert gif.n_frames == 3


@pytest.mark.parametrize('clock_hours', [0, 12])
def test_volume_html_is_offline_and_vertical_gif_has_same_window(tmp_path, clock_hours):
    from exploration import event_window, vertical_gif, volume_figure, volume_html
    from PIL import Image

    exc = np.ones((3, 2, 2, 2), bool)
    mask = make_volume_mask(exc)
    mask = xhw.XMask.from_dataset(mask.to_dataset().assign_coords(
        time=mask.dataset.time + np.timedelta64(clock_hours, 'h')))
    obj = objects(exc, mask=mask)
    graph = xhw.build_lineage(obj, field=mask.intensity())
    fig = volume_figure(mask, obj, graph, date='2000-01-01')
    assert len(fig.data[0].x) == 8
    with pytest.raises(ValueError, match='retained cells'):
        volume_figure(mask, obj, graph, date='2000-01-01', max_points=1)
    path = volume_html(mask, obj, graph, tmp_path)
    text = path.read_text(encoding='utf8')
    assert '<script src=' not in text
    assert '2000-01-03' in text and 'Plotly.newPlot' in text
    with Image.open(vertical_gif(mask, obj, graph, tmp_path)) as gif:
        assert gif.n_frames == len(event_window(graph))


def test_profile_aliases_preserve_explicit_definition_choices():
    from profiles import example_profiles

    p = example_profiles()
    assert p['mhw'].min_duration_days == 5
    assert p['mcs'].tail == 'lower' and p['mcs'].percentile == 10
    assert p['ahw_tx90'].min_duration_days == 3
    assert p['ahw_tx_then_tn90'].components[1].lag_days == 1
    assert p['mhw'].to_dict()['fields']['profile'] == 'hobday2016'


def test_cell_evidence_window_follows_selected_map_day():
    from exploration import exploration_figure

    mask = make_mask(np.ones((100, 2, 2), bool), unit_measure=True)
    fig = exploration_figure(mask, date='2000-01-02')
    times = np.asarray(fig.data[1].x).astype('datetime64[D]')
    assert times.min() == np.datetime64('2000-01-01')
    assert times.max() == np.datetime64('2000-01-31')
    assert fig.layout.shapes[0].x0 == '2000-01-02'
