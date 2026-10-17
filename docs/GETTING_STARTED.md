# Getting started

[Index](README.md) · [Next: pipeline →](PIPELINE.md)

## Install in an environment

Run commands from the repository root. Python 3.11+ works on Windows and Linux;
the tutorials use `pathlib`, without operating-system-specific data paths.

```bash
python -m venv .venv
```

| Shell | Activate |
|---|---|
| Linux/macOS bash | `source .venv/bin/activate` |
| Windows PowerShell | `.venv\Scripts\Activate.ps1` |

```bash
python -m pip install .
python examples/quickstart.py
```

The first line should read:

```text
xHW 0.0.1: known 10-day synthetic event detected
```

The following xarray table contains event starts, ends, durations and maximum
intensity. The quickstart asserts membership on all ten injected days; its noisy
reference can also produce other events. It constructs explicit `GridSpec`,
`BaselineSpec` and `DefinitionSpec` objects before calling:

```python
fitted = xhw.fit_baseline(ds, variable="sst", grid=grid,
                          baseline=baseline, definition=definition)
mask = xhw.detect(ds, variable="sst", grid=grid, definition=definition,
                  baseline_result=fitted)
catalog = xhw.build_catalog(mask, analysis_mode="retrospective")
mask.check_invariants()
catalog.check_against(mask)
```

These names are created by [the complete quickstart](../examples/quickstart.py).
A profile name is metadata; the parameters in the specs declare the actual
definition. Avoid relying on the name to fill unspecified scientific choices.

## See all stages without downloading data

```bash
python -m pip install -r examples/requirements.txt
python examples/synthetic.py --output-dir outputs/synthetic
```

This produces eight PNGs and `report.json`, with actual results and provenance.
The [synthetic notebook](../examples/synthetic.ipynb) has saved figures and tables,
then walks through calls to each stage. Its injected geometry deliberately has
one split and one merge. Its degenerate constant reference is an illustration,
not a physically representative climatology or a category-calibration example.

For interactive use:

```bash
python -m ipykernel install --user --name xhw-tutorials --display-name "xHW tutorials"
python -m jupyter lab examples/synthetic.ipynb
```

Select the kernel belonging to the environment where you installed xHW and the
requirements. Restart it, then **Run All**. The tutorials need no data files.
`XHW_OUTPUT_DIR` sets the notebook output directory; the scripts use `--output-dir`. Default notebook outputs are in
`outputs/<case>` below the repository root.

## Choose an endpoint

For cell-level duration and intensity, stop at `build_catalog`. For connected
geometry, add `build_objects`. For cross-day identity, include a track step. For
multiple-parent transitions and branch metrics, add `build_lineage`.

Installing a library does not install tutorial files into your Python environment.
Use a source checkout or the sdist to obtain the documentation and notebooks.
The wheel contains the library only.
