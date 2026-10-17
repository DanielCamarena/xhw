# Reproducing the tutorials

[Documentation index](README.md) · [Tutorials](TUTORIALS.md) · [Data](DATA.md) · [Validation](VALIDATION.md)

The two synthetic tutorials generate their own input and require no datasets,
credentials or network after installation. Saved notebooks contain figures and tables
that can be read on GitHub. Real-data examples live in
[xhw-benchmark](https://github.com/DanielCamarena/xhw-benchmark).

## Installation and scripts

From the repository root:

```bash
python -m pip install .
python -m pip install -r examples/requirements-interactive.txt
python -m pip check
python examples/quickstart.py
python examples/synthetic.py --output-dir results/synthetic
python examples/synthetic_peru.py --output-dir results/synthetic_peru
python examples/api_walkthrough.py
```

To install a built wheel, use `python -m pip install dist/xhw-0.0.1-py3-none-any.whl`.
Commands work in PowerShell and POSIX shells; quote paths containing spaces.
Each tutorial writes figures and a `report.json` with its parameters, source,
software versions, code fingerprints and review status. Execution metadata belongs
to the run that produced it; it is not inferred from an installation instruction.

## Notebooks

```bash
python examples/execute_notebooks.py --output-dir results/notebooks
python examples/execute_notebooks.py --case synthetic_peru --output-dir results/notebooks
```

The runner executes from fresh kernels using the selected Python, updates the notebooks
in place and checks their format and error outputs. Static results remain readable
without Jupyter. Install the interactive requirements for live controls and GIF generation.
Graphical helpers belong to examples, not the installed xHW API.

## Product checks

Scripts use public checks for grid, baseline, masks, temporal catalogs, objects and
genealogy. `check_against` checks correspondence between stages. Spec JSON and product
Dataset/NetCDF round trips preserve their declared fields and identifiers. All xHW
entity schemas use version `0.1`, independently of package version `0.0.1`.
See [Validation](VALIDATION.md) for test coverage and
[the API reference](API.md) for methods and failure modes.

Saved run reports are [synthetic](run_records/synthetic.json) and
[synthetic Peru](run_records/synthetic_peru.json). They preserve the actual execution
metadata of the displayed results. A new run writes its own report.

## Packaging

Build outside a synchronised folder when possible:

```bash
python -m build
python tools/audit_package.py --dist-dir dist
python -m twine check dist/xhw-0.0.1-py3-none-any.whl dist/xhw-0.0.1.tar.gz
```

The sdist includes documentation, figures, scripts, notebooks and optional example
requirements. The wheel contains the installed library and package metadata/licenses.
`tools/audit_package.py` compares wheel module bytes with the source and checks the
sdist teaching assets. Building and checking do not publish the package.

## Interpretation limits

Synthetic fields demonstrate the definitions, representations and checks; they are
not climate findings. The synthetic Peru field approximates selected OISST statistics
and does not reproduce its observations.

Short daily patches can remain in a graph after the tracker suppresses their identities.
Exporting them as a fully qualified catalog can fail its minimum-duration invariant;
see [catalog qualification](OBJECTS.md#catalog-export-qualification).
Missing observations, interpolation, joined temporal gaps and spatially added cells
remain distinct. Splits and merges describe the chosen overlap rules, not physical causes.
