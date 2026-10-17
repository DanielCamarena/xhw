# Validation

The offline suite covers scientific definitions, container contracts and independent
numerical references. It does not download data or import external comparison software.
Reference manifests contain SHA-256 checksums; absent or changed references fail validation.

```bash
python -m pip install ".[dev]"
python -m pytest
python -m ruff check .
python tools/validate_docs.py
```

Use [Reproducibility](REPRODUCIBILITY.md) for examples, environments and packaging.

## Scientific checks

- Exact runs, equality boundaries, duration-before-gap qualification, missing and
  filled days, calendar edges, censoring, heat/cold categories and daily clocks.
- Hand-written quantiles, in-base thresholds and rates; deterministic stationary
  simulations test removal of the artificial in-base/out-of-base step.
- Calendar-following seasonal phase, exact trend/harmonic recovery, causal cutoff
  invariance, residual-threshold equivalence and preservation of a one-year anomaly.
- Eager versus Dask chunked detection/baselines, explicit materialization and
  NetCDF round trips with units, flags, Specs and identifiers preserved.
- Exact spatial fixtures, morphology invariants, periodic boundaries, curvilinear
  equivalence, static sigma weights and brute-force vertical connectivity.
- Split/merge and complex transitions, graph links checked against brute force,
  footprints as unions, monitoring identity stability and independently computed metrics.
- Public API, dataclass fields, entity-schema versions and registered Spec identifiers.

## Independent numerical comparisons

| Reference | Coverage and tolerance |
|---|---|
| marineHeatWaves, pinned commit | Three synthetic seeds, both tails: climate, threshold, dates, duration, intensities, categories and rates; absolute tolerance 1e-9, relative tolerance 0. Reference-edge and missing-intensity differences are tested explicitly. |
| climdex.pcic / Climpact | TX/TN thresholds equal to 1e-9 on 337 comparable DOY positions. Year-edge/29-Feb conventions differ on 29 positions. Heatwave-day differences are restricted to those boundaries. Bootstrap annual TX90p differences are bounded by 0.15 percentage points (registered reference maximum 0.120). |
| R `stats::stl`, with stlplus contrast | Periodic, finite seasonal window and robust cases on a 366-position grid; maximum component difference below 1e-9. stlplus is a different algorithm, reported as contrast. |
| ssalib 0.1.3 | Fixed trend, annual and semiannual eigentriple groups: maximum reconstruction difference below 1e-8; leading singular values agree at relative tolerance 1e-10. Nearly degenerate noise groups are not constrained. |
| Ocetrac surface tracking and marEx labeling | Equal cell partitions under matching connectivity, wrapping and overlap. Label values may differ while the partition agrees. Full spacetime diagonal connectivity is a documented different definition. Line/volume comparisons exercise labeling primitives, not a complete upstream dimensional API. |
| marEx merge partitioning | One mask has matching merges and parent counts, with 479/486 merged cells grouped alike. Three other masks have extra merges or parents in xHW. Measured deviations are fixed in tests; their cause remains unresolved. This is not a claim of complete equivalence. |
| Ocetrac DeepTrack | Independent daily labels and tracked volumes for four seeds. Identities match at the registered low thresholds; higher-threshold deviations are fixed. DeepTrack compares strict containment against an identity's union, whereas xHW uses inclusive patch-pair overlap. |

STL and GAM also compare their vectorized engines directly with the optional
statsmodels/pygam engines in the suite. GAM agrees without the seasonal penalty;
pygam's periodic penalty differs from xHW's cyclic second difference when enabled.

## Interpretation limits

The offline suite uses synthetic fields and independent numerical references. Real
OISST, OSTIA, ERA5 and GLORYS cases are in
[xhw-benchmark](https://github.com/DanielCamarena/xhw-benchmark), where each result
keeps its scientific-review status. Reproducing a number does not grant scientific approval.

Equality with another numerical implementation does not establish physical skill.
Synthetic geometry checks do not cover every curvilinear or sigma grid. macOS,
all dependency combinations and global-domain memory costs are not certified by
these tests. Reference source and execution environments are recorded in
`tests/reference/*/manifest.json`.
