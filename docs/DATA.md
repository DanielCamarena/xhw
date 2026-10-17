# Data, preparation and provenance

[← Pipeline](PIPELINE.md) · [Index](README.md) · [Tutorials →](TUTORIALS.md)

## Bring your own data

xHW never downloads data. Open your input with xarray and declare its variable names, units,
calendar, grid, domain, cell measures and source provenance. Every tutorial in this repository
generates its input, so nothing here needs a data file or a network connection.

```python
from pathlib import Path
import hashlib
import xarray as xr
import xhw

path = Path("data/my_temperature.nc")
ds = xr.open_dataset(path)  # lazy opening; manage the dataset's lifetime
with path.open("rb") as handle:
    digest = hashlib.file_digest(handle, "sha256").hexdigest()
source = xhw.SourceRef(dataset_id="my-product-id", variable="sst",
                       accessed="2026-10-10", product_status="unknown",
                       version="actual-version", query="actual subset",
                       checksum=f"sha256:{digest}")
```

Combine this source with an explicit grid, baseline and definition: [the tutorial
helpers](../examples/_data.py) show complete specs for a surface grid, and the
[API](API.md) lists every field. `SourceRef` records identity; it does not validate scientific
quality. Treat a new input as a new, unreviewed run.

## Real data

The real cases (NOAA OISST over Peru and the equatorial Pacific, Met Office OSTIA, ERA5 and GLORYS12V1) live in the companion
project [xhw-benchmark](https://github.com/DanielCamarena/xhw-benchmark). It holds the compressed
NetCDF datasets as release assets, the scripts that download them from their sources, the processed
products and the notebooks. Each dataset is registered with two hashes: `sha256` of the bytes and
`content_sha256` of the stored values, so a recompressed copy can still be shown to carry the same
information. Download times and cautions for each source are in that project's README.

## The synthetic stand-in for OISST

`examples/_data.py::synthetic_peru` generates, in about a second, a daily SST field on the same 32 × 20
grid, with the same coast, seasonal cycle, trend and anomaly statistics as the OISST case. A few
parameters were fitted once to OISST and stored in the code; no observed value is copied, and every
random part comes from a seeded generator. It is statistically similar, not equal:

| | OISST (registered) | Synthetic reconstruction |
|---|---|---|
| Ocean cells | 349 | 349 |
| Domain-mean anomaly, sd (°C) | 1.26 | 1.32 |
| Trend (°C per year) | −0.0093 | −0.0092 |
| Per-cell MHW events | 16 702 | 14 797 |
| % of valid days in an MHW | 9.46 | 8.43 |
| Tracks / spacetime events | 118 / 136 | 112 / 140 |
| Splits / merges | 38 / 41 | 32 / 47 |
| Largest event, maximum intensity (K) | 7.3 | 9.5 |

Its tails are wider than OISST's (extremes of 11.6 and 33.4 °C against 13.4 and 30.0 °C): treat its
figures as illustrations of the API, never as climate results.

## Preparation decisions to retain

With ERA5, keep instantaneous t2m separate from forecast-derived mx2t/mn2t. Max sampled t2m
can exceed max mx2t; they are different data streams, not continuous extrema of
one identical series. Air over the ocean is not a substitute for volumetric
seawater temperature. A fixed UTC−05:00 offset is a declared day convention, not
a timezone with changing daylight-saving rules.

For a volume with retained levels, such as decimated GLORYS, thickness can be approximated by interfaces halfway between
retained depths, the upper interface at zero and the bottom extended by one half
spacing. Coarse spherical cell area times thickness gives volume. These are not
original model layer bounds, conservative remapping or vertically integrated
high-resolution truth. Decimation can alter connectivity and small-scale metrics.
No mandatory regridding is added to detection.

Every run report records the source checksum, selection, specs, method parameters,
versions, UTC date and review state. Keep the retrieval query of every input with its
checksum. A checksum identifies bytes; it does not establish that the physical selection
is appropriate.
