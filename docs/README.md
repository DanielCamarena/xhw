# xHW documentation

Use this guide to answer four questions: **what enters a stage, what leaves it,
which scientific choices affect the result, and how to reproduce it?**

| Route | Read | Outcome |
|---|---|---|
| First use | [Getting started](GETTING_STARTED.md) | Install and detect a known event without data or downloads |
| Processing | [Pipeline](PIPELINE.md) → [baselines](BASELINES.md) → [objects and genealogy](OBJECTS.md) → [metrics](METRICS.md) → [plotting](PLOTTING.md) | Understand each representation and its limits |
| Worked examples | [Tutorials](TUTORIALS.md) | Read executed notebooks or run their paired scripts |
| Your own data | [Data and provenance](DATA.md) | Prepare units, sampling, masks and physical measures; where the real-data cases live |
| Reference | [Public API](API.md) | Find signatures, inspection, serialization and errors |
| Audit | [Scientific contract](SCIENTIFIC_CONTRACT.md), [validation](VALIDATION.md), [reproducibility](REPRODUCIBILITY.md), [capability evidence](CAPABILITIES.md) | Distinguish implementation, execution and scientific review |
| Attribution | [Sources and licenses](THIRD_PARTY.md) | Identify scientific and software references |

## Find a page by task

| I want to… | Go to |
|---|---|
| See the whole pipeline in one runnable block | [README quick start](../README.md#quick-start) |
| Choose between MHW, MCS and AHW definitions | [README event types](../README.md#event-types), [API: DefinitionSpec](API.md#definitionspec) |
| Pick a baseline method or engine | [Baselines](BASELINES.md#seven-methods-at-a-glance) |
| Understand exceedance versus membership versus validity | [Pipeline: read an XMask](PIPELINE.md#read-an-xmask-correctly) |
| Turn hourly data into daily TX/TN | [Pipeline: quality](PIPELINE.md#quality-before-detection), [ERA5 tutorial](https://github.com/DanielCamarena/xhw-benchmark/blob/main/docs/TUTORIALS.md#era5) in xhw-benchmark |
| Build patches, track them, read splits and merges | [Objects and genealogy](OBJECTS.md) |
| Compute sizes, intensities and frequencies | [Metrics](METRICS.md) |
| Draw maps, series and genealogy diagrams | [Plotting](PLOTTING.md) |
| Run a real-data case or download a source | [xhw-benchmark](https://github.com/DanielCamarena/xhw-benchmark) |
| Check what was validated and how | [Validation](VALIDATION.md), [capabilities](CAPABILITIES.md) |

All public guides describe the publication installation, version **0.0.1**.
Examples use its public API and their local helper files; no development checkout
or private benchmark package is required. Notebook outputs are saved so the
examples can be read on GitHub without the original data files.

[Next: getting started →](GETTING_STARTED.md)

See [plotting and exploration](PLOTTING.md), [methodological references](REFERENCES.md) and [validation coverage](VALIDATION.md).
