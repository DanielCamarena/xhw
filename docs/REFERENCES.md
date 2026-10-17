# Methodological references

Methodological bibliography for xHW, with publisher pages and primary sources. DOI identifies
the cited work, not a software version or a dataset. A stable primary URL is used
when a DOI could not be verified. This list does not claim every scientific work has a DOI.

| Key | Reference and primary identifier | Role in xHW |
|---|---|---|
| hobday2016 | Hobday et al. (2016), *A hierarchical approach to defining marine heatwaves*, Progress in Oceanography 141, 227–238. [10.1016/j.pocean.2015.12.014](https://doi.org/10.1016/j.pocean.2015.12.014) | Marine detection, persistence and temporal metrics |
| hobday2018 | Hobday et al. (2018), *Categorizing and Naming Marine Heatwaves*, Oceanography 31(2), 162–173. [10.5670/oceanog.2018.205](https://doi.org/10.5670/oceanog.2018.205) | Warm severity categories |
| schlegel2021mcs | Schlegel, Darmaraki, Benthuysen, Filbee-Dexter and Oliver (2021), *Marine cold-spells*, Progress in Oceanography 198, 102684. [10.1016/j.pocean.2021.102684](https://doi.org/10.1016/j.pocean.2021.102684) | Cold categories: section 3, p. 6, Figure 1; not the Frontiers paper about heatwave drivers |
| perkins2013 | Perkins and Alexander (2013), *On the Measurement of Heat Waves*, Journal of Climate 26, 4500–4517. [10.1175/JCLI-D-12-00383.1](https://doi.org/10.1175/JCLI-D-12-00383.1) | Atmospheric conventions; xHW joint TX+TN profiles are explicit configurations |
| zhang2005 | Zhang et al. (2005), *Avoiding Inhomogeneity in Percentile-Based Indices of Temperature Extremes*, Journal of Climate 18, 1641–1651. [10.1175/JCLI3366.1](https://doi.org/10.1175/JCLI3366.1) | In-reference sampling corrections |
| hyndman1996 | Hyndman and Fan (1996), *Sample Quantiles in Statistical Packages*, The American Statistician 50(4), 361–365. [10.1080/00031305.1996.10473566](https://doi.org/10.1080/00031305.1996.10473566) | Explicit percentile estimators |
| cleveland1990 | Cleveland, Cleveland, McRae and Terpenning (1990), *STL: A Seasonal-Trend Decomposition Procedure Based on Loess*, Journal of Official Statistics 6(1), 3–73. [Statistics Sweden primary PDF](https://www.scb.se/contentassets/ca21efb41fee47d293bbee5bf7be7fb3/stl-a-seasonal-trend-decomposition-procedure-based-on-loess.pdf) | STL algorithm; no DOI verified for this article |
| bonino2023 | Bonino, Masina, Galimberti and Moretti (2023), *Southern Europe and western Asian marine heatwaves (SEWA-MHWs): a dataset based on macroevents*, Earth System Science Data 15, 1269–1285. [10.5194/essd-15-1269-2023](https://doi.org/10.5194/essd-15-1269-2023) | Application of STL to marine heatwaves; section 3.1.1 |
| eilers1996 | Eilers and Marx (1996), *Flexible Smoothing with B-splines and Penalties*, Statistical Science 11(2), 89–102 (discussion follows). [10.1214/ss/1038425655](https://doi.org/10.1214/ss/1038425655) | Penalized spline bases; DOI resolves to the publisher, whose full page was not readable during this check |
| hastie1990 | Hastie and Tibshirani (1990), *Generalized Additive Models*, first edition, Chapman & Hall. [Publisher](https://www.routledge.com/Generalized-Additive-Models/Hastie-Tibshirani/p/book/9780203753781); [10.1201/9780203753781](https://doi.org/10.1201/9780203753781) for the electronic edition | GAM background; publisher listing and DOI destination checked, electronic full text not accessed |
| halko2011 | Halko, Martinsson and Tropp (2011), *Finding Structure with Randomness: Probabilistic Algorithms for Constructing Approximate Matrix Decompositions*, SIAM Review 53(2), 217–288. [10.1137/090771806](https://doi.org/10.1137/090771806) | Randomized low-rank decomposition |
| vautard1989 | Vautard and Ghil (1989), *Singular spectrum analysis in nonlinear dynamics, with applications to paleoclimatic time series*, Physica D 35(3), 395–424. [10.1016/0167-2789(89)90077-8](https://doi.org/10.1016/0167-2789(89)90077-8) | SSA background |
| harvey1990 | Harvey, *Forecasting, Structural Time Series Models and the Kalman Filter*, Cambridge University Press; linked publisher catalogue lists the print edition as 1990 (earlier project citations used 1989). [10.1017/CBO9781107049994](https://doi.org/10.1017/CBO9781107049994) | Structural state-space models; identify the edition when citing |
| durbin2012 | Durbin and Koopman (2012), *Time Series Analysis by State Space Methods*, second edition, Oxford University Press. [10.1093/acprof:oso/9780199641178.001.0001](https://doi.org/10.1093/acprof:oso/9780199641178.001.0001) | State-space filtering and smoothing |
| scannell2024 | Scannell et al. (2024), *Spatiotemporal Evolution of Marine Heatwaves Globally*, Journal of Atmospheric and Oceanic Technology. [10.1175/JTECH-D-23-0126.1](https://doi.org/10.1175/JTECH-D-23-0126.1) | Spatial tracking context |

## Attribution and configuration limits

Bonino et al. use a ten-year trend smoothing and a five-year seasonal smoothing,
then a residual P90. xHW reconstructs reference = trend + seasonal and its threshold
from the residual percentile, but its default STL trend window is 7305 days and
seasonal_window="periodic". The latter repeats a fixed seasonal cycle. These defaults
do not reproduce the complete SEWA-MHW workflow or its variable seasonal amplitude.
Cleveland et al. remain the source of the STL algorithm. The Bonino citation does
not apply automatically to xHW's other dynamic methods or the detrended benchmark.

For cold categories, xHW uses I: 1≤r<2, II: 2≤r<3, III: 3≤r<4, IV: r≥4,
with r=(reference-temperature)/(reference-threshold), where the scale is nonzero.
The optional Ice flag described by Schlegel et al. is not implemented. The API
identifier is `category_scheme="schlegel2021"`. A profile name is a label;
the definition fields specify the actual detector and category choices.

`min_samples` is an explicit analysis choice; it does not define a universal physical
threshold or a universal percentage of valid observations.
Software commits, generated numerical references and dataset attribution are recorded
separately from these methodological sources.
