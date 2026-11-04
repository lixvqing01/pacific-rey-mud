# Data sources

See the accompanying manuscript's `Supplementary_Data_1`
(`Supplementary_Data_1_Geochemistry_and_Inputs.xlsx`) for the sample inventory
and source references. Sample indices are zero-based row indices
in data/geochem_class_table.parquet. The released checkpoints, fixed predictions
and map products correspond to the input table included in this package.

## Environmental sources

| Source product | Reference or provider |
|---|---|
| ETOPO1 bathymetry | Amante and Eakins (2009), NOAA, https://doi.org/10.7289/V5C8276M |
| Seafloor lithology | Dutkiewicz et al. (2015), https://doi.org/10.1130/G36883.1 |
| GlobSed sediment thickness | Straume et al. (2019), https://doi.org/10.1029/2018GC008115 |
| Long-term sediment accumulation | Average sedimentation-rate grid retained with this archive; related reconstruction: Dutkiewicz et al. (2017), https://doi.org/10.1002/2017GC007258 |
| Bottom-current variability | Thran et al. (2018), https://doi.org/10.1016/j.epsl.2018.02.044 |
| World Ocean Atlas oxygen and phosphate | NOAA World Ocean Atlas, https://www.ncei.noaa.gov/products/world-ocean-atlas ; the archived grid does not identify an atlas edition. |
| Primary productivity | Behrenfeld and Falkowski (1997), https://doi.org/10.4319/lo.1997.42.1.0001 |
| Organic carbon | Lee, Wood and Phrampus (2019), https://doi.org/10.1029/2018GB005992 |
| Benthic biomass | Wei et al. (2010), https://doi.org/10.1371/journal.pone.0015323 |
| Marine gravity | Sandwell et al. (2014), https://doi.org/10.1126/science.1258213 |
| Magnetic anomalies | Maus et al. (2009), https://doi.org/10.1029/2009GC002471 |
| Helium isotopes | Jenkins et al. (2019), https://doi.org/10.5194/essd-11-441-2019 |

The fixed 42 derived model channels and their order are recorded in
config/environment_features.json. These products provide the input context for
interpolation, derivation and standardization; the released model-ready grid
preserves the values used for the checkpoints.

## Geochemical observations

Compiled published measurements include the Pacific REY compilations of Kato
et al. (2011; https://doi.org/10.1038/ngeo1185) and the source studies cited in
the accompanying article and its supplementary geochemical data. data/README.md
describes the analysis-ready columns; config/facies_configuration.json records
the fitted classification. The analysis-ready archive does not retain a complete
row-by-row bibliography. Retain and consult the original publications when using
the underlying measured compositions.

## Units and missing values

REY is in ppm, physical sample depth in metres below seafloor, and the stratigraphic
proxy in ka. Coordinates use east-positive longitude. The standalone average
sedimentation-rate grid is in cm/kyr. Missing geochemical values remain missing
in the point archive; model-time handling and the 0.001 cm/kyr sedimentation
fallback are described in data/README.md and the dataset implementation.
