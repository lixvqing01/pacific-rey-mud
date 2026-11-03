# Data files

All file references below are relative to the archive root.

| File | Size | Contents and role |
|---|---:|---|
| `data/env_gridded.nc` | 473,733,373 bytes | Model-ready 0.1° environmental grid, 1200 latitudes × 1701 longitudes, covering 59.9°S–60.0°N and 120.0–290.0°E. It contains exactly 42 ordered model features and excludes `land_mask`. Feature order is recorded in `config/environment_features.json`. |
| `data/geochem_class_table.parquet` | 2,388,147 bytes | Analysis-ready sediment table. See the accompanying manuscript's `Supplementary_Data_1` (`Supplementary_Data_1_Geochemistry_and_Inputs.xlsx`) for its reported sample inventory and geochemical data scope. This table contains the model inputs with coordinates, sample depth, REY, available bulk geochemistry, derived process proxies, facies labels and grouping fields. Missing source measurements remain missing rather than being filled. |
| `data/Average sedimentation rate.nc` | 25,985,648 bytes | Sedimentation-rate grid used with physical sample depth to calculate the stratigraphic age proxy. The stored variable is `Average sedimentation rate`. |
| `data/boundaries.gmt` | 381,644 bytes | Plate-boundary linework used by map-plotting scripts. It is not a model input feature. |
| `data/Supplementary_Data_1_Geochemistry_and_Inputs.xlsx` | See the included workbook | Byte-identical copy of the manuscript's Supplementary Data 1; sample inventory, geochemical inputs, source information and the Data scope sheet. Runtime sample indices refer to the model-ready Parquet table. |

## Environmental model inputs

`env_gridded.nc` is the screened, model-ready input cube used in the fixed
continuous-REY workflow. The authoritative ordered list is
`config/environment_features.json`; the 43rd model channel is the derived
`age_ka_proxy`. Environmental feature names are discovered from the NetCDF at
runtime and must remain in the recorded order. `land_mask` is not a training
feature.

At a subset of sample positions, linear and nearest interpolation of the standalone
sedimentation-rate grid both remain missing. The fixed dataset implementation
handles these positions with its documented positive lower bound of 0.001
cm/ka before calculating `age_ka_proxy`. The released baseline applies this fallback
consistently during training-data preparation and test inference.

## Point-table fields

The principal fields are:

- location and sampling: `lon`, `lat`, `depth`, `core_id`, `site_hole`,
  `site_group`, `Lithology`;
- response: `rey` in ppm;
- measured REY and other available geochemistry: elemental and major-oxide
  columns such as `La`–`Lu`, `Y`, `Mo`, `V`, `Zn`, `MnO`, `Fe2O3`, `P2O5`,
  `TiO2`, `Al2O3`, `CaO`, `SiO2`, `K2O` and `MgO`;
- derived geochemical ratios and proxies: columns prefixed by `ratio_`,
  `log_ratio_` or `proxy_`;
- mechanism/facies fields: `mech_class_id`, class probabilities and retained
  compatibility labels.

data/SOURCES.md lists the source publications and the scope of the retained
provenance metadata. data/environment_channels.csv records the original
variable attributes for each of the 42 released environmental channels. Rights
and citation obligations associated with underlying third-party source data
remain with their respective providers and publications.
