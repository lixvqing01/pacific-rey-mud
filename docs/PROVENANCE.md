# Fixed population and artifact provenance

See [Supplementary Data 1](../data/Supplementary_Data_1_Geochemistry_and_Inputs.xlsx)
for the sample inventory and geochemical source information. This package
supplies the model-ready point table, checkpoints, fixed split files and
reference predictions used by the archived results.

splits/fixed/split_assignments.csv binds each zero-based dataset_index to a
train/val/test partition and core_id. The split file provides the operational
partition and grouping used for reproduction; no core overlaps partitions. All five model runs use the
same partition and seeds 310, 411, 512, 613 and 714. The five-member ensemble is
used for the reference sample evaluation; the map and SHAP analysis use seed714.

Runtime `dataset_index` values refer to rows of the model-ready Parquet table.
The supplied weights, fixed split and predictions retain the identities and
values of the archived model run.

The source release was prepared from the original project archive. Local paths
were replaced by package-relative paths in release metadata. State dictionaries,
normalization arrays, feature ordering and scientific output values were retained.
The 3D NetCDF is split without changing values. tools/verify_package.py checks
population, model metadata, fixed-split membership, asset hashes and map coordinates.
