# Pacific REY reproduction version 1.0.0

This release contains the archived model-ready table documented in
`data/Supplementary_Data_1_Geochemistry_and_Inputs.xlsx` (Data scope sheet),
five pretrained checkpoints, fixed train/validation/test
assignments and reference predictions. Source code, small tables, checkpoints
and a CPU example are in the tagged repository. Large inputs and numerical
prediction products are attached individually and checked against
release_manifest.json.

The map consists of three independently readable NetCDF files covering all
27 proxy-age layers. The input grid has 42 ordered environmental features;
the model adds one stratigraphic age-proxy channel. Grade, ordered threshold
probabilities, facies probabilities and reliability outputs remain distinct.

Start with README.md and docs/REPRODUCING.md in the source repository.
