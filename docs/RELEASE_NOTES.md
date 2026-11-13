# Data files

This repository provides the Python source code, the model-ready point table,
five pretrained checkpoints, fixed train/validation/test assignments, reference
predictions and a runnable CPU example for the accompanying article.

Large environmental input grids and prediction products are provided as
individual attachments to the Data files release. The `data` tag is a download
identifier rather than a software version number. File sizes and SHA256 hashes
are recorded in release_manifest.json.

The 3D map consists of three independently readable NetCDF files covering all
27 proxy-age layers. The input grid has 42 ordered environmental features;
the model adds one stratigraphic age-proxy channel. Grade, ordered threshold
probabilities, facies probabilities and reliability outputs remain distinct.

Start with README.md and docs/REPRODUCING.md.
