# Pacific REY mud reproduction workflow

Code, pretrained models and fixed inputs for **Mechanism-aware deep learning
predicts stratigraphic rare earth element enrichment in Pacific deep-sea mud**
by Xuqing Li and Yunying Zhang.

The workflow predicts continuous rare earth element and yttrium (REY) grade,
ordered threshold exceedance probabilities and geochemical facies from regional
environmental grids and a depth-derived stratigraphic proxy. The five-member
ensemble supplies the reference test evaluation; seed714 supplies the archived
maps and SHAP analysis.

## Start with the CPU example

Use Python 3.10 and run commands from the repository root. In a virtual or Conda
environment, install the small CPU environment and run the example:

```bash
python -m pip install numpy==1.26.4
python -m pip install "torch>=2.5,<3" --index-url https://download.pytorch.org/whl/cpu
python -B tools/quick_example.py
python -B tools/verify_package.py --source-only
```

The example runs a pretrained checkpoint on included test patches and writes
`outputs/quick_example/predictions.csv` and `verification.json`. A successful
run reports `"passed": true`. The example and source checks work directly from
the repository, before downloading the large grids.

## Reproduce the test results

Install the full analysis dependencies before running evaluation, training or
mapping:

```bash
python -m pip install -r requirements.txt
python -B tools/reproduce_evaluation.py --mode archived
```

This rebuilds the ensemble from the supplied member prediction tables and
checks six metrics against the archived reference. The reference is logREY
MAE **0.4425**, Spearman **0.7747**, R² **0.5785**, AP1000 **0.8433**,
AP1500 **0.7424** and facies macro-F1 **0.8771**.

For fresh inference from the five checkpoints, first download the input grids
from this repository's **Data files** release:

```bash
python -B tools/download_assets.py --repo lixvqing01/pacific-rey-mud --tag data --group inputs
python -B tools/reproduce_evaluation.py --mode weights --device cpu
```

The supported commands, resource requirements and training/map examples are
in [Reproducing the workflow](docs/REPRODUCING.md). Numerical tolerances and
completed checks are in [Validation](docs/VALIDATION.md).

## Files and data organization

| Location | Contents |
|---|---|
| `core/`, `scripts/`, `tools/` | Model, evaluation, training and prediction code |
| `config/` | Ordered environmental channels, facies definitions and model settings |
| `data/` | Model-ready point table, source metadata and Supplementary Data 1 |
| `models/` | Pretrained checkpoints, member predictions and ensemble reference |
| `splits/` | Fixed grouped split and balanced repeated split definitions |
| `examples/` | CPU example patches and expected predictions |
| `reference_results/` | Component comparisons and supporting diagnostics |
| `products/` | Map metadata and optional downloaded NetCDF products |

[Supplementary Data 1](data/Supplementary_Data_1_Geochemistry_and_Inputs.xlsx)
documents the compiled sample inventory and source references. Numerical
reproduction uses `data/geochem_class_table.parquet` and the row assignments in
`splits/fixed/`. Their provenance is described in [Provenance](docs/PROVENANCE.md).

Large inputs and archived maps are individual attachments in the Data files release.
`release_manifest.json` records their sizes and SHA256 hashes. To download the
optional maps and check the complete local package:

```bash
python -B tools/download_assets.py --repo lixvqing01/pacific-rey-mud --tag data --group products
python -B tools/verify_package.py
```

The 3-D map is provided in three NetCDF parts, readable with
`tools.open_map_parts.open_maps()`. See [Prediction products](products/README.md).
Generated results go to `outputs/`, which is excluded from Git.

## Citation and license

Cite the accompanying article and this repository; software
attribution is provided in `CITATION.cff`. The authors' code and documentation
use the [MIT license](LICENSE). Model, product and third-party data terms are
specified in [Data and model reuse](DATA_LICENSE.md), with source references
in [Data sources](data/SOURCES.md).

Maintainers preparing a release can follow [Publishing on GitHub](docs/PUBLISHING.md).
