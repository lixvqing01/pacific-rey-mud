# Reproduce the reported workflow

Run commands from the repository root. Python 3.10 is supported. Outputs are
written to `outputs/`; this folder is generated locally and excluded from Git.
Use `environment.yml` for the full analysis environment or
`requirements-quick.txt` for the small CPU check.

Install the full environment before the evaluation, training or mapping steps:

```bash
python -m pip install -r requirements.txt
```

`requirements-quick.txt` supplies only the CPU example dependencies. The
archived evaluation also imports the analysis workflow and uses the full
environment, although it does not need the large NetCDF downloads.

## Download and verify the full assets

The point table, model checkpoints, fixed split and example patches are available
in the source repository. Large environmental grids and map products are
optional GitHub Release downloads:

```bash
python -B tools/download_assets.py --repo lixvqing01/pacific-rey-mud --tag v1.0.1 --group inputs
python -B tools/download_assets.py --repo lixvqing01/pacific-rey-mud --tag v1.0.1 --group products
python -B tools/verify_package.py
```

The downloader checks file sizes and SHA256 hashes against
`release_manifest.json`. `--group inputs` is needed for fresh weight inference,
training and map generation. `--group products` downloads the archived maps and
is not needed to regenerate them.

## Reproduce the test metrics

```bash
python -B tools/reproduce_evaluation.py --mode archived
python -B tools/reproduce_evaluation.py --mode weights --device cpu
```

`archived` recalculates the ensemble metrics from the supplied member prediction
tables. `weights` extracts the fixed test patches from the released inputs and
runs all checkpoints. CPU is supported; use `--device cuda` when CUDA is
available. The results are compared against the fixed reference metrics; the
accepted numerical tolerances and completed local checks are recorded in
`VALIDATION.md`.

Fresh grid preparation and high-resolution mapping require approximately 12 GB
of RAM and at least 10 GB of free disk space. They are unnecessary for the
archived metric check.

The data scope and sample inventory are recorded in
[Supplementary Data 1](../data/Supplementary_Data_1_Geochemistry_and_Inputs.xlsx).
Its Data scope sheet identifies the compilation and archived model records;
runtime indices refer to `data/geochem_class_table.parquet`.

## Retrain the fixed models

```bash
python -B tools/train_fixed.py --dry-run
python -B tools/train_fixed.py
```

Training uses the included fixed grouped split and archived settings in
`config/model_configuration.json`. It trains the five released seeds and may
take substantially longer than evaluation. Fresh training outputs are written
under `outputs/training/`.

## Generate maps and figures

For a small CPU map check after downloading the inputs, run:

```bash
python -B -m scripts.predict_rey_continuous_grid --model-run-dirs models/seed714 --env-nc data/env_gridded.nc --sed-rate-nc "data/Average sedimentation rate.nc" --output-dir outputs/quick_map --age-layers-ka 0 --target-resolution-deg 2 --stride 20 --batch-size 256 --age-batch-size 1 --num-workers 0 --patch-mode fast_grid --cpu
```

From PowerShell, `./run_pretrained_prediction.ps1 -Quick` provides an alternative
preview with the wrapper's device and batch defaults.
To generate the full
archived map, run `run_pretrained_prediction.ps1` from PowerShell without
`-Quick`; the script uses the released seed-714 checkpoint and formal grid
settings. The full map is expensive to regenerate, so its three NetCDF parts
are also supplied as optional Release downloads. Read them without assembling a
second large file:

```python
from tools.open_map_parts import open_maps

with open_maps() as maps:
    print(maps.sizes)
```

Variable names, units and provenance are described in `products/README.md` and
`data/SOURCES.md`. Figure and diagnostic scripts are in `scripts/`; for example,
run `python -B -m scripts.export_geochem_facies_figures --help` or
`python -B -m scripts.explain_rey_continuous_shap --help` to see their inputs and
options. Archived component comparisons are in `reference_results/`, and
balanced repeated-split definitions are in `splits/balanced_repeated/`.
