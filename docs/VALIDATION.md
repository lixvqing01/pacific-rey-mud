# Local validation

Validation date: 2 November 2026. Hardware, software versions and the scope of
the installation check are recorded in validation_environment.json. Detailed
numerical evidence is in local_validation.json.

| Check | Observed result |
|---|---|
| CPU example | A small set of 43 × 11 × 11 archived patches; maximum logREY error 0.0001021 against archived seed714 predictions; tolerance 0.0002. |
| Archived ensemble reconstruction | The fixed held-out observations and core groups recorded in the distributed split files; all six principal metrics match within 1e-8. |
| Fresh inference from all five weights | The complete held-out partition recorded in the distributed split files; maximum principal-metric absolute difference 0.000113425; acceptance tolerance 0.005. |
| CPU map workflow | 5,246 cell-age evaluations at 2° and age 0 ka; NetCDF and PNG outputs generated; ordered threshold probabilities verified. |
| Complete 3D archive partition | All 20 data variables and all coordinates compare exactly to the original, including missing values, across all 27 age layers. |
| 2D product metadata update | An obsolete archive note was replaced with the three-part layout; all variable and coordinate values remain exactly equal to the original. |
| Map reader | All 27 layers open as Dask-backed arrays; values sampled from the first, middle and final parts. |
| Training configuration | Five dry-run commands checked; every command uses the archived fixed split arrays. |
| Source review | Python syntax, English comments/docstrings and source file checksums checked. |
| Scientific artifact identity | Sample inventory and geochemical data scope are documented in the accompanying manuscript's `Supplementary_Data_1`; disjoint fixed core split, checkpoint channels and thresholds checked. |

Inference and map tests used CPU. The existing environment includes a CUDA
development build of PyTorch; the exact version is recorded. A fresh environment
installation, full retraining, the full high-resolution mapping run and the
complete component/repeated suite were not repeated for this packaging check.
Their reference artifacts and reproduction scripts are supplied. A local source
export also exercises the quick example without the large input grids.

Public GitHub clone and Release-download checks follow publication using
docs/PUBLISHING.md. Software tests establish the tested reproduction behaviour;
the article's data-population statements must be aligned separately.
