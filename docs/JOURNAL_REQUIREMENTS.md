# Computers & Geosciences repository requirements

Basis: the journal's official Guide for Authors, saved on 27 October 2026:
https://www.sciencedirect.com/journal/computers-and-geosciences/publish/guide-for-authors

The Guide requires a publicly accessible, documented GitHub or equivalent
repository at submission, with a repository link in the manuscript. It calls
for a clear license, an English README, dependencies and computational
requirements, reproducible main results, tutorials, an input/output/options
guide, and English code comments. Source must be browsable; a single compressed
archive does not meet the repository requirement.

| Journal requirement | Material in this distribution |
|---|---|
| Open source and clear license | LICENSE (MIT); DATA_LICENSE.md separates data and model terms. |
| English installation and usage | README.md; environment.yml; requirements.txt; requirements-quick.txt. |
| Dependencies and resource requirements | docs/REPRODUCING.md; docs/validation_environment.json. |
| Runnable test case | A small set of actual normalized test patches, seed714 checkpoint, tools/quick_example.py. |
| Reproducible main test results | Five checkpoints, fixed splits, analysis-ready points, grid assets, tools/reproduce_evaluation.py. |
| Training configuration and options | config/; tools/train_fixed.py; scripts module --help output. |
| Typical workflows and input/output guide | docs/REPRODUCING.md; data/README.md; products/README.md. |
| Supporting diagnostics | Component comparisons, balanced repeated results and split definitions; figure and interpretation scripts. |
| English code comments | tools/verify_package.py scans Python comment tokens; docstrings were also reviewed. |
| Data provenance and sharing terms | docs/PROVENANCE.md; data/SOURCES.md; DATA_LICENSE.md. |
| Browsable source rather than a single archive | Regular Git files for code, small data and weights; versioned Release assets for large NetCDF files. |
| Public access and article link | Public repository; clean public clone and downloads verified; repository URL appears in the manuscript. |

Local validation checks the supplied fixed workflow. The sample inventory and
source information are referenced in the accompanying manuscript's
`Supplementary_Data_1` (`Supplementary_Data_1_Geochemistry_and_Inputs.xlsx`).
Public availability is assessed against the actual published repository.
