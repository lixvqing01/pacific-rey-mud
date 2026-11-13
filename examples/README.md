# CPU checkpoint example

test_patches.npz contains eight normalized 43-channel, 11 by 11 patches extracted
from the released model-ready grid using the archived seed714 normalization.
dataset_indices identify their original zero-based rows in the released table;
the authoritative sample inventory and scope are provided in `Supplementary_Data_1`.
expected.json contains predictions from the archived fixed-test table, not outputs
newly generated to make this example pass. Run `python tools/quick_example.py`.
Output is written to outputs/quick_example. Full preprocessing and the complete
held-out partition defined in `Supplementary_Data_1` are exercised by
`python tools/reproduce_evaluation.py --mode weights`.
