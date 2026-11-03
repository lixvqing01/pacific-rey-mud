# Trained model checkpoints

The five directories contain the official fixed-split checkpoints for seeds
310, 411, 512, 613 and 714. Each directory is a model run directory accepted by
`scripts.predict_rey_continuous_grid` and contains:

```text
models/seed<seed>/models/mechanism_aware_rey_regressor_best.pth
```

Every checkpoint stores the model state, architecture configuration,
normalization parameters, ordered 42-feature list, thresholds, mechanism class
names and selected epoch metrics. Each seed directory also contains its fixed
split, training history and train/validation/test prediction metrics. The
formal archived 3D prediction uses `models/seed714`; the other four checkpoints
support five-member sensitivity and ensemble runs. `ensemble_evaluation/`
contains the archived five-member evaluation tables and metrics.
