"""Recompute the fixed ensemble from archived tables or trained checkpoints."""
from pathlib import Path
import sys, argparse, json
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.evaluate_rey_multitask_ensemble import _average_predictions
from scripts.run_rey_continuous_multitask import (
    _evaluate_predictions, _make_base_dataset, _predict, REYMultitaskDataset,
)
from scripts.predict_rey_continuous_grid import _build_model
from torch.utils.data import DataLoader, Subset
from core.utils.serialization_utils import NumpyJSONEncoder
from core.utils.logger_config import setup_logger

SEEDS = [310, 411, 512, 613, 714]
THRESHOLDS = [600., 800., 1000., 1500.]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['archived', 'weights'], default='archived')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'outputs/evaluation')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    setup_logger(level='WARNING')
    torch.set_num_threads(min(8, torch.get_num_threads()))
    runs = [ROOT / f'models/seed{s}' for s in SEEDS]
    if args.mode == 'weights':
        device = torch.device(args.device)
        indices = np.load(ROOT / 'splits/fixed/test_indices.npy').astype(int).tolist()
        points = pd.read_parquet(ROOT / 'data/geochem_class_table.parquet')
        points['mechanism_class3'] = points['mech_class_id'].astype(int)
        prepared = args.output_dir / 'evaluation_points.parquet'
        points.to_parquet(prepared, index=False)
        fresh_runs = []
        for seed, run in zip(SEEDS, runs):
            ckpt = torch.load(run / 'models/mechanism_aware_rey_regressor_best.pth', map_location='cpu', weights_only=False)
            dataset = _make_base_dataset(
                points_path=prepared, env_nc=ROOT / 'data/env_gridded.nc',
                feature_names=ckpt['feature_names'], patch_size=11,
                sed_rate_nc=ROOT / 'data/Average sedimentation rate.nc',
                depth_feature_type='age_ka_proxy', norm_params_override=ckpt['norm_params'],
                use_target_weights=False, max_weight=1., augment=False, shift_pixels=0,
            )
            loader = DataLoader(Subset(REYMultitaskDataset(dataset, THRESHOLDS), indices), batch_size=128, shuffle=False, num_workers=0)
            pred = _predict(_build_model(ckpt, device), loader, device=device, dataset=dataset, thresholds=THRESHOLDS)
            fresh = args.output_dir / f'seed{seed}'
            (fresh / 'tables/test').mkdir(parents=True, exist_ok=True)
            pred.to_csv(fresh / 'tables/test/mechanism_aware_rey_predictions.csv', index=False)
            fresh_runs.append(fresh)
            print(f'Evaluated seed {seed}: {len(pred)} held-out observations', flush=True)
            del dataset, loader
        runs = fresh_runs
    ensemble = _average_predictions(runs, 'test', THRESHOLDS, np.full(5, .2))
    metrics = _evaluate_predictions(ensemble, THRESHOLDS)
    reference = json.loads((ROOT / 'models/ensemble_evaluation/ensemble_metrics.json').read_text(encoding='utf-8'))['test_metrics']
    paths = [('regression', 'log_mae'), ('regression', 'spearman_log'), ('regression', 'r2_log'),
             ('mechanism3', 'macro_f1'), ('thresholds', 'rey_ge_1000_ap'), ('thresholds', 'rey_ge_1500_ap')]
    tolerance = 1e-8 if args.mode == 'archived' else 0.005
    checks = {f'{section}.{key}': {'actual': metrics[section][key], 'reference': reference[section][key],
                'absolute_error': abs(metrics[section][key] - reference[section][key]),
                'pass': bool(abs(metrics[section][key] - reference[section][key]) <= tolerance)} for section, key in paths}
    refpred = pd.read_csv(ROOT / 'models/ensemble_evaluation/ensemble_test_predictions.csv')
    assert np.array_equal(ensemble.dataset_index.to_numpy(), refpred.dataset_index.to_numpy())
    cols = [f'prob_rey_ge_{int(t)}' for t in THRESHOLDS]
    assert np.all(np.diff(ensemble[cols].to_numpy(), axis=1) <= 1e-7)
    if args.mode == 'archived':
        assert np.allclose(ensemble.rey_pred, refpred.rey_pred, rtol=1e-12, atol=1e-9)
    report = {'mode': args.mode, 'n_test': len(ensemble), 'n_cores': int(ensemble.core_id.nunique()),
              'metric_absolute_tolerance': tolerance, 'checks': checks, 'all_passed': all(c['pass'] for c in checks.values())}
    ensemble.to_csv(args.output_dir / f'ensemble_{args.mode}_predictions.csv', index=False)
    (args.output_dir / f'metrics_{args.mode}.json').write_text(json.dumps(metrics, cls=NumpyJSONEncoder, indent=2), encoding='utf-8')
    (args.output_dir / f'verification_{args.mode}.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
    if not report['all_passed']:
        raise SystemExit('Reference agreement failed; see the verification report.')

if __name__ == '__main__':
    main()
