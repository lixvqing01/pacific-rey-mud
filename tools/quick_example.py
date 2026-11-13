"""Run the seed-714 model on eight archived, normalized test patches on CPU."""
from pathlib import Path
import sys, json, csv
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.models.rey_multitask import MechanismAwareREYRegressor

def main():
    torch.set_num_threads(min(4, torch.get_num_threads()))
    ckpt = torch.load(ROOT / 'models/seed714/models/mechanism_aware_rey_regressor_best.pth', map_location='cpu', weights_only=False)
    cfg = ckpt['model_config']
    model = MechanismAwareREYRegressor(input_channels=cfg['input_channels'],
        n_mechanisms=len(cfg['mechanism_class_names']), n_thresholds=len(cfg['thresholds']),
        cnn_filters=cfg['cnn_filters'], hidden_dim=cfg['hidden_dim'], dropout=cfg['dropout'],
        use_multiscale=cfg['use_multiscale'])
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    example = np.load(ROOT / 'examples/test_patches.npz', allow_pickle=False)
    with torch.no_grad():
        out = model(torch.from_numpy(example['patches']))
        probabilities = np.minimum.accumulate(torch.sigmoid(out['threshold_logits']).numpy(), axis=1)
        facies = torch.softmax(out['mechanism_logits'], dim=1).numpy()
        scale = ckpt['norm_params']['rey']
        logrey = out['mu'].numpy().reshape(-1) * float(scale['std']) + float(scale['mean'])
    expected = json.loads((ROOT / 'examples/expected.json').read_text(encoding='utf-8'))
    assert np.isfinite(logrey).all() and np.isfinite(probabilities).all()
    assert np.all(np.diff(probabilities, axis=1) <= 1e-7)
    assert np.allclose(facies.sum(axis=1), 1, atol=1e-6)
    np.testing.assert_allclose(logrey, expected['predicted_log1p_rey'], rtol=0, atol=2e-4)
    np.testing.assert_allclose(probabilities, expected['threshold_probabilities'], rtol=0, atol=2e-4)
    target = ROOT / 'outputs/quick_example'
    target.mkdir(parents=True, exist_ok=True)
    with (target / 'predictions.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['dataset_index', 'predicted_log1p_rey', 'predicted_rey_ppm', 'P600', 'P800', 'P1000', 'P1500'])
        writer.writerows([int(i), float(y), float(np.expm1(y)), *p.tolist()] for i, y, p in zip(example['dataset_indices'], logrey, probabilities))
    report = {'passed': True, 'device': 'cpu', 'n_patches': len(logrey), 'shape': list(example['patches'].shape),
              'absolute_tolerance': 2e-4, 'ordered_thresholds': True, 'maximum_log_error': float(np.max(np.abs(logrey - expected['predicted_log1p_rey'])))}
    (target / 'verification.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()
