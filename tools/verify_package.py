"""Verify source integrity, release assets and the fixed scientific contract."""
from pathlib import Path
import argparse, hashlib, json, re, tokenize, sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

def digest(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-only', action='store_true', help='Skip downloaded assets and full-environment checks.')
    args = parser.parse_args()
    checked = 0
    for line in (ROOT / 'SOURCE_SHA256SUMS.txt').read_text(encoding='utf-8').splitlines():
        expected, rel = line.split('  ', 1)
        p = ROOT / rel
        assert p.is_file() and digest(p) == expected, f'Source checksum: {rel}'
        assert p.stat().st_size < 100 * 1024**2, f'File exceeds ordinary Git limit: {rel}'
        checked += 1
    for p in [*ROOT.joinpath('core').rglob('*.py'), *ROOT.joinpath('scripts').rglob('*.py'), *ROOT.joinpath('tools').rglob('*.py')]:
        with tokenize.open(p) as f:
            for t in tokenize.generate_tokens(f.readline):
                if t.type == tokenize.COMMENT:
                    assert not re.search('[\u3400-\u9fff]', t.string), f'Non-English comment: {p.name}:{t.start[0]}'
    cfg = json.loads((ROOT / 'config/environment_features.json').read_text(encoding='utf-8'))
    assert len(cfg['environmental_features_in_order']) == 42
    assert 'land_mask' not in cfg['environmental_features_in_order']
    manifest = json.loads((ROOT / 'release_manifest.json').read_text(encoding='utf-8'))
    for asset in manifest['assets']:
        assert asset['bytes'] < 2 * 1024**3, f'Release asset exceeds 2 GiB: {asset["path"]}'
        assert not Path(asset['path']).is_absolute()
    if not args.source_only:
        import numpy as np
        import pandas as pd
        import torch
        from core.utils.netcdf_io import open_dataset
        for asset in manifest['assets']:
            p = ROOT / asset['path']
            assert p.is_file(), f'Missing asset: {p}; use tools/download_assets.py'
            assert p.stat().st_size == asset['bytes'] and digest(p) == asset['sha256'], f'Asset checksum: {p}'
        data = pd.read_parquet(ROOT / 'data/geochem_class_table.parquet')
        assert len(data) == 4385 and data.core_id.nunique() == 144 and data.site_group.nunique() == 117
        assert data.mech_class_id.value_counts().sort_index().tolist() == [916, 434, 3035]
        split = pd.read_csv(ROOT / 'splits/fixed/split_assignments.csv')
        assert split.groupby('split').size().to_dict() == {'test':753,'train':2966,'val':666}
        assert np.array_equal(data.iloc[split.dataset_index].core_id.to_numpy(), split.core_id.to_numpy())
        assert np.allclose(data.iloc[split.dataset_index].rey, split.rey)
        groups = {s:set(g.core_id) for s,g in split.groupby('split')}
        assert not (groups['train'] & groups['val'] or groups['train'] & groups['test'] or groups['val'] & groups['test'])
        for seed in [310,411,512,613,714]:
            ckpt = torch.load(ROOT / f'models/seed{seed}/models/mechanism_aware_rey_regressor_best.pth', map_location='cpu', weights_only=False)
            assert ckpt['feature_names'] == cfg['environmental_features_in_order']
            assert ckpt['model_config']['input_channels'] == 43
            assert ckpt['model_config']['thresholds'] == [600.,800.,1000.,1500.]
            member_split = pd.read_csv(ROOT / f'models/seed{seed}/splits/split_assignments.csv')
            assert split.equals(member_split)
        with open_dataset(ROOT / 'data/env_gridded.nc', decode_times=False) as ds:
            assert list(ds.data_vars) == cfg['environmental_features_in_order']
            assert dict(ds.sizes) == {'latitude':1200,'longitude':1701}
        ages = []
        for part in manifest['map_parts']:
            with open_dataset(ROOT / part['path'], decode_times=False) as ds:
                assert ds.age_ka.values.tolist() == part['age_layers_ka']
                assert ds.sizes['latitude'] == 1200 and ds.sizes['longitude'] == 1701
                ages += ds.age_ka.values.tolist()
        assert ages == [float(a) for a in range(0,2501,100)] + [2580.]
    print(json.dumps({'passed':True, 'source_files_checked':checked, 'source_only':args.source_only,
                      'release_assets':len(manifest['assets']),
                      'sample_inventory_reference':'Supplementary_Data_1'}, indent=2))

if __name__ == '__main__':
    main()
