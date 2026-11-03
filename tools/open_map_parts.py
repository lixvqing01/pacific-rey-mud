"""Open all released 3D map parts lazily; close the returned Dataset after use."""
from pathlib import Path
import json, sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import xarray as xr
from core.utils.netcdf_io import open_dataset
ROOT = Path(__file__).resolve().parents[1]

def open_maps(root=ROOT):
    root = Path(root)
    manifest = json.loads((root / 'release_manifest.json').read_text(encoding='utf-8'))
    parts = []
    try:
        for p in manifest['map_parts']:
            parts.append(open_dataset(root / p['path'], decode_times=False, chunks={}))
    except Exception:
        for part in parts:
            part.close()
        raise
    merged = xr.concat(parts, dim='age_ka', data_vars='minimal', coords='minimal', compat='equals')
    merged.set_close(lambda: [part.close() for part in parts])
    return merged

if __name__ == '__main__':
    with open_maps() as ds:
        print(ds)
