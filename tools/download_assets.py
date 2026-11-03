"""Download versioned GitHub Release assets and validate their size and SHA256."""
from pathlib import Path
import argparse, json, hashlib, urllib.request, urllib.parse, re, os
ROOT = Path(__file__).resolve().parents[1]

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', required=True, help='GitHub OWNER/REPOSITORY')
    parser.add_argument('--tag', default='v1.0.0')
    parser.add_argument('--group', choices=['inputs', 'products', 'all'], default='all')
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', args.repo):
        parser.error('Use a GitHub OWNER/REPOSITORY identifier.')
    if not re.fullmatch(r'[A-Za-z0-9_.-]+', args.tag):
        parser.error('Use an ordinary release tag.')
    manifest = json.loads((ROOT / 'release_manifest.json').read_text(encoding='utf-8'))
    for entry in manifest['assets']:
        if args.group != 'all' and entry['group'] != args.group:
            continue
        target = (ROOT / entry['path']).resolve()
        if not target.is_relative_to(ROOT):
            raise ValueError('Asset path escapes repository.')
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.stat().st_size == entry['bytes'] and digest(target) == entry['sha256']:
            print(f'Verified existing {entry["path"]}')
            continue
        asset_name = urllib.parse.quote(entry['asset_name'], safe='')
        url = f'https://github.com/{args.repo}/releases/download/{args.tag}/{asset_name}'
        partial = target.with_suffix(target.suffix + '.partial')
        print(f'Downloading {url}', flush=True)
        request = urllib.request.Request(url, headers={'User-Agent': 'pacific-rey-reproduction'})
        with urllib.request.urlopen(request, timeout=120) as response, partial.open('wb') as f:
            while block := response.read(8 * 1024 * 1024):
                f.write(block)
        if partial.stat().st_size != entry['bytes'] or digest(partial) != entry['sha256']:
            raise ValueError(f'Asset verification failed: {entry["asset_name"]}')
        os.replace(partial, target)
        print(f'Installed {entry["path"]}', flush=True)

if __name__ == '__main__':
    main()
