"""Refresh source checksums after intentional maintainer edits before tagging."""
from pathlib import Path
import hashlib

ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {'.py', '.md', '.txt', '.json', '.csv', '.yml', '.yaml', '.cff', '.ps1'}


def main():
    manifest = ROOT / 'SOURCE_SHA256SUMS.txt'
    paths = [line.split('  ', 1)[1] for line in manifest.read_text(encoding='utf-8').splitlines()]
    lines = []
    for rel in sorted(paths):
        path = (ROOT / rel).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file() or path == manifest:
            raise ValueError(f'Invalid source entry: {rel}')
        raw = path.read_bytes()
        if path.suffix in TEXT_SUFFIXES or path.name in {'.gitattributes', '.gitignore', 'LICENSE'}:
            normalized = raw.replace(b'\r\n', b'\n')
            if raw != normalized:
                path.write_bytes(normalized)
            raw = normalized
        lines.append(f'{hashlib.sha256(raw).hexdigest()}  {rel}\n')
    manifest.write_text(''.join(lines), encoding='utf-8', newline='\n')
    print(f'Refreshed {len(lines)} source checksums with LF text line endings.')


if __name__ == '__main__':
    main()
