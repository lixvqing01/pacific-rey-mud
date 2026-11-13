"""Publish this validated local candidate using the authenticated GitHub CLI."""
from pathlib import Path
import argparse, json, subprocess, sys, re
ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', required=True, help='Existing public GitHub OWNER/REPOSITORY')
    parser.add_argument('--tag', default='data')
    parser.add_argument('--execute', action='store_true', help='Actually create a release and upload assets.')
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', args.repo):
        parser.error('Use a GitHub OWNER/REPOSITORY identifier.')
    if not re.fullmatch(r'[A-Za-z0-9_.-]+', args.tag):
        parser.error('Use an ordinary release tag.')
    manifest = json.loads((ROOT / 'release_manifest.json').read_text(encoding='utf-8'))
    commands = [
        ['gh','release','create',args.tag,'--repo',args.repo,'--verify-tag','--title','Data files',
         '--notes-file','docs/RELEASE_NOTES.md'],
        ['gh','release','upload',args.tag,'--repo',args.repo,*[a['path'] for a in manifest['assets']]],
    ]
    print(json.dumps({'execute':args.execute,'prerequisite':'Push the source commit and matching tag to the public repository first.',
                      'commands':commands}, indent=2))
    if args.execute:
        info = subprocess.run(['gh','repo','view',args.repo,'--json','isPrivate'], check=True, text=True, capture_output=True)
        if json.loads(info.stdout)['isPrivate']:
            raise SystemExit('The journal requires a public repository; set its visibility before release.')
        subprocess.run([sys.executable, '-B', 'tools/verify_package.py'], cwd=ROOT, check=True)
        for cmd in commands:
            subprocess.run(cmd, cwd=ROOT, check=True)

if __name__ == '__main__':
    main()
