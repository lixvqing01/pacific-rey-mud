"""Train the archived configuration with explicit fixed split arrays."""
from pathlib import Path
import sys, json, subprocess, argparse
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_rey_paper_reproduction import _baseline_train_cmd

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--seeds', default='310,411,512,613,714')
    parser.add_argument('--output-root', type=Path, default=Path('outputs/training'))
    args = parser.parse_args()
    commands = []
    for seed in [int(s) for s in args.seeds.split(',')]:
        cmd = _baseline_train_cmd(sys.executable, env_nc=Path('data/env_gridded.nc'),
             points=Path('data/geochem_class_table.parquet'), sed_rate_nc=Path('data/Average sedimentation rate.nc'),
             output_dir=args.output_root / f'seed{seed}', seed=seed, split_seed=310, epochs=100, patience=20)
        cmd += ['--split-indices-dir', 'splits/fixed']
        commands.append(cmd)
        if not args.dry_run:
            subprocess.run(cmd, cwd=ROOT, check=True)
    print(json.dumps({'dry_run': args.dry_run, 'commands': commands}, indent=2))

if __name__ == '__main__':
    main()
