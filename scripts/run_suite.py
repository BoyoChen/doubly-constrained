"""Explicit local/server execution, independent of Prefect. Dry-run by default."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / 'code'))
from modules.experiment_helper import parse_experiment_settings

FILES = {'mnist': '01_mnist.yaml', 'structure': '01_mnist.yaml',
         'nmnist': '02_nmnist_t20.yaml', 'sender': '03_sender.yaml'}


def settings(group, label, stage='all', seed=None):
    if group == 'all':
        return [r for g in ['mnist','nmnist','sender'] for r in settings(g,label,stage,seed)]
    rows = parse_experiment_settings(ROOT / 'code/experiments' / FILES[group])
    selected = []
    for row in rows:
        structural = any(row['sub_exp_name'].startswith('mnist_'+c+'_seed') for c in
                         ['post_every','none','pre_every','alternating','simultaneous'])
        if group == 'structure' and not structural:
            continue
        if seed is not None and row['seed'] != seed:
            continue
        prefix = row['sub_exp_name'].startswith('mnist_shared_prefix_')
        if group == 'sender' and ((stage == 'prefix' and not prefix) or
                                  (stage == 'continuation' and prefix)):
            continue
        row = copy.deepcopy(row)
        if group == 'sender' and not prefix:
            checkpoint = ROOT / 'saved_models/paper_doubly_sender' / (
                f"mnist_shared_prefix_e1_seed{row['seed']}") / f'{label}.pickle'
            row['model']['load_from'] = {'path': str(checkpoint)}
        selected.append(row)
    if not selected:
        raise ValueError('No runs match the requested group/stage/seed.')
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('group', choices=['all', *FILES])
    parser.add_argument('--label', default='paper-v3')
    parser.add_argument('--stage', choices=['all', 'prefix', 'continuation'], default='all')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--execute', action='store_true', help='Actually train; Linux training host only.')
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]+', args.label):
        parser.error('label must contain only letters, digits, underscore or hyphen')
    if args.group != 'sender' and args.stage != 'all':
        parser.error('--stage applies only to sender')
    if args.group == 'all' and args.seed is not None:
        parser.error('--seed requires a specific group because datasets use different seed ranges')
    rows = settings(args.group, args.label, args.stage, args.seed)
    for row in rows:
        print(row['sub_exp_name'], 'epochs=', row['training_settings']['max_epoch'],
              'checkpoint=', row['model'].get('load_from', 'from scratch'))
    if not args.execute:
        print(f'DRY RUN: {len(rows)} runs; nothing trained or submitted.')
        return
    if sys.platform == 'win32':
        raise SystemExit('Training is restricted to the Linux training host; use Windows for checks/replots.')
    os.environ['BOYONET_COMMIT_LABEL'] = args.label
    from main import execute_sub_exp
    from modules.utils import configure_runtime_threads
    configure_runtime_threads()
    for row in rows:
        log = ROOT / 'logs' / row['experiment_name'] / row['sub_exp_name'] / args.label
        marker = log / 'reproduction_complete.json'
        digest = hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()
        checkpoint_out = ROOT/'saved_models'/row['experiment_name']/row['sub_exp_name']/f'{args.label}.pickle'
        if marker.exists():
            if json.loads(marker.read_text())['settings_sha256'] != digest:
                raise RuntimeError(f'Completed settings changed; use a new label: {log}')
            if row['training_settings'].get('save_final_model') and not checkpoint_out.is_file():
                raise RuntimeError(f'Completed run is missing its final checkpoint: {checkpoint_out}')
            print('Completed, skipping:', row['sub_exp_name'])
            continue
        if log.exists():
            raise RuntimeError(f'Incomplete/existing run directory; inspect it and use a new label: {log}')
        source = row['model'].get('load_from', {}).get('path')
        if source and not Path(source).is_file():
            raise FileNotFoundError(f'Run the shared prefix first: {source}')
        # The framework's sample-image cache is global, so reset between datasets/seeds.
        from modules import training_helper
        training_helper.cached_sample_data = None
        execute_sub_exp(row, repeat_all_subexp=True)
        if row['experiment_name'].endswith('_v3'):
            epoch = row['training_settings']['max_epoch']
            required = [f'native_decisions_test_epoch{epoch:03d}.npz',
                        f'temporal_contribution_test_epoch{epoch:03d}.npz',
                        f'root_weight_epoch{epoch:03d}.npz']
            for name in required:
                if not (log/'mechanism_artifacts'/name).is_file():
                    raise RuntimeError(f'Missing required final artifact: {log / "mechanism_artifacts" / name}')
        if row['training_settings'].get('save_final_model') and not checkpoint_out.is_file():
            raise RuntimeError(f'No final checkpoint was saved: {checkpoint_out}')
        marker.write_text(json.dumps({'settings_sha256': digest}, indent=2))


if __name__ == '__main__':
    main()
