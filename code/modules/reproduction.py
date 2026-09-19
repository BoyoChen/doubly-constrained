"""Experiment selection, checkpoint dependencies and verified completion tracking."""
import copy
import hashlib
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
HERE = ROOT / "code"
sys.path.insert(0, str(ROOT / 'code'))
from modules.experiment_helper import parse_experiment_settings

FILES = {'mnist': '01_mnist.yaml', 'structure': '01_mnist.yaml',
         'nmnist': '02_nmnist_t20.yaml', 'sender': '03_sender.yaml'}


def settings(group, stage='all', seed=None):
    if group == 'all':
        return [r for g in ['mnist','nmnist','sender'] for r in settings(g,stage,seed)]
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
                f"mnist_shared_prefix_e1_seed{row['seed']}.pickle")
            # Continuations resume the complete prefix model.  The expanded YAML
            # repeats the architecture for documentation, but construct_model
            # intentionally rejects architecture overrides when loading a model.
            row['model'] = {'load_from': {'path': str(checkpoint)}}
        selected.append(row)
    if not selected:
        raise ValueError('No runs match the requested group/stage/seed.')
    return selected


def train(rows, execute_sub_exp):
    os.environ.pop('BOYONET_COMMIT_LABEL', None)
    from modules.utils import configure_runtime_threads
    configure_runtime_threads()
    for row in rows:
        log = ROOT / 'logs' / row['experiment_name'] / row['sub_exp_name']
        marker = log / 'reproduction_complete.json'
        digest = hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()
        checkpoint_out = ROOT/'saved_models'/row['experiment_name']/f"{row['sub_exp_name']}.pickle"
        if marker.exists():
            if json.loads(marker.read_text())['settings_sha256'] != digest:
                raise RuntimeError(f'Completed settings changed; remove or archive the stale run: {log}')
            if row['training_settings'].get('save_final_model') and not checkpoint_out.is_file():
                raise RuntimeError(f'Completed run is missing its final checkpoint: {checkpoint_out}')
            print('Completed, skipping:', row['sub_exp_name'])
            continue
        if log.exists():
            raise RuntimeError(f'Incomplete/existing run directory; inspect or remove it: {log}')
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
