import copy
import torch
import os
import csv
import random
import numpy as np
import pandas as pd
import math
import time
from pathlib import Path


DEFAULT_RANDOM_SEED = 6090


def _get_positive_int_env(env_name):
    raw_value = os.environ.get(env_name)
    if raw_value is None or str(raw_value).strip() == '':
        return None
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f'{env_name} must be a positive integer') from exc
    if value < 1:
        raise ValueError(f'{env_name} must be a positive integer')
    return value


def configure_runtime_threads():
    num_threads = _get_positive_int_env('BOYONET_TORCH_NUM_THREADS')
    if num_threads is None:
        num_threads = _get_positive_int_env('TORCH_NUM_THREADS')
    if num_threads is None:
        return

    torch.set_num_threads(num_threads)
    try:
        torch.set_num_interop_threads(max(1, min(2, num_threads)))
    except RuntimeError:
        pass

    for env_name in (
        'OMP_NUM_THREADS',
        'MKL_NUM_THREADS',
        'OPENBLAS_NUM_THREADS',
        'NUMEXPR_NUM_THREADS',
    ):
        os.environ.setdefault(env_name, str(num_threads))

    print(
        f'Runtime thread cap: torch={torch.get_num_threads()}, '
        f'torch_interop={torch.get_num_interop_threads()}, '
        f'BOYONET_TORCH_NUM_THREADS={num_threads}',
        flush=True,
    )


def countdown(seconds):
    for remaining in range(seconds, 0, -10):
        print(f"resting... {remaining} seconds", end='\r', flush=True)
        time.sleep(10)
    print("Time's up!                   ")


def fast_sigmoid(x):
    """
    Approximate sigmoid using tanh: (tanh(x / 2) + 1) / 2

    Args:
        x (torch.Tensor): Input tensor

    Returns:
        torch.Tensor: Output tensor after applying fast sigmoid
    """
    return 0.5 * (torch.tanh(x / 2) + 1)


def compute_decay_rate(tau, dt):
    return math.exp(-dt / tau)


def transpose_dict_of_dict(d):
    out = {}
    for k1, inner in d.items():
        for k2, v in inner.items():
            out.setdefault(k2, {})[k1] = v
    return out


def remove_nan(tensor_with_nan, new_value):

    if isinstance(tensor_with_nan, torch.Tensor):
        nan_mask = torch.isnan(tensor_with_nan)
        tensor_without_nan = torch.where(nan_mask, new_value, tensor_with_nan)
    elif isinstance(tensor_with_nan, np.ndarray):
        tensor_without_nan = np.nan_to_num(tensor_with_nan, nan=new_value)
    else:
        raise TypeError(f"Unsupported type: {type(tensor_with_nan)}")

    return tensor_without_nan


def _recursive_update_cortex_settings(cortex, **kargs):
    for key, value in kargs.items():
        setattr(cortex, key, value)
    for subcortex in cortex.subcortexs:
        _recursive_update_cortex_settings(subcortex, **kargs)


def copied_model_with_parameters(model, **kargs):
    copied_model = copy.deepcopy(model)
    _recursive_update_cortex_settings(copied_model.cortex, **kargs)
    return copied_model


def move_to_device(obj, device):
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    elif isinstance(obj, list):
        return [move_to_device(item, device) for item in obj]
    elif isinstance(obj, dict):
        return {key: move_to_device(value, device) for key, value in obj.items()}
    elif hasattr(obj, '__dict__'):
        for key, value in obj.__dict__.items():
            setattr(obj, key, move_to_device(value, device))
        return obj
    else:
        return obj


def set_global_seed(seed, deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)


def make_torch_generator(seed):
    generator = torch.Generator(device='cpu')
    generator.manual_seed(seed)
    return generator


def write_dict_to_csv(file_path, data):
    try:
        with open(file_path, 'w', encoding='utf-8', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=data.keys())
            writer.writeheader()
            writer.writerow(data)
    except Exception as e:
        raise RuntimeError(f"Failed to write CSV to {file_path}") from e


def read_dict_from_csv(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            return next(reader)
    except Exception as e:
        raise RuntimeError(f"Failed to read CSV from {file_path}") from e


def get_score_df(log_dir):
    # Get log file paths
    score_file_paths = []
    for (path, dirs, filenames) in os.walk(log_dir):
        for filename in filenames:
            if '.csv' in filename:
                file_path = os.path.join(path, filename)
                score_file_paths.append(file_path)

    # Parse and store as list of tuples
    rows = []
    for path in score_file_paths:
        exp_name = Path(path).parent.name

        # Load the event file
        score_dict = read_dict_from_csv(path)

        rows.append([exp_name, score_dict['epoch'], score_dict['valid_score'], score_dict['GA_score']])

    # Transform into pandas DataFrame
    score_df = pd.DataFrame(rows, columns=['experiment_id', 'step', 'valid_score', 'GA_score'])
    score_df = score_df.set_index('experiment_id')
    return score_df
