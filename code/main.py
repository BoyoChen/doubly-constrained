import argparse
import os
import pprint
import json
import platform
import torch
from torch.utils.tensorboard import SummaryWriter

from modules.data_loader import get_processed_dataloaders
from modules.experiment_helper import (
    get_experiment_text_metadata,
    parse_experiment_settings,
    print_sub_exp_settings,
)
from modules.model_IO import (
    apply_data_response_receiver_refit,
    construct_model,
    prepare_model_save_path,
)
from modules.model_trainer import train_model
from modules.project_paths import get_log_path, resolve_path_from_code
from modules.utils import (
    DEFAULT_RANDOM_SEED,
    configure_runtime_threads,
    countdown,
    set_global_seed,
)


def execute_sub_exp(sub_exp_settings, repeat_all_subexp):
    experiment_name = sub_exp_settings['experiment_name']
    sub_exp_name = sub_exp_settings['sub_exp_name']
    log_path = get_log_path(experiment_name, sub_exp_name)

    print(f'Executing sub-experiment: {sub_exp_name}', flush=True)
    if not repeat_all_subexp and log_path.is_dir():
        print('Sub-experiment already done before, skipped.', flush=True)
        return 0
    print_sub_exp_settings(sub_exp_settings)

    random_seed = sub_exp_settings.get('seed', DEFAULT_RANDOM_SEED)
    set_global_seed(random_seed)

    summary_writer = SummaryWriter(str(log_path))
    hardware = {
        'hostname': platform.node(),
        'torch_version': str(torch.__version__),
        'cuda_version': torch.version.cuda,
        'gpu': torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU',
        'gpu_total_memory_bytes': (
            torch.cuda.get_device_properties(0).total_memory
            if torch.cuda.is_available() else 0
        ),
    }
    summary_writer.add_text('metadata/hardware', json.dumps(hardware), global_step=0)
    (log_path / 'hardware.json').write_text(json.dumps(hardware, indent=2), encoding='utf-8')
    summary_writer.add_scalar('metadata/random_seed/run', random_seed, global_step=0)
    sub_exp_settings_str = pprint.pformat(sub_exp_settings, indent=4)
    summary_writer.add_text('sub_exp_settings', sub_exp_settings_str, global_step=0)
    for key, value in get_experiment_text_metadata(sub_exp_settings).items():
        summary_writer.add_text(key, value, global_step=0)
    data_settings = dict(sub_exp_settings['data'])
    data_settings.setdefault('seed', random_seed)
    print('Preparing dataloaders...', flush=True)
    dataloaders = get_processed_dataloaders(**data_settings)
    print('Dataloaders ready.', flush=True)
    model_save_path = prepare_model_save_path(experiment_name, sub_exp_name)

    model = construct_model(
        sub_exp_settings['model'],
        run_seed=random_seed,
    )
    apply_data_response_receiver_refit(
        model,
        dataloaders,
        summary_writer=summary_writer,
    )
    train_model(
        model, dataloaders, summary_writer, model_save_path,
        **sub_exp_settings['training_settings']
    )

    summary_writer.close()
    return 1


def main(experiment_path, repeat_all_subexp, waiting_time_before_retry, only_this_sub_exp=''):
    experiment_path = resolve_path_from_code(experiment_path)
    while True:
        experiment_list = parse_experiment_settings(
            experiment_path,
            only_this_sub_exp=only_this_sub_exp,
        )
        if only_this_sub_exp and isinstance(experiment_list, dict):
            experiment_list = [experiment_list]

        executed_sub_exp = 0
        for sub_exp_settings in experiment_list:
            executed_sub_exp += execute_sub_exp(sub_exp_settings, repeat_all_subexp)

        if repeat_all_subexp:
            print('all experiments are done, nothing more to do!')
            break

        if executed_sub_exp == 0:
            print('all experiments are done, nothing more to do!')
            if waiting_time_before_retry == 0:
                break
            countdown(waiting_time_before_retry)


if __name__ == '__main__':
    configure_runtime_threads()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'experiment_path',
        help='name of the experiment setting, should match one of them file name in experiments folder'
    )
    parser.add_argument('-r', '--repeat_all_subexp', action='store_true')
    parser.add_argument('-d', '--CUDA_VISIBLE_DEVICES', type=str, default='')
    parser.add_argument('-w', '--waiting_time_before_retry', type=int, default=0)
    parser.add_argument('--only-sub-exp', type=str, default='')
    args = parser.parse_args()
    if args.CUDA_VISIBLE_DEVICES:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.CUDA_VISIBLE_DEVICES
    main(
        args.experiment_path,
        args.repeat_all_subexp,
        args.waiting_time_before_retry,
        args.only_sub_exp,
    )
