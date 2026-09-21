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


def main():
    parser = argparse.ArgumentParser(description='Train experiments and generate the Doubly chapter from one entry point.')
    parser.add_argument('experiment_path', nargs='?', default='experiments',
                        help='experiments (all), or one of its five YAML files')
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument('--dry-run', action='store_true', help='Show selected runs without training or downloads.')
    actions.add_argument('--analyze', action='store_true', help='Generate figures/tables from completed runs only.')
    actions.add_argument('--check', action='store_true', help='Run CPU synthetic checks only.')
    parser.add_argument('--seed', type=int, help='Select one seed within a single experiment.')
    parser.add_argument('--stage', choices=['all','prefix','continuation'], default='all', help='Sender stages only.')
    args = parser.parse_args()
    import subprocess
    import sys
    from modules.reproduction import settings, train, ROOT, FILES
    path=resolve_path_from_code(args.experiment_path).resolve()
    if path==(ROOT/'code/experiments').resolve():
        group='all'
    else:
        matches=[g for g,f in FILES.items() if g!='structure' and path==(ROOT/'code/experiments'/f).resolve()]
        if not matches:parser.error('Choose the experiments directory or one of its five YAML files.')
        group=matches[0]
    if group=='all' and args.seed is not None:parser.error('--seed requires one experiment file')
    if group!='sender' and args.stage!='all':parser.error('--stage applies only to 03_sender.yaml')
    if (args.analyze or args.check) and (args.seed is not None or args.stage!='all'):
        parser.error('--seed/--stage apply to training and dry runs only')
    if args.check:
        for name,extra in [('validate.py',['--construct-models']),('smoke.py',[]),('artifact_smoke.py',[])]:
            subprocess.run([sys.executable,str(ROOT/'tests'/name),*extra],check=True)
        return
    if args.analyze:
        from modules.paper_results import generate_results
        generate_results(sender_only=(group=='sender'))
        return
    rows=settings(group,args.stage,args.seed)
    for row in rows:
        print(row['sub_exp_name'],'epochs=',row['training_settings']['max_epoch'],flush=True)
    if args.dry_run:
        print(f'DRY RUN: {len(rows)} runs; no training, downloads, or job submission.')
        return
    if sys.platform=='win32':
        parser.error('Use a Linux training host for formal runs; Windows supports --check, --dry-run, and --analyze.')
    configure_runtime_threads()
    train(rows,execute_sub_exp)
    if group=='all' or (group=='sender' and args.seed is None and args.stage!='prefix'):
        from modules.paper_results import generate_results
        generate_results(sender_only=(group=='sender'))
    else:
        print('Selected training completed. After completing all experiments, run main.py experiments --analyze.')


if __name__ == '__main__':
    main()
