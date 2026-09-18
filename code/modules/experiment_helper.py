import yaml
import collections
import copy
from pathlib import Path
# from tensorboard.plugins.hparams import api as hp
# from modules.model_IO import get_model_gene

DEFAULT_RANDOM_SEED = 6090
def _is_int_seed(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _get_seed_list(seed):
    if seed is None:
        return [DEFAULT_RANDOM_SEED], False
    if _is_int_seed(seed):
        return [seed], False
    if isinstance(seed, list) and seed and all(_is_int_seed(item) for item in seed):
        return seed, True
    raise ValueError('seed must be an int or a non-empty list of int')


def _apply_base_model_settings(settings):
    model_settings = settings.get('model')
    if model_settings is None:
        model_settings = {}
        settings['model'] = model_settings
    return settings


def parse_experiment_settings(experiment_path, only_this_sub_exp=''):
    experiment_path = Path(experiment_path)
    if experiment_path.is_dir():
        exps_from_all_subpath = []
        subpaths = [
            subpath for subpath in experiment_path.iterdir()
            if subpath.is_dir() or subpath.suffix in {'.yaml', '.yml'}
        ]
        for subpath in subpaths:
            exps_from_all_subpath += parse_experiment_settings(subpath, only_this_sub_exp)
        return exps_from_all_subpath

    # try:
    with open(experiment_path, 'r', encoding='utf-8') as file:
        experiment_settings = yaml.safe_load(file)

    shared_settings = experiment_settings['shared_settings']
    shared_settings['experiment_name'] = experiment_settings['experiment_name']

    def deep_update(source, overrides):
        for key, value in overrides.items():
            if isinstance(value, collections.abc.Mapping) and value:
                returned = deep_update(source.get(key, {}), value)
                source[key] = returned
            else:
                source[key] = value
        return source

    exp_list = []
    for sub_exp_overrides in experiment_settings.get('sub_experiments', []):
        sub_exp_overrides = copy.deepcopy(sub_exp_overrides)
        if 'repeat' in sub_exp_overrides:
            raise ValueError('repeat is no longer supported; use seed: [6090, ...] instead')
        genetic_algorithm = sub_exp_overrides.pop('genetic_algorithm', None)
        base_sub_exp_settings = copy.deepcopy(shared_settings)
        base_sub_exp_settings = deep_update(base_sub_exp_settings, sub_exp_overrides)
        base_sub_exp_settings = _apply_base_model_settings(base_sub_exp_settings)
        seed_list, seed_from_list = _get_seed_list(base_sub_exp_settings.get('seed'))
        for seed in seed_list:
            sub_exp_settings = copy.deepcopy(shared_settings)
            sub_exp_settings = deep_update(sub_exp_settings, sub_exp_overrides)
            sub_exp_settings = _apply_base_model_settings(sub_exp_settings)
            sub_exp_settings['seed'] = seed
            if seed_from_list:
                sub_exp_settings['sub_exp_name'] += f'_seed{seed}'
            if genetic_algorithm:
                from modules.genetic_algorithm import decompress_GA_settings
                GA_exp_settings_list = decompress_GA_settings(sub_exp_settings, **genetic_algorithm)
                exp_list += GA_exp_settings_list
            else:
                exp_list.append(sub_exp_settings)

    if only_this_sub_exp:
        for sub_exp in exp_list:
            if sub_exp['sub_exp_name'] == only_this_sub_exp:
                return sub_exp
        print('sub_exp not found!')
        return []

    return exp_list

    # except (yaml.YAMLError, Exception):
    #     return []


def print_sub_exp_settings(sub_exp_settings, indent_level=0):
    for key, value in sub_exp_settings.items():
        indent = '\t' * indent_level
        if type(value) is dict:
            print(f'{indent}{key}:')
            print_sub_exp_settings(sub_exp_settings[key], indent_level+1)
        else:
            print(f'{indent}{key}: {value}')


def normalize_text_description(value):
    if value is None:
        return ''
    if isinstance(value, str):
        return value
    if isinstance(value, collections.abc.Mapping):
        sections = []
        for key, item in value.items():
            if item is None or item == '':
                continue
            title = str(key).replace('_', ' ').strip().title()
            if isinstance(item, collections.abc.Mapping):
                nested_text = normalize_text_description(item)
                if nested_text:
                    sections.append(f'{title}:\n{nested_text}')
            else:
                sections.append(f'{title}: {item}')
        return '\n\n'.join(sections)
    return str(value)


def get_experiment_text_metadata(sub_exp_settings):
    metadata = {}
    description = normalize_text_description(sub_exp_settings.get('description'))
    if description:
        metadata['description'] = description
    return metadata


def _get_flatten_gene(gene, prefix=None, max_level=2, excluded_key=None):
    prefix = [] if prefix is None else prefix
    excluded_key = ['init_cortex'] if excluded_key is None else excluded_key
    flatten_settings = {}
    for key, value in gene.items():
        if key in excluded_key:
            continue
        if type(value) is dict:
            flatten_settings.update(
                _get_flatten_gene(
                    value,
                    prefix=prefix+[key],
                    max_level=max_level,
                    excluded_key=excluded_key
                )
            )
        else:
            short_name = '/'.join(prefix[-max_level+1:]+[key])
            flatten_settings[short_name] = value
    return flatten_settings


# def record_model_settings_and_score(
#     summary_writer, model_settings, best_valid_score, best_GA_score
# ):
#     gene = get_model_gene(model_settings)
#     flatten_gene = _get_flatten_gene(gene)
#     hp.hparams(flatten_gene)
#     summary_writer.add_scalar('[HPARAMS] valid_score', best_valid_score, global_step=0)
#     summary_writer.add_scalar('[HPARAMS] GA_score', best_GA_score, global_step=0)
