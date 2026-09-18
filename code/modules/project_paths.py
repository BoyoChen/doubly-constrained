from pathlib import Path
import os


MODULES_DIR = Path(__file__).resolve().parent
CODE_ROOT = MODULES_DIR.parent
REPO_ROOT = CODE_ROOT.parent

DATA_DIR = REPO_ROOT / 'data'
LOGS_DIR = REPO_ROOT / 'logs'
SAVED_MODELS_DIR = REPO_ROOT / 'saved_models'
GENE_LIBRARY_DIR = CODE_ROOT / 'gene_library'


def get_commit_label():
    return os.getenv('BOYONET_COMMIT_LABEL', '').strip()


def resolve_path_from_code(path_like):
    path = Path(path_like)
    if path.is_absolute():
        return path
    for candidate in (path, CODE_ROOT / path, REPO_ROOT / path):
        if candidate.exists():
            return candidate.resolve()
    return CODE_ROOT / path


def get_log_path(experiment_name, sub_exp_name):
    base_path = LOGS_DIR / experiment_name / sub_exp_name
    commit_label = get_commit_label()
    return base_path / commit_label if commit_label else base_path


def get_model_save_path(experiment_name, sub_exp_name, commit_label=None):
    base_path = SAVED_MODELS_DIR / experiment_name / sub_exp_name
    if commit_label is None:
        commit_label = get_commit_label()
    return base_path / commit_label if commit_label else base_path


def get_gene_path(gene_id):
    return GENE_LIBRARY_DIR / f'{gene_id}.yaml'
