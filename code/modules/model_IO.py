from pathlib import Path
import copy
import math
import yaml
import pickle
import torch
from modules.structural_normalization import (
    calibrate_normalization_scale, fit_normalized_receiver_mixing,
)
from modules.yielding_ontology.boyonet import BoyoNet
from modules.utils import move_to_device
from modules.project_paths import (
    get_commit_label,
    get_gene_path,
    get_model_save_path,
    resolve_path_from_code,
)


def _new_distribution_accumulator(sample_cap):
    return {
        'count': 0,
        'sum': 0.0,
        'abs_sum': 0.0,
        'square_sum': 0.0,
        'positive_count': 0,
        'positive_sum': 0.0,
        'maximum': -math.inf,
        'sample_cap': int(sample_cap),
        'samples': [],
        'sample_count': 0,
    }


def _update_distribution_accumulator(accumulator, tensor):
    values = tensor.detach().float().reshape(-1)
    finite = torch.isfinite(values)
    if not bool(finite.all()):
        values = values[finite]
    if values.numel() == 0:
        return
    positive = values > 0
    packed = torch.stack([
        values.sum(),
        values.abs().sum(),
        values.square().sum(),
        positive.sum().to(values.dtype),
        values[positive].sum() if bool(positive.any()) else values.new_zeros(()),
        values.max(),
    ]).cpu().tolist()
    accumulator['count'] += int(values.numel())
    accumulator['sum'] += packed[0]
    accumulator['abs_sum'] += packed[1]
    accumulator['square_sum'] += packed[2]
    accumulator['positive_count'] += int(packed[3])
    accumulator['positive_sum'] += packed[4]
    accumulator['maximum'] = max(accumulator['maximum'], packed[5])

    remaining = accumulator['sample_cap'] - accumulator['sample_count']
    if remaining <= 0:
        return
    take = min(2048, remaining, int(values.numel()))
    if take == int(values.numel()):
        sample = values
    else:
        indices = torch.linspace(
            0,
            int(values.numel()) - 1,
            steps=take,
            device=values.device,
        ).long()
        sample = values[indices]
    sample = sample.cpu()
    accumulator['samples'].append(sample)
    accumulator['sample_count'] += int(sample.numel())


def _finalize_distribution_accumulator(accumulator):
    count = accumulator['count']
    if count <= 0:
        return {}
    mean = accumulator['sum'] / count
    rms = math.sqrt(max(accumulator['square_sum'] / count, 0.0))
    variance = max(accumulator['square_sum'] / count - mean * mean, 0.0)
    result = {
        'count': float(count),
        'mean': mean,
        'abs_mean': accumulator['abs_sum'] / count,
        'rms': rms,
        'std': math.sqrt(variance),
        'positive_fraction': accumulator['positive_count'] / count,
        'positive_mean': (
            accumulator['positive_sum'] / accumulator['positive_count']
            if accumulator['positive_count'] > 0
            else 0.0
        ),
        'max': accumulator['maximum'],
    }
    if accumulator['samples']:
        samples = torch.cat(accumulator['samples']).float()
        quantiles = torch.quantile(
            samples,
            torch.tensor([0.5, 0.9, 0.99], dtype=samples.dtype),
        ).tolist()
        result.update({'q50': quantiles[0], 'q90': quantiles[1], 'q99': quantiles[2]})
        result['sample_count'] = float(samples.numel())
    return result


def _collect_data_weighted_potential_audit(
    model,
    source,
    dataloader,
    max_samples,
    sample_cap,
):
    root = model.cortex
    inner = root.subcortexs[0]
    source_root = source.cortex
    cortexes = [('A-1', inner), ('A', root), ('source_A', source_root)]
    accumulators = {}
    for label, _ in cortexes:
        for potential_kind in ('excitatory', 'net'):
            accumulators[(label, potential_kind)] = (
                _new_distribution_accumulator(sample_cap),
                _new_distribution_accumulator(sample_cap),
            )

    device = model.device
    seen = 0
    with torch.no_grad():
        for images, _ in dataloader:
            if seen >= max_samples:
                break
            images = images[:max_samples - seen].to(device)
            seen += len(images)
            model.init_state(images.shape, device)
            source.init_state(images.shape, device)
            for spikes in model.image_encoder(images):
                source.forward_cortex(spikes, is_training=False)
                model.forward_cortex(spikes, is_training=False)
                for label, cortex in cortexes:
                    potentials = {
                        'excitatory': cortex.current_excitatory_potential,
                        'net': cortex.hidden_nv.current_potential,
                    }
                    thresholds = cortex.hidden_nv.thresholds.detach()
                    for potential_kind, potential in potentials.items():
                        if potential is None:
                            raise RuntimeError(
                                f'{label} {potential_kind} potential was not captured'
                            )
                        threshold_shape = [1] * (potential.ndim - 1) + [-1]
                        normalized = potential / thresholds.reshape(threshold_shape)
                        raw_accumulator, normalized_accumulator = accumulators[
                            (label, potential_kind)
                        ]
                        _update_distribution_accumulator(raw_accumulator, potential)
                        _update_distribution_accumulator(
                            normalized_accumulator,
                            normalized,
                        )

    results = {}
    for (label, potential_kind), (
        raw_accumulator,
        normalized_accumulator,
    ) in accumulators.items():
        results[(label, potential_kind, 'raw')] = (
            _finalize_distribution_accumulator(raw_accumulator)
        )
        results[(label, potential_kind, 'threshold_normalized')] = (
            _finalize_distribution_accumulator(normalized_accumulator)
        )
    return results


def prepare_model_save_path(experiment_name, sub_exp_name):
    model_save_path = get_model_save_path(experiment_name, sub_exp_name)
    model_save_path.parent.mkdir(parents=True, exist_ok=True)
    return model_save_path


def load_model_settings(gene_id):
    gene_path = get_gene_path(gene_id)
    if not gene_path.is_file():
        return None
    with gene_path.open('r', encoding='utf-8') as file:
        gene_model_settings = yaml.safe_load(file)
    return gene_model_settings


def save_model_settings(model_settings, gene_id):
    gene_path = get_gene_path(gene_id)

    # mkdir if not exist
    gene_path.parent.mkdir(parents=True, exist_ok=True)

    with gene_path.open('w+', encoding='utf-8') as f:
        yaml.dump(model_settings, f)
    return


def model_setting_update(model_settings, updates):
    for key, value in updates.items():
        if key == 'gene_id':
            continue
        if type(value) is dict and key in model_settings:
            model_setting_update(model_settings[key], value)
        else:
            model_settings[key] = value


def get_model_gene(model_settings):
    if 'gene_id' in model_settings:
        gene_model_settings = load_model_settings(model_settings['gene_id'])
        model_setting_update(gene_model_settings, model_settings)
        return gene_model_settings
    return model_settings


def _resolve_seed_checkpoint_reference(reference, run_seed):
    """Resolve an explicit ``{seed}`` token in a checkpoint reference."""
    resolved = copy.deepcopy(reference)
    if isinstance(resolved, (str, Path)):
        text = str(resolved)
        if '{seed}' not in text:
            return resolved
        if run_seed is None:
            raise ValueError(
                'checkpoint reference contains {seed}, but no run_seed was provided'
            )
        return text.replace('{seed}', str(int(run_seed)))
    if not isinstance(resolved, dict):
        return resolved
    for field in ('path', 'sub_exp_name'):
        value = resolved.get(field)
        if not isinstance(value, str) or '{seed}' not in value:
            continue
        if run_seed is None:
            raise ValueError(
                f'checkpoint reference {field} contains {{seed}}, '
                'but no run_seed was provided'
            )
        resolved[field] = value.replace('{seed}', str(int(run_seed)))
    return resolved


def construct_model(model_settings, run_seed=None):
    model_settings = copy.deepcopy(model_settings)
    if 'load_from' in model_settings:
        allowed_fields = {
            'load_from',
            'post_load_transforms',
            'data_response_receiver_refit',
        }
        unknown_fields = sorted(set(model_settings) - allowed_fields)
        if unknown_fields:
            raise ValueError(
                'model settings with load_from may only contain load_from '
                'post_load_transforms, and data_response_receiver_refit; '
                f'unsupported field(s): {", ".join(unknown_fields)}'
            )
        resolved_load_from = _resolve_seed_checkpoint_reference(
            model_settings['load_from'],
            run_seed,
        )
        model = construct_loaded_model(resolved_load_from)
        apply_post_load_transforms(
            model,
            model_settings.get('post_load_transforms'),
            run_seed=run_seed,
        )
        receiver_refit_settings = copy.deepcopy(
            model_settings.get('data_response_receiver_refit')
        )
        if receiver_refit_settings is not None:
            activity_calibration = receiver_refit_settings.get(
                'upstream_activity_calibration'
            )
            if activity_calibration is not None:
                activity_calibration = copy.deepcopy(activity_calibration)
                if 'reference_load_from' in activity_calibration:
                    activity_calibration['reference_load_from'] = (
                        _resolve_seed_checkpoint_reference(
                            activity_calibration['reference_load_from'],
                            run_seed,
                        )
                    )
                receiver_refit_settings['upstream_activity_calibration'] = (
                    activity_calibration
                )
        model.data_response_receiver_refit_settings = receiver_refit_settings
        model.data_response_receiver_refit_source = copy.deepcopy(
            resolved_load_from
        )
    else:
        if 'post_load_transforms' in model_settings:
            raise ValueError('model.post_load_transforms requires model.load_from')
        resolved_model_settings = get_model_gene(model_settings)
        model = BoyoNet(**resolved_model_settings)
        model.model_settings = resolved_model_settings

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.device = device
    model = move_to_device(model, device)
    return model


def _collect_activity_calibration_images(dataloader, sample_count):
    images = []
    seen = 0
    generator = getattr(dataloader, 'generator', None)
    generator_state = (
        None if generator is None else generator.get_state().clone()
    )
    try:
        for batch_images, _ in dataloader:
            if seen >= sample_count:
                break
            batch_images = batch_images[:sample_count - seen]
            images.append(batch_images.detach().cpu())
            seen += len(batch_images)
    finally:
        if generator_state is not None:
            generator.set_state(generator_state)
    if seen != sample_count:
        raise ValueError(
            'upstream activity calibration requested '
            f'{sample_count} samples but the dataloader provided {seen}'
        )
    return torch.cat(images, dim=0)


def _measure_cortex_output_activity_metric(
    model,
    images,
    cortex_id,
    metric,
    batch_size,
):
    return _measure_cortex_output_activity_metrics(
        model,
        images,
        cortex_id,
        [metric],
        batch_size,
    )[str(metric).lower()]


def _measure_cortex_output_activity_metrics(
    model,
    images,
    cortex_id,
    metrics,
    batch_size,
):
    metrics = [str(metric).lower() for metric in metrics]
    valid_metrics = {
        'output_no_spike_ratio',
        'output_mean_earliness',
        'output_active_mean_earliness',
    }
    unknown_metrics = sorted(set(metrics) - valid_metrics)
    if unknown_metrics:
        raise ValueError(
            'upstream_activity_calibration metric(s) must be '
            'output_no_spike_ratio, output_mean_earliness, or '
            'output_active_mean_earliness; got '
            + ', '.join(unknown_metrics)
        )
    device = model.device
    value_sums = {metric: 0.0 for metric in metrics}
    value_counts = {metric: 0 for metric in metrics}
    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            batch = images[start:start + batch_size].to(device)
            model.init_state(batch.shape, device)
            for spikes in model.image_encoder(batch):
                model.forward_cortex(spikes, is_training=False)
            cortex = model.get_cortex_by_id(cortex_id)
            wave = cortex.output_nv.spike_wave.detach()
            active = wave > 0
            earliness = cortex.output_nv.earliness.detach()
            for metric in metrics:
                if metric == 'output_no_spike_ratio':
                    value_sums[metric] += float((~active).sum().item())
                    value_counts[metric] += int(active.numel())
                elif metric == 'output_mean_earliness':
                    value_sums[metric] += float(earliness.sum().item())
                    value_counts[metric] += int(earliness.numel())
                else:
                    value_sums[metric] += float(earliness[active].sum().item())
                    value_counts[metric] += int(active.sum().item())
    for metric in metrics:
        if value_counts[metric] <= 0:
            raise RuntimeError(
                'upstream activity calibration metric '
                f'{metric} has no eligible values'
            )
    return {
        metric: value_sums[metric] / float(value_counts[metric])
        for metric in metrics
    }


def _apply_upstream_activity_calibration(
    model,
    dataloader,
    settings,
    summary_writer=None,
):
    if isinstance(settings, dict) and 'objectives' in settings:
        return _apply_joint_upstream_activity_calibration(
            model,
            dataloader,
            settings,
            summary_writer=summary_writer,
        )
    if not isinstance(settings, dict):
        raise TypeError(
            'data_response_receiver_refit.upstream_activity_calibration '
            'must be a mapping'
        )
    settings = copy.deepcopy(settings)
    cortex_id = str(settings.pop('cortex_id', 'A-1'))
    reference_cortex_id = str(settings.pop('reference_cortex_id', cortex_id))
    reference_load_from = settings.pop('reference_load_from', None)
    metric = str(settings.pop('metric', 'output_no_spike_ratio')).lower()
    parameter = str(settings.pop('parameter', 'amplifier')).lower()
    fit_samples = int(settings.pop('fit_samples', 128))
    audit_samples = int(settings.pop('audit_samples', 128))
    batch_size = int(settings.pop('batch_size', 64))
    search_iterations = int(settings.pop('search_iterations', 12))
    relative_lower = float(settings.pop('relative_lower', 0.125))
    relative_upper = float(settings.pop('relative_upper', 8.0))
    max_fit_abs_error = float(settings.pop('max_fit_abs_error', 0.005))
    max_audit_abs_error = float(settings.pop('max_audit_abs_error', 0.015))
    if settings:
        raise ValueError(
            'unknown upstream_activity_calibration setting(s): '
            + ', '.join(sorted(settings))
        )
    if reference_load_from is None:
        raise ValueError(
            'upstream_activity_calibration.reference_load_from is required'
        )
    if parameter not in {'amplifier', 'hidden_threshold'}:
        raise ValueError(
            'upstream_activity_calibration.parameter must be amplifier or '
            'hidden_threshold'
        )
    if (
        fit_samples <= 0
        or audit_samples <= 0
        or batch_size <= 0
        or search_iterations <= 0
        or relative_lower <= 0.0
        or relative_upper <= relative_lower
        or max_fit_abs_error < 0.0
        or max_audit_abs_error < 0.0
    ):
        raise ValueError(
            'upstream activity calibration requires positive sample, batch, '
            'iteration, and relative-bound values; upper must exceed lower; '
            'error tolerances must be nonnegative'
        )

    images = _collect_activity_calibration_images(
        dataloader,
        fit_samples + audit_samples,
    )
    fit_images = images[:fit_samples]
    audit_images = images[fit_samples:]
    reference = construct_model({'load_from': reference_load_from})
    reference_fit = _measure_cortex_output_activity_metric(
        reference,
        fit_images,
        reference_cortex_id,
        metric,
        batch_size,
    )
    reference_audit = _measure_cortex_output_activity_metric(
        reference,
        audit_images,
        reference_cortex_id,
        metric,
        batch_size,
    )
    del reference
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    cortex = model.get_cortex_by_id(cortex_id)
    if parameter == 'amplifier':
        initial_parameter = float(cortex.amplifier)
    else:
        initial_thresholds = cortex.hidden_nv.thresholds.detach()
        if not bool(torch.allclose(
            initial_thresholds,
            initial_thresholds.mean().expand_as(initial_thresholds),
            atol=1.0e-6,
            rtol=1.0e-6,
        )):
            raise ValueError(
                'hidden-threshold activity calibration requires one common '
                'initial threshold'
            )
        initial_parameter = float(initial_thresholds.mean().item())

    def set_drive_ratio(drive_ratio):
        if parameter == 'amplifier':
            value = initial_parameter * drive_ratio
            set_cortex_amplifier(model, cortex_id, value)
        else:
            value = initial_parameter / drive_ratio
            set_cortex_thresholds(model, cortex_id, hidden_threshold=value)
        return value

    def measure_ratio(drive_ratio):
        value = set_drive_ratio(drive_ratio)
        measured = _measure_cortex_output_activity_metric(
            model,
            fit_images,
            cortex_id,
            metric,
            batch_size,
        )
        return value, measured

    set_drive_ratio(1.0)
    before_fit = _measure_cortex_output_activity_metric(
        model,
        fit_images,
        cortex_id,
        metric,
        batch_size,
    )
    candidates = []
    low = relative_lower
    high = relative_upper
    low_parameter, low_metric = measure_ratio(low)
    high_parameter, high_metric = measure_ratio(high)
    candidates.extend([
        (abs(low_metric - reference_fit), low, low_parameter, low_metric),
        (abs(high_metric - reference_fit), high, high_parameter, high_metric),
    ])
    metric_increases_with_drive = metric != 'output_no_spike_ratio'
    ordered_low = low_metric if metric_increases_with_drive else high_metric
    ordered_high = high_metric if metric_increases_with_drive else low_metric
    if not ordered_low <= reference_fit <= ordered_high:
        raise ValueError(
            'upstream activity calibration could not bracket the reference '
            f'target {reference_fit:.8f}; relative bounds produced '
            f'{low_metric:.8f} and {high_metric:.8f}'
        )
    for _ in range(search_iterations):
        mid = math.sqrt(low * high)
        mid_parameter, mid_metric = measure_ratio(mid)
        candidates.append((
            abs(mid_metric - reference_fit),
            mid,
            mid_parameter,
            mid_metric,
        ))
        if metric_increases_with_drive:
            if mid_metric < reference_fit:
                low = mid
            else:
                high = mid
        else:
            if mid_metric > reference_fit:
                low = mid
            else:
                high = mid
    fit_abs_error, drive_ratio, final_parameter, calibrated_fit = min(
        candidates,
        key=lambda item: item[0],
    )
    set_drive_ratio(drive_ratio)
    calibrated_audit = _measure_cortex_output_activity_metric(
        model,
        audit_images,
        cortex_id,
        metric,
        batch_size,
    )
    audit_abs_error = abs(calibrated_audit - reference_audit)
    if fit_abs_error > max_fit_abs_error:
        raise RuntimeError(
            'upstream activity calibration fit error '
            f'{fit_abs_error:.8f} exceeds {max_fit_abs_error:.8f}'
        )
    if audit_abs_error > max_audit_abs_error:
        raise RuntimeError(
            'upstream activity calibration held-out train error '
            f'{audit_abs_error:.8f} exceeds {max_audit_abs_error:.8f}'
        )

    audit = {
        'cortex_id': cortex_id,
        'reference_cortex_id': reference_cortex_id,
        'metric': metric,
        'parameter': parameter,
        'fit_samples': fit_samples,
        'audit_samples': audit_samples,
        'reference_fit_metric': reference_fit,
        'reference_audit_metric': reference_audit,
        'candidate_before_fit_metric': before_fit,
        'candidate_after_fit_metric': calibrated_fit,
        'candidate_after_audit_metric': calibrated_audit,
        'fit_abs_error': fit_abs_error,
        'audit_abs_error': audit_abs_error,
        'initial_parameter': initial_parameter,
        'final_parameter': final_parameter,
        'drive_ratio': drive_ratio,
    }
    model.upstream_activity_calibration_audit = audit
    if summary_writer is not None:
        prefix = 'calibration/upstream_activity'
        for name, value in audit.items():
            if isinstance(value, (int, float)):
                summary_writer.add_scalar(
                    f'{prefix}/{name}',
                    value,
                    global_step=0,
                )
        summary_writer.add_text(
            f'{prefix}/metric',
            metric,
            global_step=0,
        )
        summary_writer.add_text(
            f'{prefix}/parameter',
            parameter,
            global_step=0,
        )
    return audit


def _apply_joint_upstream_activity_calibration(
    model,
    dataloader,
    settings,
    summary_writer=None,
):
    settings = copy.deepcopy(settings)
    cortex_id = str(settings.pop('cortex_id', 'A-1'))
    reference_cortex_id = str(settings.pop('reference_cortex_id', cortex_id))
    reference_load_from = settings.pop('reference_load_from', None)
    objectives = settings.pop('objectives')
    fit_samples = int(settings.pop('fit_samples', 128))
    audit_samples = int(settings.pop('audit_samples', 128))
    batch_size = int(settings.pop('batch_size', 64))
    search_iterations = int(settings.pop('search_iterations', 10))
    finite_difference_log_step = float(
        settings.pop('finite_difference_log_step', 0.08)
    )
    solver_damping = float(settings.pop('solver_damping', 1.0e-4))
    max_log_step = float(settings.pop('max_log_step', 0.7))
    relative_lower = float(settings.pop('relative_lower', 0.125))
    relative_upper = float(settings.pop('relative_upper', 8.0))
    max_fit_abs_error = float(settings.pop('max_fit_abs_error', 0.005))
    max_audit_abs_error = float(settings.pop('max_audit_abs_error', 0.015))
    if settings:
        raise ValueError(
            'unknown joint upstream_activity_calibration setting(s): '
            + ', '.join(sorted(settings))
        )
    if reference_load_from is None:
        raise ValueError(
            'joint upstream_activity_calibration.reference_load_from is '
            'required'
        )
    if not isinstance(objectives, list) or len(objectives) != 2:
        raise ValueError(
            'joint upstream_activity_calibration.objectives must contain '
            'exactly two mappings'
        )
    parsed_objectives = []
    valid_metrics = {
        'output_no_spike_ratio',
        'output_mean_earliness',
        'output_active_mean_earliness',
    }
    valid_parameters = {'amplifier', 'hidden_threshold'}
    for index, objective in enumerate(objectives):
        if not isinstance(objective, dict):
            raise TypeError(
                'joint upstream_activity_calibration objective '
                f'{index} must be a mapping'
            )
        objective = copy.deepcopy(objective)
        metric = str(objective.pop('metric')).lower()
        parameter = str(objective.pop('parameter')).lower()
        objective_lower = float(
            objective.pop('relative_lower', relative_lower)
        )
        objective_upper = float(
            objective.pop('relative_upper', relative_upper)
        )
        if objective:
            raise ValueError(
                'unknown joint upstream activity objective setting(s): '
                + ', '.join(sorted(objective))
            )
        if metric not in valid_metrics:
            raise ValueError(
                'joint upstream activity metric must be '
                'output_no_spike_ratio, output_mean_earliness, or '
                'output_active_mean_earliness'
            )
        if parameter not in valid_parameters:
            raise ValueError(
                'joint upstream activity parameter must be amplifier or '
                'hidden_threshold'
            )
        if objective_lower <= 0.0 or objective_upper <= objective_lower:
            raise ValueError(
                'joint upstream activity objective bounds must be positive '
                'and upper must exceed lower'
            )
        parsed_objectives.append({
            'metric': metric,
            'parameter': parameter,
            'relative_lower': objective_lower,
            'relative_upper': objective_upper,
        })
    if len({item['metric'] for item in parsed_objectives}) != 2:
        raise ValueError(
            'joint upstream activity calibration requires two distinct '
            'metrics'
        )
    if len({item['parameter'] for item in parsed_objectives}) != 2:
        raise ValueError(
            'joint upstream activity calibration requires amplifier and '
            'hidden_threshold exactly once each'
        )
    if (
        fit_samples <= 0
        or audit_samples <= 0
        or batch_size <= 0
        or search_iterations <= 0
        or finite_difference_log_step <= 0.0
        or solver_damping < 0.0
        or max_log_step <= 0.0
        or max_fit_abs_error < 0.0
        or max_audit_abs_error < 0.0
    ):
        raise ValueError(
            'joint upstream activity calibration requires positive sample, '
            'batch, search, finite-difference, and step values; damping and '
            'error tolerances must be nonnegative'
        )

    images = _collect_activity_calibration_images(
        dataloader,
        fit_samples + audit_samples,
    )
    fit_images = images[:fit_samples]
    audit_images = images[fit_samples:]
    reference = construct_model({'load_from': reference_load_from})
    metrics = [objective['metric'] for objective in parsed_objectives]
    reference_fit_metrics = _measure_cortex_output_activity_metrics(
        reference,
        fit_images,
        reference_cortex_id,
        metrics,
        batch_size,
    )
    reference_audit_metrics = _measure_cortex_output_activity_metrics(
        reference,
        audit_images,
        reference_cortex_id,
        metrics,
        batch_size,
    )
    for objective in parsed_objectives:
        metric = objective['metric']
        objective['reference_fit_metric'] = reference_fit_metrics[metric]
        objective['reference_audit_metric'] = reference_audit_metrics[metric]
    del reference
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    cortex = model.get_cortex_by_id(cortex_id)
    initial_amplifier = float(cortex.amplifier)
    initial_thresholds = cortex.hidden_nv.thresholds.detach()
    if not bool(torch.allclose(
        initial_thresholds,
        initial_thresholds.mean().expand_as(initial_thresholds),
        atol=1.0e-6,
        rtol=1.0e-6,
    )):
        raise ValueError(
            'joint hidden-threshold activity calibration requires one '
            'common initial threshold'
        )
    initial_threshold = float(initial_thresholds.mean().item())

    def set_parameter_drive_ratio(parameter, drive_ratio):
        if parameter == 'amplifier':
            value = initial_amplifier * drive_ratio
            set_cortex_amplifier(model, cortex_id, value)
        else:
            value = initial_threshold / drive_ratio
            set_cortex_thresholds(model, cortex_id, hidden_threshold=value)
        return value

    def get_parameter_drive_ratio(parameter):
        current = model.get_cortex_by_id(cortex_id)
        if parameter == 'amplifier':
            return float(current.amplifier) / initial_amplifier
        threshold = float(current.hidden_nv.thresholds.mean().item())
        return initial_threshold / threshold

    parameter_order = ['amplifier', 'hidden_threshold']
    objective_by_parameter = {
        objective['parameter']: objective
        for objective in parsed_objectives
    }
    log_bounds = [
        (
            math.log(objective_by_parameter[parameter]['relative_lower']),
            math.log(objective_by_parameter[parameter]['relative_upper']),
        )
        for parameter in parameter_order
    ]
    targets = torch.tensor(
        [objective['reference_fit_metric'] for objective in parsed_objectives],
        dtype=torch.float64,
    )
    evaluation_cache = {}
    candidate_records = []

    def clamp_coordinates(coordinates):
        return tuple(
            min(max(float(value), lower), upper)
            for value, (lower, upper) in zip(coordinates, log_bounds)
        )

    def evaluate(coordinates):
        coordinates = clamp_coordinates(coordinates)
        key = tuple(round(value, 12) for value in coordinates)
        if key in evaluation_cache:
            return evaluation_cache[key]
        for parameter, log_ratio in zip(parameter_order, coordinates):
            set_parameter_drive_ratio(parameter, math.exp(log_ratio))
        try:
            measured = _measure_cortex_output_activity_metrics(
                model,
                fit_images,
                cortex_id,
                metrics,
                batch_size,
            )
        except RuntimeError as error:
            if 'has no eligible values' not in str(error):
                raise
            measured = _measure_cortex_output_activity_metrics(
                model,
                fit_images,
                cortex_id,
                ['output_no_spike_ratio', 'output_mean_earliness'],
                batch_size,
            )
            for metric in metrics:
                measured.setdefault(metric, 0.0)
        values = torch.tensor(
            [measured[metric] for metric in metrics],
            dtype=torch.float64,
        )
        errors = torch.abs(values - targets)
        score = (float(errors.max()), float(torch.sum(errors.square())))
        result = {
            'coordinates': coordinates,
            'values': values,
            'errors': errors,
            'score': score,
        }
        evaluation_cache[key] = result
        candidate_records.append(result)
        return result

    current = evaluate((0.0, 0.0))
    for objective_index, objective in enumerate(parsed_objectives):
        objective['candidate_before_fit_metric'] = float(
            current['values'][objective_index]
        )

    solver_records = []
    completed_solver_iterations = 0
    for solver_index in range(search_iterations):
        if current['score'][0] <= max_fit_abs_error:
            break
        jacobian = torch.empty((2, 2), dtype=torch.float64)
        for parameter_index in range(2):
            low_coordinate = list(current['coordinates'])
            high_coordinate = list(current['coordinates'])
            low_coordinate[parameter_index] -= finite_difference_log_step
            high_coordinate[parameter_index] += finite_difference_log_step
            low_result = evaluate(low_coordinate)
            high_result = evaluate(high_coordinate)
            denominator = (
                high_result['coordinates'][parameter_index]
                - low_result['coordinates'][parameter_index]
            )
            if denominator <= 0.0:
                jacobian[:, parameter_index] = 0.0
            else:
                jacobian[:, parameter_index] = (
                    high_result['values'] - low_result['values']
                ) / denominator
        residual = targets - current['values']
        normal = jacobian.T @ jacobian
        normal += solver_damping * torch.eye(2, dtype=torch.float64)
        right = jacobian.T @ residual
        try:
            delta = torch.linalg.solve(normal, right)
        except RuntimeError:
            delta = torch.linalg.pinv(jacobian) @ residual
        largest_step = float(torch.abs(delta).max())
        if largest_step > max_log_step:
            delta *= max_log_step / largest_step

        line_candidates = []
        for line_scale in (1.0, 0.5, 0.25, 0.125):
            proposed = tuple(
                coordinate + line_scale * float(step)
                for coordinate, step in zip(current['coordinates'], delta)
            )
            line_candidates.append(evaluate(proposed))
        proposed_best = min(line_candidates, key=lambda item: item['score'])
        if proposed_best['score'] < current['score']:
            current = proposed_best
        else:
            current = min(candidate_records, key=lambda item: item['score'])
        completed_solver_iterations = solver_index + 1
        solver_records.append({
            'solver_iteration': solver_index,
            'max_fit_abs_error': current['score'][0],
            'sum_squared_fit_error': current['score'][1],
            'amplifier_drive_ratio': math.exp(current['coordinates'][0]),
            'threshold_drive_ratio': math.exp(current['coordinates'][1]),
            'jacobian_determinant': float(torch.linalg.det(jacobian)),
        })

    best = min(candidate_records, key=lambda item: item['score'])
    for parameter, log_ratio in zip(parameter_order, best['coordinates']):
        set_parameter_drive_ratio(parameter, math.exp(log_ratio))

    final_amplifier = float(model.get_cortex_by_id(cortex_id).amplifier)
    final_threshold = float(
        model.get_cortex_by_id(cortex_id).hidden_nv.thresholds.mean().item()
    )
    calibrated_fit_metrics = _measure_cortex_output_activity_metrics(
        model,
        fit_images,
        cortex_id,
        metrics,
        batch_size,
    )
    calibrated_audit_metrics = _measure_cortex_output_activity_metrics(
        model,
        audit_images,
        cortex_id,
        metrics,
        batch_size,
    )
    for objective in parsed_objectives:
        calibrated_fit = calibrated_fit_metrics[objective['metric']]
        calibrated_audit = calibrated_audit_metrics[objective['metric']]
        fit_abs_error = abs(
            calibrated_fit - objective['reference_fit_metric']
        )
        audit_abs_error = abs(
            calibrated_audit - objective['reference_audit_metric']
        )
        objective['candidate_after_fit_metric'] = calibrated_fit
        objective['candidate_after_audit_metric'] = calibrated_audit
        objective['fit_abs_error'] = fit_abs_error
        objective['audit_abs_error'] = audit_abs_error
        if fit_abs_error > max_fit_abs_error:
            raise RuntimeError(
                'joint upstream activity calibration fit error for '
                f'{objective["metric"]} {fit_abs_error:.8f} exceeds '
                f'{max_fit_abs_error:.8f}'
            )
        if audit_abs_error > max_audit_abs_error:
            raise RuntimeError(
                'joint upstream activity calibration held-out train error '
                f'for {objective["metric"]} {audit_abs_error:.8f} exceeds '
                f'{max_audit_abs_error:.8f}'
            )

    audit = {
        'cortex_id': cortex_id,
        'reference_cortex_id': reference_cortex_id,
        'fit_samples': fit_samples,
        'audit_samples': audit_samples,
        'solver_iterations': completed_solver_iterations,
        'evaluation_count': len(evaluation_cache),
        'initial_amplifier': initial_amplifier,
        'final_amplifier': final_amplifier,
        'amplifier_drive_ratio': get_parameter_drive_ratio('amplifier'),
        'initial_threshold': initial_threshold,
        'final_threshold': final_threshold,
        'threshold_drive_ratio': get_parameter_drive_ratio('hidden_threshold'),
        'objectives': parsed_objectives,
        'solver_records': solver_records,
    }
    model.upstream_activity_calibration_audit = audit
    if summary_writer is not None:
        prefix = 'calibration/upstream_activity_joint'
        for name in (
            'fit_samples',
            'audit_samples',
            'solver_iterations',
            'evaluation_count',
            'initial_amplifier',
            'final_amplifier',
            'amplifier_drive_ratio',
            'initial_threshold',
            'final_threshold',
            'threshold_drive_ratio',
        ):
            summary_writer.add_scalar(
                f'{prefix}/{name}',
                audit[name],
                global_step=0,
            )
        for index, objective in enumerate(parsed_objectives):
            objective_prefix = f'{prefix}/objective_{index}'
            for name in (
                'reference_fit_metric',
                'reference_audit_metric',
                'candidate_before_fit_metric',
                'candidate_after_fit_metric',
                'candidate_after_audit_metric',
                'fit_abs_error',
                'audit_abs_error',
            ):
                summary_writer.add_scalar(
                    f'{objective_prefix}/{name}',
                    objective[name],
                    global_step=0,
                )
            summary_writer.add_text(
                f'{objective_prefix}/metric',
                objective['metric'],
                global_step=0,
            )
            summary_writer.add_text(
                f'{objective_prefix}/parameter',
                objective['parameter'],
                global_step=0,
            )
        for record_index, record in enumerate(solver_records):
            record_prefix = f'{prefix}/solver_{record_index}'
            for name, value in record.items():
                summary_writer.add_scalar(
                    f'{record_prefix}/{name}',
                    value,
                    global_step=0,
                )
    return audit


def apply_data_response_receiver_refit(
    model,
    dataloaders,
    summary_writer=None,
):
    settings = getattr(model, 'data_response_receiver_refit_settings', None)
    if not settings:
        return model
    settings = copy.deepcopy(settings)
    source_load_from = settings.pop('source_load_from', None)
    phase = str(settings.pop('phase', 'train'))
    max_samples = int(settings.pop('max_samples', 256))
    max_rows = int(settings.pop('max_rows', 2048))
    ridge = float(settings.pop('ridge', 1.0e-3))
    blend = float(settings.pop('blend', 1.0))
    solve_mode = str(settings.pop('solve_mode', 'global_mixing')).lower()
    target_mode = str(
        settings.pop('target_mode', 'source_excitatory')
    ).lower()
    source_net_target_blend = float(
        settings.pop('source_net_target_blend', 1.0)
    )
    row_weight_mode = str(
        settings.pop('row_weight_mode', 'uniform')
    ).lower()
    nonnegative_iterations = int(
        settings.pop('nonnegative_iterations', 40)
    )
    linear_algebra_device = str(
        settings.pop('linear_algebra_device', 'model')
    ).lower()
    post_refit_branch_scale = float(
        settings.pop('post_refit_branch_scale', 1.0)
    )
    direct_residual_scale = float(
        settings.pop('direct_residual_scale', 1.0)
    )
    direct_residual_positive_scale = float(
        settings.pop('direct_residual_positive_scale', 1.0)
    )
    direct_residual_negative_scale = float(
        settings.pop('direct_residual_negative_scale', 1.0)
    )
    direct_residual_ridge_multiplier = float(
        settings.pop('direct_residual_ridge_multiplier', 1.0)
    )
    direct_residual_time_bin_count = int(settings.pop(
        'direct_residual_time_bin_count',
        1,
    ))
    branch_time_bin_count = int(settings.pop(
        'branch_time_bin_count',
        1,
    ))
    direct_residual_state_basis = str(settings.pop(
        'direct_residual_state_basis',
        'none',
    )).lower()
    direct_residual_row_sampling_mode = str(settings.pop(
        'direct_residual_row_sampling_mode',
        'legacy_global_stride',
    )).lower()
    direct_residual_time_audit_rows_per_step = int(settings.pop(
        'direct_residual_time_audit_rows_per_step',
        0,
    ))
    direct_residual_time_audit_hash_seed = int(settings.pop(
        'direct_residual_time_audit_hash_seed',
        1704000,
    ))
    direct_residual_time_audit_disjoint = bool(settings.pop(
        'direct_residual_time_audit_disjoint',
        False,
    ))
    post_refit_root_threshold = settings.pop(
        'post_refit_root_threshold', None
    )
    if post_refit_root_threshold is not None:
        post_refit_root_threshold = _positive_float(
            post_refit_root_threshold,
            'post_refit_root_threshold',
        )
    post_refit_root_input_wave_delay_steps = settings.pop(
        'post_refit_root_input_wave_delay_steps', None
    )
    if post_refit_root_input_wave_delay_steps is not None:
        post_refit_root_input_wave_delay_steps = _nonnegative_int(
            post_refit_root_input_wave_delay_steps,
            'post_refit_root_input_wave_delay_steps',
        )
    post_refit_root_input_wave_handoff_branch_activity_fraction = settings.pop(
        'post_refit_root_input_wave_handoff_branch_activity_fraction', None
    )
    if post_refit_root_input_wave_handoff_branch_activity_fraction is not None:
        post_refit_root_input_wave_handoff_branch_activity_fraction = float(
            post_refit_root_input_wave_handoff_branch_activity_fraction
        )
        if not (
            0.0
            < post_refit_root_input_wave_handoff_branch_activity_fraction
            <= 1.0
        ):
            raise ValueError(
                'post_refit_root_input_wave_handoff_branch_activity_fraction '
                'must be in (0, 1]'
            )
    post_refit_potential_gauges = settings.pop(
        'post_refit_potential_gauges', []
    )
    upstream_activity_calibration = settings.pop(
        'upstream_activity_calibration', None
    )
    normalization_scale_calibration = settings.pop('normalization_scale_calibration', None)
    normalization_aware_refit = settings.pop('normalization_aware_refit', None)
    native_onset_mode = 'none'
    native_recurrence_mode = 'fixed_surrogate'
    if normalization_aware_refit is not None:
        if solve_mode != 'global_mixing' or not isinstance(normalization_aware_refit, dict):
            raise ValueError('normalization_aware_refit requires global_mixing and a settings mapping')
        normalization_aware_refit = dict(normalization_aware_refit)
        native_onset_mode = str(normalization_aware_refit.pop(
            'native_onset_mode', 'none',
        )).lower()
        native_recurrence_mode = str(normalization_aware_refit.pop(
            'native_recurrence_mode', 'fixed_surrogate',
        )).lower()
        if native_onset_mode not in {
            'none', 'positive_only', 'pre_onset_only', 'balanced',
        }:
            raise ValueError(
                'normalization_aware_refit.native_onset_mode must be none, '
                'positive_only, pre_onset_only, or balanced'
            )
        if native_recurrence_mode not in {'fixed_surrogate', 'hard_unroll'}:
            raise ValueError(
                'normalization_aware_refit.native_recurrence_mode must be '
                'fixed_surrogate or hard_unroll'
            )
        if native_onset_mode == 'none' and native_recurrence_mode != 'fixed_surrogate':
            raise ValueError(
                'hard native recurrence requires an active native_onset_mode'
            )
    if not isinstance(post_refit_potential_gauges, list):
        raise TypeError(
            'data_response_receiver_refit.post_refit_potential_gauges '
            'must be a list of mappings'
        )
    refund_root_amplifier = bool(
        settings.pop('refund_root_amplifier', False)
    )
    if normalization_scale_calibration is not None:
        if not isinstance(normalization_scale_calibration, dict) or solve_mode != 'global_mixing':
            raise ValueError('normalization_scale_calibration requires global_mixing and a mapping')
    if normalization_aware_refit is not None or normalization_scale_calibration is not None:
        if post_refit_branch_scale != 1.0 or refund_root_amplifier or post_refit_potential_gauges:
            raise ValueError('normalization initialization cannot be mixed with other post-refit gauges')
    data_weighted_potential_audit = bool(
        settings.pop('data_weighted_potential_audit', False)
    )
    potential_audit_max_samples = int(
        settings.pop('potential_audit_max_samples', max_samples)
    )
    potential_audit_sample_cap = int(
        settings.pop('potential_audit_sample_cap', 131072)
    )
    if settings:
        raise ValueError(
            'unknown data_response_receiver_refit setting(s): '
            + ', '.join(sorted(settings))
        )
    if phase not in dataloaders:
        raise ValueError(
            f'data_response_receiver_refit phase {phase!r} is unavailable'
        )
    if (
        max_samples <= 0
        or max_rows <= 0
        or ridge <= 0.0
        or post_refit_branch_scale <= 0.0
        or direct_residual_scale < 0.0
        or direct_residual_positive_scale < 0.0
        or direct_residual_negative_scale < 0.0
        or direct_residual_ridge_multiplier <= 0.0
        or direct_residual_time_bin_count <= 0
        or branch_time_bin_count <= 0
        or direct_residual_time_audit_rows_per_step < 0
        or direct_residual_time_audit_hash_seed < 0
        or not 0.0 <= blend <= 1.0
        or nonnegative_iterations <= 0
        or potential_audit_max_samples <= 0
        or potential_audit_sample_cap <= 0
        or potential_audit_max_samples > max_samples
    ):
        raise ValueError(
            'data_response_receiver_refit requires max_samples/max_rows > 0, '
            'ridge/post_refit_branch_scale/direct_residual_ridge_multiplier/'
            'nonnegative_iterations > 0, direct residual scales >= 0, '
            'direct_residual_time_bin_count/branch_time_bin_count > 0, '
            'direct_residual_time_audit_rows_per_step >= 0, '
            'direct_residual_time_audit_hash_seed >= 0, '
            'potential_audit_max_samples/potential_audit_sample_cap > 0, '
            'potential_audit_max_samples <= max_samples, '
            'and blend in [0, 1]'
        )
    direct_residual_row_sampling_modes = {
        'legacy_global_stride',
        'even_time_balanced',
        'odd_time_balanced',
        'all_time_fixed_sample_balanced',
        'all_time_rotating_sample_balanced',
        'all_time_hashed_balanced',
    }
    if direct_residual_row_sampling_mode not in direct_residual_row_sampling_modes:
        raise ValueError(
            'data_response_receiver_refit.direct_residual_row_sampling_mode '
            'must be one of '
            + ', '.join(sorted(direct_residual_row_sampling_modes))
        )
    direct_residual_state_bases = {
        'none',
        'branch_activity_linear',
        'branch_activity_quadratic',
        'branch_activity_cubic',
        'direct_activity_linear',
        'branch_and_direct_activity_linear',
        'branch_spatial_cumulative_linear',
        'branch_spatial_cumulative_centered_linear',
        'branch_spatial_instantaneous_centered_linear',
        'branch_spatial_cumulative_and_instantaneous_centered_linear',
    }
    if direct_residual_state_basis not in direct_residual_state_bases:
        raise ValueError(
            'data_response_receiver_refit.direct_residual_state_basis '
            'must be one of '
            + ', '.join(sorted(direct_residual_state_bases))
        )
    spatial_state_requested = direct_residual_state_basis.startswith(
        'branch_spatial_'
    )
    if solve_mode not in {
        'global_mixing',
        'common_to_contrast_blocked_mixing',
        'branch_primary_residual',
        'direct_dual',
        'direct_residual_dual',
        'direct_nonnegative',
    }:
        raise ValueError(
            'data_response_receiver_refit.solve_mode must be '
            '"global_mixing", "common_to_contrast_blocked_mixing", '
            '"branch_primary_residual", "direct_dual", '
            '"direct_residual_dual", or '
            '"direct_nonnegative"'
        )
    if branch_time_bin_count != 1:
        if solve_mode != 'global_mixing':
            raise ValueError(
                'data_response_receiver_refit.branch_time_bin_count > 1 '
                'requires solve_mode="global_mixing"'
            )
        if (
            post_refit_branch_scale != 1.0
            or refund_root_amplifier
            or post_refit_potential_gauges
        ):
            raise ValueError(
                'time-binned branch offsets require an unscaled branch, no '
                'amplifier refund, and no post-refit potential gauges'
            )
    if target_mode not in {
        'source_excitatory',
        'source_net_compensated',
        'source_excitatory_common_net_contrast',
        'source_excitatory_common_net_contrast_rms_matched',
    }:
        raise ValueError(
            'data_response_receiver_refit.target_mode must be '
            '"source_excitatory", "source_net_compensated", '
            '"source_excitatory_common_net_contrast", or '
            '"source_excitatory_common_net_contrast_rms_matched"'
        )
    if row_weight_mode not in {
        'uniform',
        'source_output_onset_balanced',
        'source_output_active_balanced',
    }:
        raise ValueError(
            'data_response_receiver_refit.row_weight_mode must be '
            '"uniform", "source_output_onset_balanced", or '
            '"source_output_active_balanced"'
        )
    if not 0.0 <= source_net_target_blend <= 1.0:
        raise ValueError(
            'data_response_receiver_refit.source_net_target_blend '
            'must be in [0, 1]'
        )
    if linear_algebra_device not in {'model', 'cpu'}:
        raise ValueError(
            'data_response_receiver_refit.linear_algebra_device must be '
            '"model" or "cpu"'
        )
    if solve_mode == 'branch_primary_residual':
        if target_mode != 'source_excitatory' or row_weight_mode != 'uniform':
            raise ValueError(
                'branch_primary_residual currently requires '
                'target_mode="source_excitatory" and '
                'row_weight_mode="uniform"'
            )
        if (
            post_refit_branch_scale != 1.0
            or refund_root_amplifier
            or post_refit_potential_gauges
        ):
            raise ValueError(
                'branch_primary_residual requires an unscaled post-refit '
                'branch, no amplifier refund, and no post-refit potential '
                'gauges so the fitted direct residual remains valid'
            )
        if (
            direct_residual_state_basis != 'none'
            and direct_residual_time_bin_count != 1
        ):
            raise ValueError(
                'state-conditioned direct residual and time-binned direct '
                'residual are mutually exclusive diagnostics'
            )
    elif (
        direct_residual_scale != 1.0
        or direct_residual_positive_scale != 1.0
        or direct_residual_negative_scale != 1.0
        or direct_residual_ridge_multiplier != 1.0
        or direct_residual_time_bin_count != 1
        or direct_residual_state_basis != 'none'
        or direct_residual_row_sampling_mode != 'legacy_global_stride'
        or direct_residual_time_audit_rows_per_step != 0
    ):
        raise ValueError(
            'data_response_receiver_refit direct residual sign/scale/ridge '
            'controls are only available for '
            'solve_mode="branch_primary_residual"'
        )

    if source_load_from is None:
        source_load_from = model.data_response_receiver_refit_source
    if upstream_activity_calibration is not None:
        _apply_upstream_activity_calibration(
            model,
            dataloaders[phase],
            upstream_activity_calibration,
            summary_writer=summary_writer,
        )
    source = construct_model({'load_from': source_load_from})
    device = model.device
    source = move_to_device(source, device)
    root = model.cortex
    source_root = source.cortex
    if len(root.subcortexs) != 1:
        raise ValueError(
            'data_response_receiver_refit requires exactly one yielded subcortex'
        )
    if (
        native_recurrence_mode == 'hard_unroll'
        and float(root.hidden_nv.noise_settings.get('noise_rate', 0.0)) != 0.0
    ):
        raise ValueError('hard native recurrence requires zero root spike noise')
    root_threshold_before_refit_override = float(
        root.hidden_nv.thresholds.mean().item()
    )

    direct_len = len(root.input_nv)
    if direct_len >= int(root.kernel.weight.shape[0]):
        raise ValueError('yielded root has no branch receiver block')
    fit_device = (
        torch.device('cpu')
        if linear_algebra_device == 'cpu'
        else device
    )
    branch_weight = root.kernel.weight[direct_len:].detach().to(
        device=fit_device,
        dtype=torch.float32,
    )
    direct_weight_before = root.kernel.weight[:direct_len].detach().to(
        device=fit_device,
        dtype=torch.float32,
    )
    source_weight = source_root.kernel.weight.detach().to(
        device=fit_device,
        dtype=torch.float32,
    )
    if solve_mode == 'branch_primary_residual':
        if direct_weight_before.shape != source_weight.shape:
            raise ValueError(
                'branch_primary_residual requires a source-passthrough '
                'direct block with the same shape as the source kernel'
            )
        passthrough_error = torch.max(torch.abs(
            direct_weight_before - source_weight
        ))
        if float(passthrough_error.item()) > 1.0e-6:
            raise ValueError(
                'branch_primary_residual requires root_direct_mode='
                '"source_passthrough" before refitting the direct residual; '
                f'max source/direct error is {float(passthrough_error.item())}'
            )

    reconstructed_rows = []
    target_rows = []
    branch_refit_time_index_rows = []
    pooled_design_rows = []
    source_net_rows = []
    model_inhibition_rows = []
    source_onset_rows = []
    source_pre_onset_rows = []
    native_sequence_batch_sizes = []
    contrast_source_excitatory_rows = []
    contrast_source_net_compensated_rows = []
    row_priority_rows = []
    residual_direct_design_rows = []
    residual_branch_response_rows = []
    residual_target_rows = []
    residual_fit_time_index_rows = []
    residual_fit_sample_index_rows = []
    residual_fit_batch_position_rows = []
    residual_fit_branch_activity_rows = []
    residual_fit_branch_spatial_rows = []
    residual_fit_branch_current_spatial_rows = []
    residual_audit_direct_design_rows = []
    residual_audit_branch_response_rows = []
    residual_audit_target_rows = []
    residual_audit_time_index_rows = []
    residual_audit_sample_index_rows = []
    residual_audit_branch_activity_rows = []
    residual_audit_branch_spatial_rows = []
    residual_audit_branch_current_spatial_rows = []
    potential_audit_batches = []
    seen = 0
    seen_time_rows = 0
    simulation_time = int(model.image_encoder.simulation_time)
    if simulation_time % direct_residual_time_bin_count != 0:
        raise ValueError(
            'data_response_receiver_refit.direct_residual_time_bin_count '
            'must divide image_encoder.simulation_time exactly'
        )
    if simulation_time % branch_time_bin_count != 0:
        raise ValueError(
            'data_response_receiver_refit.branch_time_bin_count '
            'must divide image_encoder.simulation_time exactly'
        )
    expected_time_rows = max_samples * simulation_time
    row_stride = max(1, math.ceil(expected_time_rows / max_rows))

    def _balanced_sample_indices(
        active_times,
        mode,
        total_rows,
        *,
        hash_seed_base=704000,
        excluded_indices_by_time=None,
        eligible_indices_by_time=None,
    ):
        active_times = list(active_times)
        if not active_times:
            raise ValueError('direct residual sampler has no active time steps')
        base_quota, remainder = divmod(total_rows, len(active_times))
        selected = {}
        for position, time_index in enumerate(active_times):
            quota = base_quota + int(position < remainder)
            if quota <= 0:
                selected[time_index] = torch.empty(
                    0,
                    dtype=torch.long,
                    device=fit_device,
                )
                continue
            candidates = torch.arange(max_samples, dtype=torch.long)
            if eligible_indices_by_time is not None:
                eligible = eligible_indices_by_time.get(time_index)
                if eligible is not None:
                    candidates = eligible.detach().cpu().long()
            if excluded_indices_by_time is not None:
                excluded = excluded_indices_by_time.get(time_index)
                if excluded is not None and excluded.numel() > 0:
                    candidates = candidates[
                        ~torch.isin(candidates, excluded.detach().cpu().long())
                    ]
            if quota > candidates.numel():
                raise ValueError(
                    'balanced direct residual sampler requires no more than '
                    'the eligible sample count per time step'
                )
            if mode == 'hashed':
                generator = torch.Generator(device='cpu')
                generator.manual_seed(hash_seed_base + int(time_index))
                positions = torch.randperm(
                    candidates.numel(),
                    generator=generator,
                )[:quota]
                indices = candidates[positions]
            else:
                positions = torch.floor(
                    (torch.arange(quota, dtype=torch.float64) + 0.5)
                    * float(candidates.numel())
                    / float(quota)
                ).long()
                if mode == 'rotating':
                    positions = torch.remainder(
                        positions + int(time_index) * 17,
                        candidates.numel(),
                    )
                indices = candidates[positions]
            selected[time_index] = indices.to(device=fit_device)
        return selected

    audit_sample_indices_by_time = None
    if direct_residual_time_audit_rows_per_step > 0:
        audit_eligible_indices_by_time = None
        if direct_residual_time_audit_disjoint:
            configured_batch_size = int(dataloaders[phase].batch_size)
            sample_indices = torch.arange(max_samples, dtype=torch.long)
            batch_starts = torch.div(
                sample_indices,
                configured_batch_size,
                rounding_mode='floor',
            ) * configured_batch_size
            batch_sizes = torch.minimum(
                torch.full_like(batch_starts, configured_batch_size),
                max_samples - batch_starts,
            )
            batch_positions = sample_indices - batch_starts
            audit_eligible_indices_by_time = {}
            for time_index in range(simulation_time):
                legacy_row_indices = (
                    batch_starts * simulation_time
                    + time_index * batch_sizes
                    + batch_positions
                )
                audit_eligible_indices_by_time[time_index] = sample_indices[
                    torch.remainder(legacy_row_indices, row_stride).ne(0)
                ]
        audit_sample_indices_by_time = _balanced_sample_indices(
            range(simulation_time),
            'hashed',
            direct_residual_time_audit_rows_per_step * simulation_time,
            hash_seed_base=direct_residual_time_audit_hash_seed,
            eligible_indices_by_time=audit_eligible_indices_by_time,
        )
    fit_sample_indices_by_time = None
    if direct_residual_row_sampling_mode != 'legacy_global_stride':
        if direct_residual_row_sampling_mode == 'even_time_balanced':
            active_times = range(0, simulation_time, 2)
            selection_mode = 'rotating'
        elif direct_residual_row_sampling_mode == 'odd_time_balanced':
            active_times = range(1, simulation_time, 2)
            selection_mode = 'rotating'
        else:
            active_times = range(simulation_time)
            selection_mode = {
                'all_time_fixed_sample_balanced': 'fixed',
                'all_time_rotating_sample_balanced': 'rotating',
                'all_time_hashed_balanced': 'hashed',
            }[direct_residual_row_sampling_mode]
        fit_sample_indices_by_time = _balanced_sample_indices(
            active_times,
            selection_mode,
            max_rows,
            excluded_indices_by_time=(
                audit_sample_indices_by_time
                if direct_residual_time_audit_disjoint
                else None
            ),
        )
    with torch.no_grad():
        for images, _ in dataloaders[phase]:
            if seen >= max_samples:
                break
            batch_start = seen
            images = images[:max_samples - seen]
            if native_recurrence_mode == 'hard_unroll':
                native_sequence_batch_sizes.append(len(images))
            if data_weighted_potential_audit:
                potential_audit_batches.append(images.detach().cpu())
            images = images.to(device)
            seen += len(images)
            model.init_state(images.shape, device)
            source.init_state(images.shape, device)
            source_seen_output = torch.zeros(
                (len(images), branch_weight.shape[1]),
                device=fit_device,
                dtype=torch.bool,
            )
            for time_index, spikes in enumerate(model.image_encoder(images)):
                source.forward_cortex(spikes, is_training=False)
                model.forward_cortex(spikes, is_training=False)

                pooled_wave_raw = (
                    root.subcortexs[0].output_nv.spike_wave.detach().to(
                        device=fit_device,
                        dtype=torch.float32,
                    )
                )
                pooled_wave = pooled_wave_raw.reshape(len(images), -1)
                pooled_spatial_activity = None
                pooled_current_spatial_activity = None
                if spatial_state_requested:
                    pooled_current_tensor = (
                        root.subcortexs[0].output_nv.current_spikes.detach().to(
                            device=fit_device,
                            dtype=torch.float32,
                        )
                    )
                    if pooled_current_tensor.ndim < 3:
                        raise RuntimeError(
                            'state-conditioned residual expects current A-1 '
                            'output spikes ending in [pooled_patches, channels], '
                            f'got {tuple(pooled_current_tensor.shape)}'
                        )
                    pooled_patch_count = int(pooled_current_tensor.shape[-2])
                    pooled_channel_count = int(pooled_current_tensor.shape[-1])
                    pooled_current_tensor = pooled_current_tensor.reshape(
                        len(images),
                        -1,
                        pooled_patch_count,
                        pooled_channel_count,
                    )
                    if int(pooled_current_tensor.shape[1]) != 1:
                        raise RuntimeError(
                            'state-conditioned residual does not support extra '
                            'non-singleton A-1 output axes: '
                            f'{tuple(pooled_current_tensor.shape)}'
                        )
                    pooled_current_tensor = pooled_current_tensor[:, 0]
                    if pooled_wave.shape[-1] != (
                        pooled_patch_count * pooled_channel_count
                    ):
                        raise RuntimeError(
                            'cumulative and current A-1 output shapes disagree: '
                            f'{tuple(pooled_wave_raw.shape)} versus '
                            f'{tuple(pooled_current_tensor.shape)}'
                        )
                    pooled_wave_tensor = pooled_wave.reshape(
                        len(images),
                        pooled_patch_count,
                        pooled_channel_count,
                    )
                    pooled_spatial_activity = pooled_wave_tensor.mean(dim=-1)
                    pooled_current_spatial_activity = (
                        pooled_current_tensor.mean(dim=-1)
                    )
                pooled_design = pooled_wave.mul(float(root.amplifier))
                reconstructed = pooled_design @ branch_weight
                source_wave = source_root.input_nv.spike_wave.reshape(
                    len(images), -1
                ).detach().to(device=fit_device, dtype=torch.float32)
                source_excitatory = (
                    source_wave.mul(float(source_root.amplifier))
                    @ source_weight
                )
                source_net = None
                model_inhibition = None
                if (
                    target_mode != 'source_excitatory'
                    or native_onset_mode != 'none'
                ):
                    source_net = source_root.hidden_nv.current_potential.reshape(
                        len(images), -1
                    ).detach().to(device=fit_device, dtype=torch.float32)
                    model_net = root.hidden_nv.current_potential.reshape(
                        len(images), -1
                    ).detach().to(device=fit_device, dtype=torch.float32)
                    model_inhibition = reconstructed - model_net
                    source_net_compensated = source_net + model_inhibition
                    if target_mode == 'source_net_compensated':
                        target = source_excitatory + source_net_target_blend * (
                            source_net_compensated - source_excitatory
                        )
                    elif target_mode != 'source_excitatory':
                        # The common/contrast target is assembled after all
                        # sampled rows are collected so its optional RMS scale
                        # is derived once from the complete unlabeled sample.
                        target = source_excitatory
                    else:
                        target = source_excitatory
                else:
                    target = source_excitatory
                if native_onset_mode != 'none':
                    source_current_output = (
                        source_root.hidden_nv.current_spikes.reshape(
                            len(images), -1,
                        ).detach().to(device=fit_device) > 0
                    )
                    source_active_output = (
                        source_root.hidden_nv.spike_wave.reshape(
                            len(images), -1,
                        ).detach().to(device=fit_device) > 0
                    )
                    expected_output_shape = source_seen_output.shape
                    if (
                        source_current_output.shape != expected_output_shape
                        or source_active_output.shape != expected_output_shape
                    ):
                        raise RuntimeError(
                            'native onset fitting requires source output shape '
                            f'{tuple(expected_output_shape)}, got current '
                            f'{tuple(source_current_output.shape)} and active '
                            f'{tuple(source_active_output.shape)}'
                        )
                    source_onset = source_current_output & ~source_seen_output
                    source_pre_onset = (
                        ~source_seen_output & ~source_current_output
                    )
                    source_onset_rows.append(source_onset.detach())
                    source_pre_onset_rows.append(source_pre_onset.detach())
                    source_seen_output |= source_active_output
                row_priority = None
                if row_weight_mode == 'source_output_onset_balanced':
                    row_priority = (
                        source_root.output_nv.current_spikes.reshape(
                            len(images), -1
                        ) > 0
                    ).any(dim=-1)
                elif row_weight_mode == 'source_output_active_balanced':
                    row_priority = (
                        source_root.output_nv.spike_wave.reshape(
                            len(images), -1
                        ) > 0
                    ).any(dim=-1)
                if solve_mode in {
                    'direct_dual',
                    'direct_residual_dual',
                    'direct_nonnegative',
                }:
                    row_indices = torch.arange(
                        seen_time_rows,
                        seen_time_rows + len(images),
                        device=fit_device,
                    )
                    keep = (row_indices % row_stride) == 0
                    remaining = max_rows - sum(
                        int(rows.shape[0]) for rows in pooled_design_rows
                    )
                    if remaining > 0 and keep.any():
                        kept_indices = torch.nonzero(
                            keep,
                            as_tuple=False,
                        ).flatten()[:remaining]
                        pooled_design_rows.append(
                            pooled_design[kept_indices].detach()
                        )
                        reconstructed_rows.append(
                            reconstructed[kept_indices].detach()
                        )
                        target_rows.append(target[kept_indices].detach())
                        if source_net is not None:
                            source_net_rows.append(
                                source_net[kept_indices].detach()
                            )
                            model_inhibition_rows.append(
                                model_inhibition[kept_indices].detach()
                            )
                            if target_mode.startswith(
                                'source_excitatory_common_net_contrast'
                            ):
                                contrast_source_excitatory_rows.append(
                                    source_excitatory[kept_indices].detach()
                                )
                                contrast_source_net_compensated_rows.append(
                                    source_net_compensated[
                                        kept_indices
                                    ].detach()
                                )
                        if row_priority is not None:
                            row_priority_rows.append(
                                row_priority[kept_indices].detach().to(
                                    device=fit_device
                                )
                            )
                    seen_time_rows += len(images)
                else:
                    reconstructed_rows.append(reconstructed.detach())
                    target_rows.append(target.detach())
                    if branch_time_bin_count > 1:
                        branch_refit_time_index_rows.append(torch.full(
                            (len(images),),
                            time_index,
                            dtype=torch.long,
                            device=fit_device,
                        ))
                    if source_net is not None:
                        source_net_rows.append(source_net.detach())
                        model_inhibition_rows.append(
                            model_inhibition.detach()
                        )
                        if target_mode.startswith(
                            'source_excitatory_common_net_contrast'
                        ):
                            contrast_source_excitatory_rows.append(
                                source_excitatory.detach()
                            )
                            contrast_source_net_compensated_rows.append(
                                source_net_compensated.detach()
                            )
                    if row_priority is not None:
                        row_priority_rows.append(
                            row_priority.detach().to(device=fit_device)
                        )
                    if solve_mode == 'branch_primary_residual':
                        row_indices = torch.arange(
                            seen_time_rows,
                            seen_time_rows + len(images),
                            device=fit_device,
                        )
                        global_sample_indices = torch.arange(
                            batch_start,
                            batch_start + len(images),
                            device=fit_device,
                        )
                        if fit_sample_indices_by_time is None:
                            fit_keep = (row_indices % row_stride) == 0
                        else:
                            selected_samples = fit_sample_indices_by_time.get(
                                time_index
                            )
                            fit_keep = (
                                torch.zeros_like(
                                    global_sample_indices,
                                    dtype=torch.bool,
                                )
                                if selected_samples is None
                                else torch.isin(
                                    global_sample_indices,
                                    selected_samples,
                                )
                            )
                        remaining = max_rows - sum(
                            int(rows.shape[0])
                            for rows in residual_direct_design_rows
                        )
                        fit_indices = torch.empty(
                            0,
                            dtype=torch.long,
                            device=fit_device,
                        )
                        if remaining > 0 and fit_keep.any():
                            fit_indices = torch.nonzero(
                                fit_keep,
                                as_tuple=False,
                            ).flatten()[:remaining]
                        audit_indices = torch.empty(
                            0,
                            dtype=torch.long,
                            device=fit_device,
                        )
                        if audit_sample_indices_by_time is not None:
                            audit_keep = torch.isin(
                                global_sample_indices,
                                audit_sample_indices_by_time[time_index],
                            )
                            if audit_keep.any():
                                audit_indices = torch.nonzero(
                                    audit_keep,
                                    as_tuple=False,
                                ).flatten()
                        if fit_indices.numel() > 0 or audit_indices.numel() > 0:
                            direct_wave = root.input_nv.spike_wave.reshape(
                                len(images),
                                -1,
                            ).detach().to(
                                device=fit_device,
                                dtype=torch.float32,
                            )
                            direct_design = direct_wave.mul(
                                float(root.amplifier)
                            )
                        if fit_indices.numel() > 0:
                            residual_direct_design_rows.append(
                                direct_design[fit_indices].detach()
                            )
                            residual_branch_response_rows.append(
                                reconstructed[fit_indices].detach()
                            )
                            residual_target_rows.append(
                                target[fit_indices].detach()
                            )
                            residual_fit_time_index_rows.append(torch.full(
                                (len(fit_indices),),
                                time_index,
                                dtype=torch.long,
                                device=fit_device,
                            ))
                            residual_fit_sample_index_rows.append(
                                global_sample_indices[fit_indices].detach()
                            )
                            residual_fit_batch_position_rows.append(
                                fit_indices.detach()
                            )
                            residual_fit_branch_activity_rows.append(
                                pooled_wave[fit_indices].mean(dim=-1).detach()
                            )
                            if spatial_state_requested:
                                residual_fit_branch_spatial_rows.append(
                                    pooled_spatial_activity[
                                        fit_indices
                                    ].detach()
                                )
                                residual_fit_branch_current_spatial_rows.append(
                                    pooled_current_spatial_activity[
                                        fit_indices
                                    ].detach()
                                )
                        if audit_indices.numel() > 0:
                            residual_audit_direct_design_rows.append(
                                direct_design[audit_indices].detach()
                            )
                            residual_audit_branch_response_rows.append(
                                reconstructed[audit_indices].detach()
                            )
                            residual_audit_target_rows.append(
                                target[audit_indices].detach()
                            )
                            residual_audit_time_index_rows.append(torch.full(
                                (len(audit_indices),),
                                time_index,
                                dtype=torch.long,
                                device=fit_device,
                            ))
                            residual_audit_sample_index_rows.append(
                                global_sample_indices[audit_indices].detach()
                            )
                            residual_audit_branch_activity_rows.append(
                                pooled_wave[audit_indices].mean(dim=-1).detach()
                            )
                            if spatial_state_requested:
                                residual_audit_branch_spatial_rows.append(
                                    pooled_spatial_activity[
                                        audit_indices
                                    ].detach()
                                )
                                residual_audit_branch_current_spatial_rows.append(
                                    pooled_current_spatial_activity[
                                        audit_indices
                                    ].detach()
                                )
                        seen_time_rows += len(images)

    reconstructed = torch.cat(reconstructed_rows, dim=0).float()
    target = torch.cat(target_rows, dim=0).float()
    source_excitatory_common_rms = 0.0
    source_excitatory_contrast_rms = 0.0
    source_net_compensated_contrast_rms = 0.0
    source_contrast_scale = 0.0
    target_common_rms = 0.0
    target_contrast_rms = 0.0
    target_total_rms = float(torch.sqrt(torch.mean(torch.square(target))).item())
    if contrast_source_excitatory_rows:
        contrast_source_excitatory = torch.cat(
            contrast_source_excitatory_rows,
            dim=0,
        ).float()
        contrast_source_net_compensated = torch.cat(
            contrast_source_net_compensated_rows,
            dim=0,
        ).float()
        source_excitatory_common = contrast_source_excitatory.mean(
            dim=-1,
            keepdim=True,
        )
        source_excitatory_contrast = (
            contrast_source_excitatory - source_excitatory_common
        )
        source_net_compensated_contrast = (
            contrast_source_net_compensated
            - contrast_source_net_compensated.mean(dim=-1, keepdim=True)
        )
        source_excitatory_common_rms = float(torch.sqrt(torch.mean(
            torch.square(source_excitatory_common)
        )).item())
        source_excitatory_contrast_rms = float(torch.sqrt(torch.mean(
            torch.square(source_excitatory_contrast)
        )).item())
        source_net_compensated_contrast_rms = float(torch.sqrt(torch.mean(
            torch.square(source_net_compensated_contrast)
        )).item())
        source_contrast_scale = 1.0
        if target_mode.endswith('_rms_matched'):
            source_contrast_scale = (
                source_excitatory_contrast_rms
                / max(source_net_compensated_contrast_rms, 1.0e-12)
            )
        target = (
            source_excitatory_common
            + source_contrast_scale * source_net_compensated_contrast
        )
        target_common_rms = source_excitatory_common_rms
        target_contrast_rms = (
            source_contrast_scale
            * source_net_compensated_contrast_rms
        )
        target_total_rms = float(torch.sqrt(
            torch.mean(torch.square(target))
        ).item())
    before_residual = torch.mean(torch.square(reconstructed - target))
    row_weights = torch.ones(
        reconstructed.shape[0],
        dtype=reconstructed.dtype,
        device=reconstructed.device,
    )
    row_priority_fraction = 0.0
    row_priority_weight = 1.0
    row_nonpriority_weight = 1.0
    row_weight_balancing_applied = False
    if row_priority_rows:
        row_priority = torch.cat(row_priority_rows, dim=0).bool()
        row_priority_fraction = float(row_priority.float().mean().item())
        if 0.0 < row_priority_fraction < 1.0:
            row_priority_weight = 0.5 / row_priority_fraction
            row_nonpriority_weight = 0.5 / (
                1.0 - row_priority_fraction
            )
            row_weights = torch.where(
                row_priority,
                torch.full_like(row_weights, row_priority_weight),
                torch.full_like(row_weights, row_nonpriority_weight),
            )
            row_weight_balancing_applied = True
    row_weight_effective_fraction = float(
        torch.square(row_weights.sum())
        .div(torch.square(row_weights).sum() * len(row_weights))
        .item()
    )
    sqrt_row_weights = torch.sqrt(row_weights).unsqueeze(-1)
    weighted_reconstructed = reconstructed * sqrt_row_weights
    weighted_target = target * sqrt_row_weights
    weighted_before_residual = torch.sum(
        row_weights.unsqueeze(-1) * torch.square(reconstructed - target)
    ) / (row_weights.sum() * target.shape[-1])
    reconstructed_common = reconstructed.mean(dim=-1, keepdim=True)
    target_common = target.mean(dim=-1, keepdim=True)
    reconstructed_contrast = reconstructed - reconstructed_common
    target_contrast = target - target_common
    common_mse_before = torch.mean(torch.square(
        reconstructed_common - target_common
    ))
    contrast_mse_before = torch.mean(torch.square(
        reconstructed_contrast - target_contrast
    ))
    common_scale = 1.0
    common_to_contrast_leakage = 0.0
    contrast_to_common_leakage = 0.0
    if solve_mode in {
        'global_mixing',
        'common_to_contrast_blocked_mixing',
        'branch_primary_residual',
    }:
        output_count = reconstructed.shape[-1]
        identity = torch.eye(
            output_count,
            dtype=reconstructed.dtype,
            device=reconstructed.device,
        )
        common_projector = torch.full_like(
            identity,
            1.0 / float(output_count),
        )
        contrast_projector = identity - common_projector
    if solve_mode in {
        'global_mixing',
        'common_to_contrast_blocked_mixing',
        'branch_primary_residual',
    }:
        gram = weighted_reconstructed.T @ weighted_reconstructed
        cross = weighted_reconstructed.T @ weighted_target
        reference_energy = torch.trace(gram) / float(gram.shape[0])
        penalty = ridge * torch.clamp(reference_energy, min=1.0e-12)
        mixing = torch.linalg.solve(
            gram + penalty * identity,
            cross + penalty * identity,
        )
        mixing = identity + blend * (mixing - identity)
        common_scale = float(mixing.sum().div(output_count).item())
        if solve_mode == 'common_to_contrast_blocked_mixing':
            common_projector_64 = common_projector.double()
            contrast_projector_64 = contrast_projector.double()
            mixing = mixing.double()
            mixing = mixing - (
                common_projector_64 @ mixing @ contrast_projector_64
            )
            common_scale = float(mixing.sum().div(output_count).item())
            common_to_contrast_leakage = float(torch.max(torch.abs(
                common_projector_64 @ mixing @ contrast_projector_64
            )).item())
            contrast_to_common_leakage = float(torch.max(torch.abs(
                contrast_projector_64 @ mixing @ common_projector_64
            )).item())
            fitted_weight = (
                branch_weight.double() @ mixing
            ).to(branch_weight.dtype)
            fitted_response = (reconstructed.double() @ mixing).float()
        else:
            fitted_weight = branch_weight @ mixing.to(branch_weight.dtype)
            fitted_response = reconstructed @ mixing
    else:
        pooled_design = torch.cat(pooled_design_rows, dim=0).float()
        weighted_pooled_design = pooled_design * sqrt_row_weights
        dual_gram = weighted_pooled_design @ weighted_pooled_design.T
        identity = torch.eye(
            dual_gram.shape[0],
            dtype=dual_gram.dtype,
            device=dual_gram.device,
        )
        reference_energy = torch.trace(dual_gram) / float(
            dual_gram.shape[0]
        )
        penalty = ridge * torch.clamp(reference_energy, min=1.0e-12)
        dual_target = weighted_target
        if solve_mode == 'direct_residual_dual':
            dual_target = (
                weighted_target
                - weighted_pooled_design @ branch_weight.float()
            )
        dual_coefficients = torch.linalg.solve(
            dual_gram + penalty * identity,
            dual_target,
        )
        direct_weight = weighted_pooled_design.T @ dual_coefficients
        if solve_mode == 'direct_nonnegative':
            # Projected gradient solves the same unlabeled response objective
            # while keeping the native A-1 -> A projection excitatory.  The
            # scalar step uses the exact dual spectral radius, avoiding an
            # expensive feature-space eigendecomposition.
            spectral_radius = torch.linalg.eigvalsh(dual_gram).amax()
            step_size = torch.reciprocal(torch.clamp(
                spectral_radius + penalty,
                min=1.0e-12,
            ))
            direct_weight = torch.clamp(direct_weight, min=0.0)
            for _ in range(nonnegative_iterations):
                residual = (
                    weighted_pooled_design @ direct_weight
                    - weighted_target
                )
                gradient = weighted_pooled_design.T @ residual
                gradient = gradient + penalty * direct_weight
                direct_weight = torch.clamp(
                    direct_weight - step_size * gradient,
                    min=0.0,
                )
        if solve_mode == 'direct_residual_dual':
            fitted_weight = (
                branch_weight
                + blend * direct_weight.to(branch_weight.dtype)
            )
        else:
            fitted_weight = (
                (1.0 - blend) * branch_weight
                + blend * direct_weight.to(branch_weight.dtype)
            )
        fitted_response = pooled_design @ fitted_weight.float()
    if solve_mode in {'global_mixing', 'branch_primary_residual'}:
        common_to_contrast_leakage = float(torch.max(torch.abs(
            common_projector @ mixing @ contrast_projector
        )).item())
        contrast_to_common_leakage = float(torch.max(torch.abs(
            contrast_projector @ mixing @ common_projector
        )).item())
    normalization_refit_audit = None
    if normalization_aware_refit is not None:
        native_source_onset = None
        native_source_pre_onset = None
        native_model_inhibition = None
        native_threshold = None
        if native_onset_mode != 'none':
            native_source_onset = torch.cat(source_onset_rows, dim=0).bool()
            native_source_pre_onset = torch.cat(
                source_pre_onset_rows, dim=0,
            ).bool()
            native_model_inhibition = torch.cat(
                model_inhibition_rows, dim=0,
            ).float()
            native_threshold = (
                post_refit_root_threshold
                if post_refit_root_threshold is not None
                else root_threshold_before_refit_override
            )
        mixing, fitted_weight, fitted_response, normalization_refit_audit = (
            fit_normalized_receiver_mixing(
                branch_weight, reconstructed, target, mixing, row_weights,
                ridge,
                native_onset_mode=native_onset_mode,
                source_onset=native_source_onset,
                source_pre_onset=native_source_pre_onset,
                model_inhibition=native_model_inhibition,
                native_threshold=native_threshold,
                native_recurrence_mode=native_recurrence_mode,
                native_sequence_batch_sizes=native_sequence_batch_sizes,
                native_lateral_inhibition_factor=float(
                    getattr(root, 'lateral_inhibition_factor', 0.0)
                ),
                native_lateral_inhibition_power=float(
                    getattr(root, 'lateral_inhibition_power', 1.0)
                ),
                native_dominance_inhibition_factor=float(
                    getattr(root, 'dominance_inhibition_factor', 0.0)
                ),
                native_dominance_inhibition_power=float(
                    getattr(root, 'dominance_inhibition_power', 1.0)
                ),
                native_competition_group_N=int(root.competition_group_N),
                native_top_k=root.top_k,
                **normalization_aware_refit,
            )
        )
        common_scale = float(mixing.sum().div(output_count))
        common_to_contrast_leakage = float((common_projector @ mixing @ contrast_projector).abs().max())
        contrast_to_common_leakage = float((contrast_projector @ mixing @ common_projector).abs().max())
    branch_time_binned_offsets = None
    branch_time_bin_audit = {}
    branch_time_bin_mse_before = float(torch.mean(torch.square(
        fitted_response - target
    )).item())
    branch_time_bin_mse_after = branch_time_bin_mse_before
    branch_time_bin_adjacent_relative_l2 = 0.0
    branch_time_bin_offset_l2_per_output = 0.0
    branch_time_bin_column_l1_max_abs_error = 0.0
    if branch_time_bin_count > 1:
        if not branch_refit_time_index_rows:
            raise RuntimeError('time-binned branch refit collected no time indices')
        refit_time_indices = torch.cat(
            branch_refit_time_index_rows,
            dim=0,
        )
        if refit_time_indices.shape[0] != reconstructed.shape[0]:
            raise RuntimeError(
                'time-binned branch refit time-index count does not match '
                'the response rows'
            )
        fit_bin_indices = torch.div(
            refit_time_indices * branch_time_bin_count,
            simulation_time,
            rounding_mode='floor',
        )
        static_column_l1 = fitted_weight.abs().sum(dim=0)
        time_binned_response = torch.empty_like(fitted_response)
        time_binned_weights = []
        for bin_index in range(branch_time_bin_count):
            bin_mask = fit_bin_indices.eq(bin_index)
            if not bin_mask.any():
                raise RuntimeError(
                    'time-binned branch refit has no rows for bin '
                    f'{bin_index}'
                )
            bin_reconstructed = reconstructed[bin_mask]
            bin_target = target[bin_mask]
            bin_row_weights = row_weights[bin_mask]
            bin_sqrt_weights = torch.sqrt(bin_row_weights).unsqueeze(-1)
            weighted_bin_reconstructed = bin_reconstructed * bin_sqrt_weights
            weighted_bin_target = bin_target * bin_sqrt_weights
            bin_gram = (
                weighted_bin_reconstructed.T @ weighted_bin_reconstructed
            )
            bin_cross = weighted_bin_reconstructed.T @ weighted_bin_target
            bin_reference_energy = torch.trace(bin_gram) / float(
                bin_gram.shape[0]
            )
            bin_penalty = ridge * torch.clamp(
                bin_reference_energy,
                min=1.0e-12,
            )
            bin_mixing = torch.linalg.solve(
                bin_gram + bin_penalty * identity,
                bin_cross + bin_penalty * mixing.to(bin_cross.dtype),
            )
            bin_weight = branch_weight @ bin_mixing.to(branch_weight.dtype)
            bin_column_l1 = torch.clamp(
                bin_weight.abs().sum(dim=0),
                min=1.0e-12,
            )
            column_scale = static_column_l1 / bin_column_l1
            bin_mixing = bin_mixing * column_scale.unsqueeze(0).to(
                bin_mixing.dtype
            )
            bin_weight = bin_weight * column_scale.unsqueeze(0).to(
                bin_weight.dtype
            )
            bin_response = bin_reconstructed @ bin_mixing.to(
                bin_reconstructed.dtype
            )
            time_binned_response[bin_mask] = bin_response
            time_binned_weights.append(bin_weight)
            bin_denominator = bin_row_weights.sum() * bin_target.shape[-1]
            bin_mse_before = torch.sum(
                bin_row_weights.unsqueeze(-1)
                * torch.square(fitted_response[bin_mask] - bin_target)
            ) / bin_denominator
            bin_mse_after = torch.sum(
                bin_row_weights.unsqueeze(-1)
                * torch.square(bin_response - bin_target)
            ) / bin_denominator
            branch_time_bin_audit[bin_index] = {
                'row_count': int(bin_mask.sum().item()),
                'mse_before': float(bin_mse_before.item()),
                'mse_after': float(bin_mse_after.item()),
                'column_l1_max_abs_error': float(torch.max(torch.abs(
                    bin_weight.abs().sum(dim=0) - static_column_l1
                )).item()),
            }
        time_binned_weights = torch.stack(time_binned_weights, dim=0)
        branch_time_binned_offsets = (
            time_binned_weights - fitted_weight.unsqueeze(0)
        )
        fitted_response = time_binned_response
        branch_time_bin_mse_after = float(torch.sum(
            row_weights.unsqueeze(-1) * torch.square(
                fitted_response - target
            )
        ).div(row_weights.sum() * target.shape[-1]).item())
        adjacent_differences = (
            time_binned_weights[1:] - time_binned_weights[:-1]
        ).reshape(branch_time_bin_count - 1, -1)
        adjacent_references = (
            0.5
            * (
                time_binned_weights[1:].abs()
                + time_binned_weights[:-1].abs()
            )
        ).reshape(branch_time_bin_count - 1, -1)
        branch_time_bin_adjacent_relative_l2 = float(
            torch.linalg.vector_norm(adjacent_differences, dim=1)
            .div(torch.clamp(
                torch.linalg.vector_norm(adjacent_references, dim=1),
                min=1.0e-12,
            ))
            .mean()
            .item()
        )
        branch_time_bin_offset_l2_per_output = float(
            torch.linalg.vector_norm(
                branch_time_binned_offsets,
                dim=1,
            ).mean().item()
        )
        branch_time_bin_column_l1_max_abs_error = max(
            stats['column_l1_max_abs_error']
            for stats in branch_time_bin_audit.values()
        )
    branch_after_residual = torch.mean(torch.square(fitted_response - target))
    receiver_correction_relative_l2 = float(
        torch.linalg.vector_norm(fitted_weight - branch_weight)
        .div(torch.clamp(
            torch.linalg.vector_norm(branch_weight),
            min=1.0e-12,
        )).item()
    )
    direct_fitted_weight = None
    direct_time_binned_weights = None
    direct_state_residual_weights = None
    direct_residual_mse_before = None
    direct_residual_mse_after = None
    combined_response_mse_after = None
    residual_fit_row_count = 0
    direct_residual_sampling_audit = {}
    direct_residual_time_audit = {}
    direct_residual_independent_audit = {}
    direct_residual_time_bin_audit = {}
    direct_time_bin_adjacent_relative_l2 = 0.0
    direct_state_weight_l1_per_output = 0.0
    direct_state_branch_activity_mean = 0.0
    direct_state_branch_activity_std = 1.0
    direct_state_direct_activity_mean = 0.0
    direct_state_direct_activity_std = 1.0
    direct_state_spatial_patch_count = 0
    direct_state_spatial_grid_size = 0
    direct_state_spatial_direct_channel_count = 0
    direct_state_spatial_branch_channel_count = 0
    direct_state_branch_spatial_mean = None
    direct_state_branch_spatial_std = None
    direct_state_branch_spatial_centered_mean = None
    direct_state_branch_spatial_centered_std = None
    direct_state_branch_current_spatial_centered_mean = None
    direct_state_branch_current_spatial_centered_std = None
    direct_state_branch_spatial_std_mean = 0.0
    direct_state_branch_spatial_centered_std_mean = 0.0
    direct_state_branch_current_spatial_centered_std_mean = 0.0
    branch_response_rms = 0.0
    direct_residual_target_rms = 0.0
    direct_response_rms = 0.0
    combined_response_rms = 0.0
    residual_target_response_rms = 0.0
    branch_contribution_rms_fraction = 0.0
    direct_to_branch_rms_ratio = 0.0
    direct_weight_l1_per_output = 0.0
    direct_positive_weight_l1_per_output = 0.0
    direct_negative_weight_l1_per_output = 0.0
    direct_negative_weight_fraction = 0.0
    source_direct_weight_l1_per_output = float(
        direct_weight_before.abs().sum(dim=0).mean().item()
    )
    direct_weight_relative_l2_change = 0.0
    branch_weight_l1_per_output = float(
        fitted_weight.abs().sum(dim=0).mean().item()
    )
    if solve_mode == 'branch_primary_residual':
        if not residual_direct_design_rows:
            raise RuntimeError(
                'branch_primary_residual did not collect any direct-fit rows'
            )
        # Preserve the branch-first allocation explicitly.  The Tucker branch
        # receives the same global response solution as the factorized-only
        # control; the source-passthrough input block is then replaced by a
        # dual-ridge solution for only the response that branch did not
        # reconstruct.  A joint unconstrained solve would otherwise collapse
        # to the trivial source-direct solution with an unused branch.
        direct_design = torch.cat(
            residual_direct_design_rows,
            dim=0,
        ).float()
        fit_branch_activity = torch.cat(
            residual_fit_branch_activity_rows,
            dim=0,
        ).float()
        if spatial_state_requested:
            fit_branch_spatial = torch.cat(
                residual_fit_branch_spatial_rows,
                dim=0,
            ).float()
            fit_branch_current_spatial = torch.cat(
                residual_fit_branch_current_spatial_rows,
                dim=0,
            ).float()
        else:
            fit_branch_spatial = torch.zeros(
                (direct_design.shape[0], 1),
                device=direct_design.device,
                dtype=direct_design.dtype,
            )
            fit_branch_current_spatial = torch.zeros_like(
                fit_branch_spatial
            )
        fit_direct_activity = direct_design.div(
            float(root.amplifier)
        ).mean(dim=-1)
        direct_state_branch_activity_mean = float(
            fit_branch_activity.mean().item()
        )
        direct_state_branch_activity_std = float(
            fit_branch_activity.std(unbiased=False).clamp(min=1.0e-6).item()
        )
        direct_state_direct_activity_mean = float(
            fit_direct_activity.mean().item()
        )
        direct_state_direct_activity_std = float(
            fit_direct_activity.std(unbiased=False).clamp(min=1.0e-6).item()
        )

        direct_state_spatial_patch_count = int(fit_branch_spatial.shape[-1])
        direct_state_spatial_grid_size = math.isqrt(
            direct_state_spatial_patch_count
        )
        if (
            direct_state_spatial_grid_size
            * direct_state_spatial_grid_size
            != direct_state_spatial_patch_count
        ):
            raise RuntimeError(
                'spatial state requires a square pooled-patch layout'
            )
        direct_spatial_size = int(root.kernel_size)
        direct_spatial_area = direct_spatial_size * direct_spatial_size
        if direct_len % direct_spatial_area != 0:
            raise RuntimeError(
                'direct sender length is not divisible by its spatial area'
            )
        direct_state_spatial_direct_channel_count = (
            direct_len // direct_spatial_area
        )
        branch_sender_len = int(root.kernel.weight.shape[0]) - direct_len
        if branch_sender_len % direct_state_spatial_patch_count != 0:
            raise RuntimeError(
                'branch sender length is not divisible by pooled-patch count'
            )
        direct_state_spatial_branch_channel_count = (
            branch_sender_len // direct_state_spatial_patch_count
        )

        fit_branch_spatial_centered = (
            fit_branch_spatial
            - fit_branch_spatial.mean(dim=-1, keepdim=True)
        )
        fit_branch_current_spatial_centered = (
            fit_branch_current_spatial
            - fit_branch_current_spatial.mean(dim=-1, keepdim=True)
        )
        direct_state_branch_spatial_mean = fit_branch_spatial.mean(dim=0)
        direct_state_branch_spatial_std = fit_branch_spatial.std(
            dim=0,
            unbiased=False,
        ).clamp(min=1.0e-6)
        direct_state_branch_spatial_centered_mean = (
            fit_branch_spatial_centered.mean(dim=0)
        )
        direct_state_branch_spatial_centered_std = (
            fit_branch_spatial_centered.std(dim=0, unbiased=False)
            .clamp(min=1.0e-6)
        )
        direct_state_branch_current_spatial_centered_mean = (
            fit_branch_current_spatial_centered.mean(dim=0)
        )
        direct_state_branch_current_spatial_centered_std = (
            fit_branch_current_spatial_centered.std(
                dim=0,
                unbiased=False,
            ).clamp(min=1.0e-6)
        )
        direct_state_branch_spatial_std_mean = float(
            direct_state_branch_spatial_std.mean().item()
        )
        direct_state_branch_spatial_centered_std_mean = float(
            direct_state_branch_spatial_centered_std.mean().item()
        )
        direct_state_branch_current_spatial_centered_std_mean = float(
            direct_state_branch_current_spatial_centered_std.mean().item()
        )

        def _map_spatial_state_to_direct(spatial_state):
            mapped = torch.nn.functional.interpolate(
                spatial_state.reshape(
                    -1,
                    1,
                    direct_state_spatial_grid_size,
                    direct_state_spatial_grid_size,
                ),
                size=(direct_spatial_size, direct_spatial_size),
                mode='bilinear',
                align_corners=True,
            )
            mapped = mapped.expand(
                -1,
                direct_state_spatial_direct_channel_count,
                -1,
                -1,
            )
            return mapped.reshape(*spatial_state.shape[:-1], direct_len)

        def _direct_state_features(
            base_design,
            branch_activity,
            branch_spatial,
            branch_current_spatial,
        ):
            direct_activity = base_design.div(
                float(root.amplifier)
            ).mean(dim=-1)
            branch_state = (
                branch_activity - direct_state_branch_activity_mean
            ) / direct_state_branch_activity_std
            direct_state = (
                direct_activity - direct_state_direct_activity_mean
            ) / direct_state_direct_activity_std
            states = []
            if direct_residual_state_basis.startswith('branch_activity_'):
                degree = {
                    'branch_activity_linear': 1,
                    'branch_activity_quadratic': 2,
                    'branch_activity_cubic': 3,
                }[direct_residual_state_basis]
                states.extend(branch_state.pow(power) for power in range(1, degree + 1))
            elif direct_residual_state_basis == 'direct_activity_linear':
                states.append(direct_state)
            elif direct_residual_state_basis == 'branch_and_direct_activity_linear':
                states.extend((branch_state, direct_state))
            elif direct_residual_state_basis == 'branch_spatial_cumulative_linear':
                states.append(_map_spatial_state_to_direct(
                    (
                        branch_spatial - direct_state_branch_spatial_mean
                    ) / direct_state_branch_spatial_std
                ))
            elif (
                direct_residual_state_basis
                == 'branch_spatial_cumulative_centered_linear'
            ):
                centered = branch_spatial - branch_spatial.mean(
                    dim=-1,
                    keepdim=True,
                )
                states.append(_map_spatial_state_to_direct(
                    (
                        centered
                        - direct_state_branch_spatial_centered_mean
                    ) / direct_state_branch_spatial_centered_std
                ))
            elif (
                direct_residual_state_basis
                == 'branch_spatial_instantaneous_centered_linear'
            ):
                centered = branch_current_spatial - branch_current_spatial.mean(
                    dim=-1,
                    keepdim=True,
                )
                states.append(_map_spatial_state_to_direct(
                    (
                        centered
                        - direct_state_branch_current_spatial_centered_mean
                    ) / direct_state_branch_current_spatial_centered_std
                ))
            elif (
                direct_residual_state_basis
                == 'branch_spatial_cumulative_and_instantaneous_centered_linear'
            ):
                cumulative_centered = (
                    branch_spatial
                    - branch_spatial.mean(dim=-1, keepdim=True)
                )
                current_centered = (
                    branch_current_spatial
                    - branch_current_spatial.mean(dim=-1, keepdim=True)
                )
                states.extend((
                    _map_spatial_state_to_direct(
                        (
                            cumulative_centered
                            - direct_state_branch_spatial_centered_mean
                        ) / direct_state_branch_spatial_centered_std
                    ),
                    _map_spatial_state_to_direct(
                        (
                            current_centered
                            - direct_state_branch_current_spatial_centered_mean
                        ) / direct_state_branch_current_spatial_centered_std
                    ),
                ))
            feature_blocks = [base_design]
            for state in states:
                if state.ndim == base_design.ndim:
                    feature_blocks.append(base_design * state)
                else:
                    feature_blocks.append(base_design * state.unsqueeze(-1))
            return torch.cat(
                feature_blocks,
                dim=-1,
            )

        direct_feature_design = _direct_state_features(
            direct_design,
            fit_branch_activity,
            fit_branch_spatial,
            fit_branch_current_spatial,
        )
        sampled_branch_before = torch.cat(
            residual_branch_response_rows,
            dim=0,
        ).float()
        sampled_target = torch.cat(residual_target_rows, dim=0).float()
        sampled_branch_after = sampled_branch_before @ mixing.to(
            sampled_branch_before.dtype
        )
        direct_residual_target = sampled_target - sampled_branch_after
        residual_fit_row_count = int(direct_design.shape[0])
        fit_time_indices = torch.cat(residual_fit_time_index_rows, dim=0)
        fit_sample_indices = torch.cat(residual_fit_sample_index_rows, dim=0)
        fit_batch_positions = torch.cat(
            residual_fit_batch_position_rows,
            dim=0,
        )
        direct_residual_sampling_audit = {
            'row_count': residual_fit_row_count,
            'unique_sample_count': int(torch.unique(
                fit_sample_indices
            ).numel()),
            'unique_sample_fraction': float(torch.unique(
                fit_sample_indices
            ).numel()) / float(max_samples),
            'batch_position_zero_fraction': float(
                fit_batch_positions.eq(0).float().mean().item()
            ),
            'time_row_counts': {
                time_index: int(fit_time_indices.eq(time_index).sum().item())
                for time_index in range(simulation_time)
            },
        }
        if residual_audit_sample_index_rows:
            audit_time_indices_for_overlap = torch.cat(
                residual_audit_time_index_rows,
                dim=0,
            )
            audit_sample_indices_for_overlap = torch.cat(
                residual_audit_sample_index_rows,
                dim=0,
            )
            fit_keys = fit_time_indices * max_samples + fit_sample_indices
            audit_keys = (
                audit_time_indices_for_overlap * max_samples
                + audit_sample_indices_for_overlap
            )
            overlap_count = int(torch.isin(
                audit_keys,
                fit_keys,
            ).sum().item())
            direct_residual_sampling_audit.update({
                'audit_fit_overlap_count': overlap_count,
                'audit_fit_overlap_fraction': (
                    float(overlap_count) / float(audit_keys.numel())
                ),
                'audit_disjoint_requested': float(
                    direct_residual_time_audit_disjoint
                ),
            })
        direct_residual_mse_before = torch.mean(torch.square(
            direct_residual_target
        ))

        def _fit_direct_residual_weight(design, residual_target):
            direct_dual_gram = design @ design.T
            direct_identity = torch.eye(
                direct_dual_gram.shape[0],
                dtype=direct_dual_gram.dtype,
                device=direct_dual_gram.device,
            )
            direct_reference_energy = torch.trace(
                direct_dual_gram
            ) / float(direct_dual_gram.shape[0])
            direct_penalty = (
                ridge
                * direct_residual_ridge_multiplier
                * torch.clamp(
                    direct_reference_energy,
                    min=1.0e-12,
                )
            )
            direct_dual_coefficients = torch.linalg.solve(
                direct_dual_gram + direct_penalty * direct_identity,
                residual_target,
            )
            fitted_weight = (
                design.T @ direct_dual_coefficients
            ).to(direct_weight_before.dtype)
            fitted_weight = (
                torch.clamp(fitted_weight, min=0.0)
                * direct_residual_positive_scale
                + torch.clamp(fitted_weight, max=0.0)
                * direct_residual_negative_scale
            )
            # The direct input spikes arrive earlier than the yielded branch.
            # Keep causal-importance scaling explicit without changing signs.
            return fitted_weight * direct_residual_scale

        fit_bin_indices = torch.div(
            fit_time_indices * direct_residual_time_bin_count,
            simulation_time,
            rounding_mode='floor',
        )
        direct_response = torch.empty_like(direct_residual_target)
        fitted_bin_weights = []
        for bin_index in range(direct_residual_time_bin_count):
            bin_mask = fit_bin_indices.eq(bin_index)
            if not bin_mask.any():
                raise RuntimeError(
                    'direct residual time bin has no fit rows: '
                    f'{bin_index}'
                )
            bin_weight = _fit_direct_residual_weight(
                direct_feature_design[bin_mask],
                direct_residual_target[bin_mask],
            )
            fitted_bin_weights.append(bin_weight)
            direct_response[bin_mask] = (
                direct_feature_design[bin_mask] @ bin_weight.float()
            )
            direct_residual_time_bin_audit[bin_index] = {
                'row_count': int(bin_mask.sum().item()),
                'weight_l1_per_output': float(
                    bin_weight.abs().sum(dim=0).mean().item()
                ),
            }
        direct_feature_binned_weights = torch.stack(fitted_bin_weights, dim=0)
        direct_time_binned_weights = direct_feature_binned_weights[
            :, :direct_len
        ]
        if direct_feature_binned_weights.shape[1] > direct_len:
            direct_state_residual_weights = direct_feature_binned_weights[
                0, direct_len:
            ].reshape(-1, direct_len, direct_feature_binned_weights.shape[-1])
            direct_state_weight_l1_per_output = float(
                direct_state_residual_weights.abs().sum(dim=1).mean().item()
            )
        direct_fitted_weight = direct_time_binned_weights.mean(dim=0)
        if direct_residual_time_bin_count > 1:
            adjacent_differences = (
                direct_time_binned_weights[1:]
                - direct_time_binned_weights[:-1]
            ).reshape(direct_residual_time_bin_count - 1, -1)
            adjacent_references = (
                0.5
                * (
                    direct_time_binned_weights[1:].abs()
                    + direct_time_binned_weights[:-1].abs()
                )
            ).reshape(direct_residual_time_bin_count - 1, -1)
            direct_time_bin_adjacent_relative_l2 = float(
                torch.linalg.vector_norm(adjacent_differences, dim=1)
                .div(torch.clamp(
                    torch.linalg.vector_norm(adjacent_references, dim=1),
                    min=1.0e-12,
                ))
                .mean()
                .item()
            )
        combined_response = sampled_branch_after + direct_response
        direct_residual_mse_after = torch.mean(torch.square(
            direct_response - direct_residual_target
        ))
        combined_response_mse_after = torch.mean(torch.square(
            combined_response - sampled_target
        ))

        def _response_rms(value):
            return float(torch.sqrt(torch.mean(torch.square(value))).item())

        if residual_audit_direct_design_rows:
            audit_direct_design = torch.cat(
                residual_audit_direct_design_rows,
                dim=0,
            ).float()
            audit_branch_activity = torch.cat(
                residual_audit_branch_activity_rows,
                dim=0,
            ).float()
            if spatial_state_requested:
                audit_branch_spatial = torch.cat(
                    residual_audit_branch_spatial_rows,
                    dim=0,
                ).float()
                audit_branch_current_spatial = torch.cat(
                    residual_audit_branch_current_spatial_rows,
                    dim=0,
                ).float()
            else:
                audit_branch_spatial = torch.zeros(
                    (audit_direct_design.shape[0], 1),
                    device=audit_direct_design.device,
                    dtype=audit_direct_design.dtype,
                )
                audit_branch_current_spatial = torch.zeros_like(
                    audit_branch_spatial
                )
            audit_direct_feature_design = _direct_state_features(
                audit_direct_design,
                audit_branch_activity,
                audit_branch_spatial,
                audit_branch_current_spatial,
            )
            audit_branch_before = torch.cat(
                residual_audit_branch_response_rows,
                dim=0,
            ).float()
            audit_target = torch.cat(
                residual_audit_target_rows,
                dim=0,
            ).float()
            audit_time_indices = torch.cat(
                residual_audit_time_index_rows,
                dim=0,
            )
            audit_branch_after = audit_branch_before @ mixing.to(
                audit_branch_before.dtype
            )
            audit_residual_target = audit_target - audit_branch_after
            audit_bin_indices = torch.div(
                audit_time_indices * direct_residual_time_bin_count,
                simulation_time,
                rounding_mode='floor',
            )
            audit_direct_response = torch.empty_like(audit_residual_target)
            for bin_index in range(direct_residual_time_bin_count):
                bin_mask = audit_bin_indices.eq(bin_index)
                if bin_mask.any():
                    audit_direct_response[bin_mask] = (
                        audit_direct_feature_design[bin_mask]
                        @ direct_feature_binned_weights[bin_index].float()
                    )
            audit_combined = audit_branch_after + audit_direct_response
            direct_residual_independent_audit = {
                'row_count': int(audit_target.shape[0]),
                'branch_residual_rms': _response_rms(audit_residual_target),
                'direct_response_rms': _response_rms(audit_direct_response),
                'direct_residual_mse': float(torch.mean(torch.square(
                    audit_direct_response - audit_residual_target
                )).item()),
                'combined_response_mse': float(torch.mean(torch.square(
                    audit_combined - audit_target
                )).item()),
            }
            for time_index in range(simulation_time):
                time_mask = audit_time_indices.eq(time_index)
                if not time_mask.any():
                    continue
                time_residual = audit_residual_target[time_mask]
                time_direct = audit_direct_response[time_mask]
                time_combined = audit_combined[time_mask]
                cosine_denominator = torch.linalg.vector_norm(
                    time_residual
                ) * torch.linalg.vector_norm(time_direct)
                cosine = torch.sum(time_residual * time_direct) / torch.clamp(
                    cosine_denominator,
                    min=1.0e-12,
                )
                direct_residual_time_audit[time_index] = {
                    'row_count': int(time_mask.sum().item()),
                    'branch_residual_rms': _response_rms(time_residual),
                    'direct_response_rms': _response_rms(time_direct),
                    'direct_residual_cosine': float(cosine.item()),
                    'direct_residual_mse': float(torch.mean(torch.square(
                        time_direct - time_residual
                    )).item()),
                    'combined_response_mse': float(torch.mean(torch.square(
                        time_combined - audit_target[time_mask]
                    )).item()),
                }

        branch_response_rms = _response_rms(sampled_branch_after)
        direct_residual_target_rms = _response_rms(direct_residual_target)
        direct_response_rms = _response_rms(direct_response)
        combined_response_rms = _response_rms(combined_response)
        residual_target_response_rms = _response_rms(sampled_target)
        contribution_denominator = branch_response_rms + direct_response_rms
        branch_contribution_rms_fraction = (
            branch_response_rms / max(contribution_denominator, 1.0e-12)
        )
        direct_to_branch_rms_ratio = (
            direct_response_rms / max(branch_response_rms, 1.0e-12)
        )
        direct_weight_l1_per_output = float(
            direct_time_binned_weights.abs().sum(dim=1).mean().item()
        )
        direct_positive_weight_l1_per_output = float(
            torch.clamp(direct_time_binned_weights, min=0.0)
            .sum(dim=1).mean().item()
        )
        direct_negative_weight_l1_per_output = float(
            -torch.clamp(direct_time_binned_weights, max=0.0)
            .sum(dim=1).mean().item()
        )
        direct_negative_weight_fraction = float(
            (direct_time_binned_weights < 0).float().mean().item()
        )
        direct_weight_relative_l2_change = float(
            torch.linalg.vector_norm(
                direct_time_binned_weights - direct_weight_before.unsqueeze(0)
                , dim=(1, 2)
            ).div(torch.clamp(
                torch.linalg.vector_norm(direct_weight_before),
                min=1.0e-12,
            )).mean().item()
        )
    after_residual = torch.mean(torch.square(fitted_response - target))
    fitted_common = fitted_response.mean(dim=-1, keepdim=True)
    fitted_contrast = fitted_response - fitted_common
    common_mse_after = torch.mean(torch.square(
        fitted_common - target_common
    ))
    contrast_mse_after = torch.mean(torch.square(
        fitted_contrast - target_contrast
    ))
    weighted_after_residual = torch.sum(
        row_weights.unsqueeze(-1) * torch.square(fitted_response - target)
    ) / (row_weights.sum() * target.shape[-1])
    source_net_residual_before = None
    source_net_residual_after = None
    if source_net_rows:
        source_net = torch.cat(source_net_rows, dim=0).float()
        model_inhibition = torch.cat(model_inhibition_rows, dim=0).float()
        model_net_before = reconstructed - model_inhibition
        model_net_after_linearized = fitted_response - model_inhibition
        source_net_residual_before = torch.mean(torch.square(
            model_net_before - source_net
        ))
        source_net_residual_after = torch.mean(torch.square(
            model_net_after_linearized - source_net
        ))
    root_amplifier_before_gauge = float(root.amplifier)
    root_amplifier_after_gauge = root_amplifier_before_gauge
    if refund_root_amplifier:
        root_amplifier_after_gauge /= post_refit_branch_scale
    gauge_response_scale = (
        post_refit_branch_scale
        * root_amplifier_after_gauge
        / root_amplifier_before_gauge
    )
    post_gauge_residual = torch.mean(torch.square(
        fitted_response * gauge_response_scale - target
    ))
    fitted_weight = fitted_weight * post_refit_branch_scale
    with torch.no_grad():
        if direct_fitted_weight is not None:
            root.kernel.weight[:direct_len] = direct_fitted_weight.to(
                device=root.kernel.weight.device,
                dtype=root.kernel.weight.dtype,
            )
            if direct_residual_time_bin_count > 1:
                root.direct_time_binned_weights = direct_time_binned_weights.to(
                    device=root.kernel.weight.device,
                    dtype=root.kernel.weight.dtype,
                )
                root.direct_time_binned_simulation_time = simulation_time
            elif hasattr(root, 'direct_time_binned_weights'):
                del root.direct_time_binned_weights
                if hasattr(root, 'direct_time_binned_simulation_time'):
                    del root.direct_time_binned_simulation_time
            if direct_state_residual_weights is not None:
                root.direct_state_residual_weights = (
                    direct_state_residual_weights.to(
                        device=root.kernel.weight.device,
                        dtype=root.kernel.weight.dtype,
                    )
                )
                root.direct_state_residual_basis = direct_residual_state_basis
                root.direct_state_branch_activity_mean = (
                    direct_state_branch_activity_mean
                )
                root.direct_state_branch_activity_std = (
                    direct_state_branch_activity_std
                )
                root.direct_state_direct_activity_mean = (
                    direct_state_direct_activity_mean
                )
                root.direct_state_direct_activity_std = (
                    direct_state_direct_activity_std
                )
                if spatial_state_requested:
                    root.direct_state_spatial_patch_count = (
                        direct_state_spatial_patch_count
                    )
                    root.direct_state_spatial_grid_size = (
                        direct_state_spatial_grid_size
                    )
                    root.direct_state_spatial_direct_channel_count = (
                        direct_state_spatial_direct_channel_count
                    )
                    root.direct_state_spatial_branch_channel_count = (
                        direct_state_spatial_branch_channel_count
                    )
                    for attribute, value in (
                        (
                            'direct_state_branch_spatial_mean',
                            direct_state_branch_spatial_mean,
                        ),
                        (
                            'direct_state_branch_spatial_std',
                            direct_state_branch_spatial_std,
                        ),
                        (
                            'direct_state_branch_spatial_centered_mean',
                            direct_state_branch_spatial_centered_mean,
                        ),
                        (
                            'direct_state_branch_spatial_centered_std',
                            direct_state_branch_spatial_centered_std,
                        ),
                        (
                            'direct_state_branch_current_spatial_centered_mean',
                            direct_state_branch_current_spatial_centered_mean,
                        ),
                        (
                            'direct_state_branch_current_spatial_centered_std',
                            direct_state_branch_current_spatial_centered_std,
                        ),
                    ):
                        setattr(
                            root,
                            attribute,
                            value.to(
                                device=root.kernel.weight.device,
                                dtype=root.kernel.weight.dtype,
                            ),
                        )
            else:
                for attribute in (
                    'direct_state_residual_weights',
                    'direct_state_residual_basis',
                    'direct_state_branch_activity_mean',
                    'direct_state_branch_activity_std',
                    'direct_state_direct_activity_mean',
                    'direct_state_direct_activity_std',
                    'direct_state_spatial_patch_count',
                    'direct_state_spatial_grid_size',
                    'direct_state_spatial_direct_channel_count',
                    'direct_state_spatial_branch_channel_count',
                    'direct_state_branch_spatial_mean',
                    'direct_state_branch_spatial_std',
                    'direct_state_branch_spatial_centered_mean',
                    'direct_state_branch_spatial_centered_std',
                    'direct_state_branch_current_spatial_centered_mean',
                    'direct_state_branch_current_spatial_centered_std',
                ):
                    if hasattr(root, attribute):
                        delattr(root, attribute)
        root.kernel.weight[direct_len:] = fitted_weight.to(
            device=root.kernel.weight.device,
            dtype=root.kernel.weight.dtype,
        )
        if branch_time_binned_offsets is not None:
            root.branch_time_binned_offsets = branch_time_binned_offsets.to(
                device=root.kernel.weight.device,
                dtype=root.kernel.weight.dtype,
            )
            root.branch_time_binned_simulation_time = simulation_time
        else:
            for attribute in (
                'branch_time_binned_offsets',
                'branch_time_binned_simulation_time',
            ):
                if hasattr(root, attribute):
                    delattr(root, attribute)
    if refund_root_amplifier:
        set_cortex_amplifier(
            model,
            cortex_id=root.cortex_id,
            amplifier=root_amplifier_after_gauge,
        )
    if post_refit_root_threshold is not None:
        set_cortex_thresholds(
            model,
            cortex_id=root.cortex_id,
            hidden_threshold=post_refit_root_threshold,
        )
    if post_refit_root_input_wave_delay_steps is not None:
        set_cortex_input_wave_delay(
            model,
            cortex_id=root.cortex_id,
            delay_steps=post_refit_root_input_wave_delay_steps,
        )
    if post_refit_root_input_wave_handoff_branch_activity_fraction is not None:
        set_cortex_input_wave_handoff(
            model,
            cortex_id=root.cortex_id,
            branch_activity_fraction=(
                post_refit_root_input_wave_handoff_branch_activity_fraction
            ),
        )
    potential_gauge_audits = []
    for gauge_settings in post_refit_potential_gauges:
        if not isinstance(gauge_settings, dict):
            raise TypeError(
                'each post_refit_potential_gauges item must be a mapping'
            )
        potential_gauge_audits.append(apply_cortex_potential_gauge(
            model,
            **copy.deepcopy(gauge_settings),
        ))
    potential_audit = None
    if normalization_scale_calibration is not None:
        scale_audit = calibrate_normalization_scale(root, **normalization_scale_calibration)
        if hasattr(root, 'branch_time_binned_offsets'):
            with torch.no_grad():
                root.branch_time_binned_offsets.mul_(
                    float(scale_audit['kernel_scale'])
                )
        model.normalization_scale_calibration_audit = scale_audit
        if summary_writer is not None:
            for name, value in scale_audit.items():
                summary_writer.add_scalar(f'calibration/normalization_scale/{name}', value, 0)
    if normalization_refit_audit is not None:
        model.normalization_aware_refit_audit = normalization_refit_audit
        if summary_writer is not None:
            import json
            for name, value in normalization_refit_audit.items():
                if name != 'trace':
                    summary_writer.add_scalar(f'calibration/normalization_aware_refit/{name}', value, 0)
            (Path(summary_writer.log_dir) / 'normalization_aware_refit.json').write_text(
                json.dumps(normalization_refit_audit, indent=2), encoding='utf-8'
            )
    if data_weighted_potential_audit:
        potential_audit = _collect_data_weighted_potential_audit(
            model,
            source,
            [(images, None) for images in potential_audit_batches],
            max_samples=potential_audit_max_samples,
            sample_cap=potential_audit_sample_cap,
        )
    if summary_writer is not None:
        summary_writer.add_scalar(
            'calibration/data_response_mse_before',
            float(before_residual.item()),
            global_step=0,
        )
        summary_writer.add_scalar(
            'calibration/data_response_mse_after',
            float(after_residual.item()),
            global_step=0,
        )
        summary_writer.add_scalar(
            'calibration/branch_response_mse_after',
            float(branch_after_residual.item()),
            global_step=0,
        )
        summary_writer.add_scalar(
            'calibration/branch_primary_residual_enabled',
            float(solve_mode == 'branch_primary_residual'),
            global_step=0,
        )
        summary_writer.add_scalar(
            'calibration/direct_residual_prior_enabled',
            float(solve_mode == 'direct_residual_dual'),
            global_step=0,
        )
        summary_writer.add_scalar(
            'calibration/receiver_correction_relative_l2',
            receiver_correction_relative_l2,
            global_step=0,
        )
        summary_writer.add_scalar(
            'calibration/residual_fit_row_count',
            residual_fit_row_count,
            global_step=0,
        )
        if direct_residual_mse_before is not None:
            summary_writer.add_scalar(
                'calibration/direct_residual_mse_before',
                float(direct_residual_mse_before.item()),
                global_step=0,
            )
            summary_writer.add_scalar(
                'calibration/direct_residual_mse_after',
                float(direct_residual_mse_after.item()),
                global_step=0,
            )
            summary_writer.add_scalar(
                'calibration/combined_response_mse_after',
                float(combined_response_mse_after.item()),
                global_step=0,
            )
            summary_writer.add_scalar(
                'calibration/branch_response_rms',
                branch_response_rms,
                global_step=0,
            )
            summary_writer.add_scalar(
                'calibration/direct_residual_target_rms',
                direct_residual_target_rms,
                global_step=0,
            )
            summary_writer.add_scalar(
                'calibration/direct_response_rms',
                direct_response_rms,
                global_step=0,
            )
            summary_writer.add_scalar(
                'calibration/combined_response_rms',
                combined_response_rms,
                global_step=0,
            )
            summary_writer.add_scalar(
                'calibration/residual_target_response_rms',
                residual_target_response_rms,
                global_step=0,
            )
            summary_writer.add_scalar(
                'calibration/branch_contribution_rms_fraction',
                branch_contribution_rms_fraction,
                global_step=0,
            )
            summary_writer.add_scalar(
                'calibration/direct_to_branch_rms_ratio',
                direct_to_branch_rms_ratio,
                global_step=0,
            )
            summary_writer.add_scalar(
                'calibration/direct_weight_l1_per_output',
                direct_weight_l1_per_output,
                global_step=0,
            )
            summary_writer.add_scalar(
                'calibration/direct_positive_weight_l1_per_output',
                direct_positive_weight_l1_per_output,
                global_step=0,
            )
            summary_writer.add_scalar(
                'calibration/direct_negative_weight_l1_per_output',
                direct_negative_weight_l1_per_output,
                global_step=0,
            )
            summary_writer.add_scalar(
                'calibration/direct_negative_weight_fraction',
                direct_negative_weight_fraction,
                global_step=0,
            )
            summary_writer.add_scalar(
                'calibration/source_direct_weight_l1_per_output',
                source_direct_weight_l1_per_output,
                global_step=0,
            )
            summary_writer.add_scalar(
                'calibration/direct_weight_relative_l2_change',
                direct_weight_relative_l2_change,
                global_step=0,
            )
            summary_writer.add_scalar(
                'calibration/direct_residual_time_bin_count',
                direct_residual_time_bin_count,
                global_step=0,
            )
            summary_writer.add_scalar(
                'calibration/direct_time_bin_adjacent_relative_l2',
                direct_time_bin_adjacent_relative_l2,
                global_step=0,
            )
            summary_writer.add_scalar(
                'calibration/direct_state_weight_l1_per_output',
                direct_state_weight_l1_per_output,
                global_step=0,
            )
            summary_writer.add_scalar(
                'calibration/direct_state_branch_activity_mean',
                direct_state_branch_activity_mean,
                global_step=0,
            )
            summary_writer.add_scalar(
                'calibration/direct_state_branch_activity_std',
                direct_state_branch_activity_std,
                global_step=0,
            )
            summary_writer.add_scalar(
                'calibration/direct_state_direct_activity_mean',
                direct_state_direct_activity_mean,
                global_step=0,
            )
            summary_writer.add_scalar(
                'calibration/direct_state_direct_activity_std',
                direct_state_direct_activity_std,
                global_step=0,
            )
            summary_writer.add_scalar(
                'calibration/direct_state_spatial_patch_count',
                (
                    direct_state_spatial_patch_count
                    if spatial_state_requested
                    else 0
                ),
                global_step=0,
            )
            summary_writer.add_scalar(
                'calibration/direct_state_branch_spatial_std_mean',
                direct_state_branch_spatial_std_mean,
                global_step=0,
            )
            summary_writer.add_scalar(
                'calibration/direct_state_branch_spatial_centered_std_mean',
                direct_state_branch_spatial_centered_std_mean,
                global_step=0,
            )
            summary_writer.add_scalar(
                'calibration/direct_state_branch_current_spatial_centered_std_mean',
                direct_state_branch_current_spatial_centered_std_mean,
                global_step=0,
            )
            for state_basis in sorted(direct_residual_state_bases):
                summary_writer.add_scalar(
                    'calibration/direct_residual_state_basis/'
                    f'{state_basis}',
                    float(direct_residual_state_basis == state_basis),
                    global_step=0,
                )
            for bin_index, stats in direct_residual_time_bin_audit.items():
                for name, value in stats.items():
                    summary_writer.add_scalar(
                        'calibration/direct_residual_time_bin/'
                        f'b_{bin_index:02d}/{name}',
                        value,
                        global_step=0,
                    )
            summary_writer.add_scalar(
                'calibration/branch_weight_l1_per_output',
                branch_weight_l1_per_output,
                global_step=0,
            )
            for sampling_mode in sorted(direct_residual_row_sampling_modes):
                summary_writer.add_scalar(
                    'calibration/direct_residual_row_sampling_mode/'
                    f'{sampling_mode}',
                    float(direct_residual_row_sampling_mode == sampling_mode),
                    global_step=0,
                )
            for name in (
                'row_count',
                'unique_sample_count',
                'unique_sample_fraction',
                'batch_position_zero_fraction',
                'audit_fit_overlap_count',
                'audit_fit_overlap_fraction',
                'audit_disjoint_requested',
            ):
                if name in direct_residual_sampling_audit:
                    summary_writer.add_scalar(
                        f'calibration/direct_residual_sampling/{name}',
                        direct_residual_sampling_audit[name],
                        global_step=0,
                    )
            for time_index, row_count in (
                direct_residual_sampling_audit['time_row_counts'].items()
            ):
                summary_writer.add_scalar(
                    'calibration/direct_residual_fit_time_row_count/'
                    f't_{time_index:02d}',
                    row_count,
                    global_step=0,
                )
            for name, value in direct_residual_independent_audit.items():
                summary_writer.add_scalar(
                    f'calibration/direct_residual_independent_audit/{name}',
                    value,
                    global_step=0,
                )
            for time_index, stats in direct_residual_time_audit.items():
                for name, value in stats.items():
                    summary_writer.add_scalar(
                        'calibration/direct_residual_time_audit/'
                        f't_{time_index:02d}/{name}',
                        value,
                        global_step=0,
                    )
        summary_writer.add_scalar(
            'calibration/branch_time_bin_count',
            branch_time_bin_count,
            global_step=0,
        )
        summary_writer.add_scalar(
            'calibration/branch_time_bin_mse_before',
            branch_time_bin_mse_before,
            global_step=0,
        )
        summary_writer.add_scalar(
            'calibration/branch_time_bin_mse_after',
            branch_time_bin_mse_after,
            global_step=0,
        )
        summary_writer.add_scalar(
            'calibration/branch_time_bin_adjacent_relative_l2',
            branch_time_bin_adjacent_relative_l2,
            global_step=0,
        )
        summary_writer.add_scalar(
            'calibration/branch_time_bin_offset_l2_per_output',
            branch_time_bin_offset_l2_per_output,
            global_step=0,
        )
        summary_writer.add_scalar(
            'calibration/branch_time_bin_column_l1_max_abs_error',
            branch_time_bin_column_l1_max_abs_error,
            global_step=0,
        )
        for bin_index, stats in branch_time_bin_audit.items():
            for name, value in stats.items():
                summary_writer.add_scalar(
                    'calibration/branch_time_bin/'
                    f'b_{bin_index:02d}/{name}',
                    value,
                    global_step=0,
                )
        summary_writer.add_scalar(
            'calibration/data_response_mse_after_gauge',
            float(post_gauge_residual.item()),
            global_step=0,
        )
        summary_writer.add_scalar(
            'calibration/row_weight_mode_enabled',
            float(row_weight_mode != 'uniform'),
            global_step=0,
        )
        summary_writer.add_scalar(
            'calibration/row_weight_source_output_onset',
            float(row_weight_mode == 'source_output_onset_balanced'),
            global_step=0,
        )
        summary_writer.add_scalar(
            'calibration/row_weight_source_output_active',
            float(row_weight_mode == 'source_output_active_balanced'),
            global_step=0,
        )
        summary_writer.add_scalar(
            'calibration/row_weight_balancing_applied',
            float(row_weight_balancing_applied),
            global_step=0,
        )
        summary_writer.add_scalar(
            'calibration/row_priority_fraction',
            row_priority_fraction,
            global_step=0,
        )
        summary_writer.add_scalar(
            'calibration/row_priority_weight',
            row_priority_weight,
            global_step=0,
        )
        summary_writer.add_scalar(
            'calibration/row_nonpriority_weight',
            row_nonpriority_weight,
            global_step=0,
        )
        summary_writer.add_scalar(
            'calibration/row_weight_effective_fraction',
            row_weight_effective_fraction,
            global_step=0,
        )
        summary_writer.add_scalar(
            'calibration/data_response_weighted_mse_before',
            float(weighted_before_residual.item()),
            global_step=0,
        )
        summary_writer.add_scalar(
            'calibration/data_response_weighted_mse_after',
            float(weighted_after_residual.item()),
            global_step=0,
        )
        summary_writer.add_scalar(
            'calibration/common_to_contrast_blocked_mixing_enabled',
            float(solve_mode == 'common_to_contrast_blocked_mixing'),
            global_step=0,
        )
        summary_writer.add_scalar(
            'calibration/common_scale',
            common_scale,
            global_step=0,
        )
        summary_writer.add_scalar(
            'calibration/common_mse_before',
            float(common_mse_before.item()),
            global_step=0,
        )
        summary_writer.add_scalar(
            'calibration/common_mse_after',
            float(common_mse_after.item()),
            global_step=0,
        )
        summary_writer.add_scalar(
            'calibration/contrast_mse_before',
            float(contrast_mse_before.item()),
            global_step=0,
        )
        summary_writer.add_scalar(
            'calibration/contrast_mse_after',
            float(contrast_mse_after.item()),
            global_step=0,
        )
        summary_writer.add_scalar(
            'calibration/common_to_contrast_leakage_linf',
            common_to_contrast_leakage,
            global_step=0,
        )
        summary_writer.add_scalar(
            'calibration/contrast_to_common_leakage_linf',
            contrast_to_common_leakage,
            global_step=0,
        )
        summary_writer.add_scalar(
            'calibration/source_net_target_enabled',
            float(target_mode == 'source_net_compensated'),
            global_step=0,
        )
        summary_writer.add_scalar(
            'calibration/source_net_target_blend',
            source_net_target_blend,
            global_step=0,
        )
        summary_writer.add_scalar(
            'calibration/source_common_net_contrast_enabled',
            float(target_mode.startswith(
                'source_excitatory_common_net_contrast'
            )),
            global_step=0,
        )
        summary_writer.add_scalar(
            'calibration/source_common_net_contrast_rms_matched',
            float(target_mode.endswith('_rms_matched')),
            global_step=0,
        )
        summary_writer.add_scalar(
            'calibration/source_excitatory_common_rms',
            source_excitatory_common_rms,
            global_step=0,
        )
        summary_writer.add_scalar(
            'calibration/source_excitatory_contrast_rms',
            source_excitatory_contrast_rms,
            global_step=0,
        )
        summary_writer.add_scalar(
            'calibration/source_net_compensated_contrast_rms',
            source_net_compensated_contrast_rms,
            global_step=0,
        )
        summary_writer.add_scalar(
            'calibration/source_contrast_scale',
            source_contrast_scale,
            global_step=0,
        )
        summary_writer.add_scalar(
            'calibration/target_common_rms',
            target_common_rms,
            global_step=0,
        )
        summary_writer.add_scalar(
            'calibration/target_contrast_rms',
            target_contrast_rms,
            global_step=0,
        )
        summary_writer.add_scalar(
            'calibration/target_total_rms',
            target_total_rms,
            global_step=0,
        )
        if source_net_residual_before is not None:
            summary_writer.add_scalar(
                'calibration/source_net_mse_before',
                float(source_net_residual_before.item()),
                global_step=0,
            )
            summary_writer.add_scalar(
                'calibration/source_net_mse_after_linearized',
                float(source_net_residual_after.item()),
                global_step=0,
            )
        summary_writer.add_scalar(
            'calibration/post_refit_branch_scale',
            post_refit_branch_scale,
            global_step=0,
        )
        summary_writer.add_scalar(
            'calibration/direct_residual_scale',
            direct_residual_scale,
            global_step=0,
        )
        summary_writer.add_scalar(
            'calibration/direct_residual_positive_scale',
            direct_residual_positive_scale,
            global_step=0,
        )
        summary_writer.add_scalar(
            'calibration/direct_residual_negative_scale',
            direct_residual_negative_scale,
            global_step=0,
        )
        summary_writer.add_scalar(
            'calibration/direct_residual_ridge_multiplier',
            direct_residual_ridge_multiplier,
            global_step=0,
        )
        summary_writer.add_scalar(
            'calibration/post_refit_root_threshold',
            float(root.hidden_nv.thresholds.mean().item()),
            global_step=0,
        )
        summary_writer.add_scalar(
            'calibration/post_refit_root_input_wave_delay_steps',
            float(getattr(root, 'input_wave_delay_steps', 0)),
            global_step=0,
        )
        summary_writer.add_scalar(
            'calibration/'
            'post_refit_root_input_wave_handoff_branch_activity_fraction',
            float(getattr(
                root,
                'input_wave_handoff_branch_activity_fraction',
                0.0,
            ) or 0.0),
            global_step=0,
        )
        summary_writer.add_scalar(
            'calibration/runtime_root_threshold_scale',
            float(root.hidden_nv.thresholds.mean().item())
            / root_threshold_before_refit_override,
            global_step=0,
        )
        summary_writer.add_scalar(
            'calibration/root_amplifier_before_gauge',
            root_amplifier_before_gauge,
            global_step=0,
        )
        summary_writer.add_scalar(
            'calibration/root_amplifier_after_gauge',
            root_amplifier_after_gauge,
            global_step=0,
        )
        for audit in potential_gauge_audits:
            prefix = (
                'scale_allocation/post_refit_potential_gauge/'
                f"{audit['cortex_id']}"
            )
            for name, value in audit.items():
                if name == 'cortex_id':
                    continue
                summary_writer.add_scalar(
                    f'{prefix}/{name}',
                    value,
                    global_step=0,
                )
        summary_writer.add_scalar(
            'calibration/data_response_sample_time_rows',
            int(reconstructed.shape[0]),
            global_step=0,
        )
        summary_writer.add_scalar(
            'calibration/linear_algebra_on_cpu',
            float(linear_algebra_device == 'cpu'),
            global_step=0,
        )
        summary_writer.add_scalar(
            'calibration/refit_negative_weight_fraction',
            float((fitted_weight < 0).float().mean().item()),
            global_step=0,
        )
        if potential_audit is not None:
            for (cortex_label, potential_kind, scale_kind), stats in (
                potential_audit.items()
            ):
                prefix = (
                    'data_weighted_potential/'
                    f'{cortex_label}/{potential_kind}/{scale_kind}'
                )
                for name, value in stats.items():
                    summary_writer.add_scalar(
                        f'{prefix}/{name}',
                        value,
                        global_step=0,
                    )
            for potential_kind in ('excitatory', 'net'):
                for scale_kind in ('raw', 'threshold_normalized'):
                    for statistic in (
                        'abs_mean', 'rms', 'positive_mean', 'q90', 'q99', 'max'
                    ):
                        inner_value = potential_audit[
                            ('A-1', potential_kind, scale_kind)
                        ].get(statistic)
                        root_value = potential_audit[
                            ('A', potential_kind, scale_kind)
                        ].get(statistic)
                        source_value = potential_audit[
                            ('source_A', potential_kind, scale_kind)
                        ].get(statistic)
                        if inner_value is not None and abs(inner_value) > 1.0e-12:
                            summary_writer.add_scalar(
                                'data_weighted_potential_ratio/'
                                f'A_over_A-1/{potential_kind}/{scale_kind}/'
                                f'{statistic}',
                                root_value / inner_value,
                                global_step=0,
                            )
                        if source_value is not None and abs(source_value) > 1.0e-12:
                            summary_writer.add_scalar(
                                'data_weighted_potential_ratio/'
                                f'A_over_source_A/{potential_kind}/{scale_kind}/'
                                f'{statistic}',
                                root_value / source_value,
                                global_step=0,
                            )
    del source
    return model


def replace_cortex_kernel_from_checkpoint(
    model,
    cortex_id,
    source_load_from,
    source_cortex_id=None,
    target_column_l1=None,
    column_l1_epsilon=1.0e-12,
):
    """Replace one cortex kernel while retaining all target neuron state."""
    target = model.get_cortex_by_id(cortex_id)
    if target is None:
        raise ValueError(f'unknown target cortex {cortex_id!r}')
    source_model = construct_loaded_model(source_load_from)
    source_cortex_id = cortex_id if source_cortex_id is None else source_cortex_id
    source = source_model.get_cortex_by_id(source_cortex_id)
    if source is None:
        raise ValueError(f'unknown source cortex {source_cortex_id!r}')

    source_weight = source.kernel.weight.detach()
    target_weight = target.kernel.weight.detach()
    if source_weight.shape != target_weight.shape:
        raise ValueError(
            'cortex kernel transplant shape mismatch: '
            f'{tuple(source_weight.shape)} != {tuple(target_weight.shape)}'
        )
    if not torch.isfinite(source_weight).all():
        raise ValueError('source cortex kernel contains non-finite values')

    copied_weight = source_weight.to(
        device=target_weight.device,
        dtype=target_weight.dtype,
    ).clone()
    source_column_l1 = copied_weight.abs().sum(dim=0)
    if target_column_l1 is not None:
        target_column_l1 = _positive_float(
            target_column_l1,
            'target_column_l1',
        )
        column_l1_epsilon = _positive_float(
            column_l1_epsilon,
            'column_l1_epsilon',
        )
        if torch.any(source_column_l1 <= column_l1_epsilon):
            dead_columns = int(
                (source_column_l1 <= column_l1_epsilon).sum().item()
            )
            raise ValueError(
                'cannot column-L1 normalize transplanted kernel with '
                f'{dead_columns} near-zero columns'
            )
        copied_weight.mul_(
            (target_column_l1 / source_column_l1).unsqueeze(0)
        )
    copied_column_l1 = copied_weight.abs().sum(dim=0)
    target_delta = copied_weight - target_weight
    with torch.no_grad():
        target.kernel.weight = copied_weight

    audits = getattr(target, 'post_load_transform_audits', None)
    if audits is None:
        audits = []
        target.post_load_transform_audits = audits
    target_l2 = float(torch.linalg.vector_norm(target_weight).item())
    audits.append({
        'prefix': f'post_load_transform/{cortex_id}/kernel_transplant',
        'text': {
            'source_load_from': copy.deepcopy(source_load_from),
            'source_cortex_id': source_cortex_id,
            'target_column_l1': target_column_l1,
            'copied_state': 'kernel_only',
        },
        'scalars': {
            'shape_dim0': int(copied_weight.shape[0]),
            'shape_dim1': int(copied_weight.shape[1]),
            'source_column_l1_mean': float(source_column_l1.mean().item()),
            'source_column_l1_min': float(source_column_l1.min().item()),
            'source_column_l1_max': float(source_column_l1.max().item()),
            'copied_column_l1_mean': float(copied_column_l1.mean().item()),
            'copied_column_l1_min': float(copied_column_l1.min().item()),
            'copied_column_l1_max': float(copied_column_l1.max().item()),
            'target_relative_l2_change': (
                float(torch.linalg.vector_norm(target_delta).item()) / target_l2
                if target_l2 > 0.0 else 0.0
            ),
            'copied_positive_fraction': float(
                (copied_weight > 0).to(torch.float32).mean().item()
            ),
            'copied_negative_fraction': float(
                (copied_weight < 0).to(torch.float32).mean().item()
            ),
            'copied_zero_fraction': float(
                (copied_weight == 0).to(torch.float32).mean().item()
            ),
        },
    })
    del source_model
    return model


def apply_post_load_transforms(model, post_load_transforms, run_seed=None):
    if post_load_transforms is None:
        model.post_load_transforms = []
        return model
    if not isinstance(post_load_transforms, list):
        raise TypeError('model.post_load_transforms must be a list of mappings.')

    applied_transforms = []
    for transform in post_load_transforms:
        if not isinstance(transform, dict):
            raise TypeError('each model.post_load_transforms item must be a mapping.')
        transform = copy.deepcopy(transform)
        transform_type = transform.pop('type', None)
        if transform_type == 'pca_yield':
            cortex_id = transform.pop('cortex_id', 'A')
            model.pca_yield_cortex(cortex_id=cortex_id, **transform)
        elif transform_type == 'patch_pca_yield':
            cortex_id = transform.pop('cortex_id', 'A')
            model.patch_pca_yield_cortex(cortex_id=cortex_id, **transform)
        elif transform_type == 'patch_mean_pca_yield':
            cortex_id = transform.pop('cortex_id', 'A')
            model.patch_mean_pca_yield_cortex(
                cortex_id=cortex_id,
                **transform
            )
        elif transform_type == 'patch_tensor_cp_yield':
            cortex_id = transform.pop('cortex_id', 'A')
            model.patch_tensor_cp_yield_cortex(
                cortex_id=cortex_id,
                **transform
            )
        elif transform_type == 'patch_tucker_yield':
            cortex_id = transform.pop('cortex_id', 'A')
            model.patch_tucker_yield_cortex(
                cortex_id=cortex_id,
                **transform
            )
        elif transform_type == 'receiver_grouped_patch_pca_yield':
            cortex_id = transform.pop('cortex_id', 'A')
            model.receiver_grouped_patch_pca_yield_cortex(
                cortex_id=cortex_id,
                **transform
            )
        elif transform_type == 'set_cortex_thresholds':
            cortex_id = transform.pop('cortex_id')
            set_cortex_thresholds(model, cortex_id=cortex_id, **transform)
        elif transform_type == 'scale_cortex_thresholds':
            cortex_id = transform.pop('cortex_id')
            scale_cortex_thresholds(model, cortex_id=cortex_id, **transform)
        elif transform_type == 'set_cortex_amplifier':
            cortex_id = transform.pop('cortex_id')
            set_cortex_amplifier(model, cortex_id=cortex_id, **transform)
        elif transform_type == 'set_cortex_input_wave_delay':
            cortex_id = transform.pop('cortex_id')
            set_cortex_input_wave_delay(
                model,
                cortex_id=cortex_id,
                **transform,
            )
        elif transform_type == 'set_cortex_input_wave_handoff':
            cortex_id = transform.pop('cortex_id')
            set_cortex_input_wave_handoff(
                model,
                cortex_id=cortex_id,
                **transform,
            )
        elif transform_type == 'scale_cortex_inhibition':
            cortex_id = transform.pop('cortex_id')
            scale_cortex_inhibition(model, cortex_id=cortex_id, **transform)
        elif transform_type == 'set_cortex_inhibition':
            cortex_id = transform.pop('cortex_id')
            set_cortex_inhibition(model, cortex_id=cortex_id, **transform)
        elif transform_type == 'replace_cortex_kernel_from_checkpoint':
            cortex_id = transform.pop('cortex_id')
            if 'source_load_from' not in transform:
                raise ValueError(
                    'replace_cortex_kernel_from_checkpoint requires '
                    'source_load_from'
                )
            transform['source_load_from'] = _resolve_seed_checkpoint_reference(
                transform['source_load_from'],
                run_seed,
            )
            replace_cortex_kernel_from_checkpoint(
                model,
                cortex_id=cortex_id,
                **transform,
            )
        elif transform_type == 'shuffle_factorized_path':
            cortex_id = transform.pop('cortex_id', 'A')
            shuffle_factorized_path(model, cortex_id=cortex_id, **transform)
        else:
            raise ValueError(
                'unsupported post-load transform type: '
                f'{transform_type!r}'
            )
        applied = {'type': transform_type, 'cortex_id': cortex_id}
        applied.update(transform)
        applied_transforms.append(applied)

    model.post_load_transforms = applied_transforms
    return model


def _positive_float(value, field_name):
    value = float(value)
    if value <= 0:
        raise ValueError(f'{field_name} must be > 0')
    return value


def _nonnegative_int(value, field_name):
    if isinstance(value, bool):
        raise TypeError(f'{field_name} must be a non-negative integer')
    parsed = int(value)
    if float(value) != float(parsed) or parsed < 0:
        raise ValueError(f'{field_name} must be a non-negative integer')
    return parsed


def _non_negative_float(value, field_name):
    value = float(value)
    if value < 0:
        raise ValueError(f'{field_name} must be >= 0')
    return value


def _log_bound(value, dtype, device):
    return torch.tensor(
        -math.inf if value == 0 else math.log(value),
        dtype=dtype,
        device=device,
    )


def set_cortex_thresholds(
    model,
    cortex_id,
    hidden_threshold=None,
    hidden_lower_bound=None,
    hidden_upper_bound=None,
):
    if hidden_threshold is None and hidden_lower_bound is None and hidden_upper_bound is None:
        raise ValueError(
            'set_cortex_thresholds requires at least one of hidden_threshold, '
            'hidden_lower_bound, or hidden_upper_bound'
        )
    cortex = model.get_cortex_by_id(cortex_id)
    neuron_vector = cortex.hidden_nv
    dtype = neuron_vector.log_thresholds.dtype
    device = neuron_vector.log_thresholds.device

    current_lower = (
        0.0
        if torch.isneginf(neuron_vector.log_thresholds_lower_bound).item()
        else float(torch.exp(neuron_vector.log_thresholds_lower_bound).item())
    )
    current_upper = float(torch.exp(neuron_vector.log_thresholds_upper_bound).item())

    lower = (
        current_lower
        if hidden_lower_bound is None
        else _non_negative_float(hidden_lower_bound, 'hidden_lower_bound')
    )
    upper = (
        current_upper
        if hidden_upper_bound is None
        else _positive_float(hidden_upper_bound, 'hidden_upper_bound')
    )
    if lower > upper:
        raise ValueError('hidden_lower_bound must be <= hidden_upper_bound')

    neuron_vector.log_thresholds_lower_bound = _log_bound(lower, dtype, device)
    neuron_vector.log_thresholds_upper_bound = _log_bound(upper, dtype, device)
    neuron_vector.threshold_min = lower

    if hidden_threshold is not None:
        threshold = _positive_float(hidden_threshold, 'hidden_threshold')
        if threshold < lower or threshold > upper:
            raise ValueError(
                'hidden_threshold must be within hidden lower/upper bounds'
            )
        log_threshold = math.log(threshold)
        neuron_vector.log_thresholds = torch.full(
            neuron_vector.log_thresholds.shape,
            log_threshold,
            dtype=dtype,
            device=device,
        )
        neuron_vector.base_log_thresholds = neuron_vector.log_thresholds.clone()


def set_cortex_amplifier(model, cortex_id, amplifier):
    amplifier = _positive_float(amplifier, 'amplifier')
    cortex = model.get_cortex_by_id(cortex_id)
    dtype = cortex.log_amplifier.dtype
    device = cortex.log_amplifier.device
    lower = float(torch.exp(cortex.log_amplifier_lower_bound).item())
    upper = float(torch.exp(cortex.log_amplifier_upper_bound).item())
    if amplifier < lower or amplifier > upper:
        raise ValueError(
            f'amplifier must be within cortex bounds [{lower}, {upper}], '
            f'got {amplifier}'
        )
    cortex.log_amplifier = torch.tensor(
        math.log(amplifier),
        dtype=dtype,
        device=device,
    )


def set_cortex_input_wave_delay(model, cortex_id, delay_steps):
    """Delay only the cortex-owned input sender, leaving subcortex senders intact."""
    delay_steps = _nonnegative_int(delay_steps, 'delay_steps')
    cortex = model.get_cortex_by_id(cortex_id)
    cortex.input_wave_delay_base_steps = delay_steps
    cortex.input_wave_delay_per_subcortex_depth_steps = 0
    cortex.input_wave_delay_steps = delay_steps
    cortex._input_wave_delay_history = []


def set_cortex_input_wave_handoff(
    model,
    cortex_id,
    branch_activity_fraction,
):
    """Gate only the cortex-owned sender after subcortex activity accumulates."""
    branch_activity_fraction = float(branch_activity_fraction)
    if not 0.0 < branch_activity_fraction <= 1.0:
        raise ValueError('branch_activity_fraction must be in (0, 1]')
    cortex = model.get_cortex_by_id(cortex_id)
    if not cortex.subcortexs:
        raise ValueError('input-wave handoff requires a cortex with subcortexs')
    cortex.input_wave_handoff_branch_activity_fraction = (
        branch_activity_fraction
    )
    cortex.input_wave_handoff_closed_fraction = 0.0


def scale_cortex_thresholds(model, cortex_id, scale):
    """Scale hidden thresholds and their bounds without changing dispersion."""
    scale = _positive_float(scale, 'scale')
    cortex = model.get_cortex_by_id(cortex_id)
    neuron_vector = cortex.hidden_nv
    log_scale = math.log(scale)
    with torch.no_grad():
        neuron_vector.log_thresholds.add_(log_scale)
        neuron_vector.base_log_thresholds.add_(log_scale)
        if not torch.isneginf(
            neuron_vector.log_thresholds_lower_bound
        ).item():
            neuron_vector.log_thresholds_lower_bound.add_(log_scale)
        neuron_vector.log_thresholds_upper_bound.add_(log_scale)
    neuron_vector.threshold_min *= scale


def scale_cortex_inhibition(model, cortex_id, scale):
    """Scale every additive inhibition term in one cortex by one factor.

    Scaling the excitatory potential, threshold, lateral inhibition, and
    dominance inhibition together preserves the hidden spike decision.  This
    transform handles the two inhibition terms; callers remain responsible for
    the matching kernel/amplifier and threshold allocation.
    """
    scale = _positive_float(scale, 'scale')
    cortex = model.get_cortex_by_id(cortex_id)
    cortex.lateral_inhibition_factor = (
        float(getattr(cortex, 'lateral_inhibition_factor', 0.0)) * scale
    )
    cortex.dominance_inhibition_factor = (
        float(getattr(cortex, 'dominance_inhibition_factor', 0.0)) * scale
    )


def set_cortex_inhibition(
    model,
    cortex_id,
    lateral_factor=None,
    dominance_factor=None,
):
    """Set additive inhibition coefficients without applying a potential gauge."""
    if lateral_factor is None and dominance_factor is None:
        raise ValueError(
            'set_cortex_inhibition requires lateral_factor or dominance_factor'
        )
    cortex = model.get_cortex_by_id(cortex_id)
    if lateral_factor is not None:
        cortex.lateral_inhibition_factor = _non_negative_float(
            lateral_factor,
            'lateral_factor',
        )
    if dominance_factor is not None:
        cortex.dominance_inhibition_factor = _non_negative_float(
            dominance_factor,
            'dominance_factor',
        )


def apply_cortex_potential_gauge(
    model,
    cortex_id,
    target_amplifier=None,
    potential_scale=None,
    target_threshold_mean=None,
    target_kernel_per_output_l1=None,
    target_kernel_per_output_l2=None,
):
    """Reallocate one cortex scale while preserving every spike decision.

    Exactly one target axis determines the homogeneous potential scale. The
    kernel compensates any amplifier change, while thresholds and additive
    inhibition receive the same potential scale.
    """
    selectors = {
        'potential_scale': potential_scale,
        'target_threshold_mean': target_threshold_mean,
        'target_kernel_per_output_l1': target_kernel_per_output_l1,
        'target_kernel_per_output_l2': target_kernel_per_output_l2,
    }
    selected = [name for name, value in selectors.items() if value is not None]
    if len(selected) != 1:
        raise ValueError(
            'potential gauge requires exactly one of potential_scale, '
            'target_threshold_mean, target_kernel_per_output_l1, or '
            'target_kernel_per_output_l2'
        )

    cortex = model.get_cortex_by_id(cortex_id)
    weight_before = cortex.kernel.weight.detach().clone()
    amplifier_before = float(cortex.amplifier)
    threshold_before = float(cortex.hidden_nv.thresholds.mean().item())
    lateral_before = float(getattr(cortex, 'lateral_inhibition_factor', 0.0))
    dominance_before = float(getattr(
        cortex, 'dominance_inhibition_factor', 0.0
    ))
    l1_before = float(weight_before.abs().sum(dim=0).mean().item())
    l2_before = float(torch.linalg.vector_norm(
        weight_before, dim=0
    ).mean().item())

    amplifier_after = (
        amplifier_before
        if target_amplifier is None
        else _positive_float(target_amplifier, 'target_amplifier')
    )
    if potential_scale is not None:
        potential_scale = _positive_float(potential_scale, 'potential_scale')
        kernel_scale = potential_scale * amplifier_before / amplifier_after
    elif target_threshold_mean is not None:
        target_threshold_mean = _positive_float(
            target_threshold_mean, 'target_threshold_mean'
        )
        potential_scale = target_threshold_mean / threshold_before
        kernel_scale = potential_scale * amplifier_before / amplifier_after
    elif target_kernel_per_output_l1 is not None:
        target_l1 = _positive_float(
            target_kernel_per_output_l1,
            'target_kernel_per_output_l1',
        )
        if l1_before <= 0.0:
            raise ValueError(f'cortex {cortex_id} has zero per-output L1 norm')
        kernel_scale = target_l1 / l1_before
        potential_scale = kernel_scale * amplifier_after / amplifier_before
    else:
        target_l2 = _positive_float(
            target_kernel_per_output_l2,
            'target_kernel_per_output_l2',
        )
        if l2_before <= 0.0:
            raise ValueError(f'cortex {cortex_id} has zero per-output L2 norm')
        kernel_scale = target_l2 / l2_before
        potential_scale = kernel_scale * amplifier_after / amplifier_before

    with torch.no_grad():
        cortex.kernel.weight.mul_(kernel_scale)
    set_cortex_amplifier(
        model,
        cortex_id=cortex_id,
        amplifier=amplifier_after,
    )
    scale_cortex_thresholds(
        model,
        cortex_id=cortex_id,
        scale=potential_scale,
    )
    scale_cortex_inhibition(
        model,
        cortex_id=cortex_id,
        scale=potential_scale,
    )

    weight_after = cortex.kernel.weight.detach()
    expected_drive = weight_before * (amplifier_before * potential_scale)
    actual_drive = weight_after * amplifier_after
    drive_error = torch.linalg.vector_norm(actual_drive - expected_drive)
    drive_reference = torch.linalg.vector_norm(expected_drive).clamp_min(
        1.0e-12
    )
    return {
        'cortex_id': cortex_id,
        'potential_scale': potential_scale,
        'kernel_scale': kernel_scale,
        'amplifier_before': amplifier_before,
        'amplifier_after': float(cortex.amplifier),
        'threshold_mean_before': threshold_before,
        'threshold_mean_after': float(cortex.hidden_nv.thresholds.mean().item()),
        'kernel_per_output_l1_before': l1_before,
        'kernel_per_output_l1_after': float(
            weight_after.abs().sum(dim=0).mean().item()
        ),
        'kernel_per_output_l2_before': l2_before,
        'kernel_per_output_l2_after': float(torch.linalg.vector_norm(
            weight_after, dim=0
        ).mean().item()),
        'lateral_inhibition_before': lateral_before,
        'lateral_inhibition_after': float(getattr(
            cortex, 'lateral_inhibition_factor', 0.0
        )),
        'dominance_inhibition_before': dominance_before,
        'dominance_inhibition_after': float(getattr(
            cortex, 'dominance_inhibition_factor', 0.0
        )),
        'drive_relative_l2_error': float((
            drive_error / drive_reference
        ).item()),
    }


def shuffle_factorized_path(
    model,
    cortex_id='A',
    subcortex_id='A-1',
    shuffle_local_per_channel=True,
    shuffle_branch_rows=True,
):
    """Destroy factorized feature correspondence while preserving weight statistics.

    Local weights are independently permuted within each output channel, so every
    channel keeps its exact marginal distribution and norm.  Root branch rows are
    permuted as complete receiver vectors, preserving their joint class statistics.
    The source-direct block is required to be exactly zero and is never modified.
    Randomness comes from the experiment's already-seeded global torch generator.
    """
    root = model.get_cortex_by_id(cortex_id)
    subcortex = model.get_cortex_by_id(subcortex_id)
    local_weight = subcortex.kernel.weight
    root_weight = root.kernel.weight
    direct_input_len = int(root.kernel.input_len)

    if local_weight.ndim != 2 or root_weight.ndim != 2:
        raise ValueError('shuffle_factorized_path requires 2D cortex kernels')
    if direct_input_len <= 0 or direct_input_len >= root_weight.shape[0]:
        raise ValueError(
            'shuffle_factorized_path requires both direct and branch root rows'
        )
    direct_block = root_weight[:direct_input_len, :]
    if torch.count_nonzero(direct_block).item() != 0:
        raise ValueError(
            'shuffle_factorized_path requires an exactly-zero source-direct block'
        )
    if not shuffle_local_per_channel and not shuffle_branch_rows:
        raise ValueError(
            'shuffle_factorized_path requires at least one enabled shuffle'
        )

    with torch.no_grad():
        if shuffle_local_per_channel:
            local_order = torch.rand(
                local_weight.shape,
                dtype=torch.float32,
                device=local_weight.device,
            ).argsort(dim=0)
            subcortex.kernel.weight = torch.gather(
                local_weight,
                dim=0,
                index=local_order,
            )

        if shuffle_branch_rows:
            branch_weight = root_weight[direct_input_len:, :].clone()
            branch_order = torch.randperm(
                branch_weight.shape[0],
                device=branch_weight.device,
            )
            root_weight[direct_input_len:, :] = branch_weight[branch_order, :]

    model.factorized_path_randomization = {
        'type': 'matched_statistics_shuffle',
        'cortex_id': cortex_id,
        'subcortex_id': subcortex_id,
        'shuffle_local_per_channel': bool(shuffle_local_per_channel),
        'shuffle_branch_rows': bool(shuffle_branch_rows),
    }


def construct_loaded_model(load_from_settings):
    if isinstance(load_from_settings, (str, Path)):
        model = load_model_from_path(load_from_settings)
        model.loaded_from = {'path': str(load_from_settings)}
        return model

    if not isinstance(load_from_settings, dict):
        raise TypeError('model.load_from must be a mapping or a checkpoint path string.')

    allowed_fields = {'experiment_name', 'sub_exp_name', 'commit_label', 'path'}
    unknown_fields = sorted(set(load_from_settings) - allowed_fields)
    if unknown_fields:
        raise ValueError(
            'model.load_from contains unsupported field(s): '
            f'{", ".join(unknown_fields)}'
        )

    has_path = 'path' in load_from_settings
    has_named_checkpoint = (
        'experiment_name' in load_from_settings
        or 'sub_exp_name' in load_from_settings
    )
    if has_path and has_named_checkpoint:
        raise ValueError(
            'model.load_from must use either path or experiment_name/sub_exp_name, '
            'not both.'
        )
    if has_path:
        model = load_model_from_path(load_from_settings['path'])
        model.loaded_from = copy.deepcopy(load_from_settings)
        return model

    missing_fields = [
        field for field in ('experiment_name', 'sub_exp_name')
        if field not in load_from_settings
    ]
    if missing_fields:
        raise ValueError(
            'model.load_from is missing required field(s): '
            f'{", ".join(missing_fields)}'
        )

    model = load_model(
        load_from_settings['experiment_name'],
        load_from_settings['sub_exp_name'],
        load_from_settings.get('commit_label'),
    )
    model.loaded_from = copy.deepcopy(load_from_settings)
    return model


def save_model(model, model_save_path):
    if model_save_path is None:
        return
    original_device = getattr(
        model,
        'device',
        torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    )
    sample_shape = getattr(model, '_state_sample_shape', None)
    if sample_shape:
        model.init_state((0, *sample_shape[1:]), torch.device('cpu'))
    model = move_to_device(model, torch.device('cpu'))

    # mkdir if not exist
    model_save_path = Path(model_save_path)
    model_save_path.parent.mkdir(parents=True, exist_ok=True)

    with model_save_path.with_suffix('.pickle').open('wb') as fp:
        pickle.dump(model, fp, protocol=pickle.HIGHEST_PROTOCOL)

    model = move_to_device(model, original_device)
    if sample_shape:
        model.init_state(sample_shape, original_device)


def _pickle_path(path):
    path = Path(path)
    return path if path.suffix == '.pickle' else path.with_suffix('.pickle')


def _load_model_pickle(model_pickle_path):
    if not model_pickle_path.is_file():
        raise FileNotFoundError(f'Model checkpoint does not exist: {model_pickle_path}')
    with model_pickle_path.open('rb') as fp:
        return pickle.load(fp)


def load_model_from_path(load_from_path):
    model_pickle_path = _pickle_path(resolve_path_from_code(load_from_path))
    return _load_model_pickle(model_pickle_path)


def load_model(load_from_experiment, load_from_sub_exp, commit_label=None):
    candidate_paths = []

    if commit_label is None:
        current_commit_label = get_commit_label()
        if current_commit_label:
            candidate_paths.append(
                get_model_save_path(
                    load_from_experiment,
                    load_from_sub_exp,
                    commit_label=current_commit_label,
                )
            )
    else:
        candidate_paths.append(
            get_model_save_path(
                load_from_experiment,
                load_from_sub_exp,
                commit_label=commit_label,
            )
        )

    candidate_paths.append(
        get_model_save_path(
            load_from_experiment,
            load_from_sub_exp,
            commit_label='',
        )
    )

    for load_from_path in candidate_paths:
        model_pickle_path = _pickle_path(load_from_path)
        if model_pickle_path.is_file():
            return _load_model_pickle(model_pickle_path)

    searched_paths = ', '.join(str(_pickle_path(path)) for path in candidate_paths)

    raise FileNotFoundError(
        f'No saved model found for experiment={load_from_experiment}, '
        f'sub_exp={load_from_sub_exp}, commit_label={commit_label or get_commit_label()!r}. '
        f'Searched: {searched_paths}'
    )
