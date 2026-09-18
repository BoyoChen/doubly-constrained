import arrow
import copy
import sys
import time
import torch
from modules.neuron_experience import configure_neuron_experience, log_neuron_experience
from modules.training_helper import (
    get_model_device,
    inspect_model,
    log_normalization_mechanism_weight_artifact,
    write_best_score_to_csv,
)
from modules.model_IO import save_model
from tqdm import tqdm
import pprint


def _format_duration(seconds):
    seconds = max(int(seconds), 0)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f'{hours:02d}:{minutes:02d}:{seconds:02d}'
    return f'{minutes:02d}:{seconds:02d}'


def _iter_with_progress(iterable, desc):
    if sys.stderr.isatty():
        yield from tqdm(iterable, desc=desc)
        return

    total = len(iterable) if hasattr(iterable, '__len__') else None
    start_time = time.monotonic()
    last_log_time = start_time
    log_interval_seconds = 60
    log_interval_steps = max(1, total // 20) if total else 100

    for index, item in enumerate(iterable, 1):
        yield item
        now = time.monotonic()
        is_done = total is not None and index >= total
        should_log = (
            index == 1
            or is_done
            or index % log_interval_steps == 0
            or now - last_log_time >= log_interval_seconds
        )
        if not should_log:
            continue

        elapsed = now - start_time
        rate = index / max(elapsed, 1.0)
        if total:
            remaining = max(total - index, 0) / max(rate, 1.0e-9)
            print(
                f'{desc}: {100.0 * index / total:.1f}% {index}/{total} '
                f'elapsed {_format_duration(elapsed)} '
                f'ETA {_format_duration(remaining)}, {rate:.2f}it/s',
                flush=True
            )
        else:
            print(
                f'{desc}: {index} batches '
                f'elapsed {_format_duration(elapsed)}, {rate:.2f}it/s',
                flush=True
            )
        last_log_time = now


def timeit(method):
    def timed_function(*args, **kw):
        t1 = arrow.now()
        result = method(*args, **kw)
        t2 = arrow.now()
        print(f'{method.__name__}, runtime: {t2-t1}')
        return result
    return timed_function


def remember(
    model, train_dataloaders, STDP_interval, stdp_update_mode='online',
    simulation_forward_mode='streaming'
):
    device = get_model_device(model)
    for images, labels in _iter_with_progress(train_dataloaders, desc='rem'):
        images, labels = images.to(device), labels.to(device)
        model.remember(
            images,
            labels,
            STDP_interval,
            stdp_update_mode=stdp_update_mode,
            simulation_forward_mode=simulation_forward_mode,
        )


@timeit
def evaluate(
    model, dataloaders, summary_writer, epoch_index, readout_params, level,
    diagnostic_settings=None, phases=None, evaluation_micro_batch_size=None
):
    print(f'Now evaluating: epoch #{epoch_index}')
    inspect_result = inspect_model(
        model, dataloaders, summary_writer, epoch_index, readout_params,
        level=level, diagnostic_settings=diagnostic_settings, phases=phases,
        evaluation_micro_batch_size=evaluation_micro_batch_size
    )
    if isinstance(inspect_result, tuple):
        if len(inspect_result) == 3:
            accuracy_dict, GA_score, selected_valid_score = inspect_result
        else:
            accuracy_dict, GA_score = inspect_result
            selected_valid_score = accuracy_dict['valid']
    else:
        accuracy_dict = inspect_result
        GA_score = accuracy_dict['valid']
        selected_valid_score = accuracy_dict['valid']

    return selected_valid_score, GA_score


@timeit
def sleep(model):
    model.sleep()


def _reset_remember_peak_memory(model):
    if not torch.cuda.is_available():
        return
    device = getattr(model, 'device', torch.device('cpu'))
    if device.type != 'cuda':
        return
    torch.cuda.reset_peak_memory_stats(device)


def _log_remember_resources(summary_writer, model, epoch_index, seconds):
    summary_writer.add_scalar(
        'runtime/remember_epoch_seconds',
        float(seconds),
        global_step=epoch_index
    )
    if not torch.cuda.is_available():
        return
    device = getattr(model, 'device', torch.device('cpu'))
    if device.type != 'cuda':
        return
    summary_writer.add_scalar(
        'memory/remember_epoch_peak_allocated_MB',
        torch.cuda.max_memory_allocated(device) / 1024**2,
        global_step=epoch_index
    )
    summary_writer.add_scalar(
        'memory/remember_epoch_peak_reserved_MB',
        torch.cuda.max_memory_reserved(device) / 1024**2,
        global_step=epoch_index
    )


def _log_post_load_transform_audits(summary_writer, model):
    root = getattr(model, 'cortex', None)
    if root is None:
        return
    audit_index = 0
    for cortex in root.iter_cortex_tree():
        audits = getattr(cortex, 'post_load_transform_audits', [])
        for audit in audits:
            prefix = audit.get(
                'prefix',
                f'post_load_transform/{cortex.cortex_id}/{audit_index}',
            )
            text = audit.get('text')
            if text:
                summary_writer.add_text(
                    f'{prefix}/settings',
                    pprint.pformat(text, indent=2),
                    global_step=0,
                )
            for name, scalar in audit.get('scalars', {}).items():
                try:
                    value = float(scalar)
                except (TypeError, ValueError):
                    continue
                if not torch.isfinite(torch.tensor(value)):
                    continue
                summary_writer.add_scalar(
                    f'{prefix}/{name}',
                    value,
                    global_step=0,
                )
            audit_index += 1


def _get_training_invariant_audit_settings(diagnostic_settings):
    raw_settings = (diagnostic_settings or {}).get(
        'training_invariant_audit',
        False,
    )
    if not raw_settings:
        return None
    if raw_settings is True:
        raw_settings = {}
    if not isinstance(raw_settings, dict):
        raise ValueError(
            'diagnostic_settings.training_invariant_audit must be bool or mapping'
        )
    settings = {
        'cortex_ids': ['A-1'],
        'root_cortex_id': 'A',
        'audit_root_direct': True,
        'audit_root_branch': False,
    }
    unknown = sorted(set(raw_settings) - set(settings))
    if unknown:
        raise ValueError(
            'training_invariant_audit contains unsupported field(s): '
            + ', '.join(unknown)
        )
    settings.update(raw_settings)
    settings['cortex_ids'] = list(settings['cortex_ids'])
    settings['audit_root_direct'] = bool(settings['audit_root_direct'])
    settings['audit_root_branch'] = bool(settings['audit_root_branch'])
    return settings


def _capture_training_invariant_audit(model, diagnostic_settings):
    settings = _get_training_invariant_audit_settings(diagnostic_settings)
    if settings is None:
        return None
    snapshot = {'settings': settings, 'cortex': {}}
    with torch.no_grad():
        for cortex_id in settings['cortex_ids']:
            cortex = model.get_cortex_by_id(cortex_id)
            snapshot['cortex'][cortex_id] = {
                'kernel': cortex.kernel.weight.detach().clone(),
                'previous_kernel': cortex.kernel.weight.detach().clone(),
                'threshold': torch.exp(
                    cortex.hidden_nv.log_thresholds.detach()
                ).clone(),
                'amplifier': torch.exp(
                    cortex.log_amplifier.detach()
                ).clone(),
                'lateral_inhibition_factor': float(getattr(
                    cortex, 'lateral_inhibition_factor', 0.0
                )),
                'dominance_inhibition_factor': float(getattr(
                    cortex, 'dominance_inhibition_factor', 0.0
                )),
            }
        if settings['audit_root_direct']:
            root = model.get_cortex_by_id(settings['root_cortex_id'])
            direct_input_len = int(root.kernel.input_len)
            snapshot['root_direct'] = {
                'cortex_id': settings['root_cortex_id'],
                'direct_input_len': direct_input_len,
                'weight': root.kernel.weight[
                    :direct_input_len, :
                ].detach().clone(),
                'threshold': torch.exp(
                    root.hidden_nv.log_thresholds.detach()
                ).clone(),
                'amplifier': torch.exp(
                    root.log_amplifier.detach()
                ).clone(),
                'lateral_inhibition_factor': float(getattr(
                    root, 'lateral_inhibition_factor', 0.0
                )),
                'dominance_inhibition_factor': float(getattr(
                    root, 'dominance_inhibition_factor', 0.0
                )),
            }
        if settings['audit_root_branch']:
            root = model.get_cortex_by_id(settings['root_cortex_id'])
            direct_input_len = int(root.kernel.input_len)
            branch_weight = root.kernel.weight[
                direct_input_len:, :
            ].detach().clone()
            snapshot['root_branch'] = {
                'cortex_id': settings['root_cortex_id'],
                'direct_input_len': direct_input_len,
                'weight': branch_weight,
                'previous_weight': branch_weight.clone(),
            }
    return snapshot


def _tensor_linf(value):
    if value.numel() == 0:
        return 0.0
    return float(value.detach().abs().max().item())


def _log_tensor_scale_summary(summary_writer, prefix, value, epoch_index):
    value = value.detach()
    flat = value.reshape(-1)
    if flat.numel() == 0:
        return
    abs_flat = flat.abs()
    summary_writer.add_scalar(
        f'{prefix}/numel', float(flat.numel()), global_step=epoch_index
    )
    if value.ndim >= 1:
        summary_writer.add_scalar(
            f'{prefix}/shape_dim0', float(value.shape[0]),
            global_step=epoch_index,
        )
    if value.ndim >= 2:
        summary_writer.add_scalar(
            f'{prefix}/shape_dim1', float(value.shape[1]),
            global_step=epoch_index,
        )
    for name, scalar in (
        ('mean', flat.mean()),
        ('std', flat.std(unbiased=False)),
        ('abs_mean', abs_flat.mean()),
        ('l1', abs_flat.sum()),
        ('l2', torch.linalg.vector_norm(flat)),
        ('linf', abs_flat.max()),
        ('positive_fraction', (flat > 0).float().mean()),
        ('negative_fraction', (flat < 0).float().mean()),
        ('zero_fraction', (flat == 0).float().mean()),
    ):
        summary_writer.add_scalar(
            f'{prefix}/{name}', float(scalar.item()),
            global_step=epoch_index,
        )


def _log_root_branch_scale_summary(
    summary_writer, prefix, value, epoch_index
):
    _log_tensor_scale_summary(summary_writer, prefix, value, epoch_index)
    if value.ndim != 2 or value.numel() == 0:
        return
    per_output_l1 = value.abs().sum(dim=0)
    per_output_l2 = torch.linalg.vector_norm(value, dim=0)
    for norm_name, values in (
        ('l1', per_output_l1),
        ('l2', per_output_l2),
    ):
        for name, scalar in (
            ('mean', values.mean()),
            ('std', values.std(unbiased=False)),
            ('min', values.min()),
            ('max', values.max()),
        ):
            summary_writer.add_scalar(
                f'{prefix}/per_output_{norm_name}_{name}',
                float(scalar.item()),
                global_step=epoch_index,
            )


def _multiply_kernel_by_amplifier(kernel, amplifier):
    amplifier = amplifier.detach()
    if amplifier.numel() == 1:
        return kernel * amplifier.reshape(())
    if kernel.ndim >= 1 and amplifier.numel() == kernel.shape[-1]:
        shape = [1] * kernel.ndim
        shape[-1] = amplifier.numel()
        return kernel * amplifier.reshape(shape)
    raise ValueError(
        'amplifier must be scalar or match the kernel output dimension: '
        f'kernel_shape={tuple(kernel.shape)}, '
        f'amplifier_shape={tuple(amplifier.shape)}'
    )


def _log_training_invariant_audit(
    summary_writer,
    model,
    epoch_index,
    snapshot,
):
    if snapshot is None:
        return
    with torch.no_grad():
        for cortex_id, initial in snapshot['cortex'].items():
            cortex = model.get_cortex_by_id(cortex_id)
            current_threshold = torch.exp(cortex.hidden_nv.log_thresholds.detach())
            current_amplifier = torch.exp(cortex.log_amplifier.detach())
            current_kernel = cortex.kernel.weight.detach()
            kernel_delta = current_kernel - initial['kernel']
            incremental_kernel_delta = current_kernel - initial['previous_kernel']
            effective_kernel_delta = _multiply_kernel_by_amplifier(
                kernel_delta, current_amplifier
            )
            effective_incremental_kernel_delta = (
                _multiply_kernel_by_amplifier(
                    incremental_kernel_delta, current_amplifier
                )
            )
            _log_root_branch_scale_summary(
                summary_writer,
                f'parameter_scale/cortex_kernel/{cortex_id}/initial',
                initial['kernel'],
                epoch_index,
            )
            _log_root_branch_scale_summary(
                summary_writer,
                f'parameter_scale/cortex_kernel/{cortex_id}/current',
                current_kernel,
                epoch_index,
            )
            _log_root_branch_scale_summary(
                summary_writer,
                f'parameter_scale/cortex_kernel/{cortex_id}/delta',
                kernel_delta,
                epoch_index,
            )
            _log_root_branch_scale_summary(
                summary_writer,
                f'parameter_scale/cortex_kernel/{cortex_id}/incremental_delta',
                incremental_kernel_delta,
                epoch_index,
            )
            _log_root_branch_scale_summary(
                summary_writer,
                f'parameter_scale/effective_kernel_delta/{cortex_id}/delta',
                effective_kernel_delta,
                epoch_index,
            )
            _log_root_branch_scale_summary(
                summary_writer,
                f'parameter_scale/effective_kernel_delta/{cortex_id}/incremental_delta',
                effective_incremental_kernel_delta,
                epoch_index,
            )
            initial_kernel_l2 = float(initial['kernel'].norm().item())
            kernel_relative_l2 = (
                float(kernel_delta.norm().item()) / initial_kernel_l2
                if initial_kernel_l2 > 0.0 else 0.0
            )
            incremental_kernel_relative_l2 = (
                float(incremental_kernel_delta.norm().item()) / initial_kernel_l2
                if initial_kernel_l2 > 0.0 else 0.0
            )
            summary_writer.add_scalar(
                f'training_invariant/kernel_linf_drift/{cortex_id}',
                _tensor_linf(kernel_delta),
                global_step=epoch_index,
            )
            summary_writer.add_scalar(
                f'training_invariant/kernel_relative_l2_drift/{cortex_id}',
                kernel_relative_l2,
                global_step=epoch_index,
            )
            summary_writer.add_scalar(
                f'training_invariant/kernel_incremental_relative_l2_drift/{cortex_id}',
                incremental_kernel_relative_l2,
                global_step=epoch_index,
            )
            summary_writer.add_scalar(
                f'training_invariant/effective_kernel_delta_l2/{cortex_id}',
                float(effective_kernel_delta.norm().item()),
                global_step=epoch_index,
            )
            summary_writer.add_scalar(
                f'training_invariant/effective_kernel_incremental_delta_l2/{cortex_id}',
                float(effective_incremental_kernel_delta.norm().item()),
                global_step=epoch_index,
            )
            summary_writer.add_scalar(
                f'training_invariant/threshold_linf_drift/{cortex_id}',
                _tensor_linf(current_threshold - initial['threshold']),
                global_step=epoch_index,
            )
            summary_writer.add_scalar(
                f'training_invariant/amplifier_abs_drift/{cortex_id}',
                _tensor_linf(current_amplifier - initial['amplifier']),
                global_step=epoch_index,
            )
            current_lateral_inhibition = float(getattr(
                cortex, 'lateral_inhibition_factor', 0.0
            ))
            current_dominance_inhibition = float(getattr(
                cortex, 'dominance_inhibition_factor', 0.0
            ))
            for inhibition_name, initial_value, current_value in (
                (
                    'lateral', initial['lateral_inhibition_factor'],
                    current_lateral_inhibition,
                ),
                (
                    'dominance', initial['dominance_inhibition_factor'],
                    current_dominance_inhibition,
                ),
            ):
                summary_writer.add_scalar(
                    f'parameter_scale/cortex_inhibition/{cortex_id}/'
                    f'{inhibition_name}/initial',
                    initial_value,
                    global_step=epoch_index,
                )
                summary_writer.add_scalar(
                    f'parameter_scale/cortex_inhibition/{cortex_id}/'
                    f'{inhibition_name}/current',
                    current_value,
                    global_step=epoch_index,
                )
                summary_writer.add_scalar(
                    f'training_invariant/{inhibition_name}_inhibition_abs_drift/'
                    f'{cortex_id}',
                    abs(current_value - initial_value),
                    global_step=epoch_index,
                )
            summary_writer.add_scalar(
                f'training_invariant/learning_enabled/{cortex_id}',
                float(bool(cortex.learning_enabled)),
                global_step=epoch_index,
            )
            summary_writer.add_scalar(
                f'training_invariant/homeostasis_enabled/{cortex_id}',
                float(bool(cortex.homeostasis_enabled)),
                global_step=epoch_index,
            )
            summary_writer.add_scalar(
                f'training_invariant/amplifier_homeostasis_enabled/{cortex_id}',
                float(bool(getattr(
                    cortex, 'amplifier_homeostasis_enabled', True
                ))),
                global_step=epoch_index,
            )
            summary_writer.add_scalar(
                f'training_invariant/input_wave_delay_steps/{cortex_id}',
                float(getattr(cortex, 'input_wave_delay_steps', 0)),
                global_step=epoch_index,
            )
            summary_writer.add_scalar(
                f'training_invariant/'
                f'temporal_receiver_phase_count/{cortex_id}',
                float(getattr(cortex, 'temporal_receiver_phase_count', 1)),
                global_step=epoch_index,
            )
            summary_writer.add_scalar(
                f'training_invariant/'
                f'temporal_receiver_active_fraction/{cortex_id}',
                float(getattr(
                    cortex,
                    'temporal_receiver_active_fraction',
                    1.0,
                )),
                global_step=epoch_index,
            )
            temporal_block_count = int(getattr(
                cortex,
                'temporal_basis_update_block_count',
                1,
            ))
            temporal_specific_scale = float(getattr(
                cortex,
                'temporal_basis_specific_update_scale',
                1.0,
            ))
            summary_writer.add_scalar(
                f'temporal_basis_coupling/{cortex_id}/block_count',
                float(temporal_block_count),
                global_step=epoch_index,
            )
            summary_writer.add_scalar(
                f'temporal_basis_coupling/{cortex_id}/specific_update_scale',
                temporal_specific_scale,
                global_step=epoch_index,
            )
            summary_writer.add_scalar(
                f'temporal_basis_coupling/{cortex_id}/shared_initialization',
                float(bool(getattr(
                    cortex,
                    'temporal_basis_shared_initialization',
                    False,
                ))),
                global_step=epoch_index,
            )
            if temporal_block_count > 1:
                direct_input_len = int(cortex.kernel.input_len)
                if direct_input_len % temporal_block_count != 0:
                    raise RuntimeError(
                        'temporal basis coupling block count no longer '
                        'divides direct input rows'
                    )
                block_weights = current_kernel[:direct_input_len].reshape(
                    temporal_block_count,
                    direct_input_len // temporal_block_count,
                    current_kernel.shape[1],
                )
                shared_weight = block_weights.mean(dim=0, keepdim=True)
                shared_l2 = torch.linalg.vector_norm(shared_weight)
                specific_l2 = torch.linalg.vector_norm(
                    block_weights - shared_weight
                )
                relative_specific_l2 = float(
                    specific_l2.item() / max(shared_l2.item(), 1.0e-12)
                )
                summary_writer.add_scalar(
                    f'temporal_basis_coupling/{cortex_id}/'
                    'block_specific_relative_l2',
                    relative_specific_l2,
                    global_step=epoch_index,
                )
            spatial_diffusion_strength = float(getattr(
                cortex,
                'spatial_update_diffusion_strength',
                0.0,
            ))
            spatial_height = int(getattr(
                cortex,
                'spatial_update_diffusion_input_height',
                1,
            ))
            spatial_width = int(getattr(
                cortex,
                'spatial_update_diffusion_input_width',
                1,
            ))
            summary_writer.add_scalar(
                f'spatial_update_diffusion/{cortex_id}/strength',
                spatial_diffusion_strength,
                global_step=epoch_index,
            )
            summary_writer.add_scalar(
                f'spatial_update_diffusion/{cortex_id}/input_height',
                float(spatial_height),
                global_step=epoch_index,
            )
            summary_writer.add_scalar(
                f'spatial_update_diffusion/{cortex_id}/input_width',
                float(spatial_width),
                global_step=epoch_index,
            )
            spatial_area = spatial_height * spatial_width
            direct_input_len = int(cortex.kernel.input_len)
            if spatial_area > 1:
                if direct_input_len % spatial_area != 0:
                    raise RuntimeError(
                        'spatial update diffusion shape no longer divides '
                        'direct input rows'
                    )
                spatial_weights = current_kernel[:direct_input_len].reshape(
                    direct_input_len // spatial_area,
                    spatial_height,
                    spatial_width,
                    current_kernel.shape[1],
                )
                roughness_squared = (
                    torch.square(
                        spatial_weights[:, 1:, :, :]
                        - spatial_weights[:, :-1, :, :]
                    ).sum()
                    + torch.square(
                        spatial_weights[:, :, 1:, :]
                        - spatial_weights[:, :, :-1, :]
                    ).sum()
                )
                relative_roughness = float(
                    torch.sqrt(roughness_squared).item()
                    / max(torch.linalg.vector_norm(spatial_weights).item(), 1.0e-12)
                )
                summary_writer.add_scalar(
                    f'spatial_update_diffusion/{cortex_id}/'
                    'weight_relative_l2_roughness',
                    relative_roughness,
                    global_step=epoch_index,
                )
            spatial_weight_prox_strength = float(getattr(
                cortex,
                'spatial_weight_prox_strength',
                0.0,
            ))
            spatial_weight_prox_height = int(getattr(
                cortex,
                'spatial_weight_prox_input_height',
                1,
            ))
            spatial_weight_prox_width = int(getattr(
                cortex,
                'spatial_weight_prox_input_width',
                1,
            ))
            summary_writer.add_scalar(
                f'spatial_weight_prox/{cortex_id}/strength',
                spatial_weight_prox_strength,
                global_step=epoch_index,
            )
            summary_writer.add_scalar(
                f'spatial_weight_prox/{cortex_id}/input_height',
                float(spatial_weight_prox_height),
                global_step=epoch_index,
            )
            summary_writer.add_scalar(
                f'spatial_weight_prox/{cortex_id}/input_width',
                float(spatial_weight_prox_width),
                global_step=epoch_index,
            )
            summary_writer.add_scalar(
                f'spatial_weight_prox/{cortex_id}/application_count',
                float(getattr(
                    cortex,
                    'spatial_weight_prox_application_count',
                    0,
                )),
                global_step=epoch_index,
            )
            handoff_mode = str(getattr(
                cortex, 'subcortex_handoff_mode', 'cumulative'
            ))
            summary_writer.add_scalar(
                f'training_invariant/subcortex_handoff_scale/{cortex_id}',
                float(getattr(cortex, 'subcortex_handoff_scale', 1.0)),
                global_step=epoch_index,
            )
            for mode_name in ('cumulative', 'onset', 'balanced_trace'):
                summary_writer.add_scalar(
                    f'training_invariant/subcortex_handoff_mode_{mode_name}/'
                    f'{cortex_id}',
                    float(handoff_mode == mode_name),
                    global_step=epoch_index,
                )
            summary_writer.add_scalar(
                f'training_invariant/'
                f'input_wave_handoff_branch_activity_fraction/{cortex_id}',
                float(getattr(
                    cortex,
                    'input_wave_handoff_branch_activity_fraction',
                    0.0,
                ) or 0.0),
                global_step=epoch_index,
            )
            summary_writer.add_scalar(
                f'activity/input_wave_handoff_closed_fraction/{cortex_id}',
                float(getattr(
                    cortex,
                    'input_wave_handoff_closed_fraction',
                    0.0,
                )),
                global_step=epoch_index,
            )
            summary_writer.add_scalar(
                f'training_invariant/kernel_regulation_enabled/{cortex_id}',
                float(bool(getattr(
                    cortex, 'kernel_regulation_enabled', True
                ))),
                global_step=epoch_index,
            )
            propagated_mix = getattr(
                cortex,
                'propagated_label_update_mix',
                None,
            )
            summary_writer.add_scalar(
                f'training_invariant/propagated_label_update_mix/{cortex_id}',
                -1.0 if propagated_mix is None else float(propagated_mix),
                global_step=epoch_index,
            )
            propagated_mode = getattr(
                cortex,
                'propagated_label_update_mode',
                'linear',
            )
            summary_writer.add_scalar(
                f'training_invariant/'
                f'propagated_label_update_mode/{cortex_id}',
                {'linear': 0.0, 'orthogonal': 1.0}[propagated_mode],
                global_step=epoch_index,
            )
            propagated_metrics = {
                'raw_l2_ratio': getattr(
                    cortex,
                    'propagated_label_update_raw_l2_ratio',
                    None,
                ),
                'cosine': getattr(
                    cortex,
                    'propagated_label_update_cosine',
                    None,
                ),
                'normalized_l2_ratio': getattr(
                    cortex,
                    'propagated_label_update_normalized_l2_ratio',
                    None,
                ),
                'mixed_l2_ratio': getattr(
                    cortex,
                    'propagated_label_update_mixed_l2_ratio',
                    None,
                ),
                'orthogonal_l2_ratio': getattr(
                    cortex,
                    'propagated_label_update_orthogonal_l2_ratio',
                    None,
                ),
                'label_free_projection': getattr(
                    cortex,
                    'propagated_label_update_label_free_projection',
                    None,
                ),
                'degenerate_fallback': getattr(
                    cortex,
                    'propagated_label_update_degenerate_fallback',
                    None,
                ),
                'degenerate_fallback_count': getattr(
                    cortex,
                    'propagated_label_update_degenerate_fallback_count',
                    0,
                ),
                'batch_count': getattr(
                    cortex,
                    'propagated_label_update_batch_count',
                    0,
                ),
            }
            for metric_name, metric_value in propagated_metrics.items():
                if metric_value is None:
                    continue
                summary_writer.add_scalar(
                    f'training_invariant/propagated_label_update/'
                    f'{cortex_id}/{metric_name}',
                    float(metric_value),
                    global_step=epoch_index,
                )
            operation_counts = getattr(
                cortex.kernel, 'normalization_operation_counts', {}
            )
            for operation_name in (
                'post', 'pre', 'global_l1_sham', 'none'
            ):
                summary_writer.add_scalar(
                    f'normalization_audit/{cortex_id}/count/{operation_name}',
                    float(operation_counts.get(operation_name, 0)),
                    global_step=epoch_index,
                )
            normalize_settings = cortex.kernel.normalize_settings
            summary_writer.add_scalar(
                f'normalization_audit/{cortex_id}/dim_0_mode',
                {
                    'per_receiver': 0.0,
                    'per_receiver_blockwise': 1.0,
                }[str(normalize_settings.get('dim_0_mode', 'per_receiver'))],
                global_step=epoch_index,
            )
            summary_writer.add_scalar(
                f'normalization_audit/{cortex_id}/dim_1_mode',
                {
                    'per_sender': 0.0,
                    'global_l1_sham': 1.0,
                    'per_sender_blockwise': 2.0,
                }[str(normalize_settings.get('dim_1_mode', 'per_sender'))],
                global_step=epoch_index,
            )
            summary_writer.add_scalar(
                f'normalization_audit/{cortex_id}/global_l1',
                float(current_kernel.abs().sum().item()),
                global_step=epoch_index,
            )
            row_l1 = current_kernel.abs().sum(dim=1)
            col_l1 = current_kernel.abs().sum(dim=0)
            summary_writer.add_scalar(
                f'normalization_audit/{cortex_id}/row_l1_cv',
                float(
                    row_l1.std(unbiased=False).item()
                    / max(row_l1.mean().item(), 1.0e-12)
                ),
                global_step=epoch_index,
            )
            summary_writer.add_scalar(
                f'training_invariant/'
                f'temporal_receiver_event_progress_basis/{cortex_id}',
                float(
                    getattr(
                        cortex,
                        'temporal_receiver_phase_basis',
                        'clock',
                    ) == 'event_progress'
                ),
                global_step=epoch_index,
            )
            summary_writer.add_scalar(
                f'normalization_audit/{cortex_id}/column_l1_cv',
                float(
                    col_l1.std(unbiased=False).item()
                    / max(col_l1.mean().item(), 1.0e-12)
                ),
                global_step=epoch_index,
            )
            sender_block_lengths = tuple(getattr(
                cortex.kernel,
                'sender_block_lengths',
                (int(cortex.kernel.input_len),),
            ))
            block_start = 0
            for block_index, block_length in enumerate(sender_block_lengths):
                block_end = block_start + int(block_length)
                block_weight = current_kernel[block_start:block_end]
                block_row_l1 = block_weight.abs().sum(dim=1)
                block_name = (
                    'direct' if block_index == 0 else f'branch_{block_index}'
                )
                summary_writer.add_scalar(
                    f'normalization_audit/{cortex_id}/source_block/'
                    f'{block_name}/row_l1_mean',
                    float(block_row_l1.mean().item()),
                    global_step=epoch_index,
                )
                summary_writer.add_scalar(
                    f'normalization_audit/{cortex_id}/source_block/'
                    f'{block_name}/total_l1',
                    float(block_weight.abs().sum().item()),
                    global_step=epoch_index,
                )
                block_start = block_end
            sham_target = getattr(
                cortex.kernel, 'last_global_l1_sham_target', None
            )
            if sham_target is not None:
                summary_writer.add_scalar(
                    f'normalization_audit/{cortex_id}/'
                    'last_global_l1_sham_target_before_out_scale',
                    float(sham_target),
                    global_step=epoch_index,
                )
            initial['previous_kernel'].copy_(current_kernel)

        root_direct = snapshot.get('root_direct')
        if root_direct is not None:
            root_id = root_direct['cortex_id']
            root = model.get_cortex_by_id(root_id)
            direct_input_len = root_direct['direct_input_len']
            current_direct = root.kernel.weight[:direct_input_len, :]
            current_threshold = torch.exp(root.hidden_nv.log_thresholds.detach())
            current_amplifier = torch.exp(root.log_amplifier.detach())
            _log_tensor_scale_summary(
                summary_writer,
                f'parameter_scale/root_threshold/{root_id}/initial',
                root_direct['threshold'],
                epoch_index,
            )
            _log_tensor_scale_summary(
                summary_writer,
                f'parameter_scale/root_threshold/{root_id}/current',
                current_threshold,
                epoch_index,
            )
            _log_tensor_scale_summary(
                summary_writer,
                f'parameter_scale/root_threshold/{root_id}/delta',
                current_threshold - root_direct['threshold'],
                epoch_index,
            )
            _log_tensor_scale_summary(
                summary_writer,
                f'parameter_scale/root_amplifier/{root_id}/initial',
                root_direct['amplifier'].reshape(-1),
                epoch_index,
            )
            _log_tensor_scale_summary(
                summary_writer,
                f'parameter_scale/root_amplifier/{root_id}/current',
                current_amplifier.reshape(-1),
                epoch_index,
            )
            summary_writer.add_scalar(
                f'training_invariant/root_direct_abs_linf/{root_id}',
                _tensor_linf(current_direct),
                global_step=epoch_index,
            )
            summary_writer.add_scalar(
                f'training_invariant/root_direct_linf_drift/{root_id}',
                _tensor_linf(current_direct - root_direct['weight']),
                global_step=epoch_index,
            )
            summary_writer.add_scalar(
                f'training_invariant/root_threshold_linf_drift/{root_id}',
                _tensor_linf(current_threshold - root_direct['threshold']),
                global_step=epoch_index,
            )
            summary_writer.add_scalar(
                f'training_invariant/root_amplifier_abs_drift/{root_id}',
                _tensor_linf(current_amplifier - root_direct['amplifier']),
                global_step=epoch_index,
            )
            summary_writer.add_scalar(
                f'training_invariant/root_input_wave_delay_steps/{root_id}',
                float(getattr(root, 'input_wave_delay_steps', 0)),
                global_step=epoch_index,
            )
            summary_writer.add_scalar(
                f'training_invariant/'
                f'root_input_wave_handoff_branch_activity_fraction/{root_id}',
                float(getattr(
                    root,
                    'input_wave_handoff_branch_activity_fraction',
                    0.0,
                ) or 0.0),
                global_step=epoch_index,
            )
            summary_writer.add_scalar(
                f'activity/root_input_wave_handoff_closed_fraction/{root_id}',
                float(getattr(
                    root,
                    'input_wave_handoff_closed_fraction',
                    0.0,
                )),
                global_step=epoch_index,
            )
            current_root_lateral = float(getattr(
                root, 'lateral_inhibition_factor', 0.0
            ))
            current_root_dominance = float(getattr(
                root, 'dominance_inhibition_factor', 0.0
            ))
            for inhibition_name, initial_value, current_value in (
                (
                    'lateral',
                    root_direct['lateral_inhibition_factor'],
                    current_root_lateral,
                ),
                (
                    'dominance',
                    root_direct['dominance_inhibition_factor'],
                    current_root_dominance,
                ),
            ):
                summary_writer.add_scalar(
                    f'parameter_scale/root_inhibition/{root_id}/'
                    f'{inhibition_name}/initial',
                    initial_value,
                    global_step=epoch_index,
                )
                summary_writer.add_scalar(
                    f'parameter_scale/root_inhibition/{root_id}/'
                    f'{inhibition_name}/current',
                    current_value,
                    global_step=epoch_index,
                )
                summary_writer.add_scalar(
                    f'training_invariant/root_{inhibition_name}_inhibition_'
                    f'abs_drift/{root_id}',
                    abs(current_value - initial_value),
                    global_step=epoch_index,
                )
            summary_writer.add_scalar(
                f'training_invariant/root_learning_enabled/{root_id}',
                float(bool(root.learning_enabled)),
                global_step=epoch_index,
            )
            summary_writer.add_scalar(
                f'training_invariant/root_homeostasis_enabled/{root_id}',
                float(bool(root.homeostasis_enabled)),
                global_step=epoch_index,
            )
            summary_writer.add_scalar(
                f'training_invariant/root_amplifier_homeostasis_enabled/{root_id}',
                float(bool(getattr(
                    root, 'amplifier_homeostasis_enabled', True
                ))),
                global_step=epoch_index,
            )
            summary_writer.add_scalar(
                f'training_invariant/root_kernel_regulation_enabled/{root_id}',
                float(bool(getattr(
                    root, 'kernel_regulation_enabled', True
                ))),
                global_step=epoch_index,
            )
            norm_mode = getattr(root, 'kernel_norm_preservation_mode', 'none')
            summary_writer.add_scalar(
                f'training_invariant/root_kernel_norm_preservation_mode/{root_id}',
                {
                    'none': 0.0,
                    'global_l2': 1.0,
                    'per_output_l2': 2.0,
                }[norm_mode],
                global_step=epoch_index,
            )
            preconditioner_mode = getattr(
                root, 'kernel_update_preconditioner_mode', 'none'
            )
            summary_writer.add_scalar(
                f'training_invariant/root_update_preconditioner_mode/{root_id}',
                {
                    'none': 0.0,
                    'feature_row_relative': 1.0,
                    'feature_row_update_rms': 2.0,
                    'feature_row_sender_activity_rms': 3.0,
                    'feature_row_sender_activity_rms_class_contrast_supported': 5.0,
                }[preconditioner_mode],
                global_step=epoch_index,
            )
            preconditioner = getattr(root, 'kernel_update_preconditioner', None)
            if preconditioner is not None:
                summary_writer.add_scalar(
                    f'training_invariant/root_update_preconditioner_min/{root_id}',
                    float(preconditioner[preconditioner > 0].min().item()),
                    global_step=epoch_index,
                )
                summary_writer.add_scalar(
                    f'training_invariant/root_update_preconditioner_max/{root_id}',
                    float(preconditioner.max().item()),
                    global_step=epoch_index,
                )
            observed_min = getattr(
                root,
                'kernel_update_preconditioner_observed_min',
                None,
            )
            observed_max = getattr(
                root,
                'kernel_update_preconditioner_observed_max',
                None,
            )
            l2_ratio = getattr(
                root,
                'kernel_update_preconditioner_l2_ratio',
                None,
            )
            if observed_min is not None:
                summary_writer.add_scalar(
                    f'training_invariant/root_update_preconditioner_observed_min/{root_id}',
                    observed_min,
                    global_step=epoch_index,
                )
                summary_writer.add_scalar(
                    f'training_invariant/root_update_preconditioner_observed_max/{root_id}',
                    observed_max,
                    global_step=epoch_index,
                )
            if l2_ratio is not None:
                summary_writer.add_scalar(
                    f'training_invariant/root_update_preconditioner_l2_ratio/{root_id}',
                    l2_ratio,
                    global_step=epoch_index,
                )
            class_contrast_removed = getattr(
                root,
                'kernel_update_class_contrast_removed_fraction',
                None,
            )
            if class_contrast_removed is not None:
                class_contrast_metrics = {
                    'removed_fraction': class_contrast_removed,
                    'mean_abs_max': getattr(
                        root,
                        'kernel_update_class_contrast_mean_abs_max',
                    ),
                    'centered_mean_abs_max': getattr(
                        root,
                        'kernel_update_class_contrast_centered_mean_abs_max',
                    ),
                    'l2_ratio_before_restore': getattr(
                        root,
                        'kernel_update_class_contrast_l2_ratio_before_restore',
                    ),
                    'l2_ratio_after_restore': getattr(
                        root,
                        'kernel_update_class_contrast_l2_ratio_after_restore',
                    ),
                    'cosine': getattr(
                        root,
                        'kernel_update_class_contrast_cosine',
                    ),
                    'support_fraction': getattr(
                        root,
                        'kernel_update_class_contrast_support_fraction',
                    ),
                    'support_count_mean': getattr(
                        root,
                        'kernel_update_class_contrast_support_count_mean',
                    ),
                    'support_centerable_fraction': getattr(
                        root,
                        'kernel_update_class_contrast_support_centerable_fraction',
                    ),
                    'support_centerable_fraction_max': getattr(
                        root,
                        'kernel_update_class_contrast_support_centerable_fraction_max',
                    ),
                    'support_singleton_fraction': getattr(
                        root,
                        'kernel_update_class_contrast_support_singleton_fraction',
                    ),
                    'support_singleton_l2_fraction': getattr(
                        root,
                        'kernel_update_class_contrast_support_singleton_l2_fraction',
                    ),
                    'support_singleton_l2_fraction_max': getattr(
                        root,
                        'kernel_update_class_contrast_support_singleton_l2_fraction_max',
                    ),
                    'support_full_fraction': getattr(
                        root,
                        'kernel_update_class_contrast_support_full_fraction',
                    ),
                    'support_leak_fraction': getattr(
                        root,
                        'kernel_update_class_contrast_support_leak_fraction',
                    ),
                    'support_leak_abs_max': getattr(
                        root,
                        'kernel_update_class_contrast_support_leak_abs_max',
                    ),
                    'total_global_scale': getattr(
                        root,
                        'kernel_update_class_contrast_total_global_scale',
                    ),
                    'total_degenerate_fallback': getattr(
                        root,
                        'kernel_update_class_contrast_total_degenerate_fallback',
                    ),
                    'total_degenerate_fallback_count': getattr(
                        root,
                        'kernel_update_class_contrast_total_degenerate_fallback_count',
                    ),
                }
                for metric_name, metric_value in class_contrast_metrics.items():
                    summary_writer.add_scalar(
                        f'training_invariant/root_update_class_contrast_'
                        f'{metric_name}/{root_id}',
                        metric_value,
                        global_step=epoch_index,
                    )
                signed_component_metrics = {
                    'signed_reconstruction_error': getattr(
                        root,
                        'kernel_update_class_contrast_signed_reconstruction_error',
                        None,
                    ),
                    'signed_positive_l2_fraction': getattr(
                        root,
                        'kernel_update_class_contrast_signed_positive_l2_fraction',
                        None,
                    ),
                    'signed_negative_l2_fraction': getattr(
                        root,
                        'kernel_update_class_contrast_signed_negative_l2_fraction',
                        None,
                    ),
                }
                for metric_name, metric_value in signed_component_metrics.items():
                    if metric_value is not None:
                        summary_writer.add_scalar(
                            f'training_invariant/root_update_class_contrast_'
                            f'{metric_name}/{root_id}',
                            metric_value,
                            global_step=epoch_index,
                        )
            learning_mask = getattr(root, 'kernel_learning_mask', None)
            if learning_mask is not None:
                summary_writer.add_scalar(
                    f'training_invariant/root_direct_learning_mask_max/{root_id}',
                    _tensor_linf(learning_mask[:direct_input_len, :]),
                    global_step=epoch_index,
                )
                summary_writer.add_scalar(
                    f'training_invariant/root_direct_learning_mask_min/{root_id}',
                    float(
                        learning_mask[:direct_input_len, :]
                        .detach()
                        .min()
                        .item()
                    ),
                    global_step=epoch_index,
                )
                summary_writer.add_scalar(
                    f'training_invariant/root_branch_learning_mask_min/{root_id}',
                    float(
                        learning_mask[direct_input_len:, :]
                        .detach()
                        .min()
                        .item()
                    ),
                    global_step=epoch_index,
                )
                summary_writer.add_scalar(
                    f'training_invariant/root_branch_learning_mask_max/{root_id}',
                    _tensor_linf(learning_mask[direct_input_len:, :]),
                    global_step=epoch_index,
                )

        root_branch = snapshot.get('root_branch')
        if root_branch is not None:
            root_id = root_branch['cortex_id']
            root = model.get_cortex_by_id(root_id)
            direct_input_len = root_branch['direct_input_len']
            current_branch = root.kernel.weight[direct_input_len:, :]
            branch_delta = current_branch - root_branch['weight']
            incremental_delta = (
                current_branch - root_branch['previous_weight']
            )
            _log_root_branch_scale_summary(
                summary_writer,
                f'parameter_scale/root_branch/{root_id}/initial',
                root_branch['weight'],
                epoch_index,
            )
            _log_root_branch_scale_summary(
                summary_writer,
                f'parameter_scale/root_branch/{root_id}/current',
                current_branch,
                epoch_index,
            )
            _log_root_branch_scale_summary(
                summary_writer,
                f'parameter_scale/root_branch/{root_id}/delta',
                branch_delta,
                epoch_index,
            )
            _log_root_branch_scale_summary(
                summary_writer,
                f'parameter_scale/root_branch/{root_id}/incremental_delta',
                incremental_delta,
                epoch_index,
            )
            initial_l2 = float(root_branch['weight'].norm().item())
            relative_l2 = (
                float(branch_delta.norm().item()) / initial_l2
                if initial_l2 > 0.0 else 0.0
            )
            summary_writer.add_scalar(
                f'training_invariant/root_branch_linf_drift/{root_id}',
                _tensor_linf(branch_delta),
                global_step=epoch_index,
            )
            summary_writer.add_scalar(
                f'training_invariant/root_branch_relative_l2_drift/{root_id}',
                relative_l2,
                global_step=epoch_index,
            )
            root_branch['previous_weight'].copy_(current_branch)

            root_direct = snapshot.get('root_direct')
            if root_direct is not None:
                for inner_id in snapshot['cortex']:
                    inner = model.get_cortex_by_id(inner_id)
                    inner_kernel = inner.kernel.weight.detach()
                    inner_l1 = inner_kernel.abs().sum(dim=0).mean()
                    inner_l2 = torch.linalg.vector_norm(
                        inner_kernel, dim=0
                    ).mean()
                    root_l1 = current_branch.abs().sum(dim=0).mean()
                    root_l2 = torch.linalg.vector_norm(
                        current_branch, dim=0
                    ).mean()
                    inner_threshold = torch.exp(
                        inner.hidden_nv.log_thresholds.detach()
                    ).mean()
                    root_threshold = torch.exp(
                        root.hidden_nv.log_thresholds.detach()
                    ).mean()
                    inner_amplifier = torch.exp(
                        inner.log_amplifier.detach()
                    )
                    root_amplifier = torch.exp(root.log_amplifier.detach())
                    prefix = f'layer_scale_balance/{root_id}_over_{inner_id}'
                    ratios = {
                        'kernel_per_output_l1': root_l1 / inner_l1,
                        'kernel_per_output_l2': root_l2 / inner_l2,
                        'amplifier': root_amplifier / inner_amplifier,
                        'threshold_mean': root_threshold / inner_threshold,
                        'l1_drive_proxy': (
                            root_l1 * root_amplifier / root_threshold
                        ) / (
                            inner_l1 * inner_amplifier / inner_threshold
                        ),
                        'l2_drive_proxy': (
                            root_l2 * root_amplifier / root_threshold
                        ) / (
                            inner_l2 * inner_amplifier / inner_threshold
                        ),
                    }
                    for name, ratio in ratios.items():
                        summary_writer.add_scalar(
                            f'{prefix}/{name}',
                            float(ratio.item()),
                            global_step=epoch_index,
                        )
                    root_lateral = float(getattr(
                        root, 'lateral_inhibition_factor', 0.0
                    ))
                    inner_lateral = float(getattr(
                        inner, 'lateral_inhibition_factor', 0.0
                    ))
                    if inner_lateral != 0.0:
                        summary_writer.add_scalar(
                            f'{prefix}/lateral_inhibition',
                            root_lateral / inner_lateral,
                            global_step=epoch_index,
                        )
                        summary_writer.add_scalar(
                            f'{prefix}/lateral_inhibition_over_threshold',
                            (
                                root_lateral / float(root_threshold.item())
                            ) / (
                                inner_lateral / float(inner_threshold.item())
                            ),
                            global_step=epoch_index,
                        )
                    root_dominance = float(getattr(
                        root, 'dominance_inhibition_factor', 0.0
                    ))
                    inner_dominance = float(getattr(
                        inner, 'dominance_inhibition_factor', 0.0
                    ))
                    if inner_dominance != 0.0:
                        summary_writer.add_scalar(
                            f'{prefix}/dominance_inhibition',
                            root_dominance / inner_dominance,
                            global_step=epoch_index,
                        )
                        summary_writer.add_scalar(
                            f'{prefix}/dominance_inhibition_over_threshold',
                            (
                                root_dominance / float(root_threshold.item())
                            ) / (
                                inner_dominance / float(
                                    inner_threshold.item()
                                )
                            ),
                            global_step=epoch_index,
                        )

def _get_training_schedule_settings(training_schedule):
    settings = {
        'enabled': False,
        'stages': [],
    }
    if training_schedule:
        settings.update(training_schedule)
    return settings


def _log_presynaptic_class_preference(model, summary_writer, epoch_index):
    metrics = getattr(
        model, 'presynaptic_class_preference_metrics', None
    )
    if metrics is None:
        return
    metrics = dict(metrics)
    weight_metrics = getattr(
        model, '_presynaptic_class_preference_weight_metrics', None
    )
    if callable(weight_metrics):
        metrics.update(weight_metrics())
    settings = model._get_presynaptic_class_preference_settings()
    target_id = (
        settings.get('target_cortex_id', 'unknown')
        if settings is not None
        else 'unknown'
    )
    for metric_name, value in metrics.items():
        summary_writer.add_scalar(
            f'class_preference_stdp/{target_id}/{metric_name}',
            float(value),
            global_step=epoch_index,
        )
    if settings is not None:
        target = model.get_cortex_by_id(target_id)
        for metric_name, value in (
            (
                'application_count',
                getattr(
                    target,
                    'presynaptic_modulation_application_count',
                    0,
                ),
            ),
            (
                'last_abs_deviation',
                getattr(
                    target,
                    'presynaptic_modulation_last_abs_deviation',
                    0.0,
                ),
            ),
            (
                'sender_application_count',
                getattr(
                    target,
                    'presynaptic_sender_modulation_application_count',
                    0,
                ),
            ),
            (
                'sender_last_abs_deviation',
                getattr(
                    target,
                    'presynaptic_sender_modulation_last_abs_deviation',
                    0.0,
                ),
            ),
            (
                'receiver_application_count',
                getattr(
                    target,
                    'presynaptic_receiver_modulation_application_count',
                    0,
                ),
            ),
            (
                'receiver_last_abs_deviation',
                getattr(
                    target,
                    'presynaptic_receiver_modulation_last_abs_deviation',
                    0.0,
                ),
            ),
            (
                'selective_application_count',
                getattr(
                    target,
                    'presynaptic_selective_modulation_application_count',
                    0,
                ),
            ),
            (
                'selective_last_abs_deviation',
                getattr(
                    target,
                    'presynaptic_selective_modulation_last_abs_deviation',
                    0.0,
                ),
            ),
            (
                'selective_last_gate_fraction',
                getattr(
                    target,
                    'presynaptic_selective_modulation_last_gate_fraction',
                    0.0,
                ),
            ),
            (
                'selective_receiver_application_count',
                getattr(
                    target,
                    'presynaptic_selective_receiver_modulation_application_count',
                    0,
                ),
            ),
            (
                'selective_receiver_last_abs_deviation',
                getattr(
                    target,
                    'presynaptic_selective_receiver_modulation_last_abs_deviation',
                    0.0,
                ),
            ),
            (
                'selective_receiver_last_gate_fraction',
                getattr(
                    target,
                    'presynaptic_selective_receiver_modulation_last_gate_fraction',
                    0.0,
                ),
            ),
            (
                'forward_build_count',
                getattr(
                    target,
                    'class_preference_forward_build_count',
                    0,
                ),
            ),
            (
                'forward_application_count',
                getattr(
                    target,
                    'class_preference_forward_application_count',
                    0,
                ),
            ),
            (
                'forward_last_abs_deviation',
                getattr(
                    target,
                    'class_preference_forward_last_abs_deviation',
                    0.0,
                ),
            ),
            (
                'forward_multiplier_mean',
                getattr(
                    target,
                    'class_preference_forward_multiplier_mean',
                    1.0,
                ),
            ),
            (
                'forward_multiplier_q10',
                getattr(
                    target,
                    'class_preference_forward_multiplier_q10',
                    1.0,
                ),
            ),
            (
                'forward_multiplier_q90',
                getattr(
                    target,
                    'class_preference_forward_multiplier_q90',
                    1.0,
                ),
            ),
            (
                'forward_multiplier_min',
                getattr(
                    target,
                    'class_preference_forward_multiplier_min',
                    1.0,
                ),
            ),
            (
                'forward_multiplier_max',
                getattr(
                    target,
                    'class_preference_forward_multiplier_max',
                    1.0,
                ),
            ),
            (
                'forward_multiplier_abs_deviation',
                getattr(
                    target,
                    'class_preference_forward_multiplier_abs_deviation',
                    0.0,
                ),
            ),
            (
                'forward_time_bin_count',
                getattr(
                    target,
                    'class_preference_forward_time_bin_count',
                    1.0,
                ),
            ),
            (
                'forward_time_permutation_identity',
                getattr(
                    target,
                    'class_preference_forward_time_permutation_identity',
                    1.0,
                ),
            ),
            (
                'forward_active_mass_application_count',
                getattr(
                    target,
                    'class_preference_forward_active_mass_application_count',
                    0,
                ),
            ),
            (
                'forward_active_mass_scale_mean',
                getattr(
                    target,
                    'class_preference_forward_active_mass_scale_mean',
                    1.0,
                ),
            ),
            (
                'forward_active_mass_scale_min',
                getattr(
                    target,
                    'class_preference_forward_active_mass_scale_min',
                    1.0,
                ),
            ),
            (
                'forward_active_mass_scale_max',
                getattr(
                    target,
                    'class_preference_forward_active_mass_scale_max',
                    1.0,
                ),
            ),
            (
                'forward_active_mass_matched_ratio',
                getattr(
                    target,
                    'class_preference_forward_active_mass_matched_ratio',
                    1.0,
                ),
            ),
            (
                'forward_active_mass_valid_fraction',
                getattr(
                    target,
                    'class_preference_forward_active_mass_valid_fraction',
                    0.0,
                ),
            ),
        ):
            summary_writer.add_scalar(
                f'class_preference_stdp/{target_id}/{metric_name}',
                float(value),
                global_step=epoch_index,
            )


def _log_native_hidden_class_aggregation(
    model, summary_writer, epoch_index
):
    metrics = getattr(
        model, 'native_hidden_class_aggregation_metrics', None
    )
    if metrics is None:
        return
    settings = model._get_native_hidden_class_aggregation_settings()
    target_id = (
        settings.get('target_cortex_id', 'unknown')
        if settings is not None
        else 'unknown'
    )
    for metric_name, value in dict(metrics).items():
        summary_writer.add_scalar(
            f'native_hidden_class_aggregation/{target_id}/{metric_name}',
            float(value),
            global_step=epoch_index,
        )
    if settings is None:
        return
    target = model.get_cortex_by_id(target_id)
    for metric_name, value in (
        (
            'build_count',
            getattr(
                target,
                'native_hidden_class_aggregation_build_count',
                0,
            ),
        ),
        (
            'application_count',
            getattr(
                target,
                'native_hidden_class_aggregation_application_count',
                0,
            ),
        ),
    ):
        summary_writer.add_scalar(
            f'native_hidden_class_aggregation/{target_id}/{metric_name}',
            float(value),
            global_step=epoch_index,
        )


def _deep_update_settings(settings, updates):
    merged = copy.deepcopy(settings)
    for key, value in updates.items():
        if (
            isinstance(value, dict)
            and isinstance(merged.get(key), dict)
        ):
            merged[key] = _deep_update_settings(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _get_base_STDP_mechanism(model):
    if not hasattr(model, '_training_schedule_base_STDP_mechanism'):
        model._training_schedule_base_STDP_mechanism = copy.deepcopy(
            model.STDP_mechanism
        )
    return copy.deepcopy(model._training_schedule_base_STDP_mechanism)


def _get_active_training_stage(training_schedule, epoch_index):
    active_stage = {}
    stages = sorted(
        training_schedule['stages'],
        key=lambda stage: int(stage.get('start_epoch', 1))
    )
    for stage in stages:
        if epoch_index < int(stage.get('start_epoch', 1)):
            continue
        active_stage.update(
            {
                key: value
                for key, value in stage.items()
                if key != 'start_epoch'
            }
        )
    return active_stage


def apply_training_schedule(model, summary_writer, epoch_index, training_schedule):
    settings = _get_training_schedule_settings(training_schedule)
    if not settings['enabled']:
        return

    active_stage = _get_active_training_stage(settings, epoch_index)
    model.STDP_mechanism = _get_base_STDP_mechanism(model)
    model.cortex.set_learning_enabled_tree(True)
    model.cortex.set_homeostasis_enabled_tree(True)
    model.cortex.set_amplifier_homeostasis_enabled_tree(True)
    model.cortex.set_kernel_regulation_enabled_tree(True)
    for cortex in model.cortex.iter_cortex_tree():
        cortex.kernel_normalization_only = False
        if not hasattr(cortex, '_schedule_base_normalization_settings'):
            cortex._schedule_base_normalization_settings = copy.deepcopy(
                cortex.kernel.normalize_settings
            )
        cortex.kernel.normalize_settings = copy.deepcopy(
            cortex._schedule_base_normalization_settings
        )
    model.cortex.set_kernel_norm_preservation_mode_tree('none')
    model.cortex.set_kernel_update_preconditioner_mode_tree('none')
    model.cortex.set_STDP_rate_scale_tree(1.0)
    model.cortex.reset_DP_ratio_tree()
    model.cortex.set_propagated_label_update_mix_tree(None)
    model.cortex.set_propagated_label_update_mode_tree('linear')

    allowed_stage_fields = {
        'STDP_mechanism',
        'disable_amplifier_homeostasis_cortex_ids',
        'disable_homeostasis_cortex_ids',
        'frozen_cortex_ids',
        'no_skip_links',
        'skip_kernel_regulation_cortex_ids',
        'normalization_only',
        'kernel_normalization',
        'kernel_norm_preservation',
        'kernel_update_preconditioner',
        'propagated_label_update_mix',
        'propagated_label_update_mode',
        'stdp_rate_scale',
        'stdp_DP_ratio',
        'train_only_cortex_ids',
    }
    unknown_stage_fields = sorted(
        key for key in active_stage
        if key not in allowed_stage_fields
    )
    if unknown_stage_fields:
        unknown = ', '.join(unknown_stage_fields)
        raise ValueError(
            'training_schedule contains unsupported stage field(s): '
            f'{unknown}'
        )

    if 'no_skip_links' in active_stage:
        model.cortex.set_no_skip_links_tree(active_stage['no_skip_links'])

    train_only_cortex_ids = active_stage.get('train_only_cortex_ids')
    if train_only_cortex_ids is not None:
        model.cortex.set_learning_enabled_tree(False)
        model.cortex.set_learning_enabled_tree(True, cortex_ids=train_only_cortex_ids)

    frozen_cortex_ids = active_stage.get('frozen_cortex_ids', [])
    if frozen_cortex_ids:
        model.cortex.set_learning_enabled_tree(False, cortex_ids=frozen_cortex_ids)
        model.cortex.set_homeostasis_enabled_tree(False, cortex_ids=frozen_cortex_ids)
        model.cortex.set_amplifier_homeostasis_enabled_tree(
            False,
            cortex_ids=frozen_cortex_ids,
        )

    disable_homeostasis_cortex_ids = active_stage.get(
        'disable_homeostasis_cortex_ids', []
    )
    if disable_homeostasis_cortex_ids:
        model.cortex.set_homeostasis_enabled_tree(
            False,
            cortex_ids=disable_homeostasis_cortex_ids,
        )

    disable_amplifier_homeostasis_cortex_ids = active_stage.get(
        'disable_amplifier_homeostasis_cortex_ids', []
    )
    if disable_amplifier_homeostasis_cortex_ids:
        model.cortex.set_amplifier_homeostasis_enabled_tree(
            False,
            cortex_ids=disable_amplifier_homeostasis_cortex_ids,
        )

    skip_kernel_regulation_cortex_ids = active_stage.get(
        'skip_kernel_regulation_cortex_ids', []
    )
    if skip_kernel_regulation_cortex_ids:
        model.cortex.set_kernel_regulation_enabled_tree(
            False,
            cortex_ids=skip_kernel_regulation_cortex_ids,
        )

    kernel_norm_preservation = active_stage.get('kernel_norm_preservation')
    normalization_only = active_stage.get('normalization_only', {})
    kernel_normalization = active_stage.get('kernel_normalization', {})
    if not isinstance(kernel_normalization, dict):
        raise ValueError('kernel_normalization must map cortex ids to settings')
    if not isinstance(normalization_only, dict):
        raise ValueError('normalization_only must map cortex ids to normalization settings')
    if set(kernel_normalization) & set(normalization_only):
        raise ValueError('kernel_normalization and normalization_only overlap')
    for cortex_id, normalization in {**kernel_normalization, **normalization_only}.items():
        cortex = model.get_cortex_by_id(cortex_id)
        if cortex is None:
            raise ValueError(f'unknown normalization cortex {cortex_id!r}')
        if cortex_id in skip_kernel_regulation_cortex_ids:
            raise ValueError('normalization_only conflicts with skip_kernel_regulation')
        allowed = {'dim_0_freq', 'dim_1_freq', 'freq_diff', 'norm_p',
                   'out_weight_len', 'use_shape_ratio', 'dim_0_mode',
                   'dim_1_mode'}
        if not isinstance(normalization, dict) or set(normalization) - allowed:
            raise ValueError('normalization_only contains invalid normalization settings')
        cortex.kernel.normalize_settings.update(copy.deepcopy(normalization))
        cortex.kernel_normalization_only = cortex_id in normalization_only
        if epoch_index == 1:
            cortex.kernel.normalize_count_up = 0
        if summary_writer is not None:
            summary_writer.add_scalar(
                f'normalization_audit/{cortex_id}/normalization_only',
                float(cortex.kernel_normalization_only), epoch_index
            )
    if kernel_norm_preservation is not None:
        if not isinstance(kernel_norm_preservation, dict):
            raise ValueError('kernel_norm_preservation must map cortex ids to modes')
        for cortex_id, mode in kernel_norm_preservation.items():
            model.cortex.set_kernel_norm_preservation_mode_tree(
                mode,
                cortex_ids=[cortex_id],
            )

    kernel_update_preconditioner = active_stage.get(
        'kernel_update_preconditioner'
    )
    if kernel_update_preconditioner is not None:
        if not isinstance(kernel_update_preconditioner, dict):
            raise ValueError(
                'kernel_update_preconditioner must map cortex ids to modes'
            )
        for cortex_id, mode in kernel_update_preconditioner.items():
            model.cortex.set_kernel_update_preconditioner_mode_tree(
                mode,
                cortex_ids=[cortex_id],
            )

    if 'stdp_rate_scale' in active_stage:
        stdp_rate_scale = active_stage['stdp_rate_scale']
        if isinstance(stdp_rate_scale, dict):
            for cortex_id, scale in stdp_rate_scale.items():
                model.cortex.set_STDP_rate_scale_tree(
                    scale,
                    cortex_ids=[cortex_id],
                )
        else:
            model.cortex.set_STDP_rate_scale_tree(stdp_rate_scale)

    if 'propagated_label_update_mix' in active_stage:
        mixes = active_stage['propagated_label_update_mix']
        if not isinstance(mixes, dict):
            raise ValueError(
                'propagated_label_update_mix must map cortex ids to values'
            )
        for cortex_id, mix in mixes.items():
            model.cortex.set_propagated_label_update_mix_tree(
                mix,
                cortex_ids=[cortex_id],
            )

    if 'propagated_label_update_mode' in active_stage:
        modes = active_stage['propagated_label_update_mode']
        if not isinstance(modes, dict):
            raise ValueError(
                'propagated_label_update_mode must map cortex ids to modes'
            )
        for cortex_id, mode in modes.items():
            model.cortex.set_propagated_label_update_mode_tree(
                mode,
                cortex_ids=[cortex_id],
            )

    if 'stdp_DP_ratio' in active_stage:
        ratios = active_stage['stdp_DP_ratio']
        if isinstance(ratios, dict):
            for cortex_id, ratio in ratios.items():
                model.cortex.set_DP_ratio_tree(ratio, cortex_ids=[cortex_id])
        else:
            model.cortex.set_DP_ratio_tree(ratios)

    for cortex in model.cortex.iter_cortex_tree():
        for name in ('potentiation_rate', 'DP_ratio', 'depression_rate'):
            summary_writer.add_scalar(
                f'training_schedule/{cortex.cortex_id}/{name}',
                float(getattr(cortex, name)), epoch_index,
            )

    if 'STDP_mechanism' in active_stage:
        model.STDP_mechanism = _deep_update_settings(
            model.STDP_mechanism,
            active_stage['STDP_mechanism']
        )

    summary_writer.add_text(
        'training_schedule/active_stage',
        pprint.pformat(active_stage, indent=2),
        global_step=epoch_index
    )


def train_model(
    model, dataloaders, summary_writer, model_save_path,
    max_epoch, min_epoch, sleep_freq, evaluate_freq, STDP_interval, save_best_model,
    evaluation_level, readout_params=None, training_schedule=None, diagnostic_settings=None,
    evaluation_phases=None, final_evaluation_phases=None, stdp_update_mode='online',
    simulation_forward_mode='streaming', save_final_model=False,
    evaluation_micro_batch_size=None, epoch_offset=0
):
    if int(epoch_offset) != epoch_offset or epoch_offset < 0:
        raise ValueError('epoch_offset must be a nonnegative integer')
    epoch_offset = int(epoch_offset)
    final_epoch = epoch_offset + max_epoch
    readout_params = {} if readout_params is None else dict(readout_params)
    best_valid_accuracy = 0
    best_GA_score = 0
    model.simulation_forward_mode = simulation_forward_mode
    configure_neuron_experience(model, diagnostic_settings, stdp_update_mode, simulation_forward_mode)
    if diagnostic_settings and diagnostic_settings.get('sender_decision_study'):
        from modules.sender_decision_study import log_initial_sender_parameters
        log_initial_sender_parameters(model, summary_writer, epoch_offset)
    _log_post_load_transform_audits(summary_writer, model)
    invariant_audit = _capture_training_invariant_audit(
        model,
        diagnostic_settings,
    )
    _log_training_invariant_audit(
        summary_writer,
        model,
        0,
        invariant_audit,
    )
    if (
        diagnostic_settings
        and diagnostic_settings.get('normalization_mechanism_artifacts', False)
    ):
        log_normalization_mechanism_weight_artifact(
            model,
            summary_writer,
            0,
            diagnostic_settings,
        )
    if max_epoch == 0:
        active_evaluation_phases = (
            final_evaluation_phases
            if final_evaluation_phases is not None
            else evaluation_phases
        )
        valid_score, GA_score = evaluate(
            model, dataloaders, summary_writer, 0, readout_params,
            level=evaluation_level, diagnostic_settings=diagnostic_settings,
            phases=active_evaluation_phases,
            evaluation_micro_batch_size=evaluation_micro_batch_size
        )
        if min_epoch <= 0:
            best_valid_accuracy = valid_score
            best_GA_score = GA_score
            if save_best_model:
                save_model(model, model_save_path)
            write_best_score_to_csv(
                summary_writer,
                0,
                valid_score,
                GA_score
            )
        if save_final_model and model_save_path is not None:
            save_model(model, model_save_path)
            final_model_path = str(model_save_path.with_suffix('.pickle'))
            print(f'Saved final model checkpoint to {final_model_path}', flush=True)
            summary_writer.add_text(
                'checkpoint/final_model_path',
                final_model_path,
                global_step=0
            )
        return float(best_valid_accuracy), float(best_GA_score)

    for epoch_index in range(epoch_offset+1, final_epoch+1):
        apply_training_schedule(
            model,
            summary_writer,
            epoch_index,
            training_schedule
        )
        model.cortex.begin_threshold_epoch_tree(epoch_index)
        _reset_remember_peak_memory(model)
        remember_start_time = time.monotonic()
        remember(
            model,
            dataloaders['train'],
            STDP_interval,
            stdp_update_mode=stdp_update_mode,
            simulation_forward_mode=simulation_forward_mode,
        )
        for cortex in model.cortex.iter_cortex_tree():
            apply_prox = getattr(cortex, 'apply_spatial_weight_prox', None)
            if apply_prox is not None:
                apply_prox()
        _log_presynaptic_class_preference(
            model, summary_writer, epoch_index
        )
        _log_native_hidden_class_aggregation(
            model, summary_writer, epoch_index
        )
        log_neuron_experience(model, summary_writer, epoch_index)
        _log_remember_resources(
            summary_writer,
            model,
            epoch_index,
            time.monotonic() - remember_start_time
        )
        _log_training_invariant_audit(
            summary_writer,
            model,
            epoch_index,
            invariant_audit,
        )

        if epoch_index % evaluate_freq == 0:
            active_evaluation_phases = evaluation_phases
            if epoch_index == final_epoch and final_evaluation_phases is not None:
                active_evaluation_phases = final_evaluation_phases
            valid_score, GA_score = evaluate(
                model, dataloaders, summary_writer, epoch_index, readout_params,
                level=evaluation_level, diagnostic_settings=diagnostic_settings,
                phases=active_evaluation_phases,
                evaluation_micro_batch_size=evaluation_micro_batch_size
            )
            if epoch_index >= min_epoch and best_valid_accuracy < valid_score:
                best_valid_accuracy = valid_score
                best_GA_score = GA_score
                if save_best_model:
                    save_model(model, model_save_path)
                write_best_score_to_csv(summary_writer, epoch_index, valid_score, GA_score)

        if diagnostic_settings and diagnostic_settings.get('sender_decision_study'):
            from modules.sender_decision_study import log_sender_decision_study
            log_sender_decision_study(
                model, dataloaders, summary_writer, epoch_index,
                diagnostic_settings['sender_decision_study'],
            )

        if epoch_index % sleep_freq == 0 and epoch_index != final_epoch:
            sleep(model)
            # remember with STDP_interval=None, only for adjust thresholds before evaluate
            remember(
                model,
                dataloaders['train'],
                STDP_interval=None,
                stdp_update_mode=stdp_update_mode,
                simulation_forward_mode=simulation_forward_mode,
            )
            valid_score, GA_score = evaluate(
                model, dataloaders, summary_writer, epoch_index+1,
                readout_params, level=evaluation_level,
                diagnostic_settings=diagnostic_settings,
                phases=evaluation_phases,
                evaluation_micro_batch_size=evaluation_micro_batch_size
            )
            # given that after sleep the parameter is not at a stable stage,
            # never use the model here.
            # if epoch_index >= min_epoch and best_valid_accuracy < valid_score:
            #     best_valid_accuracy = valid_score
            #     save_model(model, model_save_path)

    if save_final_model and model_save_path is not None:
        save_model(model, model_save_path)
        final_model_path = str(model_save_path.with_suffix('.pickle'))
        print(f'Saved final model checkpoint to {final_model_path}', flush=True)
        summary_writer.add_text(
            'checkpoint/final_model_path',
            final_model_path,
            global_step=final_epoch
        )

    return float(best_valid_accuracy), float(best_GA_score)
