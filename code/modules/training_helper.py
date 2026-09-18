import pprint
import torch
import numpy as np
from collections import defaultdict
from modules.model_inspector import get_cortex_dict, analysis_spike_activity, \
    draw_heat_map, get_nested_shape, draw_bar_chart, \
    get_cortex_top_maps, draw_4D_tensor_in_2D, draw_singular_value_spectrum
from modules.utils import write_dict_to_csv, transpose_dict_of_dict
from modules.linear_probe import readout_test
from pathlib import Path

cached_sample_data = None


def get_model_device(model):
    return getattr(
        model,
        'device',
        torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    )


def get_sample_data(dataloaders=None, model=None, use_cache=True, max_num=3):
    global cached_sample_data
    device = get_model_device(model)

    if use_cache is True and cached_sample_data:
        return cached_sample_data

    sample_data_dict = {}
    for phase in dataloaders:
        for images, labels in dataloaders[phase]:
            images, labels = images.to(device), labels.to(device)
            sample_data_dict[phase] = (images[:max_num], labels[:max_num])
            break

    if use_cache is True:
        cached_sample_data = sample_data_dict
    return sample_data_dict


def _mean_cortex_channels(values):
    """Average spatial positions while preserving feature-map channels."""
    batch_size = values.shape[0]
    neuron_num = values.shape[-1]
    return values.reshape(batch_size, -1, neuron_num).mean(dim=1)


def _cortex_earliness_features(model, cortex_wave_sums):
    return {
        cortex_name: _mean_cortex_channels(
            model._earliness_from_wave_sum(wave_sum)
        )
        for cortex_name, wave_sum in cortex_wave_sums.items()
    }


def _forward_model_and_collect(
    model, images, cortex_dict=None, collect_native_temporal=False,
    collect_root_receiver_activity=False
):
    device = get_model_device(model)
    model.init_state(images.shape, device)

    earliness_scores = None
    first_spike_times = None
    cortex_wave_sums = None if cortex_dict is None else {}
    root_receiver_wave_sum = None
    with torch.no_grad():
        for time_step, spikes in enumerate(model.image_encoder(images)):
            output_wave = model.forward_cortex(spikes, is_training=False)
            if earliness_scores is None:
                earliness_scores = torch.zeros_like(output_wave)
            earliness_scores += output_wave / float(model.image_encoder.simulation_time)

            if collect_native_temporal:
                if first_spike_times is None:
                    first_spike_times = torch.full(
                        output_wave.shape,
                        -1,
                        dtype=torch.long,
                        device=output_wave.device
                    )
                first_now = (output_wave > 0) & (first_spike_times < 0)
                first_spike_times[first_now] = time_step

            if collect_root_receiver_activity:
                receiver_wave = model.cortex.hidden_nv.spike_wave
                if root_receiver_wave_sum is None:
                    root_receiver_wave_sum = torch.zeros_like(receiver_wave)
                root_receiver_wave_sum += receiver_wave

            if cortex_dict is not None:
                for cortex_name, cortex in cortex_dict.items():
                    output_wave = cortex.output_nv.spike_wave.reshape(
                        *cortex.output_nv.spike_wave.shape[:-2],
                        -1
                    )
                    if cortex_name not in cortex_wave_sums:
                        cortex_wave_sums[cortex_name] = torch.zeros_like(
                            output_wave
                        )
                    cortex_wave_sums[cortex_name] += output_wave

        if earliness_scores is None:
            earliness_scores = torch.zeros(
                images.shape[0],
                len(model.cortex.output_nv),
                device=device
            )
        grouped_pred = model.score_output_neuron_scores(earliness_scores)
    pred_classes = torch.argmax(grouped_pred, dim=1)
    has_decision = grouped_pred.max(dim=1).values > 0

    class_first_spike_earliness = None
    if collect_native_temporal:
        if first_spike_times is None:
            first_spike_times = torch.full(
                earliness_scores.shape,
                -1,
                dtype=torch.long,
                device=earliness_scores.device
            )
        has_first_spike = first_spike_times >= 0
        first_spike_earliness = (
            float(model.image_encoder.simulation_time)
            - first_spike_times.clamp_min(0).to(dtype=torch.float32)
        ) / float(model.image_encoder.simulation_time)
        first_spike_earliness = torch.where(
            has_first_spike,
            first_spike_earliness,
            torch.zeros_like(first_spike_earliness)
        )
        class_first_spike_earliness = model.score_output_neuron_scores(
            first_spike_earliness
        )

    readout_feature_dict = {}
    if cortex_wave_sums is not None:
        readout_feature_dict = _cortex_earliness_features(
            model, cortex_wave_sums
        )
    root_receiver_features = None
    if root_receiver_wave_sum is not None:
        root_receiver_features = _mean_cortex_channels(
            model._earliness_from_wave_sum(root_receiver_wave_sum)
        )

    return (
        pred_classes,
        readout_feature_dict,
        has_decision,
        class_first_spike_earliness,
        root_receiver_features,
    )


def _build_native_temporal_sample_metrics(
    class_first_spike_earliness, labels, pred_classes, has_decision,
    num_classes
):
    if class_first_spike_earliness is None:
        raise ValueError('native temporal sample metrics were not collected')
    if class_first_spike_earliness.shape != (len(labels), num_classes):
        raise ValueError(
            'native class first-spike earliness must have shape '
            f'({len(labels)}, {num_classes}); got '
            f'{tuple(class_first_spike_earliness.shape)}'
        )
    class_ids = torch.arange(num_classes, device=labels.device)
    true_class_mask = labels.unsqueeze(1) == class_ids.unsqueeze(0)
    T_correct = class_first_spike_earliness.gather(
        1, labels.unsqueeze(1)
    ).squeeze(1)
    if num_classes > 1:
        T_wrong = class_first_spike_earliness.masked_fill(
            true_class_mask,
            float('-inf')
        ).amax(dim=1)
        T_wrong = torch.where(
            torch.isfinite(T_wrong),
            T_wrong,
            torch.zeros_like(T_wrong)
        )
    else:
        T_wrong = torch.zeros_like(T_correct)
    return {
        'T_correct': T_correct,
        'T_wrong': T_wrong,
        'T_gap': T_correct - T_wrong,
        'predicted_class': pred_classes,
        'has_decision': has_decision,
    }


def _build_decision_confusion_matrix(labels, pred_classes, has_decision, num_classes):
    no_decision_class = num_classes
    labels = labels.detach().cpu().long()
    pred_classes = pred_classes.detach().cpu().long()
    has_decision = has_decision.detach().cpu().bool()

    effective_preds = pred_classes.clone()
    effective_preds[~has_decision] = no_decision_class

    valid_rows = (labels >= 0) & (labels < num_classes)
    valid_cols = (effective_preds >= 0) & (effective_preds <= no_decision_class)
    valid = valid_rows & valid_cols

    matrix = torch.zeros(num_classes, num_classes + 1, dtype=torch.long)
    if valid.any():
        matrix.index_put_(
            (labels[valid], effective_preds[valid]),
            torch.ones(int(valid.sum().item()), dtype=torch.long),
            accumulate=True
        )
    return matrix


def _validate_micro_batch_size(micro_batch_size):
    if micro_batch_size is not None and micro_batch_size <= 0:
        raise ValueError('evaluation_micro_batch_size must be positive')


def _initial_micro_batch_size(labels, micro_batch_size):
    batch_size = len(labels)
    if micro_batch_size is None:
        return batch_size
    return min(micro_batch_size, batch_size)


def collect_phase_metrics(
    model, dataloaders, phases, cortex_dict=None, with_confusion_matrix=True,
    micro_batch_size=None, collect_native_temporal=False,
    collect_root_receiver_activity=False
):
    _validate_micro_batch_size(micro_batch_size)
    device = get_model_device(model)
    accuracy_dict = {}
    confusion_matrices = {} if with_confusion_matrix else None
    cortex_phase_readout_feature_dict = defaultdict(lambda: defaultdict(list))
    all_labels = {}
    all_native_correct = {}
    all_native_temporal = {}
    root_phase_receiver_features = defaultdict(list)

    for phase in phases:
        correct_pred_count = 0
        all_pred_count = 0
        all_batch_labels = []
        all_batch_native_correct = []
        phase_native_temporal = defaultdict(list)
        for images, labels in dataloaders[phase]:
            images, labels = images.to(device), labels.to(device)
            fixed_micro_batch_size = _initial_micro_batch_size(
                labels, micro_batch_size
            )
            start = 0
            while start < len(labels):
                end = min(start + fixed_micro_batch_size, len(labels))
                micro_images = images[start:end]
                micro_labels = labels[start:end]
                (
                    pred_classes,
                    readout_feature_dict,
                    has_decision,
                    class_first_spike_earliness,
                    root_receiver_features,
                ) = _forward_model_and_collect(
                    model,
                    micro_images,
                    cortex_dict=cortex_dict,
                    collect_native_temporal=collect_native_temporal,
                    collect_root_receiver_activity=(
                        collect_root_receiver_activity
                    ),
                )
                correct_pred_count += (
                    (pred_classes == micro_labels) & has_decision
                ).sum().item()
                all_pred_count += len(micro_labels)
                all_batch_labels.append(micro_labels.detach().to('cpu'))
                all_batch_native_correct.append(
                    ((pred_classes == micro_labels) & has_decision).detach().to('cpu')
                )

                if collect_native_temporal:
                    native_temporal = _build_native_temporal_sample_metrics(
                        class_first_spike_earliness,
                        micro_labels,
                        pred_classes,
                        has_decision,
                        model.num_classes
                    )
                    for metric_name, values in native_temporal.items():
                        phase_native_temporal[metric_name].append(
                            values.detach().to('cpu')
                        )

                if with_confusion_matrix:
                    batch_confusion = _build_decision_confusion_matrix(
                        micro_labels,
                        pred_classes,
                        has_decision,
                        model.num_classes
                    )
                    if phase not in confusion_matrices:
                        confusion_matrices[phase] = batch_confusion
                    else:
                        confusion_matrices[phase] += batch_confusion

                for cortex_name, readout_feature in readout_feature_dict.items():
                    cortex_phase_readout_feature_dict[cortex_name][phase].append(
                        readout_feature.detach().to('cpu')
                    )
                if root_receiver_features is not None:
                    root_phase_receiver_features[phase].append(
                        root_receiver_features.detach().to('cpu')
                    )
                start = end

        accuracy_dict[phase] = correct_pred_count / all_pred_count
        if all_batch_labels:
            all_labels[phase] = torch.concat(all_batch_labels, axis=0)
            all_native_correct[phase] = torch.concat(
                all_batch_native_correct, axis=0
            )
            if collect_native_temporal:
                all_native_temporal[phase] = {
                    metric_name: torch.concat(values, axis=0)
                    for metric_name, values in phase_native_temporal.items()
                }

    return (
        accuracy_dict,
        confusion_matrices,
        cortex_phase_readout_feature_dict,
        all_labels,
        all_native_correct,
        all_native_temporal,
        root_phase_receiver_features,
    )


def evaluate_accuracy(model, dataloaders, valid_only=False, with_CM=True):
    if valid_only:
        phases = ['valid']
    else:
        phases = ['train', 'test', 'valid']
    result, confusion_matrix, _, _, _, _, _ = collect_phase_metrics(
        model, dataloaders, phases, cortex_dict=None, with_confusion_matrix=with_CM
    )

    if with_CM:
        return result, confusion_matrix

    return result


def collect_phase_readout_features(model, dataloaders, cortex_dict, phases):
    (
        _, _, cortex_phase_readout_feature_dict, all_labels, _, _, _
    ) = collect_phase_metrics(
        model, dataloaders, phases, cortex_dict=cortex_dict, with_confusion_matrix=False
    )
    return cortex_phase_readout_feature_dict, all_labels


def _concat_phase_features(phase_feature_list):
    return torch.concat(
        [feature.detach().to('cpu') for feature in phase_feature_list],
        axis=0
    )


def get_readout_accuracy(
    cortex_phase_readout_feature_dict, all_labels, readout_params,
    return_predictions=False, return_parameters=False
):
    required_phases = ['train', 'valid', 'test']
    for phase in required_phases:
        if phase not in all_labels:
            raise ValueError(f'Missing phase "{phase}" for readout evaluation')

    cpu_labels = {
        phase: all_labels[phase].detach().to('cpu')
        for phase in required_phases
    }
    cortex_phase_readout_acc = {}
    cortex_phase_readout_predictions = {}
    cortex_readout_parameters = {}
    for cortex_name, phase_feature_dict in cortex_phase_readout_feature_dict.items():
        train_features = _concat_phase_features(phase_feature_dict['train'])
        valid_features = _concat_phase_features(phase_feature_dict['valid'])
        test_features = _concat_phase_features(phase_feature_dict['test'])
        readout_result = readout_test(
            train_features,
            cpu_labels['train'],
            valid_features,
            cpu_labels['valid'],
            test_features,
            cpu_labels['test'],
            return_predictions=return_predictions,
            return_parameters=return_parameters,
            **readout_params
        )
        if return_predictions and return_parameters:
            (
                cortex_phase_readout_acc[cortex_name],
                cortex_phase_readout_predictions[cortex_name],
                cortex_readout_parameters[cortex_name],
            ) = readout_result
        elif return_predictions:
            (
                cortex_phase_readout_acc[cortex_name],
                cortex_phase_readout_predictions[cortex_name],
            ) = readout_result
        elif return_parameters:
            (
                cortex_phase_readout_acc[cortex_name],
                cortex_readout_parameters[cortex_name],
            ) = readout_result
        else:
            cortex_phase_readout_acc[cortex_name] = readout_result

    phase_cortex_readout_acc = transpose_dict_of_dict(cortex_phase_readout_acc)

    if return_predictions and return_parameters:
        return (
            phase_cortex_readout_acc,
            cortex_phase_readout_predictions,
            cortex_readout_parameters,
        )
    if return_predictions:
        return phase_cortex_readout_acc, cortex_phase_readout_predictions
    if return_parameters:
        return phase_cortex_readout_acc, cortex_readout_parameters

    return phase_cortex_readout_acc


def get_readout_fusion_accuracy(
    cortex_phase_readout_feature_dict, all_labels, readout_params,
    cortex_names=None
):
    required_phases = ['train', 'valid', 'test']
    for phase in required_phases:
        if phase not in all_labels:
            raise ValueError(f'Missing phase "{phase}" for readout fusion')

    if cortex_names is None:
        cortex_names = sorted(cortex_phase_readout_feature_dict.keys())
    else:
        cortex_names = list(cortex_names)

    if len(cortex_names) < 2:
        raise ValueError('readout_fusion requires at least two cortexes')

    missing_cortexes = [
        cortex_name for cortex_name in cortex_names
        if cortex_name not in cortex_phase_readout_feature_dict
    ]
    if missing_cortexes:
        raise ValueError(
            f'readout_fusion missing cortex feature(s): {missing_cortexes}'
        )

    phase_features = {}
    for phase in required_phases:
        phase_features[phase] = torch.concat([
            _concat_phase_features(
                cortex_phase_readout_feature_dict[cortex_name][phase]
            )
            for cortex_name in cortex_names
        ], dim=1)

    cpu_labels = {
        phase: all_labels[phase].detach().to('cpu')
        for phase in required_phases
    }
    fusion_name = '+'.join(cortex_names)
    fusion_readout_acc = {
        fusion_name: readout_test(
            phase_features['train'],
            cpu_labels['train'],
            phase_features['valid'],
            cpu_labels['valid'],
            phase_features['test'],
            cpu_labels['test'],
            **readout_params
        )
    }
    return transpose_dict_of_dict(fusion_readout_acc)


def _get_diagnostic_settings(diagnostic_settings):
    settings = {
        'readout': False,
        'readout_error_overlap': False,
        'readout_error_overlap_temporal': False,
        'readout_error_overlap_temporal_phases': None,
        'readout_receiver_alignment': False,
        'probe_route_conflict': False,
        'probe_route_conflict_phases': ('valid', 'test'),
        'temporal_receiver_phase_metrics': False,
        'temporal_receiver_phase_metric_phases': ('valid', 'test'),
        'temporal_receiver_vote_metrics': False,
        'temporal_receiver_vote_metric_phases': ('valid', 'test'),
        'temporal_receiver_progress_metrics': False,
        'temporal_receiver_progress_metric_phases': ('valid', 'test'),
        'subcortex_handoff_metrics': False,
        'subcortex_handoff_metric_phases': ('valid', 'test'),
        'cdna_assignment_readout': False,
        'cdna_assignment_readout_phases': ('valid', 'test'),
        'readout_fusion': False,
        'confusion_matrix': False,
        'cortex_statistics': True,
        'threshold_distribution': False,
        'ablation_chart': False,
        'path_ablation': False,
        'path_ablation_phases': ('valid', 'test'),
        'singular_value_spectrum': False,
        'top_maps_plot': False,
        'sample_activity_images': False,
        'temporal_sample_metrics': True,
        'gpu_usage': False,
        'model_shape': False,
        'kernel_heatmaps': False,
        'hinge_temporal_metrics': False,
        'hinge_temporal_phases': None,
        'per_neuron_activity': False,
        'per_neuron_activity_phases': None,
        'per_neuron_activity_max_batches': None,
        'normalization_mechanism_artifacts': False,
        'normalization_mechanism_trajectory_epochs': (),
        'normalization_mechanism_trajectory_phases': ('valid', 'test'),
        'normalization_mechanism_artifact_micro_batch_size': 20,
        'normalization_mechanism_save_full_weight': False,
        'normalization_mechanism_full_weight_epochs': (2, 8),
    }
    if diagnostic_settings:
        settings.update(diagnostic_settings)
    return settings


def _gini(values):
    values = values.detach().to(dtype=torch.float64).flatten().clamp_min(0)
    total = values.sum()
    if values.numel() == 0 or total <= 0:
        return 0.0
    sorted_values = torch.sort(values).values
    ranks = torch.arange(
        1,
        sorted_values.numel() + 1,
        dtype=sorted_values.dtype,
        device=sorted_values.device,
    )
    n = float(sorted_values.numel())
    return float(
        ((2.0 * ranks - n - 1.0) * sorted_values).sum().item()
        / (n * total.item())
    )


def _top_fraction_mass_share(values, fraction=0.10):
    values = values.detach().to(dtype=torch.float64).flatten().clamp_min(0)
    total = values.sum()
    if values.numel() == 0 or total <= 0:
        return 0.0
    top_count = max(1, int(np.ceil(values.numel() * fraction)))
    return float(torch.topk(values, top_count).values.sum().item() / total.item())


def _root_learnable_branch_weight(model):
    root = model.cortex
    if len(root.subcortexs) == 0:
        if root.kernel.no_skip_links:
            raise ValueError(
                'single-cortex normalization mechanism artifacts require '
                'no_skip_links=false'
            )
        return root, root.kernel.weight[:int(root.kernel.input_len)]
    if len(root.subcortexs) != 1:
        raise ValueError(
            'normalization mechanism artifacts support zero or one root subcortex'
        )
    if not root.kernel.no_skip_links:
        raise ValueError(
            'normalization mechanism artifacts require no_skip_links=true '
            'so the learnable root branch block is unambiguous'
        )
    direct_input_len = int(root.kernel.input_len)
    return root, root.kernel.weight[direct_input_len:]


def log_normalization_mechanism_weight_artifact(
    model, summary_writer, epoch_index, diagnostic_settings
):
    if not diagnostic_settings.get('normalization_mechanism_artifacts', False):
        return

    root, branch_weight = _root_learnable_branch_weight(model)
    weight = branch_weight.detach()
    row_l1 = weight.abs().sum(dim=1)
    column_l1 = weight.abs().sum(dim=0)
    row_mean = row_l1.mean()
    column_mean = column_l1.mean()
    row_cv = row_l1.std(unbiased=False) / row_mean.clamp_min(1.0e-12)
    column_cv = (
        column_l1.std(unbiased=False) / column_mean.clamp_min(1.0e-12)
    )
    metrics = {
        'global_l1': weight.abs().sum().item(),
        'row_l1_cv': row_cv.item(),
        'row_l1_gini': _gini(row_l1),
        'row_l1_top10_mass_share': _top_fraction_mass_share(row_l1),
        'column_l1_cv': column_cv.item(),
        'column_l1_gini': _gini(column_l1),
        'column_l1_top10_mass_share': _top_fraction_mass_share(column_l1),
    }
    for metric_name, value in metrics.items():
        summary_writer.add_scalar(
            f'normalization_mechanism/root_A/{metric_name}',
            float(value),
            global_step=epoch_index,
        )

    artifact_dir = Path(summary_writer.log_dir) / 'mechanism_artifacts'
    artifact_dir.mkdir(parents=True, exist_ok=True)
    arrays = {
        'epoch': np.asarray(epoch_index, dtype=np.int64),
        'root_branch_row_l1': row_l1.detach().cpu().float().numpy(),
        'root_branch_column_l1': column_l1.detach().cpu().float().numpy(),
        'root_branch_row_order_desc': (
            torch.argsort(row_l1, descending=True).detach().cpu().numpy()
        ),
    }
    full_weight_epochs = set(
        int(value) for value in diagnostic_settings.get(
            'normalization_mechanism_full_weight_epochs', (2, 8)
        )
    )
    if (
        diagnostic_settings.get(
            'normalization_mechanism_save_full_weight', False
        )
        and int(epoch_index) in full_weight_epochs
    ):
        arrays['root_branch_weight'] = weight.cpu().float().numpy()
    np.savez_compressed(
        artifact_dir / f'root_weight_epoch{int(epoch_index):03d}.npz',
        **arrays,
    )


def _root_branch_sender_wave(root, expected_sender_count, direct_spikes=None):
    if len(root.subcortexs) == 0:
        if direct_spikes is None:
            raise ValueError(
                'single-cortex normalization artifact requires direct input spikes'
            )
        wave = direct_spikes
    else:
        child = root.subcortexs[0]
        wave = child.output_nv.spike_wave
    batch_size = wave.shape[0]
    wave = wave.reshape(batch_size, -1)
    if wave.shape[1] != expected_sender_count:
        raise ValueError(
            'root branch sender wave does not match learnable branch rows: '
            f'{wave.shape[1]} versus {expected_sender_count}'
        )
    return wave


def _root_class_weight_sums(model, root, branch_weight):
    receiver_classes = model._get_output_label_classes(
        branch_weight.device,
        output_len=branch_weight.shape[1],
    )
    class_weight = torch.zeros(
        branch_weight.shape[0],
        model.num_classes,
        dtype=branch_weight.dtype,
        device=branch_weight.device,
    )
    class_weight.index_add_(1, receiver_classes, branch_weight)
    amplifier = root.amplifier.detach()
    if amplifier.numel() != 1:
        raise ValueError(
            'normalization mechanism artifacts currently require a scalar '
            'root amplifier'
        )
    return class_weight * amplifier.reshape(())


def _collect_normalization_mechanism_phase_artifact(
    model, dataloader, phase, micro_batch_size
):
    device = get_model_device(model)
    root, branch_weight = _root_learnable_branch_weight(model)
    branch_weight = branch_weight.detach()
    sender_count = branch_weight.shape[0]
    time_count = int(model.image_encoder.simulation_time)
    class_count = int(model.num_classes)
    branch_time_binned_offsets = getattr(
        root,
        'branch_time_binned_offsets',
        None,
    )
    if branch_time_binned_offsets is None:
        branch_weights_by_time = [branch_weight] * time_count
    else:
        branch_time_binned_offsets = branch_time_binned_offsets.detach()
        if (
            branch_time_binned_offsets.ndim != 3
            or tuple(branch_time_binned_offsets.shape[1:])
            != tuple(branch_weight.shape)
        ):
            raise ValueError(
                'normalization mechanism artifact found invalid '
                'branch_time_binned_offsets shape'
            )
        bin_count = int(branch_time_binned_offsets.shape[0])
        branch_weights_by_time = [
            branch_weight
            + branch_time_binned_offsets[
                min(bin_count - 1, time_step * bin_count // time_count)
            ]
            for time_step in range(time_count)
        ]
    class_weights_by_time = [
        _root_class_weight_sums(model, root, time_weight)
        for time_weight in branch_weights_by_time
    ]

    sender_activity_sum = torch.zeros(sender_count, dtype=torch.float64)
    class_sender_time_margin_sum = torch.zeros(
        class_count, time_count, sender_count, dtype=torch.float64
    )
    class_sample_count = torch.zeros(class_count, dtype=torch.long)
    all_labels = []
    all_wrong_classes = []
    all_first_spike_times = []
    all_correct_instant = []
    all_wrong_instant = []
    sender_time_true_onset_margin_sum = torch.zeros(
        time_count, sender_count, dtype=torch.float64
    )
    sender_time_wrong_onset_margin_sum = torch.zeros(
        time_count, sender_count, dtype=torch.float64
    )
    true_onset_count = torch.zeros(time_count, dtype=torch.long)
    wrong_onset_count = torch.zeros(time_count, dtype=torch.long)
    # First spikes of each SSIF population, not its persistent spike wave.
    layer_spike_counts = {'input': torch.zeros(time_count, dtype=torch.long)}
    # Per-neuron first-spike incidence in the learned root receiver population.
    root_receiver_first_counts = None
    for cortex in root.iter_cortex_tree():
        for population in ('hidden', 'output'):
            layer_spike_counts[f'{population}_{cortex.cortex_id}'] = torch.zeros(
                time_count, dtype=torch.long
            )
    sample_count = 0

    with torch.no_grad():
        for images, labels in dataloader:
            start = 0
            while start < len(labels):
                end = min(start + micro_batch_size, len(labels))
                micro_images = images[start:end].to(device)
                micro_labels = labels[start:end].to(device)
                batch_size = len(micro_labels)
                model.init_state(micro_images.shape, device)
                sender_steps = []
                first_spike_times = None
                previous_layer_waves = {}
                for time_step, spikes in enumerate(model.image_encoder(micro_images)):
                    output_wave = model.forward_cortex(spikes, is_training=False)
                    layer_spike_counts['input'][time_step] += int((spikes > 0).sum())
                    for cortex in root.iter_cortex_tree():
                        for population in ('hidden', 'output'):
                            key = f'{population}_{cortex.cortex_id}'
                            wave = getattr(cortex, f'{population}_nv').spike_wave
                            previous = previous_layer_waves.get(key)
                            first = wave > 0 if previous is None else (wave > 0) & ~previous
                            layer_spike_counts[key][time_step] += int(first.sum())
                            if cortex is root and population == 'hidden':
                                counts = first.reshape(batch_size, -1).sum(dim=0).cpu().long()
                                if root_receiver_first_counts is None:
                                    root_receiver_first_counts = torch.zeros_like(counts)
                                root_receiver_first_counts += counts
                            previous_layer_waves[key] = (wave > 0).clone()
                    class_wave = model.score_output_neuron_scores(output_wave)
                    if first_spike_times is None:
                        first_spike_times = torch.full(
                            class_wave.shape,
                            -1,
                            dtype=torch.long,
                            device=device,
                        )
                    first_now = (class_wave > 0) & (first_spike_times < 0)
                    first_spike_times[first_now] = time_step
                    sender_steps.append(
                        _root_branch_sender_wave(
                            root,
                            sender_count,
                            direct_spikes=spikes,
                        ).detach().clone()
                    )

                has_spike = first_spike_times >= 0
                class_earliness = (
                    time_count
                    - first_spike_times.clamp_min(0).to(dtype=torch.float32)
                ) / float(time_count)
                class_earliness = torch.where(
                    has_spike,
                    class_earliness,
                    torch.zeros_like(class_earliness),
                )
                true_mask = torch.nn.functional.one_hot(
                    micro_labels,
                    num_classes=class_count,
                ).bool()
                strongest_wrong = class_earliness.masked_fill(
                    true_mask,
                    float('-inf'),
                ).argmax(dim=1)
                sample_indices = torch.arange(batch_size, device=device)
                true_first_spike_time = first_spike_times[
                    sample_indices, micro_labels
                ]
                wrong_first_spike_time = first_spike_times[
                    sample_indices, strongest_wrong
                ]
                correct_steps = []
                wrong_steps = []
                for time_step, sender_wave in enumerate(sender_steps):
                    class_weight = class_weights_by_time[time_step]
                    true_weight = class_weight[:, micro_labels].transpose(0, 1)
                    wrong_weight = class_weight[
                        :, strongest_wrong
                    ].transpose(0, 1)
                    sender_activity_sum += sender_wave.abs().sum(dim=0).cpu().double()
                    correct_contribution = sender_wave * true_weight
                    wrong_contribution = sender_wave * wrong_weight
                    margin_contribution = correct_contribution - wrong_contribution
                    true_onset_mask = true_first_spike_time.eq(time_step)
                    if true_onset_mask.any():
                        sender_time_true_onset_margin_sum[time_step] += (
                            margin_contribution[true_onset_mask]
                            .sum(dim=0)
                            .cpu()
                            .double()
                        )
                        true_onset_count[time_step] += int(true_onset_mask.sum())
                    wrong_onset_mask = wrong_first_spike_time.eq(time_step)
                    if wrong_onset_mask.any():
                        sender_time_wrong_onset_margin_sum[time_step] += (
                            margin_contribution[wrong_onset_mask]
                            .sum(dim=0)
                            .cpu()
                            .double()
                        )
                        wrong_onset_count[time_step] += int(wrong_onset_mask.sum())
                    correct_steps.append(correct_contribution.sum(dim=1).cpu())
                    wrong_steps.append(wrong_contribution.sum(dim=1).cpu())
                    for class_id in range(class_count):
                        class_mask = micro_labels.eq(class_id)
                        if class_mask.any():
                            class_sender_time_margin_sum[class_id, time_step] += (
                                margin_contribution[class_mask]
                                .sum(dim=0)
                                .cpu()
                                .double()
                            )

                cpu_labels = micro_labels.cpu()
                class_sample_count += torch.bincount(
                    cpu_labels,
                    minlength=class_count,
                )
                all_labels.append(cpu_labels)
                all_wrong_classes.append(strongest_wrong.cpu())
                all_first_spike_times.append(first_spike_times.cpu())
                all_correct_instant.append(torch.stack(correct_steps, dim=1))
                all_wrong_instant.append(torch.stack(wrong_steps, dim=1))
                sample_count += batch_size
                start = end

    if sample_count == 0:
        raise ValueError(f'no samples available for mechanism artifact phase {phase}')
    class_denominator = class_sample_count.clamp_min(1).to(torch.float64)
    class_sender_time_margin_mean = (
        class_sender_time_margin_sum
        / class_denominator[:, None, None]
    )
    sender_activity_mean = sender_activity_sum / float(sample_count * time_count)
    row_l1 = torch.stack([
        time_weight.abs().sum(dim=1)
        for time_weight in branch_weights_by_time
    ]).mean(dim=0).cpu().double()
    effective_influence = sender_activity_mean * row_l1
    correct_instant = torch.cat(all_correct_instant, dim=0)
    wrong_instant = torch.cat(all_wrong_instant, dim=0)
    return {
        'phase': np.asarray(phase),
        'sample_count': np.asarray(sample_count, dtype=np.int64),
        'root_receiver_first_spike_count': root_receiver_first_counts.numpy(),
        **{f'layer_first_spike_count_{key}': value.numpy()
           for key, value in layer_spike_counts.items()},
        'labels': torch.cat(all_labels).numpy(),
        'strongest_wrong_class': torch.cat(all_wrong_classes).numpy(),
        'class_first_spike_time': (
            torch.cat(all_first_spike_times).to(torch.int16).numpy()
        ),
        'sender_activity_mean': sender_activity_mean.float().numpy(),
        'effective_sender_influence': effective_influence.float().numpy(),
        'class_sample_count': class_sample_count.numpy(),
        'class_sender_time_margin_mean': (
            class_sender_time_margin_mean.float().numpy()
        ),
        'sender_time_true_onset_margin_mean_per_sample': (
            sender_time_true_onset_margin_sum.float().numpy() / float(sample_count)
        ),
        'sender_time_wrong_onset_margin_mean_per_sample': (
            sender_time_wrong_onset_margin_sum.float().numpy() / float(sample_count)
        ),
        'true_onset_count': true_onset_count.numpy(),
        'wrong_onset_count': wrong_onset_count.numpy(),
        'correct_evidence_instant': correct_instant.float().numpy(),
        'wrong_evidence_instant': wrong_instant.float().numpy(),
        'correct_evidence_cumulative': correct_instant.cumsum(dim=1).float().numpy(),
        'wrong_evidence_cumulative': wrong_instant.cumsum(dim=1).float().numpy(),
        'margin_evidence_cumulative': (
            (correct_instant - wrong_instant).cumsum(dim=1).float().numpy()
        ),
    }


def log_normalization_mechanism_trajectory_artifacts(
    model, dataloaders, summary_writer, epoch_index, diagnostic_settings, phases
):
    if not diagnostic_settings.get('normalization_mechanism_artifacts', False):
        return
    trajectory_epochs = set(
        int(value) for value in diagnostic_settings.get(
            'normalization_mechanism_trajectory_epochs', ()
        )
    )
    if int(epoch_index) not in trajectory_epochs:
        return
    requested_phases = list(diagnostic_settings.get(
        'normalization_mechanism_trajectory_phases', ('valid', 'test')
    ))
    unavailable = sorted(set(requested_phases) - set(phases))
    if unavailable:
        raise ValueError(
            'normalization mechanism artifact phases must be included in '
            f'evaluation phases: {unavailable}'
        )
    micro_batch_size = int(diagnostic_settings.get(
        'normalization_mechanism_artifact_micro_batch_size', 20
    ))
    if micro_batch_size <= 0:
        raise ValueError(
            'normalization_mechanism_artifact_micro_batch_size must be positive'
        )
    artifact_dir = Path(summary_writer.log_dir) / 'mechanism_artifacts'
    artifact_dir.mkdir(parents=True, exist_ok=True)
    for phase in requested_phases:
        arrays = _collect_normalization_mechanism_phase_artifact(
            model,
            dataloaders[phase],
            phase,
            micro_batch_size,
        )
        arrays['epoch'] = np.asarray(epoch_index, dtype=np.int64)
        np.savez_compressed(
            artifact_dir
            / f'temporal_contribution_{phase}_epoch{int(epoch_index):03d}.npz',
            **arrays,
        )


def _normalize_evaluation_phases(phases):
    if phases is None:
        return ['train', 'test', 'valid']
    phases = list(phases)
    if 'valid' not in phases:
        raise ValueError('evaluation_phases must include "valid"')
    allowed_phases = {'train', 'test', 'valid'}
    unknown_phases = [phase for phase in phases if phase not in allowed_phases]
    if unknown_phases:
        raise ValueError(f'Unsupported evaluation phase(s): {unknown_phases}')
    return phases


def ablation_test(model, dataloaders, cortex_dict, valid_score):
    ablation_score_dict = {}
    for cortex_id, cortex in cortex_dict.items():
        backup_cortex_kernel = cortex.kernel.weight.clone()
        scheduled_direct_weights = getattr(
            cortex,
            'direct_time_binned_weights',
            None,
        )
        backup_scheduled_direct_weights = (
            scheduled_direct_weights.clone()
            if scheduled_direct_weights is not None
            else None
        )
        scheduled_branch_offsets = getattr(
            cortex,
            'branch_time_binned_offsets',
            None,
        )
        backup_scheduled_branch_offsets = (
            scheduled_branch_offsets.clone()
            if scheduled_branch_offsets is not None
            else None
        )
        state_direct_weights = getattr(
            cortex,
            'direct_state_residual_weights',
            None,
        )
        backup_state_direct_weights = (
            state_direct_weights.clone()
            if state_direct_weights is not None
            else None
        )
        in_len, out_len = cortex.shape
        cortex.kernel.weight[:in_len, :out_len] = 0
        if scheduled_direct_weights is not None:
            cortex.direct_time_binned_weights.zero_()
        if scheduled_branch_offsets is not None:
            cortex.branch_time_binned_offsets.zero_()
        if state_direct_weights is not None:
            cortex.direct_state_residual_weights.zero_()
        ablation_score_dict[cortex_id] = valid_score - evaluate_accuracy(
            model, dataloaders, valid_only=True, with_CM=False
        )['valid']
        cortex.kernel.weight = backup_cortex_kernel
        if backup_scheduled_direct_weights is not None:
            cortex.direct_time_binned_weights = backup_scheduled_direct_weights
        if backup_scheduled_branch_offsets is not None:
            cortex.branch_time_binned_offsets = backup_scheduled_branch_offsets
        if backup_state_direct_weights is not None:
            cortex.direct_state_residual_weights = backup_state_direct_weights
    return ablation_score_dict


def path_ablation_test(model, dataloaders, phases, micro_batch_size=None):
    """Measure direct-only and branch-only native decisions at the root."""
    root = model.cortex
    if not root.subcortexs:
        raise ValueError('path ablation requires a root cortex with subcortexs')
    direct_len = int(root.kernel.input_len)
    if direct_len <= 0 or direct_len >= root.kernel.weight.shape[0]:
        raise ValueError('path ablation requires non-empty direct and branch blocks')

    backup = root.kernel.weight.clone()
    result = {}
    try:
        for variant in ('direct_only', 'branch_only'):
            root.kernel.weight = backup.clone()
            if variant == 'direct_only':
                root.kernel.weight[direct_len:] = 0
            else:
                root.kernel.weight[:direct_len] = 0
            (
                accuracy,
                _,
                _,
                _,
                _,
                native_temporal,
                _,
            ) = collect_phase_metrics(
                model,
                dataloaders,
                phases,
                cortex_dict=None,
                with_confusion_matrix=False,
                micro_batch_size=micro_batch_size,
                collect_native_temporal=True,
            )
            result[variant] = {}
            for phase in phases:
                metrics = native_temporal[phase]
                T_correct = metrics['T_correct'].to(dtype=torch.float64)
                T_wrong = metrics['T_wrong'].to(dtype=torch.float64)
                T_gap = metrics['T_gap'].to(dtype=torch.float64)
                has_decision = metrics['has_decision'].to(dtype=torch.float64)
                result[variant][phase] = {
                    'accuracy': float(accuracy[phase]),
                    'no_decision': float(1.0 - has_decision.mean().item()),
                    'mean_T_correct': float(T_correct.mean().item()),
                    'mean_T_wrong': float(T_wrong.mean().item()),
                    'T_gap': float(T_gap.mean().item()),
                    'negative_fraction': float((T_gap < 0).to(
                        dtype=torch.float64
                    ).mean().item()),
                }
    finally:
        root.kernel.weight = backup
    return result


def log_path_ablation_metrics(
    model, dataloaders, summary_writer, epoch_index, phases,
    micro_batch_size=None,
):
    result = path_ablation_test(
        model,
        dataloaders,
        phases,
        micro_batch_size=micro_batch_size,
    )
    for variant, phase_metrics in result.items():
        for phase, metrics in phase_metrics.items():
            for metric, value in metrics.items():
                summary_writer.add_scalar(
                    f'path_ablation/{variant}/{metric}/{phase}',
                    value,
                    global_step=epoch_index,
                )


def draw_ablation_chart(ablation_score_dict, valid_score, cortex_dict, show):
    output_nv_len = [
        len(cortex_dict[cortex_name].hidden_nv)
        for cortex_name in ablation_score_dict
    ]
    ablation_chart = draw_bar_chart(
        ablation_score_dict, hlines=[valid_score], bottom=0.1,
        labels=output_nv_len, show=show,
        title='Ablation test'
    )
    return ablation_chart


def draw_readout_chart(readout_score_dict, valid_score, cortex_dict, show):
    output_nv_len = [
        len(cortex_dict[cortex_name].hidden_nv)
        for cortex_name in readout_score_dict
    ]
    readout_chart = draw_bar_chart(
        readout_score_dict, hlines=[valid_score], bottom=0,
        labels=output_nv_len, show=show,
        title='linear probe on earliness'
    )
    return readout_chart


def draw_top_maps_plot(model):
    top_maps = get_cortex_top_maps(model.cortex)
    top_maps_plot = draw_4D_tensor_in_2D(top_maps, title='Top maps', show=False)
    return top_maps_plot


def write_best_score_to_csv(summary_writer, epoch, valid_score, GA_score):
    log_dir = Path(summary_writer.log_dir)
    csv_path = log_dir / 'best_score.csv'
    record = {
        'epoch': epoch,
        'valid_score': valid_score,
        'GA_score': GA_score
    }
    write_dict_to_csv(csv_path, record)


def log_accuracy(summary_writer, accuracy_dict, epoch_index):
    for phase, value in accuracy_dict.items():
        summary_writer.add_scalar(
            f'accuracy/top_spike/{phase}',
            value,
            global_step=epoch_index
        )


def _normalize_diagnostic_phase_subset(requested_phases, evaluation_phases):
    if requested_phases is None:
        return list(evaluation_phases)
    requested_phases = list(requested_phases)
    evaluation_phase_set = set(evaluation_phases)
    unknown_phases = [
        phase for phase in requested_phases
        if phase not in evaluation_phase_set
    ]
    if unknown_phases:
        raise ValueError(
            'diagnostic phase(s) must be included in evaluation_phases: '
            f'{unknown_phases}'
        )
    return requested_phases


def log_hinge_temporal_metrics(
    summary_writer, epoch_index, model, dataloaders, phases
):
    device = get_model_device(model)
    for phase in phases:
        T_correct_sum = 0.0
        T_wrong_sum = 0.0
        count = 0
        per_class_count = torch.zeros(model.num_classes, dtype=torch.float64)
        per_class_T_correct_sum = torch.zeros_like(per_class_count)
        per_class_T_wrong_sum = torch.zeros_like(per_class_count)
        sample_T_gap_values = []
        group_neuron_slot_count = 0
        group_neuron_active_count = None
        group_neuron_earliness_sum = None
        with torch.no_grad():
            for images, labels in dataloaders[phase]:
                images, labels = images.to(device), labels.to(device)
                stats = model.get_hinge_temporal_margin_stats(images, labels)
                T_correct_sum += stats['T_correct_sum']
                T_wrong_sum += stats['T_wrong_sum']
                count += stats['count']
                sample_T_gap_values.append(stats['T_gap_values'])
                if 'per_class_count' in stats:
                    per_class_count += stats['per_class_count']
                    per_class_T_correct_sum += stats['per_class_T_correct_sum']
                    per_class_T_wrong_sum += stats['per_class_T_wrong_sum']
                group_stats = stats.get('output_group_neuron_stats') or {}
                if group_stats:
                    group_neuron_slot_count += group_stats['slot_count']
                    if group_neuron_active_count is None:
                        group_neuron_active_count = group_stats['active_count'].clone()
                        group_neuron_earliness_sum = group_stats['earliness_sum'].clone()
                    else:
                        group_neuron_active_count += group_stats['active_count']
                        group_neuron_earliness_sum += group_stats['earliness_sum']
        if count == 0:
            continue
        mean_T_correct = T_correct_sum / count
        mean_T_wrong = T_wrong_sum / count
        summary_writer.add_scalar(
            f'hinge/mean_T_correct/{phase}',
            mean_T_correct,
            global_step=epoch_index
        )
        summary_writer.add_scalar(
            f'hinge/mean_T_wrong/{phase}',
            mean_T_wrong,
            global_step=epoch_index
        )
        summary_writer.add_scalar(
            f'hinge/T_gap/{phase}',
            mean_T_correct - mean_T_wrong,
            global_step=epoch_index
        )
        sample_T_gap = torch.cat(sample_T_gap_values).to(dtype=torch.float64)
        for quantile in (0.10, 0.50, 0.90):
            summary_writer.add_scalar(
                f'hinge/sample_T_gap/{phase}/q{int(quantile * 100):02d}',
                torch.quantile(sample_T_gap, quantile).item(),
                global_step=epoch_index,
            )
        summary_writer.add_scalar(
            f'hinge/sample_T_gap/{phase}/negative_fraction',
            (sample_T_gap < 0).to(dtype=torch.float64).mean().item(),
            global_step=epoch_index,
        )
        _log_per_class_hinge_temporal_metrics(
            summary_writer,
            epoch_index,
            phase,
            per_class_count,
            per_class_T_correct_sum,
            per_class_T_wrong_sum
        )
        _log_output_group_neuron_temporal_metrics(
            summary_writer,
            epoch_index,
            phase,
            group_neuron_slot_count,
            group_neuron_active_count,
            group_neuron_earliness_sum
        )


def _log_per_class_hinge_temporal_metrics(
    summary_writer,
    epoch_index,
    phase,
    per_class_count,
    per_class_T_correct_sum,
    per_class_T_wrong_sum
):
    for class_id, class_count in enumerate(per_class_count.tolist()):
        summary_writer.add_scalar(
            f'hinge/per_true_class/count/{phase}/class_{class_id}',
            class_count,
            global_step=epoch_index
        )
        if class_count <= 0:
            continue
        mean_T_correct = per_class_T_correct_sum[class_id].item() / class_count
        mean_T_wrong = per_class_T_wrong_sum[class_id].item() / class_count
        summary_writer.add_scalar(
            f'hinge/per_true_class/mean_T_correct/{phase}/class_{class_id}',
            mean_T_correct,
            global_step=epoch_index
        )
        summary_writer.add_scalar(
            f'hinge/per_true_class/mean_T_wrong/{phase}/class_{class_id}',
            mean_T_wrong,
            global_step=epoch_index
        )
        summary_writer.add_scalar(
            f'hinge/per_true_class/T_gap/{phase}/class_{class_id}',
            mean_T_correct - mean_T_wrong,
            global_step=epoch_index
        )


def _log_output_group_neuron_temporal_metrics(
    summary_writer,
    epoch_index,
    phase,
    slot_count,
    active_count,
    earliness_sum
):
    if slot_count <= 0 or active_count is None:
        return
    active_fraction = active_count / float(slot_count)
    mean_earliness = earliness_sum / float(slot_count)
    active_mean_earliness = earliness_sum / active_count.clamp_min(1.0)
    active_mean_earliness = torch.where(
        active_count > 0,
        active_mean_earliness,
        torch.zeros_like(active_mean_earliness)
    )

    title = 'rows: output classes, columns: neurons inside each class group'
    summary_writer.add_image(
        f'output_group_neuron_active_fraction/{phase}',
        draw_heat_map(active_fraction, title=title),
        global_step=epoch_index,
        dataformats='HWC'
    )
    summary_writer.add_image(
        f'output_group_neuron_mean_earliness/{phase}',
        draw_heat_map(mean_earliness, title=title),
        global_step=epoch_index,
        dataformats='HWC'
    )

    for class_id in range(active_fraction.shape[0]):
        _log_distribution_scalars(
            summary_writer,
            f'output_group_neuron_active_fraction/{phase}/class_{class_id}',
            active_fraction[class_id],
            epoch_index
        )
        _log_distribution_scalars(
            summary_writer,
            f'output_group_neuron_mean_earliness/{phase}/class_{class_id}',
            mean_earliness[class_id],
            epoch_index
        )
        _log_distribution_scalars(
            summary_writer,
            f'output_group_neuron_active_mean_earliness/{phase}/class_{class_id}',
            active_mean_earliness[class_id],
            epoch_index
        )


def _collect_hidden_earliness_by_cortex(model, images, cortex_dict):
    device = get_model_device(model)
    images = images.to(device)
    model.init_state(images.shape, device)
    with torch.no_grad():
        for spikes in model.image_encoder(images):
            model.forward_cortex(spikes, is_training=False)

    return {
        cortex_name: cortex.hidden_nv.earliness.detach()
        for cortex_name, cortex in cortex_dict.items()
    }


def _update_per_neuron_activity_state(state, earliness):
    flat_earliness = earliness.reshape(-1, earliness.shape[-1])
    active = flat_earliness > 0
    active_count = active.sum(dim=0).detach().to('cpu', dtype=torch.float64)
    earliness_sum = flat_earliness.sum(dim=0).detach().to(
        'cpu',
        dtype=torch.float64
    )
    if state['active_count'] is None:
        state['active_count'] = active_count
        state['earliness_sum'] = earliness_sum
    else:
        state['active_count'] += active_count
        state['earliness_sum'] += earliness_sum
    state['slot_count'] += int(flat_earliness.shape[0])


def _log_distribution_scalars(summary_writer, tag_prefix, values, epoch_index):
    values = values.detach().to('cpu', dtype=torch.float64).flatten()
    if values.numel() == 0:
        return
    summary_writer.add_scalar(
        f'{tag_prefix}/mean',
        values.mean().item(),
        global_step=epoch_index
    )
    summary_writer.add_scalar(
        f'{tag_prefix}/std',
        values.std(unbiased=False).item(),
        global_step=epoch_index
    )
    quantiles = {
        f'q{percentile:02d}': percentile / 100.0
        for percentile in range(0, 101, 10)
    }
    for name, q in quantiles.items():
        summary_writer.add_scalar(
            f'{tag_prefix}/{name}',
            torch.quantile(values, q).item(),
            global_step=epoch_index
        )
    summary_writer.add_scalar(
        f'{tag_prefix}/min',
        torch.quantile(values, 0.0).item(),
        global_step=epoch_index
    )
    summary_writer.add_scalar(
        f'{tag_prefix}/max',
        torch.quantile(values, 1.0).item(),
        global_step=epoch_index
    )


def _log_per_neuron_activity_summary(
    summary_writer, epoch_index, phase, cortex_name, state
):
    slot_count = state['slot_count']
    if slot_count <= 0 or state['active_count'] is None:
        return

    active_count = state['active_count']
    earliness_sum = state['earliness_sum']
    active_fraction = active_count / float(slot_count)
    mean_earliness = earliness_sum / float(slot_count)
    active_neuron_mask = active_count > 0
    active_mean_earliness = (
        earliness_sum[active_neuron_mask]
        / active_count[active_neuron_mask].clamp_min(1.0)
    )

    base = f'hidden/{phase}/{cortex_name}'
    active_prefix = f'per_neuron_active_fraction/{base}'
    _log_distribution_scalars(
        summary_writer,
        active_prefix,
        active_fraction,
        epoch_index
    )
    for threshold_text, threshold in (
        ('eq_zero', 0.0),
        ('lt_0p001', 0.001),
        ('lt_0p005', 0.005),
        ('lt_0p010', 0.010),
        ('lt_0p050', 0.050),
    ):
        if threshold == 0.0:
            mask = active_fraction == 0
        else:
            mask = active_fraction < threshold
        summary_writer.add_scalar(
            f'{active_prefix}/{threshold_text}',
            mask.to(torch.float64).mean().item(),
            global_step=epoch_index
        )

    _log_distribution_scalars(
        summary_writer,
        f'per_neuron_mean_earliness/{base}',
        mean_earliness,
        epoch_index
    )
    if active_mean_earliness.numel() > 0:
        _log_distribution_scalars(
            summary_writer,
            f'per_neuron_active_mean_earliness/{base}',
            active_mean_earliness,
            epoch_index
        )


def log_per_neuron_activity_metrics(
    summary_writer, epoch_index, model, dataloaders, cortex_dict,
    phases, max_batches=None, micro_batch_size=None
):
    _validate_micro_batch_size(micro_batch_size)
    for phase in phases:
        phase_state = {
            cortex_name: {
                'slot_count': 0,
                'active_count': None,
                'earliness_sum': None,
            }
            for cortex_name in cortex_dict
        }
        with torch.no_grad():
            for batch_index, (images, _) in enumerate(dataloaders[phase]):
                if max_batches is not None and batch_index >= max_batches:
                    break
                fixed_micro_batch_size = _initial_micro_batch_size(
                    images, micro_batch_size
                )
                for start in range(0, len(images), fixed_micro_batch_size):
                    hidden_earliness = _collect_hidden_earliness_by_cortex(
                        model,
                        images[start:start + fixed_micro_batch_size],
                        cortex_dict
                    )
                    for cortex_name, earliness in hidden_earliness.items():
                        _update_per_neuron_activity_state(
                            phase_state[cortex_name],
                            earliness
                        )

        for cortex_name, state in phase_state.items():
            _log_per_neuron_activity_summary(
                summary_writer,
                epoch_index,
                phase,
                cortex_name,
                state
            )


def log_decision_confusion_matrices(summary_writer, epoch_index, confusion_matrices):
    if not confusion_matrices:
        return
    for phase, matrix in confusion_matrices.items():
        summary_writer.add_image(
            f'decision_confusion_matrix/{phase}',
            draw_heat_map(
                matrix,
                title='columns: classes 0-9, last column: no-decision'
            ),
            global_step=epoch_index,
            dataformats='HWC'
        )
        _log_decision_distribution_scalars(
            summary_writer,
            epoch_index,
            phase,
            matrix
        )


def _log_decision_distribution_scalars(
    summary_writer, epoch_index, phase, matrix
):
    matrix = matrix.detach().to('cpu', dtype=torch.float64)
    if matrix.numel() == 0:
        return
    total = matrix.sum().item()
    if total <= 0:
        return

    num_classes = matrix.shape[0]
    pred_counts = matrix.sum(dim=0)
    for class_id in range(num_classes):
        summary_writer.add_scalar(
            f'decision_distribution/predicted_fraction/{phase}/class_{class_id}',
            pred_counts[class_id].item() / total,
            global_step=epoch_index
        )
        summary_writer.add_scalar(
            f'decision_distribution/predicted_count/{phase}/class_{class_id}',
            pred_counts[class_id].item(),
            global_step=epoch_index
        )

    no_decision_count = pred_counts[num_classes].item()
    summary_writer.add_scalar(
        f'decision_distribution/predicted_fraction/{phase}/no_decision',
        no_decision_count / total,
        global_step=epoch_index
    )
    summary_writer.add_scalar(
        f'decision_distribution/predicted_count/{phase}/no_decision',
        no_decision_count,
        global_step=epoch_index
    )

    row_counts = matrix.sum(dim=1)
    for class_id in range(num_classes):
        class_total = row_counts[class_id].item()
        if class_total <= 0:
            continue
        summary_writer.add_scalar(
            f'decision_distribution/per_true_class_accuracy/{phase}/class_{class_id}',
            matrix[class_id, class_id].item() / class_total,
            global_step=epoch_index
        )
        summary_writer.add_scalar(
            f'decision_distribution/per_true_class_no_decision/{phase}/class_{class_id}',
            matrix[class_id, num_classes].item() / class_total,
            global_step=epoch_index
        )


def log_level_one_diagnostics(
    summary_writer, epoch_index, cortex_dict,
    accuracy_dict, phase_cortex_readout_acc, show_fig
):
    for phase in ['train', 'valid', 'test']:
        readout_chart = draw_readout_chart(
            phase_cortex_readout_acc[phase], accuracy_dict[phase], cortex_dict, show=show_fig
        )
        for cortex_name, readout_acc in phase_cortex_readout_acc[phase].items():
            summary_writer.add_scalar(
                f'accuracy/readout/{cortex_name}/{phase}',
                readout_acc,
                global_step=epoch_index
            )
        if 'A-1' in phase_cortex_readout_acc[phase]:
            summary_writer.add_scalar(
                f'accuracy/readout/{phase}',
                phase_cortex_readout_acc[phase]['A-1'],
                global_step=epoch_index
            )
        summary_writer.add_image(
            f'readout_score/{phase}',
            readout_chart, global_step=epoch_index, dataformats='HWC'
        )

    if 'A-1' in phase_cortex_readout_acc['valid']:
        GA_score = phase_cortex_readout_acc['valid']['A-1']
    else:
        GA_score = max(phase_cortex_readout_acc['valid'].values())
    summary_writer.add_scalar('score/ga/valid', GA_score, global_step=epoch_index)

    return GA_score


def log_readout_fusion_accuracy(summary_writer, epoch_index, phase_fusion_readout_acc):
    for phase, fusion_readout_acc in phase_fusion_readout_acc.items():
        for fusion_name, readout_acc in fusion_readout_acc.items():
            summary_writer.add_scalar(
                f'accuracy/readout_fusion/{fusion_name}/{phase}',
                readout_acc,
                global_step=epoch_index
            )


def compute_readout_receiver_alignment(
    readout_weight,
    root_branch_weight,
    num_classes,
    competition_group_N,
):
    """Compare probe class directions with native branch-receiver directions."""
    readout_weight = readout_weight.detach().to('cpu', dtype=torch.float64)
    root_branch_weight = root_branch_weight.detach().to(
        'cpu', dtype=torch.float64
    )
    if readout_weight.ndim != 2:
        raise ValueError('readout weight must have shape [classes, features]')
    if root_branch_weight.ndim != 2:
        raise ValueError('root branch weight must have shape [inputs, receivers]')

    class_count, feature_count = readout_weight.shape
    expected_receivers = int(num_classes) * int(competition_group_N)
    if class_count != int(num_classes):
        raise ValueError(
            'readout class count does not match the native model: '
            f'{class_count} != {num_classes}'
        )
    if root_branch_weight.shape[1] != expected_receivers:
        raise ValueError(
            'root receiver count does not match class grouping: '
            f'{root_branch_weight.shape[1]} != {expected_receivers}'
        )
    if root_branch_weight.shape[0] != feature_count:
        raise ValueError(
            'root branch inputs must match the flattened subcortex readout '
            f'features: {root_branch_weight.shape[0]} != {feature_count}'
        )

    receiver_weight = root_branch_weight.transpose(0, 1)
    receiver_weight = receiver_weight.reshape(
        int(num_classes),
        int(competition_group_N),
        feature_count,
    )
    native_centroid = receiver_weight.mean(dim=1)

    readout_direction = readout_weight - readout_weight.mean(dim=0, keepdim=True)
    native_direction = native_centroid - native_centroid.mean(
        dim=0, keepdim=True
    )
    epsilon = torch.finfo(readout_direction.dtype).eps
    readout_norm = readout_direction.norm(dim=1)
    native_norm = native_direction.norm(dim=1)
    if (readout_norm <= epsilon).any() or (native_norm <= epsilon).any():
        raise ValueError('readout/native class direction has zero norm')

    readout_unit = readout_direction / readout_norm.unsqueeze(1)
    native_unit = native_direction / native_norm.unsqueeze(1)
    cosine_matrix = native_unit @ readout_unit.transpose(0, 1)
    matched_cosine = cosine_matrix.diagonal()
    diagonal_mask = torch.eye(
        int(num_classes), dtype=torch.bool, device=cosine_matrix.device
    )
    max_wrong_cosine = cosine_matrix.masked_fill(
        diagonal_mask, float('-inf')
    ).amax(dim=1)
    class_margin = matched_cosine - max_wrong_cosine

    global_receiver_mean = receiver_weight.reshape(
        expected_receivers, feature_count
    ).mean(dim=0, keepdim=True)
    receiver_direction = receiver_weight - global_receiver_mean.reshape(
        1, 1, feature_count
    )
    receiver_norm = receiver_direction.norm(dim=2).clamp_min(epsilon)
    within_group_cosine = (
        receiver_direction * native_unit.unsqueeze(1)
    ).sum(dim=2) / receiver_norm

    tensors = {
        'matched_cosine': matched_cosine,
        'max_wrong_cosine': max_wrong_cosine,
        'class_margin': class_margin,
        'within_group_cosine': within_group_cosine.reshape(-1),
        'cosine_matrix': cosine_matrix,
    }
    for name, values in tensors.items():
        if not torch.isfinite(values).all():
            raise ValueError(f'non-finite readout receiver alignment: {name}')
    return {
        **tensors,
        'feature_count': int(feature_count),
    }


def log_readout_receiver_alignment(
    summary_writer,
    epoch_index,
    model,
    cortex_dict,
    cortex_readout_parameters,
):
    if len(model.cortex.subcortexs) != 1:
        raise ValueError(
            'readout_receiver_alignment requires exactly one root subcortex'
        )
    child = model.cortex.subcortexs[0]
    child_name = child.cortex_id
    if child_name not in cortex_dict or child_name not in cortex_readout_parameters:
        raise ValueError(
            f'missing readout features or parameters for root child {child_name}'
        )

    root = model.cortex
    direct_input_len = int(root.kernel.input_len)
    branch_weight = root.kernel.weight[direct_input_len:]
    metrics = compute_readout_receiver_alignment(
        cortex_readout_parameters[child_name]['weight'],
        branch_weight,
        model.num_classes,
        model.competition_group_N,
    )
    prefix = f'readout_receiver_alignment/{child_name}_to_{root.cortex_id}'
    for metric_name in ('matched_cosine', 'max_wrong_cosine', 'class_margin'):
        values = metrics[metric_name]
        summary_writer.add_scalar(
            f'{prefix}/{metric_name}/mean',
            values.mean(),
            global_step=epoch_index,
        )
        summary_writer.add_scalar(
            f'{prefix}/{metric_name}/min',
            values.min(),
            global_step=epoch_index,
        )
    summary_writer.add_scalar(
        f'{prefix}/class_margin/positive_fraction',
        (metrics['class_margin'] > 0).to(torch.float64).mean(),
        global_step=epoch_index,
    )
    within_group = metrics['within_group_cosine']
    for statistic, value in (
        ('mean', within_group.mean()),
        ('min', within_group.min()),
        ('q10', torch.quantile(within_group, 0.10)),
        ('q50', torch.quantile(within_group, 0.50)),
        ('q90', torch.quantile(within_group, 0.90)),
    ):
        summary_writer.add_scalar(
            f'{prefix}/within_group_cosine/{statistic}',
            value,
            global_step=epoch_index,
        )
    summary_writer.add_scalar(
        f'{prefix}/feature_count',
        metrics['feature_count'],
        global_step=epoch_index,
    )
    for class_index in range(model.num_classes):
        summary_writer.add_scalar(
            f'{prefix}/per_class/matched_cosine/class_{class_index}',
            metrics['matched_cosine'][class_index],
            global_step=epoch_index,
        )
        summary_writer.add_scalar(
            f'{prefix}/per_class/class_margin/class_{class_index}',
            metrics['class_margin'][class_index],
            global_step=epoch_index,
        )


def _profile_similarity(left_profile, right_profile):
    """Return distribution overlap and cosine for non-negative profiles."""
    left_profile = left_profile.clamp_min(0)
    right_profile = right_profile.clamp_min(0)
    epsilon = torch.finfo(left_profile.dtype).eps
    left_mass = left_profile.sum()
    right_mass = right_profile.sum()
    left_norm = left_profile.norm()
    right_norm = right_profile.norm()
    if (
        left_mass <= epsilon
        or right_mass <= epsilon
        or left_norm <= epsilon
        or right_norm <= epsilon
    ):
        nan = left_profile.new_tensor(float('nan'))
        return nan, nan
    overlap = torch.minimum(
        left_profile / left_mass,
        right_profile / right_mass,
    ).sum()
    cosine = torch.dot(left_profile, right_profile) / (
        left_norm * right_norm
    )
    return overlap, cosine


def compute_probe_route_conflict(
    child_features,
    root_features,
    labels,
    native_predictions,
    native_has_decision,
    native_correct,
    child_probe_predictions,
    root_branch_weight,
    num_classes,
    competition_group_N,
):
    """Compare false-positive and legitimate routes for each native class."""
    child_features = child_features.detach().to('cpu', dtype=torch.float64)
    root_features = root_features.detach().to('cpu', dtype=torch.float64)
    labels = labels.detach().to('cpu', dtype=torch.long)
    native_predictions = native_predictions.detach().to('cpu', dtype=torch.long)
    native_has_decision = native_has_decision.detach().to('cpu').bool()
    native_correct = native_correct.detach().to('cpu').bool()
    child_probe_predictions = child_probe_predictions.detach().to(
        'cpu', dtype=torch.long
    )
    root_branch_weight = root_branch_weight.detach().to(
        'cpu', dtype=torch.float64
    )

    if child_features.ndim != 2 or root_features.ndim != 2:
        raise ValueError('route-conflict features must have shape [samples, features]')
    sample_count = len(labels)
    sample_tensors = (
        child_features,
        root_features,
        native_predictions,
        native_has_decision,
        native_correct,
        child_probe_predictions,
    )
    if any(len(values) != sample_count for values in sample_tensors):
        raise ValueError('route-conflict tensors must share sample order and length')

    class_count = int(num_classes)
    group_size = int(competition_group_N)
    expected_receivers = class_count * group_size
    if root_features.shape[1] != expected_receivers:
        raise ValueError(
            'root route features must match native class receivers: '
            f'{root_features.shape[1]} != {expected_receivers}'
        )
    if root_branch_weight.shape != (
        child_features.shape[1], expected_receivers
    ):
        raise ValueError(
            'root branch weight must map child features to native receivers: '
            f'{tuple(root_branch_weight.shape)} != '
            f'{(child_features.shape[1], expected_receivers)}'
        )

    probe_only = (
        child_probe_predictions.eq(labels)
        & ~native_correct
        & native_has_decision
        & native_predictions.ne(labels)
    )
    per_class_probe_only_count = torch.zeros(class_count, dtype=torch.long)
    per_class_legitimate_count = torch.zeros(class_count, dtype=torch.long)
    metric_names = (
        'receiver_l1_overlap',
        'receiver_cosine',
        'sender_l1_overlap',
        'sender_cosine',
        'sender_relative_l2',
        'sender_wrong_excess_fraction',
        'weighted_sender_wrong_excess_fraction',
    )
    per_class = {
        name: torch.full((class_count,), float('nan'), dtype=torch.float64)
        for name in metric_names
    }
    epsilon = torch.finfo(torch.float64).eps

    for class_index in range(class_count):
        false_positive = probe_only & native_predictions.eq(class_index)
        legitimate = (
            native_correct
            & child_probe_predictions.eq(labels)
            & labels.eq(class_index)
        )
        false_count = int(false_positive.sum().item())
        legitimate_count = int(legitimate.sum().item())
        per_class_probe_only_count[class_index] = false_count
        per_class_legitimate_count[class_index] = legitimate_count
        if false_count == 0 or legitimate_count == 0:
            continue

        receiver_slice = slice(
            class_index * group_size,
            (class_index + 1) * group_size,
        )
        false_receiver = root_features[
            false_positive, receiver_slice
        ].mean(dim=0)
        legitimate_receiver = root_features[
            legitimate, receiver_slice
        ].mean(dim=0)
        (
            per_class['receiver_l1_overlap'][class_index],
            per_class['receiver_cosine'][class_index],
        ) = _profile_similarity(false_receiver, legitimate_receiver)

        false_sender = child_features[false_positive].mean(dim=0).clamp_min(0)
        legitimate_sender = child_features[legitimate].mean(dim=0).clamp_min(0)
        (
            per_class['sender_l1_overlap'][class_index],
            per_class['sender_cosine'][class_index],
        ) = _profile_similarity(false_sender, legitimate_sender)
        sender_norm_sum = false_sender.norm() + legitimate_sender.norm()
        if sender_norm_sum > epsilon:
            per_class['sender_relative_l2'][class_index] = (
                2.0 * (false_sender - legitimate_sender).norm()
                / sender_norm_sum
            )
        wrong_excess = (false_sender - legitimate_sender).clamp_min(0)
        false_sender_mass = false_sender.sum()
        if false_sender_mass > epsilon:
            per_class['sender_wrong_excess_fraction'][class_index] = (
                wrong_excess.sum() / false_sender_mass
            )
        branch_importance = root_branch_weight[
            :, receiver_slice
        ].abs().mean(dim=1)
        weighted_false_mass = torch.dot(branch_importance, false_sender)
        if weighted_false_mass > epsilon:
            per_class[
                'weighted_sender_wrong_excess_fraction'
            ][class_index] = (
                torch.dot(branch_importance, wrong_excess)
                / weighted_false_mass
            )

    usable_class_mask = torch.stack([
        torch.isfinite(values) for values in per_class.values()
    ]).all(dim=0)
    macro = {}
    probe_only_weighted = {}
    false_counts = per_class_probe_only_count.to(torch.float64)
    for metric_name, values in per_class.items():
        finite = torch.isfinite(values)
        if finite.any():
            macro[metric_name] = values[finite].mean()
            weights = false_counts[finite]
            probe_only_weighted[metric_name] = (
                (values[finite] * weights).sum() / weights.sum()
            )
        else:
            macro[metric_name] = values.new_tensor(float('nan'))
            probe_only_weighted[metric_name] = values.new_tensor(float('nan'))

    return {
        'probe_only_count': int(probe_only.sum().item()),
        'per_class_probe_only_count': per_class_probe_only_count,
        'per_class_legitimate_count': per_class_legitimate_count,
        'usable_class_mask': usable_class_mask,
        'per_class': per_class,
        'macro': macro,
        'probe_only_weighted': probe_only_weighted,
    }


def log_probe_route_conflict(
    summary_writer,
    epoch_index,
    model,
    cortex_phase_readout_feature_dict,
    root_phase_receiver_features,
    all_labels,
    all_native_correct,
    all_native_temporal,
    cortex_phase_readout_predictions,
    phases,
):
    if len(model.cortex.subcortexs) != 1:
        raise ValueError(
            'probe_route_conflict requires exactly one root subcortex'
        )
    root = model.cortex
    child = root.subcortexs[0]
    root_name = root.cortex_id
    child_name = child.cortex_id
    if child_name not in cortex_phase_readout_feature_dict:
        raise ValueError(
            f'missing route-conflict features for cortex {child_name}'
        )
    if child_name not in cortex_phase_readout_predictions:
        raise ValueError(
            f'missing route-conflict probe predictions for cortex {child_name}'
        )

    direct_input_len = int(root.kernel.input_len)
    root_branch_weight = root.kernel.weight[direct_input_len:]
    for phase in phases:
        native_temporal = all_native_temporal[phase]
        metrics = compute_probe_route_conflict(
            _concat_phase_features(
                cortex_phase_readout_feature_dict[child_name][phase]
            ),
            _concat_phase_features(root_phase_receiver_features[phase]),
            all_labels[phase],
            native_temporal['predicted_class'],
            native_temporal['has_decision'],
            all_native_correct[phase],
            cortex_phase_readout_predictions[child_name][phase],
            root_branch_weight,
            model.num_classes,
            model.competition_group_N,
        )
        prefix = f'probe_route_conflict/{child_name}_to_{root_name}/{phase}'
        usable_count = int(metrics['usable_class_mask'].sum().item())
        summary_writer.add_scalar(
            f'{prefix}/probe_only_count',
            metrics['probe_only_count'],
            global_step=epoch_index,
        )
        summary_writer.add_scalar(
            f'{prefix}/usable_class_count',
            usable_count,
            global_step=epoch_index,
        )
        summary_writer.add_scalar(
            f'{prefix}/usable_class_fraction',
            usable_count / model.num_classes,
            global_step=epoch_index,
        )
        for aggregate_name in ('macro', 'probe_only_weighted'):
            for metric_name, value in metrics[aggregate_name].items():
                if torch.isfinite(value):
                    summary_writer.add_scalar(
                        f'{prefix}/{aggregate_name}/{metric_name}',
                        value,
                        global_step=epoch_index,
                    )
        for class_index in range(model.num_classes):
            class_prefix = f'{prefix}/per_class/class_{class_index}'
            summary_writer.add_scalar(
                f'{class_prefix}/probe_only_count',
                metrics['per_class_probe_only_count'][class_index],
                global_step=epoch_index,
            )
            summary_writer.add_scalar(
                f'{class_prefix}/legitimate_count',
                metrics['per_class_legitimate_count'][class_index],
                global_step=epoch_index,
            )
            for metric_name, values in metrics['per_class'].items():
                value = values[class_index]
                if torch.isfinite(value):
                    summary_writer.add_scalar(
                        f'{class_prefix}/{metric_name}',
                        value,
                        global_step=epoch_index,
                    )


def log_temporal_receiver_phase_metrics(
    summary_writer,
    epoch_index,
    model,
    root_phase_receiver_features,
    all_labels,
    phases,
):
    root = model.cortex
    phase_count = int(getattr(root, 'temporal_receiver_phase_count', 1))
    group_size = int(model.competition_group_N)
    if group_size % phase_count != 0:
        raise ValueError(
            'temporal receiver phase count must divide competition group size'
        )
    phase_width = group_size // phase_count
    for split in phases:
        features = _concat_phase_features(
            root_phase_receiver_features[split]
        ).to(dtype=torch.float64)
        labels = all_labels[split].to(dtype=torch.long)
        expected_width = model.num_classes * group_size
        if features.shape != (len(labels), expected_width):
            raise ValueError(
                'root receiver features must have shape '
                f'({len(labels)}, {expected_width}); got '
                f'{tuple(features.shape)}'
            )
        grouped = features.reshape(
            len(labels), model.num_classes, group_size
        )
        class_ids = torch.arange(model.num_classes)
        true_mask = labels.unsqueeze(1) == class_ids.unsqueeze(0)
        for phase_index in range(phase_count):
            start = phase_index * phase_width
            end = start + phase_width
            bank = grouped[:, :, start:end]
            active = bank > 0
            per_receiver_active = active.to(torch.float64).mean(dim=0)
            active_values = bank[active]
            class_scores = bank.mean(dim=2)
            correct = class_scores.gather(1, labels.unsqueeze(1)).squeeze(1)
            wrong = class_scores.masked_fill(
                true_mask,
                float('-inf'),
            ).amax(dim=1)
            gap = correct - wrong
            prefix = (
                f'temporal_receiver_phase/{root.cortex_id}/{split}/'
                f'phase_{phase_index}'
            )
            for metric_name, value in (
                ('active_fraction', active.to(torch.float64).mean()),
                ('dead_receiver_fraction', (
                    per_receiver_active == 0
                ).to(torch.float64).mean()),
                ('mean_earliness', bank.mean()),
                ('active_mean_earliness', (
                    active_values.mean()
                    if active_values.numel() > 0
                    else bank.new_tensor(0.0)
                )),
                ('T_correct', correct.mean()),
                ('T_wrong', wrong.mean()),
                ('T_gap', gap.mean()),
                ('negative_gap_fraction', (
                    gap < 0
                ).to(torch.float64).mean()),
            ):
                summary_writer.add_scalar(
                    f'{prefix}/{metric_name}',
                    value,
                    global_step=epoch_index,
                )
            for class_index in range(model.num_classes):
                class_mask = labels == class_index
                if class_mask.any():
                    summary_writer.add_scalar(
                        f'{prefix}/per_true_class_T_gap/class_{class_index}',
                        gap[class_mask].mean(),
                        global_step=epoch_index,
                    )


def _compute_cdna_class_preferences(features, labels, num_classes):
    """Learn per-neuron class preferences from training activity only."""
    features = features.detach().to('cpu', dtype=torch.float64).clamp_min(0)
    labels = labels.detach().to('cpu', dtype=torch.long)
    if features.ndim != 2 or features.shape[0] != len(labels):
        raise ValueError(
            'CDNA assignment features must have shape [samples, neurons]'
        )
    class_means = torch.zeros(
        int(num_classes), features.shape[1], dtype=features.dtype
    )
    for class_index in range(int(num_classes)):
        class_mask = labels == class_index
        if not class_mask.any():
            raise ValueError(
                f'CDNA assignment training split has no class {class_index}'
            )
        class_means[class_index] = features[class_mask].mean(dim=0)

    preference_mass = class_means.sum(dim=0)
    assigned = preference_mass > 0
    normalized = class_means / preference_mass.clamp_min(1.0e-12)
    assignment = normalized.argmax(dim=0)
    assignment = torch.where(
        assigned,
        assignment,
        torch.full_like(assignment, -1),
    )
    return normalized, assignment, assigned


def _score_cdna_assignment_readouts(
    features, normalized_preferences, assignment, num_classes
):
    features = features.detach().to('cpu', dtype=torch.float64).clamp_min(0)
    class_count = int(num_classes)
    class_ids = torch.arange(class_count, dtype=torch.long)
    assignment_mask = assignment.unsqueeze(0) == class_ids.unsqueeze(1)

    winner_neuron = features.argmax(dim=1)
    winner_prediction = assignment[winner_neuron]
    winner_has_decision = (
        (features.amax(dim=1) > 0) & (winner_prediction >= 0)
    )

    capacity = assignment_mask.sum(dim=1).to(dtype=features.dtype)
    population_scores = (
        features @ assignment_mask.to(dtype=features.dtype).T
    ) / capacity.clamp_min(1.0).unsqueeze(0)

    soft_denominator = normalized_preferences.sum(dim=1).clamp_min(1.0)
    soft_scores = (
        features @ normalized_preferences.T
    ) / soft_denominator.unsqueeze(0)

    def decision(scores):
        return scores.argmax(dim=1), scores.amax(dim=1) > 0

    population_prediction, population_has_decision = decision(
        population_scores
    )
    soft_prediction, soft_has_decision = decision(soft_scores)
    return {
        'winner': (winner_prediction, winner_has_decision),
        'population': (population_prediction, population_has_decision),
        'soft_vote': (soft_prediction, soft_has_decision),
    }


def _log_cdna_decision_metrics(
    summary_writer, prefix, epoch_index, predictions, has_decision, labels,
    num_classes
):
    labels = labels.detach().to('cpu', dtype=torch.long)
    predictions = predictions.detach().to('cpu', dtype=torch.long)
    has_decision = has_decision.detach().to('cpu', dtype=torch.bool)
    correct = (predictions == labels) & has_decision
    summary_writer.add_scalar(
        f'{prefix}/accuracy',
        correct.to(torch.float64).mean(),
        global_step=epoch_index,
    )
    summary_writer.add_scalar(
        f'{prefix}/no_decision_fraction',
        (~has_decision).to(torch.float64).mean(),
        global_step=epoch_index,
    )
    for class_index in range(int(num_classes)):
        predicted_fraction = (
            (predictions == class_index) & has_decision
        ).to(torch.float64).mean()
        summary_writer.add_scalar(
            f'{prefix}/predicted_fraction/class_{class_index}',
            predicted_fraction,
            global_step=epoch_index,
        )
        true_class = labels == class_index
        if true_class.any():
            summary_writer.add_scalar(
                f'{prefix}/per_true_class_accuracy/class_{class_index}',
                correct[true_class].to(torch.float64).mean(),
                global_step=epoch_index,
            )


def log_cdna_assignment_readout_metrics(
    summary_writer,
    epoch_index,
    model,
    root_phase_receiver_features,
    all_labels,
    phases,
):
    """Evaluate train-only neuronal class assignment without a learned head."""
    if 'train' not in root_phase_receiver_features or 'train' not in all_labels:
        raise ValueError('CDNA assignment readout requires the training phase')
    train_features = _concat_phase_features(
        root_phase_receiver_features['train']
    )
    normalized, assignment, assigned = _compute_cdna_class_preferences(
        train_features,
        all_labels['train'],
        model.num_classes,
    )
    group_size = int(model.competition_group_N)
    expected_width = int(model.num_classes) * group_size
    if train_features.shape[1] != expected_width:
        raise ValueError(
            'CDNA assignment readout requires root receiver width '
            f'{expected_width}; got {train_features.shape[1]}'
        )

    base_prefix = f'cdna_assignment/{model.cortex.cortex_id}'
    assignment_prefix = f'{base_prefix}/assignment'
    unassigned_fraction = (~assigned).to(torch.float64).mean()
    summary_writer.add_scalar(
        f'{assignment_prefix}/unassigned_fraction',
        unassigned_fraction,
        global_step=epoch_index,
    )
    assigned_count = assigned.sum()
    class_ids = torch.arange(model.num_classes, dtype=torch.long)
    assigned_capacity = (
        assignment.unsqueeze(0) == class_ids.unsqueeze(1)
    ).sum(dim=1).to(torch.float64)
    capacity_distribution = assigned_capacity / assigned_count.clamp_min(1)
    capacity_entropy = -(
        capacity_distribution
        * capacity_distribution.clamp_min(1.0e-12).log()
    ).sum()
    if model.num_classes > 1:
        capacity_entropy = capacity_entropy / np.log(float(model.num_classes))
    summary_writer.add_scalar(
        f'{assignment_prefix}/capacity_entropy_normalized',
        capacity_entropy,
        global_step=epoch_index,
    )
    fixed_assignment = class_ids.repeat_interleave(group_size)
    agreement = (
        (assignment[assigned] == fixed_assignment[assigned])
        .to(torch.float64).mean()
        if assigned.any()
        else train_features.new_tensor(0.0)
    )
    summary_writer.add_scalar(
        f'{assignment_prefix}/fixed_group_agreement',
        agreement,
        global_step=epoch_index,
    )
    sorted_preferences = normalized.sort(dim=0, descending=True).values
    top_margin = sorted_preferences[0] - sorted_preferences[1]
    preference_entropy = -(
        normalized * normalized.clamp_min(1.0e-12).log()
    ).sum(dim=0)
    if model.num_classes > 1:
        preference_entropy = preference_entropy / np.log(float(model.num_classes))
    for metric_name, values in (
        ('preference_top1_margin_mean', top_margin),
        ('preference_entropy_normalized_mean', preference_entropy),
    ):
        value = (
            values[assigned].mean()
            if assigned.any()
            else train_features.new_tensor(0.0)
        )
        summary_writer.add_scalar(
            f'{assignment_prefix}/{metric_name}',
            value,
            global_step=epoch_index,
        )
    for class_index in range(model.num_classes):
        summary_writer.add_scalar(
            f'{assignment_prefix}/assigned_fraction/class_{class_index}',
            assigned_capacity[class_index] / float(expected_width),
            global_step=epoch_index,
        )

    for split in phases:
        features = _concat_phase_features(
            root_phase_receiver_features[split]
        )
        readouts = _score_cdna_assignment_readouts(
            features,
            normalized,
            assignment,
            model.num_classes,
        )
        for rule, (predictions, has_decision) in readouts.items():
            _log_cdna_decision_metrics(
                summary_writer,
                f'{base_prefix}/{split}/{rule}',
                epoch_index,
                predictions,
                has_decision,
                all_labels[split],
                model.num_classes,
            )


def log_temporal_receiver_progress_metrics(
    summary_writer,
    epoch_index,
    model,
    dataloaders,
    phases,
):
    """Log how clock or event-progress routing partitions each sample."""
    root = model.cortex
    phase_count = int(getattr(root, 'temporal_receiver_phase_count', 1))
    phase_basis = str(getattr(
        root,
        'temporal_receiver_phase_basis',
        'clock',
    ))
    encoder = model.image_encoder
    temporal_bins = int(getattr(encoder, 'temporal_bins', 0))
    channels_per_bin = int(getattr(encoder, 'channels_per_bin', 0))
    if temporal_bins <= 0 or channels_per_bin <= 0:
        raise ValueError(
            'temporal receiver progress metrics require an event encoder'
        )

    for split in phases:
        assignment_batches = []
        zero_event_batches = []
        for images, _ in dataloaders[split]:
            expected_channels = temporal_bins * channels_per_bin
            if images.ndim != 4 or images.shape[1] != expected_channels:
                raise ValueError(
                    'temporal receiver progress metrics expected event frames '
                    f'with {expected_channels} channels; got '
                    f'{tuple(images.shape)}'
                )
            event_bins = images.reshape(
                images.shape[0],
                temporal_bins,
                channels_per_bin,
                *images.shape[2:],
            )
            event_mass = (event_bins > 0).to(torch.float64).flatten(2).sum(dim=2)
            total_event_mass = event_mass.sum(dim=1, keepdim=True)
            zero_event = total_event_mass.squeeze(1) == 0
            if phase_basis == 'clock':
                assignment = (
                    torch.arange(temporal_bins, device=images.device)
                    * phase_count
                    // temporal_bins
                ).unsqueeze(0).expand(len(images), -1)
            elif phase_basis == 'event_progress':
                progress = (
                    event_mass.cumsum(dim=1)
                    / total_event_mass.clamp_min(1.0)
                )
                assignment = torch.clamp(
                    (progress * phase_count).to(torch.long),
                    min=0,
                    max=phase_count - 1,
                )
            else:
                raise ValueError(
                    'temporal receiver phase basis must be clock or '
                    'event_progress'
                )
            assignment_batches.append(assignment.cpu())
            zero_event_batches.append(zero_event.cpu())

        assignments = torch.cat(assignment_batches, dim=0)
        zero_event = torch.cat(zero_event_batches, dim=0)
        prefix = f'temporal_receiver_progress/{root.cortex_id}/{split}'
        summary_writer.add_scalar(
            f'{prefix}/zero_event_fraction',
            zero_event.to(torch.float64).mean(),
            global_step=epoch_index,
        )
        distinct_phase_count = torch.tensor([
            torch.unique(row).numel()
            for row in assignments
        ], dtype=torch.float64)
        summary_writer.add_scalar(
            f'{prefix}/distinct_phase_count_mean',
            distinct_phase_count.mean(),
            global_step=epoch_index,
        )
        for phase_index in range(phase_count):
            summary_writer.add_scalar(
                f'{prefix}/phase_{phase_index}/step_fraction_mean',
                (assignments == phase_index).to(torch.float64).mean(),
                global_step=epoch_index,
            )
        for boundary_index in range(1, phase_count):
            reached = assignments >= boundary_index
            has_crossing = reached.any(dim=1)
            crossing_step = reached.to(torch.long).argmax(dim=1)
            crossing_step = crossing_step[has_crossing].to(torch.float64)
            boundary_prefix = f'{prefix}/boundary_{boundary_index}'
            summary_writer.add_scalar(
                f'{boundary_prefix}/unreached_fraction',
                (~has_crossing).to(torch.float64).mean(),
                global_step=epoch_index,
            )
            if crossing_step.numel() == 0:
                continue
            denominator = max(temporal_bins - 1, 1)
            for metric_name, value in (
                ('crossing_step_mean', crossing_step.mean()),
                ('crossing_step_q10', torch.quantile(crossing_step, 0.10)),
                ('crossing_step_q50', torch.quantile(crossing_step, 0.50)),
                ('crossing_step_q90', torch.quantile(crossing_step, 0.90)),
                ('crossing_fraction_mean', crossing_step.mean() / denominator),
            ):
                summary_writer.add_scalar(
                    f'{boundary_prefix}/{metric_name}',
                    value,
                    global_step=epoch_index,
                )


def log_subcortex_handoff_metrics(
    summary_writer,
    epoch_index,
    model,
    dataloaders,
    phases,
    micro_batch_size=None,
):
    """Measure when A1 emits new features and how much reaches root A."""
    root = model.cortex
    if len(root.subcortexs) != 1:
        raise ValueError(
            'subcortex handoff metrics require exactly one root subcortex'
        )
    time_count = int(model.image_encoder.simulation_time)
    if micro_batch_size is None:
        micro_batch_size = 2 ** 31 - 1
    micro_batch_size = int(micro_batch_size)
    if micro_batch_size <= 0:
        raise ValueError('evaluation micro batch size must be positive')
    device = get_model_device(model)

    for split in phases:
        onset_step_sums = torch.zeros(time_count, dtype=torch.float64)
        handoff_step_sums = torch.zeros(time_count, dtype=torch.float64)
        integrated_samples = []
        sample_count = 0
        with torch.no_grad():
            for images, _ in dataloaders[split]:
                for start in range(0, len(images), micro_batch_size):
                    micro_images = images[
                        start:start + micro_batch_size
                    ].to(device)
                    model.init_state(micro_images.shape, device)
                    integrated = torch.zeros(
                        len(micro_images), dtype=torch.float64, device=device
                    )
                    for time_step, spikes in enumerate(
                        model.image_encoder(micro_images)
                    ):
                        model.forward_cortex(spikes, is_training=False)
                        a1 = root.subcortexs[0]
                        onset = a1.output_nv.current_spikes.reshape(
                            len(micro_images), -1
                        )
                        handoff = root.latest_subcortex_handoff_waves[0].reshape(
                            len(micro_images), -1
                        )
                        onset_per_sample = (onset > 0).to(
                            torch.float64
                        ).mean(dim=1)
                        handoff_per_sample = handoff.to(
                            torch.float64
                        ).mean(dim=1)
                        onset_step_sums[time_step] += onset_per_sample.sum().cpu()
                        handoff_step_sums[time_step] += handoff_per_sample.sum().cpu()
                        integrated += handoff_per_sample
                    integrated_samples.append(integrated.cpu())
                    sample_count += len(micro_images)

        if sample_count == 0:
            continue
        integrated = torch.cat(integrated_samples)
        onset_step_mean = onset_step_sums / float(sample_count)
        handoff_step_mean = handoff_step_sums / float(sample_count)
        prefix = f'subcortex_handoff/{root.cortex_id}/{split}'
        cumulative_handoff = torch.cumsum(handoff_step_mean, dim=0)
        for time_step in range(time_count):
            summary_writer.add_scalar(
                f'{prefix}/step_{time_step}/A1_new_onset_fraction',
                onset_step_mean[time_step],
                global_step=epoch_index,
            )
            summary_writer.add_scalar(
                f'{prefix}/step_{time_step}/handoff_mean',
                handoff_step_mean[time_step],
                global_step=epoch_index,
            )
            summary_writer.add_scalar(
                f'{prefix}/step_{time_step}/handoff_cumulative_mean',
                cumulative_handoff[time_step],
                global_step=epoch_index,
            )
        for name, value in (
            ('mean', integrated.mean()),
            ('q10', torch.quantile(integrated, 0.10)),
            ('q50', torch.quantile(integrated, 0.50)),
            ('q90', torch.quantile(integrated, 0.90)),
        ):
            summary_writer.add_scalar(
                f'{prefix}/integrated_handoff_mass_{name}',
                value,
                global_step=epoch_index,
            )


def _log_fixed_temporal_decision(
    summary_writer,
    prefix,
    epoch_index,
    class_scores,
    labels,
    has_decision,
    predicted_class=None,
):
    if predicted_class is None:
        predicted_class = class_scores.argmax(dim=1)
    correct = (predicted_class == labels) & has_decision
    summary_writer.add_scalar(
        f'{prefix}/accuracy',
        correct.to(torch.float64).mean(),
        global_step=epoch_index,
    )
    summary_writer.add_scalar(
        f'{prefix}/no_decision_fraction',
        (~has_decision).to(torch.float64).mean(),
        global_step=epoch_index,
    )
    for class_index in range(class_scores.shape[1]):
        predicted_fraction = (
            (predicted_class == class_index) & has_decision
        ).to(torch.float64).mean()
        summary_writer.add_scalar(
            f'{prefix}/predicted_fraction/class_{class_index}',
            predicted_fraction,
            global_step=epoch_index,
        )
        true_class_mask = labels == class_index
        if true_class_mask.any():
            summary_writer.add_scalar(
                f'{prefix}/per_true_class_accuracy/class_{class_index}',
                correct[true_class_mask].to(torch.float64).mean(),
                global_step=epoch_index,
            )


def log_temporal_receiver_vote_metrics(
    summary_writer,
    epoch_index,
    model,
    root_phase_receiver_features,
    all_labels,
    all_native_temporal,
    phases,
):
    """Evaluate fixed within-phase rank voting without a learned readout."""
    root = model.cortex
    phase_count = int(getattr(root, 'temporal_receiver_phase_count', 1))
    group_size = int(model.competition_group_N)
    if group_size % phase_count != 0:
        raise ValueError(
            'temporal receiver phase count must divide competition group size'
        )
    phase_width = group_size // phase_count
    for split in phases:
        features = _concat_phase_features(
            root_phase_receiver_features[split]
        ).to(dtype=torch.float64)
        labels = all_labels[split].to(dtype=torch.long)
        grouped = features.reshape(
            len(labels), model.num_classes, group_size
        )
        phase_scores = torch.stack([
            grouped[
                :,
                :,
                phase_index * phase_width:(phase_index + 1) * phase_width,
            ].mean(dim=2)
            for phase_index in range(phase_count)
        ], dim=1)
        valid_phase = phase_scores.amax(dim=2) > 0
        base_prefix = f'temporal_receiver_vote/{root.cortex_id}/{split}'
        for phase_index in range(phase_count):
            scores = phase_scores[:, phase_index, :]
            _log_fixed_temporal_decision(
                summary_writer,
                f'{base_prefix}/phase_{phase_index}',
                epoch_index,
                scores,
                labels,
                valid_phase[:, phase_index],
            )

        pairwise = (
            phase_scores.unsqueeze(-1) - phase_scores.unsqueeze(-2)
        )
        borda_points = (
            (pairwise > 0).to(torch.float64).sum(dim=-1)
            + 0.5 * (
                (pairwise == 0).to(torch.float64).sum(dim=-1) - 1.0
            )
        )
        borda_points = borda_points * valid_phase.unsqueeze(-1)
        borda_scores = borda_points.sum(dim=1)
        raw_scores = phase_scores.mean(dim=1)
        best_borda = borda_scores.amax(dim=1, keepdim=True)
        tied_for_best = borda_scores == best_borda
        tie_break_scores = raw_scores.masked_fill(
            ~tied_for_best,
            float('-inf'),
        )
        predicted_class = tie_break_scores.argmax(dim=1)
        has_decision = valid_phase.any(dim=1)
        _log_fixed_temporal_decision(
            summary_writer,
            f'{base_prefix}/borda',
            epoch_index,
            borda_scores,
            labels,
            has_decision,
            predicted_class=predicted_class,
        )
        summary_writer.add_scalar(
            f'{base_prefix}/borda/top_tie_fraction',
            ((tied_for_best.sum(dim=1) > 1) & has_decision)
            .to(torch.float64).mean(),
            global_step=epoch_index,
        )

        native_temporal = all_native_temporal[split]
        raw_prediction = raw_scores.argmax(dim=1)
        raw_has_decision = raw_scores.amax(dim=1) > 0
        native_prediction = native_temporal['predicted_class'].to(torch.long)
        native_has_decision = native_temporal['has_decision'].to(torch.bool)
        summary_writer.add_scalar(
            f'{base_prefix}/raw_native_prediction_match_fraction',
            (
                (raw_prediction == native_prediction)
                & (raw_has_decision == native_has_decision)
            ).to(torch.float64).mean(),
            global_step=epoch_index,
        )


def log_readout_error_overlap(
    summary_writer, epoch_index, all_labels, all_native_correct,
    cortex_phase_readout_predictions
):
    """Attribute native errors to probe-solvable versus shared-error samples."""
    for cortex_name, phase_predictions in cortex_phase_readout_predictions.items():
        for phase, readout_predictions in phase_predictions.items():
            labels = all_labels[phase]
            native_correct = all_native_correct[phase].bool()
            readout_correct = readout_predictions.eq(labels)
            sample_count = len(labels)
            if sample_count == 0:
                continue

            both_correct = native_correct & readout_correct
            native_only_correct = native_correct & ~readout_correct
            readout_only_correct = ~native_correct & readout_correct
            both_wrong = ~native_correct & ~readout_correct
            native_wrong_count = (~native_correct).sum().item()
            readout_wrong_count = (~readout_correct).sum().item()

            prefix = f'readout_error_overlap/{cortex_name}/{phase}'
            fractions = {
                'both_correct_fraction': both_correct.float().mean().item(),
                'native_only_correct_fraction': native_only_correct.float().mean().item(),
                'readout_only_correct_fraction': readout_only_correct.float().mean().item(),
                'both_wrong_fraction': both_wrong.float().mean().item(),
                'probe_rescue_fraction_of_native_errors': (
                    readout_only_correct.sum().item() / native_wrong_count
                    if native_wrong_count else 0.0
                ),
                'native_rescue_fraction_of_probe_errors': (
                    native_only_correct.sum().item() / readout_wrong_count
                    if readout_wrong_count else 0.0
                ),
            }
            for metric_name, value in fractions.items():
                summary_writer.add_scalar(
                    f'{prefix}/{metric_name}', value, global_step=epoch_index
                )


def _log_overlap_temporal_values(
    summary_writer, epoch_index, prefix, T_correct, T_wrong
):
    T_gap = T_correct - T_wrong
    for metric_name, values in (
        ('T_correct', T_correct),
        ('T_wrong', T_wrong),
        ('T_gap', T_gap),
    ):
        summary_writer.add_scalar(
            f'{prefix}/{metric_name}/mean',
            values.mean().item(),
            global_step=epoch_index
        )
        for quantile in (0.10, 0.50, 0.90):
            summary_writer.add_scalar(
                f'{prefix}/{metric_name}/q{int(quantile * 100):02d}',
                torch.quantile(values, quantile).item(),
                global_step=epoch_index
            )
    summary_writer.add_scalar(
        f'{prefix}/T_gap/negative_fraction',
        (T_gap < 0).to(dtype=torch.float64).mean().item(),
        global_step=epoch_index
    )


def log_readout_error_overlap_temporal(
    summary_writer, epoch_index, all_labels, all_native_correct,
    all_native_temporal, cortex_phase_readout_predictions, phases,
    num_classes
):
    """Log native timing and decisions within each probe/native overlap stratum."""
    for cortex_name, phase_predictions in cortex_phase_readout_predictions.items():
        for phase in phases:
            labels = all_labels[phase]
            native_correct = all_native_correct[phase].bool()
            readout_correct = phase_predictions[phase].eq(labels)
            native_temporal = all_native_temporal[phase]
            T_correct = native_temporal['T_correct']
            T_wrong = native_temporal['T_wrong']
            predicted_class = native_temporal['predicted_class'].long()
            has_decision = native_temporal['has_decision'].bool()
            sample_count = len(labels)
            for values in (
                native_correct,
                readout_correct,
                T_correct,
                T_wrong,
                predicted_class,
                has_decision,
            ):
                if len(values) != sample_count:
                    raise ValueError(
                        'probe/native overlap temporal tensors must share '
                        f'sample order and length for phase "{phase}"'
                    )

            strata = {
                'both_correct': native_correct & readout_correct,
                'native_only_correct': native_correct & ~readout_correct,
                'probe_only_correct': ~native_correct & readout_correct,
                'both_wrong': ~native_correct & ~readout_correct,
            }
            for stratum_name, stratum_mask in strata.items():
                prefix = (
                    f'readout_error_overlap_temporal/{cortex_name}/'
                    f'{phase}/{stratum_name}'
                )
                stratum_count = int(stratum_mask.sum().item())
                summary_writer.add_scalar(
                    f'{prefix}/count', stratum_count,
                    global_step=epoch_index
                )
                summary_writer.add_scalar(
                    f'{prefix}/fraction',
                    stratum_count / sample_count if sample_count else 0.0,
                    global_step=epoch_index
                )
                if stratum_count == 0:
                    continue

                _log_overlap_temporal_values(
                    summary_writer,
                    epoch_index,
                    prefix,
                    T_correct[stratum_mask],
                    T_wrong[stratum_mask]
                )

                stratum_has_decision = has_decision[stratum_mask]
                stratum_predictions = predicted_class[stratum_mask]
                summary_writer.add_scalar(
                    f'{prefix}/native_prediction/no_decision_fraction',
                    (~stratum_has_decision).to(dtype=torch.float64).mean().item(),
                    global_step=epoch_index
                )
                for class_id in range(num_classes):
                    predicted_fraction = (
                        (stratum_predictions == class_id) & stratum_has_decision
                    ).to(dtype=torch.float64).mean().item()
                    summary_writer.add_scalar(
                        f'{prefix}/native_prediction/class_{class_id}_fraction',
                        predicted_fraction,
                        global_step=epoch_index
                    )

                for class_id in range(num_classes):
                    class_mask = stratum_mask & labels.eq(class_id)
                    class_count = int(class_mask.sum().item())
                    class_prefix = f'{prefix}/true_class_{class_id}'
                    summary_writer.add_scalar(
                        f'{class_prefix}/count',
                        class_count,
                        global_step=epoch_index
                    )
                    if class_count == 0:
                        continue
                    class_T_correct = T_correct[class_mask]
                    class_T_wrong = T_wrong[class_mask]
                    class_T_gap = class_T_correct - class_T_wrong
                    for metric_name, values in (
                        ('T_correct', class_T_correct),
                        ('T_wrong', class_T_wrong),
                        ('T_gap', class_T_gap),
                    ):
                        summary_writer.add_scalar(
                            f'{class_prefix}/{metric_name}/mean',
                            values.mean().item(),
                            global_step=epoch_index
                        )
                    summary_writer.add_scalar(
                        f'{class_prefix}/T_gap/negative_fraction',
                        (class_T_gap < 0).to(dtype=torch.float64).mean().item(),
                        global_step=epoch_index
                    )


def log_ablation_chart(
    model, dataloaders, summary_writer, epoch_index, cortex_dict,
    accuracy_dict, show_fig
):
    ablation_score_dict = ablation_test(model, dataloaders, cortex_dict, accuracy_dict['valid'])
    ablation_chart = draw_ablation_chart(
        ablation_score_dict, accuracy_dict['valid'], cortex_dict, show=show_fig
    )
    summary_writer.add_image(
        'Ablation chart',
        ablation_chart, global_step=epoch_index, dataformats='HWC'
    )


def log_level_four_diagnostics(
    model, summary_writer, epoch_index, cortex_dict, show_fig, diagnostic_settings
):
    if diagnostic_settings['singular_value_spectrum']:
        log_kernel_visualizations(summary_writer, epoch_index, cortex_dict)

    if diagnostic_settings['top_maps_plot']:
        log_top_maps(summary_writer, epoch_index, model)


def log_kernel_visualizations(summary_writer, epoch_index, cortex_dict):
    for cortex_name, cortex in cortex_dict.items():
        summary_writer.add_image(
            f'singular_value_spectrum/{cortex_name}',
            draw_singular_value_spectrum(cortex.kernel.weight, cortex_name), global_step=epoch_index, dataformats='HWC'
        )


def log_top_maps(summary_writer, epoch_index, model):
    summary_writer.add_image(
        'top_maps_plot',
        draw_top_maps_plot(model), global_step=epoch_index, dataformats='HWC'
    )


def log_cortex_statistics(
    summary_writer, epoch_index, cortex_dict, log_threshold_distribution=False
):
    cortex_amplifier = {
        cortex_name: cortex.amplifier
        for cortex_name, cortex in cortex_dict.items()
    }
    for cortex_name, amplifier in cortex_amplifier.items():
        summary_writer.add_scalar(
            f'amplifier/cortex/{cortex_name}',
            amplifier, global_step=epoch_index
        )

    cortex_hidden_thresholds = {
        cortex_name: cortex.hidden_nv.thresholds
        for cortex_name, cortex in cortex_dict.items()
    }
    for cortex_name, threshold in cortex_hidden_thresholds.items():
        summary_writer.add_scalar(
            f'threshold_mean/hidden/{cortex_name}',
            torch.mean(threshold).item(),
            global_step=epoch_index
        )
        if log_threshold_distribution:
            _log_distribution_scalars(
                summary_writer,
                f'per_neuron_threshold/hidden/{cortex_name}',
                threshold,
                epoch_index
            )
    for cortex_name, cortex in cortex_dict.items():
        train_threshold = cortex.hidden_nv.thresholds
        base_threshold = cortex.hidden_nv.base_thresholds
        summary_writer.add_scalar(
            f'threshold_delta_mean/hidden_minus_base/{cortex_name}',
            torch.mean(train_threshold - base_threshold).item(),
            global_step=epoch_index
        )
        summary_writer.add_scalar(
            f'threshold_delta_max/hidden_minus_base/{cortex_name}',
            torch.max(train_threshold - base_threshold).item(),
            global_step=epoch_index
        )
        if log_threshold_distribution:
            _log_distribution_scalars(
                summary_writer,
                f'per_neuron_threshold_delta/hidden_minus_base/{cortex_name}',
                train_threshold - base_threshold,
                epoch_index
            )

    for cortex_name, cortex in cortex_dict.items():
        sender_mean_earliness = getattr(cortex, 'sender_mean_earliness', None)
        if sender_mean_earliness is not None:
            summary_writer.add_scalar(
                f'homeostasis/sender_mean_earliness/{cortex_name}',
                sender_mean_earliness.detach().mean().item(),
                global_step=epoch_index
            )
        input_source_earliness = getattr(
            cortex,
            'input_source_mean_first_spike_earliness',
            None,
        )
        if input_source_earliness is not None:
            source_names = ['direct_input'] + [
                subcortex.cortex_id for subcortex in cortex.subcortexs
            ]
            source_values = torch.as_tensor(
                input_source_earliness
            ).detach().reshape(-1)
            if source_values.numel() != len(source_names):
                raise RuntimeError(
                    f'{cortex_name} input-source earliness has '
                    f'{source_values.numel()} values for '
                    f'{len(source_names)} sources'
                )
            for source_name, source_value in zip(
                source_names,
                source_values,
            ):
                summary_writer.add_scalar(
                    'homeostasis/input_source_mean_first_spike_earliness/'
                    f'{cortex_name}/{source_name}',
                    source_value.item(),
                    global_step=epoch_index,
                )
        for attr_name, tag_name in (
            (
                'latest_input_mean_first_spike_earliness',
                'latest_input_mean_first_spike_earliness'
            ),
            (
                'mean_first_spike_earliness',
                'mean_first_spike_earliness'
            ),
            (
                'hidden_mean_first_spike_earliness',
                'hidden_mean_first_spike_earliness'
            ),
            (
                'output_mean_first_spike_earliness',
                'output_mean_first_spike_earliness'
            ),
            (
                'optimal_first_spike_earliness',
                'optimal_first_spike_earliness'
            ),
            (
                'optimal_mean_spike_earliness',
                'optimal_mean_spike_earliness'
            ),
            (
                'amplifier_target_mean_first_spike_earliness',
                'amplifier_target_mean_first_spike_earliness'
            ),
        ):
            value = getattr(cortex, attr_name, None)
            if value is not None:
                summary_writer.add_scalar(
                    f'homeostasis/{tag_name}/{cortex_name}',
                    torch.as_tensor(value).detach().mean().item(),
                    global_step=epoch_index
                )
        target_mean_earliness = getattr(
            cortex.hidden_nv,
            'last_target_mean_earliness',
            None
        )
        if target_mean_earliness is not None:
            summary_writer.add_scalar(
                f'homeostasis/target_mean_earliness/{cortex_name}',
                torch.as_tensor(target_mean_earliness).detach().mean().item(),
                global_step=epoch_index
            )


def _activity_scalar_tag(cortex_name, stat_name):
    if cortex_name == 'image_encoder':
        return {
            'mean earliness': 'earliness_mean/image_encoder/input',
            'mean first spike earliness': 'earliness_first_spike_mean/image_encoder/input',
            'no spike ratio': 'no_spike_ratio/image_encoder/input',
        }[stat_name]

    return {
        'mean hidden earliness': f'earliness_mean/hidden/{cortex_name}',
        'hidden mean first spike earliness': f'earliness_first_spike_mean/hidden/{cortex_name}',
        'hidden no spike ratio': f'no_spike_ratio/hidden/{cortex_name}',
        'mean output earliness': f'earliness_mean/output/{cortex_name}',
        'output mean first spike earliness': f'earliness_first_spike_mean/output/{cortex_name}',
        'output no spike ratio': f'no_spike_ratio/output/{cortex_name}',
    }[stat_name]


def log_sample_activity(
    summary_writer, epoch_index, dataloaders, model,
    log_images=True, log_temporal_metrics=True, phases=None
):
    if not log_images and not log_temporal_metrics:
        return

    sample_data_dict = get_sample_data(dataloaders, model=model)
    if phases is not None:
        phase_set = set(phases)
        sample_data_dict = {
            phase: data
            for phase, data in sample_data_dict.items()
            if phase in phase_set
        }
    spike_train_image_dict, cortex_activity_stats_dict = \
        analysis_spike_activity(sample_data_dict, model, make_plots=log_images)
    if log_images:
        for figure_name, figure in spike_train_image_dict.items():
            summary_writer.add_image(
                figure_name, figure,
                global_step=epoch_index, dataformats='NHWC'
            )

    if log_temporal_metrics:
        for cortex_name, activity_stats in cortex_activity_stats_dict.items():
            for stat_name, value in activity_stats.items():
                summary_writer.add_scalar(
                    _activity_scalar_tag(cortex_name, stat_name),
                    value,
                    global_step=epoch_index
                )


def log_gpu_usage(summary_writer, epoch_index, model):
    if not torch.cuda.is_available():
        return
    if getattr(model, 'device', torch.device('cpu')).type != 'cuda':
        return

    allocated = torch.cuda.memory_allocated(model.device) / 1024**2
    max_allocated = torch.cuda.max_memory_allocated(model.device) / 1024**2
    reserved = torch.cuda.memory_reserved(model.device) / 1024**2
    max_reserved = torch.cuda.max_memory_reserved(model.device) / 1024**2
    summary_writer.add_scalar('memory/allocated/current_MB', allocated, global_step=epoch_index)
    summary_writer.add_scalar('memory/allocated/max_MB', max_allocated, global_step=epoch_index)
    summary_writer.add_scalar('memory/reserved/current_MB', reserved, global_step=epoch_index)
    summary_writer.add_scalar('memory/reserved/max_MB', max_reserved, global_step=epoch_index)
    torch.cuda.reset_peak_memory_stats(model.device)


def log_model_shape(summary_writer, epoch_index, model):
    nested_shape = get_nested_shape(model.cortex)
    model_shape_info = pprint.pformat(nested_shape, indent=4)
    print('---------------- model shape ----------------')
    print(model_shape_info)
    print('---------------------------------------------')
    summary_writer.add_text('model_shape_info', model_shape_info, global_step=epoch_index)


def log_kernel_heatmaps(summary_writer, epoch_index, cortex_dict, show_fig):
    cortex_kernel_heat_map = {
        cortex_name: draw_heat_map(cortex.kernel.weight, show=show_fig)
        for cortex_name, cortex in cortex_dict.items()
    }
    for cortex_name, figure in cortex_kernel_heat_map.items():
        summary_writer.add_image(
            f'cortex kernel heat map/{cortex_name}',
            figure, global_step=epoch_index, dataformats='HWC'
        )


def inspect_model(
    model, dataloaders, summary_writer, epoch_index, readout_params, level=5,
    show_fig=False, diagnostic_settings=None, phases=None,
    evaluation_micro_batch_size=None
):
    phases = _normalize_evaluation_phases(phases)
    diagnostic_settings = _get_diagnostic_settings(diagnostic_settings)
    if (
        diagnostic_settings['readout_error_overlap']
        and not diagnostic_settings['readout']
    ):
        raise ValueError('readout_error_overlap requires readout=true')
    if (
        diagnostic_settings['readout_error_overlap_temporal']
        and not diagnostic_settings['readout_error_overlap']
    ):
        raise ValueError(
            'readout_error_overlap_temporal requires '
            'readout_error_overlap=true'
        )
    if (
        diagnostic_settings['readout_receiver_alignment']
        and not diagnostic_settings['readout']
    ):
        raise ValueError('readout_receiver_alignment requires readout=true')
    if diagnostic_settings['probe_route_conflict']:
        required_route_diagnostics = (
            'readout',
            'readout_error_overlap',
            'readout_error_overlap_temporal',
        )
        missing_route_diagnostics = [
            name for name in required_route_diagnostics
            if not diagnostic_settings[name]
        ]
        if missing_route_diagnostics:
            raise ValueError(
                'probe_route_conflict requires enabled diagnostics: '
                + ', '.join(missing_route_diagnostics)
            )
    if (
        (diagnostic_settings['readout'] or diagnostic_settings['readout_fusion'])
        and set(phases) != {'train', 'test', 'valid'}
    ):
        raise ValueError(
            'readout diagnostics require evaluation_phases to include '
            'train, test, and valid'
        )
    if (
        diagnostic_settings['cdna_assignment_readout']
        and set(phases) != {'train', 'test', 'valid'}
    ):
        raise ValueError(
            'CDNA assignment readout requires evaluation_phases to include '
            'train, test, and valid'
        )

    needs_cortex_dict = (
        (level > 0 and diagnostic_settings['cortex_statistics'])
        or diagnostic_settings['per_neuron_activity']
        or (level > 1 and diagnostic_settings['readout'])
        or (level > 1 and diagnostic_settings['readout_fusion'])
        or (level > 1 and diagnostic_settings['ablation_chart'])
        or (level > 3 and (
            diagnostic_settings['singular_value_spectrum']
            or diagnostic_settings['top_maps_plot']
        ))
        or (level > 6 and diagnostic_settings['kernel_heatmaps'])
    )
    cortex_dict = get_cortex_dict(model.cortex) if needs_cortex_dict else None
    collect_cortex_activity = (
        (level > 1 and diagnostic_settings['readout'])
        or (level > 1 and diagnostic_settings['readout_fusion'])
    )

    (
        accuracy_dict,
        confusion_matrices,
        cortex_phase_readout_feature_dict,
        all_labels,
        all_native_correct,
        all_native_temporal,
        root_phase_receiver_features,
    ) = collect_phase_metrics(
        model,
        dataloaders,
        phases,
        cortex_dict=(cortex_dict if collect_cortex_activity else None),
        with_confusion_matrix=(level > 0 and diagnostic_settings['confusion_matrix']),
        micro_batch_size=evaluation_micro_batch_size,
        collect_native_temporal=(
            diagnostic_settings['readout_error_overlap_temporal']
            or diagnostic_settings['temporal_receiver_vote_metrics']
            or (diagnostic_settings['normalization_mechanism_artifacts']
                and epoch_index in diagnostic_settings['normalization_mechanism_trajectory_epochs'])
        ),
        collect_root_receiver_activity=(
            diagnostic_settings['probe_route_conflict']
            or diagnostic_settings['temporal_receiver_phase_metrics']
            or diagnostic_settings['temporal_receiver_vote_metrics']
            or diagnostic_settings['cdna_assignment_readout']
        ),
    )
    log_accuracy(summary_writer, accuracy_dict, epoch_index)
    if (diagnostic_settings['normalization_mechanism_artifacts']
            and epoch_index in diagnostic_settings['normalization_mechanism_trajectory_epochs']):
        artifact_dir = Path(summary_writer.log_dir) / 'mechanism_artifacts'
        artifact_dir.mkdir(parents=True, exist_ok=True)
        for phase in phases:
            native = all_native_temporal[phase]
            np.savez_compressed(
                artifact_dir / f'native_decisions_{phase}_epoch{int(epoch_index):03d}.npz',
                accuracy=np.asarray(accuracy_dict[phase]),
                labels=all_labels[phase].cpu().numpy(),
                **{key: value.cpu().numpy() for key, value in native.items()},
            )
    if diagnostic_settings['temporal_receiver_progress_metrics']:
        log_temporal_receiver_progress_metrics(
            summary_writer,
            epoch_index,
            model,
            dataloaders,
            _normalize_diagnostic_phase_subset(
                diagnostic_settings[
                    'temporal_receiver_progress_metric_phases'
                ],
                phases,
            ),
        )
    if diagnostic_settings['subcortex_handoff_metrics']:
        log_subcortex_handoff_metrics(
            summary_writer,
            epoch_index,
            model,
            dataloaders,
            _normalize_diagnostic_phase_subset(
                diagnostic_settings['subcortex_handoff_metric_phases'],
                phases,
            ),
            micro_batch_size=evaluation_micro_batch_size,
        )
    if diagnostic_settings['temporal_receiver_phase_metrics']:
        log_temporal_receiver_phase_metrics(
            summary_writer,
            epoch_index,
            model,
            root_phase_receiver_features,
            all_labels,
            _normalize_diagnostic_phase_subset(
                diagnostic_settings[
                    'temporal_receiver_phase_metric_phases'
                ],
                phases,
            ),
        )
    if diagnostic_settings['temporal_receiver_vote_metrics']:
        log_temporal_receiver_vote_metrics(
            summary_writer,
            epoch_index,
            model,
            root_phase_receiver_features,
            all_labels,
            all_native_temporal,
            _normalize_diagnostic_phase_subset(
                diagnostic_settings[
                    'temporal_receiver_vote_metric_phases'
                ],
                phases,
            ),
        )
    if diagnostic_settings['cdna_assignment_readout']:
        log_cdna_assignment_readout_metrics(
            summary_writer,
            epoch_index,
            model,
            root_phase_receiver_features,
            all_labels,
            _normalize_diagnostic_phase_subset(
                diagnostic_settings['cdna_assignment_readout_phases'],
                phases,
            ),
        )
    if diagnostic_settings['gpu_usage']:
        log_gpu_usage(summary_writer, epoch_index, model)
    if diagnostic_settings['hinge_temporal_metrics']:
        log_hinge_temporal_metrics(
            summary_writer,
            epoch_index,
            model,
            dataloaders,
            _normalize_diagnostic_phase_subset(
                diagnostic_settings['hinge_temporal_phases'],
                phases
            )
        )
    if diagnostic_settings['per_neuron_activity']:
        log_per_neuron_activity_metrics(
            summary_writer,
            epoch_index,
            model,
            dataloaders,
            cortex_dict,
            _normalize_diagnostic_phase_subset(
                diagnostic_settings['per_neuron_activity_phases'],
                phases
            ),
            max_batches=diagnostic_settings['per_neuron_activity_max_batches'],
            micro_batch_size=evaluation_micro_batch_size,
        )

    if level <= 0:
        return accuracy_dict, accuracy_dict['valid']

    selected_valid_score = accuracy_dict['valid']
    GA_score = selected_valid_score

    if diagnostic_settings['confusion_matrix']:
        log_decision_confusion_matrices(summary_writer, epoch_index, confusion_matrices)

    if diagnostic_settings['cortex_statistics']:
        log_cortex_statistics(
            summary_writer,
            epoch_index,
            cortex_dict,
            log_threshold_distribution=diagnostic_settings['threshold_distribution']
        )

    if level <= 1:
        return accuracy_dict, GA_score, selected_valid_score

    if diagnostic_settings['readout']:
        return_readout_predictions = diagnostic_settings[
            'readout_error_overlap'
        ]
        return_readout_parameters = diagnostic_settings[
            'readout_receiver_alignment'
        ]
        readout_result = get_readout_accuracy(
            cortex_phase_readout_feature_dict,
            all_labels,
            readout_params,
            return_predictions=return_readout_predictions,
            return_parameters=return_readout_parameters,
        )
        cortex_readout_parameters = None
        if return_readout_predictions and return_readout_parameters:
            (
                phase_cortex_readout_acc,
                cortex_phase_readout_predictions,
                cortex_readout_parameters,
            ) = readout_result
        elif return_readout_predictions:
            (
                phase_cortex_readout_acc,
                cortex_phase_readout_predictions,
            ) = readout_result
        elif return_readout_parameters:
            (
                phase_cortex_readout_acc,
                cortex_readout_parameters,
            ) = readout_result
        else:
            phase_cortex_readout_acc = readout_result
        GA_score = log_level_one_diagnostics(
            summary_writer, epoch_index, cortex_dict,
            accuracy_dict, phase_cortex_readout_acc, show_fig
        )
        if diagnostic_settings['readout_error_overlap']:
            log_readout_error_overlap(
                summary_writer,
                epoch_index,
                all_labels,
                all_native_correct,
                cortex_phase_readout_predictions,
            )
        if diagnostic_settings['readout_error_overlap_temporal']:
            log_readout_error_overlap_temporal(
                summary_writer,
                epoch_index,
                all_labels,
                all_native_correct,
                all_native_temporal,
                cortex_phase_readout_predictions,
                _normalize_diagnostic_phase_subset(
                    diagnostic_settings[
                        'readout_error_overlap_temporal_phases'
                    ],
                    phases
                ),
                model.num_classes
            )
        if diagnostic_settings['probe_route_conflict']:
            log_probe_route_conflict(
                summary_writer,
                epoch_index,
                model,
                cortex_phase_readout_feature_dict,
                root_phase_receiver_features,
                all_labels,
                all_native_correct,
                all_native_temporal,
                cortex_phase_readout_predictions,
                _normalize_diagnostic_phase_subset(
                    diagnostic_settings['probe_route_conflict_phases'],
                    phases,
                ),
            )
        if diagnostic_settings['readout_receiver_alignment']:
            log_readout_receiver_alignment(
                summary_writer,
                epoch_index,
                model,
                cortex_dict,
                cortex_readout_parameters,
            )

    if diagnostic_settings['readout_fusion']:
        phase_fusion_readout_acc = get_readout_fusion_accuracy(
            cortex_phase_readout_feature_dict, all_labels, readout_params
        )
        log_readout_fusion_accuracy(
            summary_writer, epoch_index, phase_fusion_readout_acc
        )

    if diagnostic_settings['ablation_chart']:
        log_ablation_chart(
            model, dataloaders, summary_writer, epoch_index, cortex_dict,
            accuracy_dict, show_fig
        )

    if diagnostic_settings['path_ablation']:
        log_path_ablation_metrics(
            model,
            dataloaders,
            summary_writer,
            epoch_index,
            _normalize_diagnostic_phase_subset(
                diagnostic_settings['path_ablation_phases'],
                phases,
            ),
            micro_batch_size=evaluation_micro_batch_size,
        )

    if level <= 2:
        return accuracy_dict, GA_score, selected_valid_score

    log_sample_activity(
        summary_writer, epoch_index, dataloaders, model,
        log_images=diagnostic_settings['sample_activity_images'],
        log_temporal_metrics=diagnostic_settings['temporal_sample_metrics'],
        phases=diagnostic_settings.get('sample_activity_phases', phases)
    )

    if diagnostic_settings['normalization_mechanism_artifacts']:
        log_normalization_mechanism_weight_artifact(
            model,
            summary_writer,
            epoch_index,
            diagnostic_settings,
        )
        log_normalization_mechanism_trajectory_artifacts(
            model,
            dataloaders,
            summary_writer,
            epoch_index,
            diagnostic_settings,
            phases,
        )

    if level <= 3:
        return accuracy_dict, GA_score, selected_valid_score

    log_level_four_diagnostics(
        model, summary_writer, epoch_index, cortex_dict, show_fig, diagnostic_settings
    )

    if level <= 4:
        return accuracy_dict, GA_score, selected_valid_score

    if level <= 5:
        return accuracy_dict, GA_score, selected_valid_score

    if diagnostic_settings['model_shape']:
        log_model_shape(summary_writer, epoch_index, model)

    if level <= 6:
        return accuracy_dict, GA_score, selected_valid_score

    if diagnostic_settings['kernel_heatmaps']:
        log_kernel_heatmaps(summary_writer, epoch_index, cortex_dict, show_fig)

    return accuracy_dict, GA_score, selected_valid_score

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
