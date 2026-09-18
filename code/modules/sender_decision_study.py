"""Decision-associated sender drive and native forward interventions.

Drive shares are descriptive. Necessity is tested by rerunning the unmodified
SSIF decision dynamics with masked sender inputs; no proxy is called accuracy.
Restricted to the audited single-branch, global-root, persistent SSIF family.
"""
import csv
import hashlib
import json
import math
import random
from pathlib import Path

import numpy as np
import torch


def parameter_fingerprint(model):
    result = {}
    for cortex in model.cortex.iter_cortex_tree():
        for name, value in (
            ('weight', cortex.kernel.weight),
            ('threshold', cortex.hidden_nv.log_thresholds),
            ('amplifier', cortex.log_amplifier),
        ):
            raw = value.detach().cpu().contiguous().numpy().tobytes()
            result[f'{cortex.cortex_id}/{name}'] = hashlib.sha256(raw).hexdigest()
    return result


def log_initial_sender_parameters(model, writer, epoch_offset):
    folder = Path(writer.log_dir)/'sender_decision'
    folder.mkdir(parents=True, exist_ok=True)
    (folder/'initial_parameters.json').write_text(json.dumps({
        'epoch_offset': epoch_offset,
        'loaded_from': getattr(model, 'loaded_from', None),
        'parameters': parameter_fingerprint(model),
    }, indent=2), encoding='utf-8')


def validate_family(model):
    root = model.cortex
    if len(root.subcortexs) != 1 or not root.kernel.no_skip_links:
        raise ValueError('sender study requires one branch and no direct bypass')
    if torch.count_nonzero(root.kernel.weight[:root.kernel.input_len]):
        raise ValueError('sender study found nonzero direct weights')
    for name in ('branch_time_binned_offsets', 'scheduled_direct_weights',
                 'scheduled_branch_offsets', 'state_conditioned_direct_weights',
                 'presynaptic_class_preference', 'branch_class_preference'):
        if getattr(root, name, None) is not None:
            raise ValueError(f'sender study does not support {name}')
    if root.hidden_nv.resting_length < model.image_encoder.simulation_time:
        raise ValueError('sender study requires persistent single-spike receivers')
    return root


@torch.no_grad()
def capture_forward(model, images, sender_mask=None, capture=True):
    root = validate_family(model)
    device = root.kernel.weight.device
    images = images.to(device)
    batch = len(images)
    direct = int(root.kernel.input_len)
    weights = root.kernel.weight[direct:]
    original_project = root._project_sender_waves
    had_override = '_project_sender_waves' in root.__dict__
    old_override = root.__dict__.get('_project_sender_waves')
    senders, excitations, potentials, receiver_waves = [], [], [], []

    def project(waves):
        if waves.numel() != batch * root.kernel.weight.shape[0]:
            raise ValueError('sender study requires exactly one root spatial patch')
        if sender_mask is not None:
            waves = waves.clone()
            waves[..., direct:] *= sender_mask.to(waves).reshape(batch, 1, -1)
        value = original_project(waves)
        if capture:
            branch = waves.reshape(batch, -1)[:, direct:]
            reconstructed = branch @ weights
            if not torch.allclose(reconstructed, value.reshape(batch, -1), atol=2e-5, rtol=2e-5):
                raise RuntimeError('sender drive does not reconstruct native excitation')
            senders.append(branch.detach().cpu())
        return value

    root._project_sender_waves = project
    try:
        model.init_state(images.shape, device)
        score = None
        for spikes in model.image_encoder(images):
            wave = model.forward_cortex(spikes, is_training=False)
            score = wave.clone() if score is None else score + wave
            if capture:
                excitations.append(root.current_excitatory_potential.reshape(batch, -1).detach().cpu())
                potentials.append(root.hidden_nv.current_potential.reshape(batch, -1).detach().cpu())
                receiver_waves.append(root.hidden_nv.spike_wave.reshape(batch, -1).detach().cpu())
        time_count = int(model.image_encoder.simulation_time)
        scores = model.score_output_neuron_scores(score / time_count).detach().cpu()
        result = {'scores': scores}
        if capture:
            receiver_wave = torch.stack(receiver_waves, dim=1)
            fired = receiver_wave.gt(0)
            first = fired.to(torch.int64).argmax(dim=1)
            first[~fired.any(dim=1)] = -1
            earliness = torch.where(first >= 0, (time_count-first).float()/time_count, 0.)
            reconstructed_scores = earliness.reshape(batch, model.num_classes, -1).mean(-1)
            if not torch.allclose(scores, reconstructed_scores, atol=2e-6, rtol=2e-6):
                raise RuntimeError('receiver first spikes do not reconstruct native class scores')
            result.update(
                sender=torch.stack(senders, dim=1),
                excitation=torch.stack(excitations, dim=1),
                potential=torch.stack(potentials, dim=1),
                first=first,
                receiver_earliness=earliness,
                weight=weights.detach().cpu().clone(),
                threshold=root.hidden_nv.get_thresholds_for_forward(
                    use_training_thresholds=False).detach().cpu().reshape(-1).clone(),
            )
        return result
    finally:
        if had_override:
            root._project_sender_waves = old_override
        else:
            del root._project_sender_waves


def drive_at_onset(record, classes, num_classes, support_normalization='threshold'):
    """Signed onset excitation weighted by native score share; optionally /theta.

    This scaling defines diagnostic support only, never neuron dynamics.
    """
    if support_normalization not in ('threshold', 'none'):
        raise ValueError('support_normalization must be threshold or none')
    batch, _, senders = record['sender'].shape
    group = record['weight'].shape[1] // num_classes
    positive = torch.zeros(batch, senders)
    negative = torch.zeros_like(positive)
    for b in range(batch):
        receivers = torch.arange(int(classes[b])*group, (int(classes[b])+1)*group)
        for r in receivers:
            t = int(record['first'][b, r])
            if t < 0:
                continue
            contribution = record['sender'][b, t] * record['weight'][:, r]
            expected = record['excitation'][b, t, r]
            if not torch.allclose(contribution.sum(), expected, atol=2e-5, rtol=2e-5):
                raise RuntimeError('onset sender contributions fail excitation conservation')
            if support_normalization == 'threshold':
                contribution = contribution / record['threshold'][r].clamp_min(1e-12)
            contribution *= record['receiver_earliness'][b, r] / group
            positive[b] += contribution.clamp_min(0)
            negative[b] += contribution.clamp_max(0)
    return positive, negative


def concentration(values):
    total = float(values.sum())
    if total <= 0:
        return {'has_support': 0, 'k80': '', 'effective_senders': '', 'top10_share': ''}
    share = values / total
    ranked = share.sort(descending=True).values
    eligible = int((values > 0).sum())
    return {
        'has_support': 1,
        'k80': int(torch.searchsorted(ranked.cumsum(0), .8)) + 1,
        'effective_senders': float(1 / share.square().sum()),
        'top10_share': float(ranked[:max(1, math.ceil(eligible*.1))].sum()),
    }


def deletion_masks(positive, fraction, seed):
    top = torch.ones_like(positive)
    random_mask = torch.ones_like(positive)
    matched = torch.ones_like(positive)
    generator = torch.Generator().manual_seed(int(seed))
    for b, values in enumerate(positive):
        eligible = torch.where(values > 0)[0]
        if not len(eligible):
            continue
        count = max(1, math.ceil(len(eligible)*fraction))
        chosen = eligible[torch.topk(values[eligible], count).indices]
        top[b, chosen] = 0
        random_chosen = eligible[torch.randperm(len(eligible), generator=generator)[:count]]
        random_mask[b, random_chosen] = 0
        removed_fraction = values[chosen].sum() / values.sum()
        matched[b] = 1 - removed_fraction
    return {'top_support': top, 'random_support': random_mask, 'equal_drive_attenuation': matched}


def write_csv(path, rows):
    if not rows:
        raise ValueError(f'empty sender study CSV: {path}')
    with path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def select_samples(dataset, num_classes, per_class):
    counts = [0] * num_classes
    chosen = []
    for index in range(len(dataset)):
        image, label = dataset[index]
        label = int(label)
        if counts[label] < per_class:
            chosen.append((index, image, label))
            counts[label] += 1
        if min(counts) >= per_class:
            break
    if min(counts) < per_class:
        raise ValueError(f'insufficient class-balanced diagnostic samples: {counts}')
    return chosen


def decision_rows(scores, labels, fixed_wrong, ids, variant, reference=None):
    rows = []
    wrong_scores = scores.clone()
    wrong_scores[torch.arange(len(labels)), labels] = -torch.inf
    reselected = wrong_scores.argmax(1)
    prediction = scores.argmax(1)
    prediction[scores.max(1).values <= 0] = -1
    for b, index in enumerate(ids):
        y, w, other = int(labels[b]), int(fixed_wrong[b]), int(reselected[b])
        row = {
            'sample_index': index, 'true_class': y, 'variant': variant,
            'prediction': int(prediction[b]), 'correct': int(prediction[b] == y),
            'no_decision': int(prediction[b] == -1),
            'fixed_wrong_class': w, 'reselected_wrong_class': other,
            'T_correct': float(scores[b, y]), 'T_wrong_fixed': float(scores[b, w]),
            'T_wrong_reselected': float(scores[b, other]),
            'T_gap': float(scores[b, y]-scores[b, other]),
        }
        if reference is not None:
            row['delta_T_correct'] = float(scores[b, y]-reference[b, y])
            row['delta_T_wrong_fixed'] = float(scores[b, w]-reference[b, w])
        else:
            row['delta_T_correct'] = 0.
            row['delta_T_wrong_fixed'] = 0.
        rows.append(row)
    return rows


def collect_phase(model, dataloader, output, epoch, settings):
    support_normalization = settings.get('support_normalization', 'threshold')
    if support_normalization not in ('threshold', 'none'):
        raise ValueError('support_normalization must be threshold or none')
    root = validate_family(model)
    before = parameter_fingerprint(model)
    selected = select_samples(dataloader.dataset, model.num_classes, int(settings['samples_per_class']))
    rows, support_rows, top_rows, trace_rows, budget_rows = [], [], [], [], []
    sender_trace_rows = []
    all_positive = {role: [] for role in ('true', 'predicted', 'wrong')}
    all_negative = {role: [] for role in all_positive}
    ids_all, labels_all, scores_all = [], [], []
    masks_all = {'top_removed': [], 'random_removed': [], 'uniform_multiplier': []}
    micro = int(settings['micro_batch_size'])
    direct = int(root.kernel.input_len)
    original_weight = root.kernel.weight.detach().clone()
    try:
        for start in range(0, len(selected), micro):
            subset = selected[start:start+micro]
            ids = [item[0] for item in subset]
            images = torch.stack([item[1] for item in subset])
            labels = torch.tensor([item[2] for item in subset])
            base = capture_forward(model, images)
            scores = base['scores']
            wrong_scores = scores.clone()
            wrong_scores[torch.arange(len(labels)), labels] = -torch.inf
            wrong = wrong_scores.argmax(1)
            predicted = scores.argmax(1)
            rows.extend(decision_rows(scores, labels, wrong, ids, 'identity'))
            positive_true = None
            for role, classes in (('true', labels), ('predicted', predicted), ('wrong', wrong)):
                positive, negative = drive_at_onset(
                    base, classes, model.num_classes, support_normalization)
                all_positive[role].append(positive.numpy())
                all_negative[role].append(negative.numpy())
                if role == 'true':
                    positive_true = positive
                for b, index in enumerate(ids):
                    support_rows.append({
                        'sample_index': index, 'true_class': int(labels[b]),
                        'role': role, 'target_class': int(classes[b]),
                        'positive_drive': float(positive[b].sum()),
                        'negative_drive': float(negative[b].sum()),
                        **concentration(positive[b]),
                    })
                    ranks = torch.argsort(positive[b], descending=True)[:32]
                    total = positive[b].sum().clamp_min(1e-30)
                    for rank, sender in enumerate(ranks):
                        top_rows.append({
                            'sample_index': index, 'role': role, 'rank': rank+1,
                            'sender_index': int(sender),
                            'positive_share': float(positive[b, sender]/total),
                            'signed_drive': float(positive[b, sender]+negative[b, sender]),
                        })
                        if role in ('true', 'wrong') and positive[b, sender] > 0:
                            for t in range(base['sender'].shape[1]):
                                sender_trace_rows.append({
                                    'sample_index': index, 'role': role,
                                    'sender_index': int(sender), 'rank': rank+1,
                                    'time_bin': t,
                                    'amplified_sender_input': float(base['sender'][b,t,sender]),
                                })
            masks = deletion_masks(positive_true, float(settings['deletion_fraction']),
                                   int(settings['mask_seed'])+start)
            masks_all['top_removed'].append((masks['top_support'] == 0).numpy())
            masks_all['random_removed'].append((masks['random_support'] == 0).numpy())
            masks_all['uniform_multiplier'].append(masks['equal_drive_attenuation'][:, 0].numpy())
            for variant, mask in masks.items():
                changed = capture_forward(model, images, sender_mask=mask, capture=False)['scores']
                these = decision_rows(changed, labels, wrong, ids, variant, scores)
                for b, row in enumerate(these):
                    row['removed_positive_drive'] = float(((1-mask[b])*positive_true[b]).sum())
                rows.extend(these)
            # Immediate, zero-learning interventions on the common epoch-1 state.
            if epoch == int(settings.get('common_prefix_epoch', 1)):
                w = original_weight[direct:]
                total = w.abs().sum()
                normalized = torch.nn.functional.normalize(w, p=1, dim=1)
                normalized *= total / normalized.abs().sum().clamp_min(1e-30)
                restored_columns = torch.nn.functional.normalize(normalized, p=1, dim=0)
                restored_columns *= w.abs().sum(dim=0, keepdim=True)
                for variant, transformed in (
                    ('immediate_pre_total_matched', normalized),
                    ('immediate_pre_column_restored', restored_columns),
                ):
                    root.kernel.weight[direct:] = transformed
                    changed = capture_forward(model, images, capture=False)['scores']
                    rows.extend(decision_rows(changed, labels, wrong, ids, variant, scores))
                    root.kernel.weight.copy_(original_weight)
                    if start == 0:
                        budget_rows.append({
                            'variant': variant, 'total_l1': float(transformed.abs().sum()),
                            'reference_total_l1': float(total),
                            'row_l1_cv': float(transformed.abs().sum(1).std()/transformed.abs().sum(1).mean()),
                            'max_column_l1_delta': float((transformed.abs().sum(0)-w.abs().sum(0)).abs().max()),
                            'max_row_l1_delta': float((transformed.abs().sum(1)-w.abs().sum(1)).abs().max()),
                        })
            # Real traces: all sampled receivers, not a selected success example.
            for b, index in enumerate(ids):
                for target in sorted({int(labels[b]), int(wrong[b])}):
                    group = int(root.competition_group_N)
                    for receiver in range(target*group, (target+1)*group):
                        for t in range(base['sender'].shape[1]):
                            trace_rows.append({
                                'sample_index': index, 'true_class': int(labels[b]),
                                'target_class': target, 'receiver_index': receiver,
                                'time_bin': t, 'excitation': float(base['excitation'][b,t,receiver]),
                                'potential': float(base['potential'][b,t,receiver]),
                                'inhibition': float(base['excitation'][b,t,receiver]-base['potential'][b,t,receiver]),
                                'threshold': float(base['threshold'][receiver]),
                                'first_spike_bin': int(base['first'][b,receiver]),
                            })
            ids_all.extend(ids); labels_all.extend(labels.tolist()); scores_all.append(scores.numpy())
    finally:
        root.kernel.weight.copy_(original_weight)
    if parameter_fingerprint(model) != before:
        raise RuntimeError('sender diagnostic mutated learned parameters')
    for row in rows:
        row.setdefault('removed_positive_drive', 0.)
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output/'native_interventions.csv', rows)
    write_csv(output/'support_concentration.csv', support_rows)
    write_csv(output/'top_senders.csv', top_rows)
    write_csv(output/'receiver_traces.csv', trace_rows)
    if sender_trace_rows:
        write_csv(output/'sender_activity_traces.csv', sender_trace_rows)
    if budget_rows:
        write_csv(output/'immediate_budget_audit.csv', budget_rows)
    arrays = {'sample_index': np.asarray(ids_all), 'true_class': np.asarray(labels_all),
              'class_scores': np.concatenate(scores_all)}
    for role in all_positive:
        arrays[role+'_positive_drive'] = np.concatenate(all_positive[role])
        arrays[role+'_negative_drive'] = np.concatenate(all_negative[role])
    np.savez_compressed(output/'sender_drive.npz', **arrays)
    np.savez_compressed(output/'sender_masks.npz',
                        sample_index=np.asarray(ids_all),
                        **{key: np.concatenate(values) for key, values in masks_all.items()})
    sender_count = int(root.kernel.weight.shape[0])-direct
    channels = int(root.subcortexs[0].output_channel)
    child = root.subcortexs[0]
    patch_h, patch_w, _ = child._cached_init_patch_grid
    pool_h, pool_w, _ = child.pooling.get_pooled_grid(patch_h, patch_w)
    if sender_count != pool_h*pool_w*channels:
        raise RuntimeError('sender spatial identity contract failed')
    mapping = [{'sender_index': s, 'source_cortex': child.cortex_id,
                'spatial_flat_index': s//channels, 'channel_index': s%channels,
                'spatial_row': (s//channels)//pool_w, 'spatial_column': (s//channels)%pool_w}
               for s in range(sender_count)]
    write_csv(output/'sender_identity.csv', mapping)
    (output/'manifest.json').write_text(json.dumps({
        'schema': 2, 'epoch': epoch, 'sample_count': len(selected),
        'selection': 'first dataset indices, equal count per true class; no outcome selection',
        'parameters_before_and_after': before, 'sender_count': sender_count,
        'support_normalization': support_normalization,
        'definition': ('signed excitation at receiver first spike'
                       + (' / threshold' if support_normalization == 'threshold' else '')
                       + ', weighted by receiver earliness / class group size'),
        'interpretation': 'drive shares are descriptive; masking measures context-dependent necessity',
        'settings': settings,
    }, indent=2), encoding='utf-8')
    return rows


def log_sender_decision_study(model, dataloaders, writer, epoch, settings):
    if int(epoch) not in settings.get('epochs', []):
        return
    rng_numpy, rng_python = np.random.get_state(), random.getstate()
    cuda_devices = [model.cortex.kernel.weight.device.index] if model.cortex.kernel.weight.is_cuda else []
    studies = [('sender_decision', settings)]
    reference = settings.get('reference_support_normalization')
    if reference is not None:
        if reference == settings.get('support_normalization', 'threshold'):
            raise ValueError('reference support normalization must differ from primary')
        reference_settings = dict(settings, support_normalization=reference)
        reference_settings.pop('reference_support_normalization')
        studies.append(('sender_decision_reference', reference_settings))
    try:
        with torch.random.fork_rng(devices=cuda_devices):
            for namespace, study_settings in studies:
                for phase in study_settings.get('phases', ['valid', 'test']):
                    folder = Path(writer.log_dir)/namespace/f'epoch{epoch:03d}'/phase
                    rows = collect_phase(model, dataloaders[phase], folder, int(epoch), study_settings)
                    if namespace == 'sender_decision_reference':
                        primary = Path(writer.log_dir)/'sender_decision'/f'epoch{epoch:03d}'/phase
                        with np.load(primary/'sender_drive.npz') as a, np.load(folder/'sender_drive.npz') as b:
                            for key in ('sample_index', 'true_class', 'class_scores'):
                                if not np.array_equal(a[key], b[key]):
                                    raise RuntimeError(f'paired support diagnostics differ in {key}')
                        with np.load(primary/'sender_masks.npz') as a, np.load(folder/'sender_masks.npz') as b:
                            if not np.array_equal(a['random_removed'], b['random_removed']):
                                raise RuntimeError('paired support diagnostics changed random-removal masks')
                    for variant in sorted({row['variant'] for row in rows}):
                        selected = [row for row in rows if row['variant'] == variant]
                        for metric in ('correct', 'no_decision', 'T_correct', 'T_wrong_fixed', 'T_gap'):
                            writer.add_scalar(f'{namespace}/{phase}/{variant}/{metric}',
                                              float(np.mean([row[metric] for row in selected])), epoch)
    finally:
        np.random.set_state(rng_numpy)
        random.setstate(rng_python)
