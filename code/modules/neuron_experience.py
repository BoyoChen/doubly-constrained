"""Opt-in, read-only training history at the shared receiver-column level.

Counts apply to the learning pass only, not the two-pass decision replay or
evaluation. A spatial neuron has its own firing count but shares the channel's
incoming weights. Update counts are applied-update events (batches in two-pass),
not sample eligibility counts. Delta paths are actual floating-point changes;
the separately named P/D term totals are proposed updates before cancellation.
"""
import csv
import json
from pathlib import Path

import numpy as np
import torch


class NeuronExperience:
    def __init__(self, cortex):
        self.initial_weight = cortex.kernel.weight.detach().clone()
        mask = getattr(cortex, 'kernel_learning_mask', None)
        self.support = (
            torch.ones_like(self.initial_weight, dtype=torch.bool)
            if mask is None else mask.to(dtype=torch.bool).clone()
        )
        if getattr(cortex.kernel, 'has_subcortexs', False) and cortex.kernel.no_skip_links:
            self.support[:cortex.kernel.input_len, :cortex.kernel.output_len] = False
        self.values = {}
        self.spatial_fire = None
        self.samples = 0
        self.batches = 0
        self.steps = 0
        self.previous = {}
        self.previous_samples = 0
        self.previous_batches = 0

    def add(self, name, value):
        value = value.detach().to(dtype=torch.float64)
        if name not in self.values:
            self.values[name] = torch.zeros_like(value)
        self.values[name].add_(value)

    def fire(self, spikes):
        # Last axis is hidden_channel, including separate NCG members in A.
        counts = (spikes > 0).sum(dim=0)
        if self.spatial_fire is None:
            self.spatial_fire = torch.zeros_like(counts, dtype=torch.int64)
        if counts.shape != self.spatial_fire.shape:
            raise ValueError('neuron experience requires fixed spatial geometry')
        self.spatial_fire.add_(counts)
        self.add('fire_count', counts.reshape(-1, counts.shape[-1]).sum(0))
        self.steps += 1

    def stdp_terms(self, potentiation, depression):
        # Uncancelled, rate-scaled terms before normalization or preconditioning.
        # These are proposed terms, not actual floating-point applied weight paths.
        for name, term in (('stdp_p_term_l1', potentiation), ('stdp_d_term_l1', depression)):
            self.add(name, torch.where(self.support, term, 0.).abs().sum(0, dtype=torch.float64))

    def delta(self, stage, delta):
        # Reduction stays on the training device; no per-step CPU synchronization.
        self.add(f'{stage}_excluded_l1', torch.where(self.support, 0., delta).abs().sum(0, dtype=torch.float64))
        delta = torch.where(self.support, delta, 0.)
        changed = delta != 0
        self.add(f'{stage}_update_count', changed.any(dim=0))
        self.add(f'{stage}_synapse_update_count', changed.sum(dim=0))
        self.add(f'{stage}_l1', delta.abs().sum(dim=0, dtype=torch.float64))
        self.add(f'{stage}_positive', delta.clamp_min(0).sum(dim=0, dtype=torch.float64))
        self.add(f'{stage}_negative', (-delta.clamp_max(0)).sum(dim=0, dtype=torch.float64))

    def before_regulation(self, weight_before, weight_after):
        self.delta('stdp', weight_after - weight_before)
        return weight_after.detach().clone()

    def after_regulation(self, weight_before, after_stdp, weight_after):
        self.delta('regulation', weight_after - after_stdp)
        self.delta('net', weight_after - weight_before)

    def export(self, cortex, writer, epoch):
        weight = cortex.kernel.weight.detach()
        columns = weight.shape[1]
        zeros = np.zeros(columns, dtype=np.float64)
        arrays = {key: value.cpu().numpy().copy() for key, value in self.values.items()}
        arrays.setdefault('fire_count', zeros.copy())
        for stage in ('stdp', 'regulation', 'net'):
            for suffix in ('update_count', 'synapse_update_count', 'l1', 'positive', 'negative', 'excluded_l1'):
                arrays.setdefault(f'{stage}_{suffix}', zeros.copy())
        cumulative = dict(arrays)
        for key, value in cumulative.items():
            arrays[f'epoch_{key}'] = value - self.previous.get(key, zeros)
        self.previous = {key: value.copy() for key, value in cumulative.items()}
        fan_in = self.support.sum(0).cpu().numpy()
        initial_l1 = torch.where(self.support, self.initial_weight, 0.).abs().sum(0).cpu().numpy()
        arrays.update(
            effective_fan_in=fan_in,
            initial_weight_l1=initial_l1,
            current_weight_l1=torch.where(self.support, weight, 0.).abs().sum(0).cpu().numpy(),
            net_displacement_l1=torch.where(self.support, weight - self.initial_weight, 0.).abs().sum(0).cpu().numpy(),
        )
        decay_mode = getattr(cortex, 'stdp_decay_mode', 'none')
        if decay_mode != 'none':
            decay_age = cortex.get_stdp_decay_age()
            decay_multiplier = cortex.get_stdp_decay_multiplier()
            if torch.is_tensor(decay_age):
                decay_age = decay_age.detach().cpu().numpy().copy()
            else:
                decay_age = np.full(columns, float(decay_age), dtype=np.float64)
            if torch.is_tensor(decay_multiplier):
                decay_multiplier = decay_multiplier.detach().cpu().numpy().copy()
            else:
                decay_multiplier = np.full(
                    columns,
                    float(decay_multiplier),
                    dtype=np.float64,
                )
            arrays['stdp_decay_age'] = decay_age
            arrays['stdp_decay_multiplier'] = decay_multiplier
        locations = 0 if self.spatial_fire is None else self.spatial_fire.numel() // columns
        arrays['fire_per_sample_per_location'] = arrays['fire_count'] / max(self.samples * locations, 1)
        for stage in ('stdp', 'regulation', 'net'):
            arrays[f'{stage}_l1_per_synapse'] = arrays[f'{stage}_l1'] / np.maximum(fan_in, 1)
            arrays[f'{stage}_l1_over_initial_l1'] = np.divide(
                arrays[f'{stage}_l1'], initial_l1,
                out=np.full(columns, np.nan), where=initial_l1 > 0,
            )
        metadata = {
            'epoch': epoch, 'cortex': cortex.cortex_id, 'samples': self.samples,
            'batches': self.batches, 'learning_pass_steps': self.steps,
            'epoch_samples': self.samples - self.previous_samples,
            'epoch_batches': self.batches - self.previous_batches,
            'spatial_locations_per_channel': locations, 'channels': columns,
            'learning_enabled': bool(cortex.learning_enabled),
            'potentiation_rate': float(cortex.potentiation_rate),
            'DP_ratio': float(cortex.DP_ratio),
            'depression_rate': float(cortex.depression_rate),
            'p_d_terms_scope': 'fast-path proposed terms before cancellation/preconditioning/normalization; root signed_ltp P term includes label sign, not purely LTP',
            'normalization_operation_counts': dict(getattr(
                cortex.kernel, 'normalization_operation_counts',
                {'post': 0, 'pre': 0, 'global_l1_sham': 0, 'none': 0},
            )),
            'count_scope': 'training learning pass only; hidden pre-pooling; shared receiver columns',
            'update_count_unit': 'applied update with any nonzero actual delta in receiver column',
            'total_unit': 'sum over applied updates and incoming synapses of abs(actual delta)',
            'regulation_scope': 'norm preservation plus kernel regulation, including normalization',
            'weight_support': 'learning mask intersect enabled connectivity; no-skip direct rows excluded and logged separately as excluded_l1',
            'stdp_decay_mode': decay_mode,
            'stdp_decay_configured_mode': getattr(
                cortex,
                'stdp_decay_configured_mode',
                decay_mode,
            ),
            'stdp_decay_cortex_ids': getattr(
                cortex,
                'stdp_decay_cortex_ids',
                None,
            ),
            'stdp_decay_reference': getattr(cortex, 'stdp_decay_reference', None),
            'stdp_decay_exponent': getattr(cortex, 'stdp_decay_exponent', 1.0),
            'stdp_decay_warmup_updates': getattr(
                cortex,
                'stdp_decay_warmup_updates',
                0,
            ),
            'stdp_decay_applied_update_count': getattr(
                cortex,
                'stdp_decay_applied_update_count',
                0,
            ),
            'stdp_decay_age_unit': (
                'cumulative actual pre-regulation STDP L1 / initial effective incoming-weight L1'
                if decay_mode == 'receiver_relative_stdp_l1'
                else (
                    'applied training update batches'
                    if decay_mode == 'global_update_count'
                    else 'none'
                )
            ),
        }
        self.previous_samples, self.previous_batches = self.samples, self.batches
        target = Path(writer.log_dir) / 'neuron_experience'
        target.mkdir(parents=True, exist_ok=True)
        stem = f'{cortex.cortex_id}_epoch{epoch:03d}'
        with (target / f'{stem}.csv').open('w', newline='', encoding='utf-8') as handle:
            output = csv.DictWriter(handle, fieldnames=['channel', *arrays])
            output.writeheader()
            for channel in range(columns):
                output.writerow({'channel': channel, **{key: value[channel] for key, value in arrays.items()}})
        (target / f'{stem}.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
        if self.spatial_fire is not None:
            np.savez_compressed(target / f'{stem}_spatial.npz',
                                fire_count=self.spatial_fire.cpu().numpy())
        for key, value in arrays.items():
            if key.startswith('epoch_') or key in (
                'fire_count',
                'stdp_l1',
                'regulation_l1',
                'net_displacement_l1',
                'stdp_decay_age',
                'stdp_decay_multiplier',
            ):
                prefix = f'neuron_experience/{cortex.cortex_id}/{key}'
                for statistic, number in (
                    ('mean', np.mean(value)), ('p50', np.quantile(value, .5)),
                    ('p90', np.quantile(value, .9)), ('max', np.max(value)),
                    ('zero_fraction', np.mean(value == 0)),
                ):
                    writer.add_scalar(f'{prefix}/{statistic}', float(number), epoch)
        writer.add_scalar(f'neuron_experience/{cortex.cortex_id}/samples', self.samples, epoch)


def configure_neuron_experience(model, settings, stdp_update_mode, simulation_forward_mode):
    enabled = (settings or {}).get('neuron_experience', False)
    if not isinstance(enabled, bool):
        raise ValueError('diagnostic_settings.neuron_experience must be bool')
    if enabled and (stdp_update_mode != 'two_pass_decision_contingent' or simulation_forward_mode != 'streaming'):
        raise ValueError('neuron experience currently requires streaming two_pass_decision_contingent')
    if enabled and model.self_cleaning_after_STDP:
        raise ValueError('neuron experience requires self_cleaning_after_STDP=false')
    for cortex in model.cortex.iter_cortex_tree():
        cortex.neuron_experience = NeuronExperience(cortex) if enabled else None


def log_neuron_experience(model, writer, epoch):
    for cortex in model.cortex.iter_cortex_tree():
        observer = getattr(cortex, 'neuron_experience', None)
        if observer is not None:
            observer.export(cortex, writer, epoch)
