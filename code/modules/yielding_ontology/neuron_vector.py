import math
import torch


class SpikeHolder():
    def __init__(self, length, spike_decay_rate, balanced_spike_trace, simulation_time):
        self.spike_decay_rate = spike_decay_rate
        self.length = length
        self.simulation_time = int(simulation_time)
        if self.simulation_time <= 0:
            raise ValueError('simulation_time must be positive')

        if balanced_spike_trace:
            self.spike_trace_factor = (1-spike_decay_rate)
        else:
            self.spike_trace_factor = 1

    def __len__(self):
        return self.length

    def init_state(self, outer_dim, device):
        self.spike_trace = torch.zeros([*outer_dim, self.length], device=device)
        self.current_spikes = torch.zeros([*outer_dim, self.length], device=device)
        self.spike_wave = torch.zeros([*outer_dim, self.length], device=device)
        self.earliness = torch.zeros([*outer_dim, self.length], device=device)

    def clear_state(self):
        self.spike_trace = None
        self.current_spikes = None
        self.spike_wave = None
        self.earliness = None

    def _update_earliness(self):
        if self.earliness is None:
            self.earliness = torch.zeros_like(self.spike_wave)
        self.earliness += self.spike_wave / float(self.simulation_time)

    def get_activity(self):
        return (
            self.current_spikes.reshape(*self.current_spikes.shape[:-2], -1),
            self.spike_trace.reshape(*self.spike_trace.shape[:-2], -1)
        )

    def stimulate_by_spike(self, current_spikes, is_training='this para is just dummy'):
        if self.spike_wave is None:
            self.spike_wave = torch.zeros_like(current_spikes)
        if self.spike_trace is None:
            self.spike_trace = torch.zeros_like(current_spikes)
        self.spike_wave = torch.maximum(self.spike_wave, current_spikes)
        self.spike_trace *= self.spike_decay_rate
        self.spike_trace += current_spikes * self.spike_trace_factor
        self.current_spikes = current_spikes
        self._update_earliness()

    def stimulate_by_wave(self, spike_wave, is_training='this para is just dummy'):
        if self.spike_wave is None:
            self.spike_wave = torch.zeros_like(spike_wave)
        if self.spike_trace is None:
            self.spike_trace = torch.zeros_like(spike_wave)
        current_spikes = spike_wave - self.spike_wave
        self.spike_wave = spike_wave
        self.spike_trace *= self.spike_decay_rate
        self.spike_trace += current_spikes * self.spike_trace_factor
        self.current_spikes = current_spikes
        self._update_earliness()

    def stimulate(self, current_spikes, is_training='this para is just dummy'):
        self.stimulate_by_spike(current_spikes, is_training=is_training)


class NeuronVector(SpikeHolder):
    def __init__(
        self, length, spike_trace_settings, reset_state, potential_decay_rate,
        threshold_settings,
        resting_length, simulation_time,
        noise_settings=None
    ):
        super().__init__(
            length,
            simulation_time=simulation_time,
            **spike_trace_settings
        )
        self.reset_state = reset_state
        self.potential_decay_rate = potential_decay_rate
        self.noise_settings = {
            'noise_rate': 0.0,
            'noise_type': 'or',
        }
        if noise_settings is not None:
            self.noise_settings.update(noise_settings)
        self.log_thresholds = torch.zeros(length)
        self.base_log_thresholds = self.log_thresholds.clone()

        self.resting_length = resting_length
        self.homeostasis_step_scale = 1.0
        self.current_potential = None

        def threshold_related_setup(
            homeostasis_ema_decay, log_thresholds_delta, upper_bound, lower_bound,
            target_mean_earliness=0.12,
            target_active_fraction=0.02,
            initial_threshold=1.0,
            threshold_update_rule='earliness_homeostasis',
            reset_training_threshold_each_epoch=False,
            s2_ncg_threshold_lr=0.0,
            s2_ncg_threshold_lr_anneal=1.0,
            s2_ncg_skip_silent_samples=True,
            active_earliness_signal_weight=1.0
        ):
            self.homeostasis_ema_decay = homeostasis_ema_decay
            self.log_thresholds_delta = log_thresholds_delta
            lower_bound = float(lower_bound)
            upper_bound = float(upper_bound)
            if lower_bound < 0:
                raise ValueError('threshold lower_bound must be >= 0')
            if upper_bound <= 0:
                raise ValueError('threshold upper_bound must be > 0')
            if lower_bound > upper_bound:
                raise ValueError(
                    'threshold lower_bound must be <= upper_bound'
                )
            self.log_thresholds_upper_bound = torch.tensor(math.log(upper_bound))
            self.log_thresholds_lower_bound = torch.tensor(
                -math.inf
                if lower_bound == 0
                else math.log(lower_bound)
            )
            self.threshold_min = lower_bound
            initial_threshold = float(initial_threshold)
            if initial_threshold <= 0:
                raise ValueError('initial_threshold must be > 0')
            if initial_threshold < lower_bound or initial_threshold > upper_bound:
                raise ValueError(
                    'initial_threshold must be within threshold lower/upper bounds'
                )
            initial_log_threshold = math.log(initial_threshold)
            self.log_thresholds = torch.full(
                (self.length,),
                initial_log_threshold,
                dtype=self.log_thresholds.dtype
            )
            self.base_log_thresholds = self.log_thresholds.clone()
            if threshold_update_rule not in {
                'earliness_homeostasis',
                'active_earliness_homeostasis',
                'active_fraction_homeostasis',
                'active_fraction_earliness_homeostasis',
                's2_ncg',
                'none',
            }:
                raise ValueError(
                    'threshold_update_rule must be one of '
                    '{"earliness_homeostasis", "active_earliness_homeostasis", '
                    '"active_fraction_homeostasis", '
                    '"active_fraction_earliness_homeostasis", '
                    '"s2_ncg", "none"}'
                )
            self.threshold_update_rule = threshold_update_rule
            self.reset_training_threshold_each_epoch = bool(
                reset_training_threshold_each_epoch
            )
            self.s2_ncg_threshold_lr = float(s2_ncg_threshold_lr)
            if self.s2_ncg_threshold_lr < 0:
                raise ValueError('s2_ncg_threshold_lr must be >= 0')
            self.s2_ncg_threshold_lr_anneal = float(s2_ncg_threshold_lr_anneal)
            if not 0.0 < self.s2_ncg_threshold_lr_anneal <= 1.0:
                raise ValueError(
                    's2_ncg_threshold_lr_anneal must satisfy 0 < value <= 1'
                )
            self.s2_ncg_current_threshold_lr = self.s2_ncg_threshold_lr
            self.s2_ncg_skip_silent_samples = bool(s2_ncg_skip_silent_samples)
            self.active_earliness_signal_weight = float(
                active_earliness_signal_weight
            )
            self.target_active_fraction = float(target_active_fraction)
            if not 0.0 < self.target_active_fraction <= 1.0:
                raise ValueError(
                    'target_active_fraction must satisfy 0 < value <= 1'
                )
            self._setup_target_mean_earliness(target_mean_earliness)

        threshold_related_setup(**threshold_settings)

    def _setup_target_mean_earliness(self, target_mean_earliness):
        self.target_mean_earliness_delay_ratio = None
        self.target_mean_earliness_min = None
        self.target_mean_earliness_max = None

        if isinstance(target_mean_earliness, dict):
            settings = dict(target_mean_earliness)
            delay_ratio = settings.pop('delay_ratio', None)
            fixed_target = settings.pop('fixed', None)
            min_target = settings.pop('min_target_earliness', None)
            max_target = settings.pop('max_target_earliness', None)
            if settings:
                unknown = ', '.join(sorted(settings))
                raise ValueError(
                    f'unknown target_mean_earliness settings: {unknown}'
                )
            if fixed_target is not None and delay_ratio is not None:
                raise ValueError(
                    'target_mean_earliness.fixed and delay_ratio are mutually exclusive'
                )
            if fixed_target is not None:
                if min_target is None:
                    min_target = fixed_target
                if max_target is None:
                    max_target = fixed_target
            if min_target is None or max_target is None:
                raise ValueError(
                    'target_mean_earliness must define min_target_earliness '
                    'and max_target_earliness'
                )
            min_target = float(min_target)
            max_target = float(max_target)
            if not 0.0 <= min_target <= max_target <= 1.0:
                raise ValueError(
                    'target_mean_earliness clamp must satisfy '
                    '0 <= min_target_earliness <= max_target_earliness <= 1'
                )
            if delay_ratio is None:
                if min_target != max_target:
                    raise ValueError(
                        'fixed target_mean_earliness requires '
                        'min_target_earliness == max_target_earliness'
                    )
                self.target_mean_earliness = min_target
            else:
                delay_ratio = float(delay_ratio)
                if delay_ratio <= 0:
                    raise ValueError(
                        'target_mean_earliness.delay_ratio must be > 0'
                    )
                self.target_mean_earliness = None
                self.target_mean_earliness_delay_ratio = delay_ratio
            self.target_mean_earliness_min = min_target
            self.target_mean_earliness_max = max_target
        else:
            self.target_mean_earliness = float(target_mean_earliness)
            if not 0.0 <= self.target_mean_earliness <= 1.0:
                raise ValueError('target_mean_earliness must be between 0 and 1')
            self.target_mean_earliness_min = self.target_mean_earliness
            self.target_mean_earliness_max = self.target_mean_earliness

        self.last_target_mean_earliness = torch.tensor(
            float(self.target_mean_earliness_min)
            if self.target_mean_earliness is None
            else float(self.target_mean_earliness)
        )

    def _resolve_target_mean_earliness(self, reference_mean_earliness, mean_earliness):
        if self.target_mean_earliness_delay_ratio is None:
            target = torch.as_tensor(
                self.target_mean_earliness,
                dtype=mean_earliness.dtype,
                device=mean_earliness.device
            )
        else:
            if reference_mean_earliness is None:
                raise ValueError(
                    'dynamic target_mean_earliness requires a sender mean '
                    'earliness reference'
                )
            target = torch.as_tensor(
                reference_mean_earliness,
                dtype=mean_earliness.dtype,
                device=mean_earliness.device
            )
            target = target * float(self.target_mean_earliness_delay_ratio)
            target = torch.clamp(
                target,
                min=float(self.target_mean_earliness_min),
                max=float(self.target_mean_earliness_max)
            )
        self.last_target_mean_earliness = target.detach()
        return target

    @property
    def thresholds(self):
        return torch.exp(self.log_thresholds)

    @property
    def base_thresholds(self):
        return torch.exp(self.base_log_thresholds)

    def get_thresholds_for_forward(self, use_training_thresholds=True):
        return self.thresholds

    def begin_training_epoch(self, epoch_index):
        if self.reset_training_threshold_each_epoch:
            self.log_thresholds = self.base_log_thresholds.clone()
            if hasattr(self, 'earliness_homeostasis_ema'):
                del self.earliness_homeostasis_ema

        epoch_offset = max(int(epoch_index) - 1, 0)
        self.s2_ncg_current_threshold_lr = (
            self.s2_ncg_threshold_lr
            * (self.s2_ncg_threshold_lr_anneal ** epoch_offset)
        )

    def _current_fire_time(self, dtype, device):
        return torch.tensor(
            float(self.simulation_time - self.temporal_step),
            dtype=dtype,
            device=device
        )

    def _adjust_earliness_thresholds(self, target_mean_earliness_reference=None):
        if self.threshold_update_rule not in {
            'earliness_homeostasis',
            'active_earliness_homeostasis',
            'active_fraction_homeostasis',
            'active_fraction_earliness_homeostasis',
        }:
            return
        reduce_dims = tuple(range(self.earliness.ndim - 1))
        if self.threshold_update_rule in {
            'active_fraction_homeostasis',
            'active_fraction_earliness_homeostasis',
        }:
            active_mask = self.earliness > 0
            active_count = active_mask.to(self.earliness.dtype).sum(
                dim=reduce_dims
            )
            active_slots = 1
            for dim in reduce_dims:
                active_slots *= self.earliness.shape[dim]
            active_fraction = active_count / float(active_slots)
            target_active_fraction = torch.ones_like(active_fraction) * (
                self.target_active_fraction
            )
            fraction_signal = active_fraction - target_active_fraction
            if self.threshold_update_rule == 'active_fraction_homeostasis':
                homeostasis_signal = fraction_signal
            else:
                active_sum = self.earliness.sum(dim=reduce_dims)
                mean_earliness = active_sum / active_count.clamp_min(1.0)
                target_mean_earliness = self._resolve_target_mean_earliness(
                    target_mean_earliness_reference,
                    mean_earliness
                )
                earliness_signal = (
                    mean_earliness
                    - target_mean_earliness
                )
                earliness_component = (
                    self.active_earliness_signal_weight * earliness_signal
                )
                homeostasis_signal = fraction_signal + earliness_component
        else:
            if self.threshold_update_rule == 'active_earliness_homeostasis':
                active_mask = self.earliness > 0
                active_count = active_mask.to(self.earliness.dtype).sum(
                    dim=reduce_dims
                )
                active_sum = self.earliness.sum(dim=reduce_dims)
                mean_earliness = active_sum / active_count.clamp_min(1.0)
            else:
                active_count = None
                mean_earliness = torch.mean(
                    self.earliness,
                    dim=reduce_dims
                )
            target_mean_earliness = self._resolve_target_mean_earliness(
                target_mean_earliness_reference,
                mean_earliness
            )
            homeostasis_signal = (
                mean_earliness
                - target_mean_earliness
            )
        if self.threshold_update_rule == 'active_earliness_homeostasis':
            active_now = active_count > 0
            homeostasis_signal = torch.where(
                active_now,
                homeostasis_signal,
                torch.zeros_like(homeostasis_signal)
            )

        if not hasattr(self, 'earliness_homeostasis_ema'):
            self.earliness_homeostasis_ema = homeostasis_signal
        else:
            self.earliness_homeostasis_ema = \
                self.homeostasis_ema_decay * self.earliness_homeostasis_ema \
                + (1.0 - self.homeostasis_ema_decay) * homeostasis_signal

        self.log_thresholds += (
            self.log_thresholds_delta
            * self.homeostasis_step_scale
            * self.earliness_homeostasis_ema
        )

        self.log_thresholds = torch.clamp(
            self.log_thresholds,
            min=self.log_thresholds_lower_bound,
            max=self.log_thresholds_upper_bound
        )

    def adjust_s2_ncg_thresholds(self, labels, num_classes, competition_group_N):
        if self.threshold_update_rule != 's2_ncg':
            return False
        lr = float(self.s2_ncg_current_threshold_lr)
        if lr == 0.0:
            return False
        if self.earliness is None:
            return False

        labels = labels.to(device=self.log_thresholds.device, dtype=torch.long)
        if labels.ndim != 1:
            raise ValueError('S2-NCG threshold update requires 1D labels')
        if self.earliness.shape[0] != labels.shape[0]:
            raise ValueError(
                'S2-NCG threshold update label count must match batch size'
            )

        earliness = self.earliness.detach().to(self.log_thresholds.device)
        if earliness.ndim == 2:
            channel_earliness = earliness
        else:
            channel_earliness = earliness.reshape(
                earliness.shape[0],
                -1,
                earliness.shape[-1]
            ).amax(dim=1)

        group_width = int(num_classes) * int(competition_group_N)
        if group_width <= 0 or self.length % group_width != 0:
            raise ValueError(
                'S2-NCG threshold update requires hidden length to be a '
                'multiple of num_classes * competition_group_N'
            )
        repeats = self.length // group_width
        batch_size = labels.shape[0]
        group_N = int(competition_group_N)
        grouped = channel_earliness.reshape(
            batch_size,
            repeats,
            int(num_classes),
            group_N
        )
        batch_indices = torch.arange(batch_size, device=labels.device)
        repeat_indices = torch.arange(repeats, device=labels.device)
        target_earliness = grouped[
            batch_indices.view(-1, 1),
            repeat_indices.view(1, -1),
            labels.view(-1, 1),
            :
        ]
        active = torch.ones(
            target_earliness.shape[:-1],
            device=labels.device,
            dtype=torch.bool
        )
        if self.s2_ncg_skip_silent_samples:
            active = target_earliness.amax(dim=-1) > 0

        winners = target_earliness.argmax(dim=-1)
        target_updates = torch.full(
            target_earliness.shape,
            -lr / float(group_N),
            device=labels.device,
            dtype=self.log_thresholds.dtype
        )
        target_updates.scatter_add_(
            -1,
            winners.unsqueeze(-1),
            torch.full(
                winners.unsqueeze(-1).shape,
                lr,
                device=labels.device,
                dtype=self.log_thresholds.dtype
            )
        )
        target_updates = target_updates * active.unsqueeze(-1).to(
            target_updates.dtype
        )

        sample_updates = torch.zeros(
            grouped.shape,
            device=labels.device,
            dtype=self.log_thresholds.dtype
        )
        sample_updates.scatter_add_(
            2,
            labels.view(batch_size, 1, 1, 1).expand(
                batch_size,
                repeats,
                1,
                group_N
            ),
            target_updates.unsqueeze(2)
        )
        total_updates = sample_updates.sum(dim=0)

        threshold_values = self.thresholds.detach().reshape(
            repeats,
            int(num_classes),
            group_N
        )
        base_values = self.base_thresholds.detach().reshape(
            repeats,
            int(num_classes),
            group_N
        )
        new_values = threshold_values + total_updates
        upper_bound = torch.exp(self.log_thresholds_upper_bound).to(
            device=new_values.device,
            dtype=new_values.dtype
        )
        new_values = torch.maximum(new_values, base_values)
        new_values = torch.minimum(new_values, upper_bound)
        self.log_thresholds = torch.log(new_values.reshape(-1).clamp_min(1.0e-12))
        return True

    def init_state(self, outer_dim, device):
        self.temporal_step = 0
        self.current_potential = None
        super().init_state(outer_dim, device)

    def clear_state(self):
        self.temporal_step = 0
        self.current_potential = None
        super().clear_state()

    def _add_noise(self, spike_vector, noise_type, noise_rate):
        if noise_rate == 0:
            return spike_vector
        # Generate a noise vector of the same shape as spike_vector with random values
        noise_vector = torch.rand(spike_vector.shape, device=spike_vector.device) < noise_rate
        bool_spike_vector = spike_vector.bool()

        if noise_type == 'or':
            bool_noisy_spikes = torch.logical_or(bool_spike_vector, noise_vector)
        elif noise_type == 'xor':
            bool_noisy_spikes = torch.logical_xor(bool_spike_vector, noise_vector)
        else:
            raise ValueError(f'Unsupported noise type: {noise_type}')

        noisy_spikes = bool_noisy_spikes.float()
        return noisy_spikes

    def _apply_current_top_k(self, candidate_wave, potential, top_k, used_wave=None):
        if top_k is None:
            return candidate_wave

        top_k = int(top_k)
        if top_k >= candidate_wave.shape[-1]:
            return candidate_wave

        if used_wave is None:
            used_wave = self.spike_wave
        new_candidate = candidate_wave.bool() & ~(used_wave > 0)
        scores = potential.masked_fill(~new_candidate, -torch.inf)
        if top_k == 1:
            top_values, top_indices = torch.max(scores, dim=-1, keepdim=True)
            selected_candidates = torch.zeros_like(new_candidate)
            selected_candidates.scatter_(
                -1,
                top_indices,
                torch.isfinite(top_values)
            )
            del scores, top_values, top_indices, new_candidate
            return candidate_wave.mul_(selected_candidates)

        top_values, top_indices = torch.topk(
            scores,
            top_k,
            dim=-1,
            sorted=False
        )
        finite_top_values = torch.isfinite(top_values)
        del scores, top_values
        selected_candidates = torch.zeros_like(new_candidate)
        selected_candidates.scatter_(
            -1,
            top_indices,
            finite_top_values
        )
        del top_indices, finite_top_values, new_candidate
        return candidate_wave.mul_(selected_candidates)

    def stimulate_by_potential(
        self, potential, is_training=False, top_k=None,
        target_mean_earliness_reference=None,
        use_training_thresholds=None
    ):
        self.current_potential = potential
        if use_training_thresholds is None:
            use_training_thresholds = is_training
        TH = self.get_thresholds_for_forward(
            use_training_thresholds=use_training_thresholds
        )

        candidate_wave = (potential >= TH).float()
        candidate_wave = self._add_noise(candidate_wave, **self.noise_settings)
        candidate_wave = self._apply_current_top_k(
            candidate_wave,
            potential,
            top_k,
            used_wave=self.spike_wave
        )

        previous_spike_wave = self.spike_wave
        spike_wave = torch.maximum(previous_spike_wave, candidate_wave)
        self.current_spikes = spike_wave - previous_spike_wave
        self.spike_wave = spike_wave
        self.spike_trace *= self.spike_decay_rate
        self.spike_trace += self.current_spikes * self.spike_trace_factor
        self._update_earliness()

        self.temporal_step += 1
        if is_training and self.temporal_step >= self.simulation_time:
            self._adjust_earliness_thresholds(
                target_mean_earliness_reference=target_mean_earliness_reference
            )

    def stimulate(self, input_spikes, is_training=False, top_k=None):
        self.stimulate_by_potential(
            input_spikes,
            is_training=is_training,
            top_k=top_k
        )

    def delete_small_threshold_neurons(self):
        # Create a boolean mask for thresholds greater than the minimal threshold
        # !!! remove equal neurons because all thresholds lower than self.threshold_min will be clamp to threshold_min !!!
        valid_thresholds_mask = self.thresholds > self.threshold_min

        # Find the indices where thresholds are smaller than the minimal threshold
        removed_indices = torch.where(~valid_thresholds_mask)[0]

        # Apply the mask to log_thresholds to filter out small thresholds
        self.log_thresholds = self.log_thresholds[valid_thresholds_mask]
        self.length = len(self.log_thresholds)

        return removed_indices
