import copy
import math
import torch
from modules.yielding_ontology.cortex_kernel import CortexKernel
from modules.yielding_ontology.cortex_pooling import CortexPooling
from modules.yielding_ontology.neuron_vector import NeuronVector, SpikeHolder


class BaseCortex():
    FIXED_AMPLIFIER_HOMEOSTASIS_EMA_DECAY = 0.8
    FIXED_AMPLIFIER_THRESHOLD_TARGET = 1.0

    def __init__(
        self, cortex_id,
        kernel_size, stride,
        input_channel, output_channel,
        neuron_vectors_spec,
        spike_trace_settings,
        base_cortex_settings,
        simulation_time,
        subcortexs=None,
        kernel=None,
        init_amplifier=1,
        pooling_settings=None,
        competition_group_N=None,
        competition_group_aggregation_stage=None,
        cover_edges=False,
    ):

        self.simulation_time = int(simulation_time)

        kernel_settings = base_cortex_settings['kernel_settings']
        normalize_settings = copy.deepcopy(kernel_settings['normalize_settings'])
        normalize_defaults = {
            'dim_0_freq': 0,
            'dim_1_freq': 0,
            'freq_diff': 0,
            'norm_p': 1,
            'out_weight_len': 1.0,
            'use_shape_ratio': False,
            'dim_0_mode': 'per_receiver',
            'dim_1_mode': 'per_sender',
        }
        for setting_name, default_value in normalize_defaults.items():
            normalize_settings[setting_name] = self._resolve_cortex_setting(
                normalize_settings,
                setting_name,
                cortex_id,
                default=default_value,
            )
            normalize_settings.pop(f'{setting_name}_by_cortex', None)
        amplifier_settings = base_cortex_settings['amplifier_settings']
        lateral_inhibition_settings = base_cortex_settings.get(
            'lateral_inhibition'
        )
        if lateral_inhibition_settings is not None:
            if not isinstance(lateral_inhibition_settings, dict):
                raise ValueError(
                    'base_cortex_settings.lateral_inhibition must be a mapping'
                )
            if (
                'lateral_inhibition_factor' in base_cortex_settings
                or 'lateral_inhibition_power' in base_cortex_settings
            ):
                raise ValueError(
                    'use base_cortex_settings.lateral_inhibition.factor/power '
                    'instead of scalar lateral_inhibition_factor/power'
                )
            self.lateral_inhibition_factor = float(self._resolve_cortex_setting(
                lateral_inhibition_settings,
                'factor',
                cortex_id,
                default=0.0
            ))
            self.lateral_inhibition_power = float(self._resolve_cortex_setting(
                lateral_inhibition_settings,
                'power',
                cortex_id,
                default=1.0
            ))
        else:
            self.lateral_inhibition_factor = float(self._resolve_cortex_setting(
                base_cortex_settings,
                'lateral_inhibition_factor',
                cortex_id,
                default=base_cortex_settings.get('lateral_inhibition_factor', 0.0)
            ))
            self.lateral_inhibition_power = float(self._resolve_cortex_setting(
                base_cortex_settings,
                'lateral_inhibition_power',
                cortex_id,
                default=1.0
            ))
        if self.lateral_inhibition_power <= 0:
            raise ValueError(
                'base_cortex_settings.lateral_inhibition.power must be > 0'
            )
        removed_queue_settings = {
            'dominance_inhibition_source',
            'shadow_queue_occupancy_top_k',
            'admission_queue_occupancy_top_k',
            'admission_queue_occupancy_source',
        } & set(base_cortex_settings)
        if removed_queue_settings:
            removed = ', '.join(sorted(removed_queue_settings))
            raise ValueError(
                'group-cap replacement settings were removed: '
                f'{removed}'
            )
        removed_base_settings = {
            'unify_thresholds',
            'unify_thresholds_mode',
        } & set(base_cortex_settings)
        if removed_base_settings:
            removed = ', '.join(sorted(removed_base_settings))
            raise ValueError(
                'removed base_cortex_settings field(s): '
                f'{removed}'
            )
        removed_amplifier_settings = {
            'mean_threshold_ema_decay',
            'optimal_output_thresholds',
        } & set(amplifier_settings)
        if removed_amplifier_settings:
            removed = ', '.join(sorted(removed_amplifier_settings))
            raise ValueError(
                'removed amplifier_settings field(s): '
                f'{removed}'
            )
        self.dominance_inhibition_factor = float(self._resolve_cortex_setting(
            base_cortex_settings,
            'dominance_inhibition_factor',
            cortex_id,
            default=0.0
        ))
        self.dominance_inhibition_power = float(self._resolve_cortex_setting(
            base_cortex_settings,
            'dominance_inhibition_power',
            cortex_id,
            default=1.0
        ))
        if self.dominance_inhibition_power <= 0:
            raise ValueError(
                'base_cortex_settings.dominance_inhibition_power must be > 0'
            )
        self.top_k = self._resolve_cortex_setting(
            base_cortex_settings,
            'top_k',
            cortex_id,
            default=None
        )
        if self.top_k is not None:
            self.top_k = int(self.top_k)
            if self.top_k <= 0:
                raise ValueError('base_cortex_settings.top_k must be positive or null')
        input_wave_delay_steps = self._resolve_cortex_setting(
            base_cortex_settings,
            'input_wave_delay_steps',
            cortex_id,
            default=0
        )
        input_wave_delay_steps = int(input_wave_delay_steps)
        if input_wave_delay_steps < 0:
            raise ValueError(
                'base_cortex_settings.input_wave_delay_steps must be >= 0'
            )
        input_wave_delay_per_subcortex_depth_steps = (
            self._resolve_cortex_setting(
                base_cortex_settings,
                'input_wave_delay_per_subcortex_depth_steps',
                cortex_id,
                default=0
            )
        )
        input_wave_delay_per_subcortex_depth_steps = int(
            input_wave_delay_per_subcortex_depth_steps
        )
        if input_wave_delay_per_subcortex_depth_steps < 0:
            raise ValueError(
                'base_cortex_settings.'
                'input_wave_delay_per_subcortex_depth_steps must be >= 0'
            )
        input_wave_handoff_branch_activity_fraction = (
            self._resolve_cortex_setting(
                base_cortex_settings,
                'input_wave_handoff_branch_activity_fraction',
                cortex_id,
                default=None,
            )
        )
        if input_wave_handoff_branch_activity_fraction is not None:
            input_wave_handoff_branch_activity_fraction = float(
                input_wave_handoff_branch_activity_fraction
            )
            if not 0.0 < input_wave_handoff_branch_activity_fraction <= 1.0:
                raise ValueError(
                    'base_cortex_settings.'
                    'input_wave_handoff_branch_activity_fraction must be '
                    'in (0, 1] or null'
                )
        subcortex_handoff_mode = str(self._resolve_cortex_setting(
            base_cortex_settings,
            'subcortex_handoff_mode',
            cortex_id,
            default='cumulative',
        ))
        if subcortex_handoff_mode not in {
            'cumulative', 'onset', 'balanced_trace'
        }:
            raise ValueError(
                'subcortex_handoff_mode must be cumulative, onset, or '
                'balanced_trace'
            )
        subcortex_handoff_scale = float(self._resolve_cortex_setting(
            base_cortex_settings,
            'subcortex_handoff_scale',
            cortex_id,
            default=1.0,
        ))
        if not math.isfinite(subcortex_handoff_scale) or subcortex_handoff_scale <= 0:
            raise ValueError('subcortex_handoff_scale must be finite and > 0')
        temporal_receiver_phase_count = int(self._resolve_cortex_setting(
            base_cortex_settings,
            'temporal_receiver_phase_count',
            cortex_id,
            default=1,
        ))
        if temporal_receiver_phase_count <= 0:
            raise ValueError(
                'base_cortex_settings.temporal_receiver_phase_count '
                'must be positive'
            )
        temporal_receiver_phase_basis = str(self._resolve_cortex_setting(
            base_cortex_settings,
            'temporal_receiver_phase_basis',
            cortex_id,
            default='clock',
        ))
        if temporal_receiver_phase_basis not in {'clock', 'event_progress'}:
            raise ValueError(
                'temporal_receiver_phase_basis must be clock or '
                'event_progress'
            )

        self.first_spike_delay_factor = self._resolve_cortex_setting(
            base_cortex_settings,
            'first_spike_delay_factor',
            cortex_id,
            default=None
        )
        self.mean_spike_delay_factor = self._resolve_cortex_setting(
            base_cortex_settings,
            'mean_spike_delay_factor',
            cortex_id,
            default=None
        )
        if (
            self.first_spike_delay_factor is None
            and self.mean_spike_delay_factor is not None
        ) or (
            self.first_spike_delay_factor is not None
            and self.mean_spike_delay_factor is None
        ):
            raise ValueError(
                'base_cortex_settings.first_spike_delay_factor and '
                'mean_spike_delay_factor must be set together'
            )
        self.uses_dynamic_spike_homeostasis = (
            self.first_spike_delay_factor is not None
        )
        if self.uses_dynamic_spike_homeostasis:
            self.first_spike_delay_factor = float(self.first_spike_delay_factor)
            self.mean_spike_delay_factor = float(self.mean_spike_delay_factor)
            if self.first_spike_delay_factor <= 0:
                raise ValueError(
                    'base_cortex_settings.first_spike_delay_factor must be > 0'
                )
            if self.mean_spike_delay_factor <= 0:
                raise ValueError(
                    'base_cortex_settings.mean_spike_delay_factor must be > 0'
                )

        # ======== amplifier_settings ========
        self.amplifier_first_spike_target_settings = None
        self.threshold_target_mean_first_spike_factor = None
        amplifier_first_spike_target = self._resolve_cortex_setting(
            amplifier_settings,
            'target_mean_first_spike_earliness',
            cortex_id,
            default=None
        )
        if isinstance(amplifier_first_spike_target, dict):
            self.amplifier_target_mean_first_spike_earliness = None
            self.amplifier_first_spike_target_settings = (
                self._normalize_source_relative_first_spike_target(
                    amplifier_first_spike_target,
                    'base_cortex_settings.amplifier_settings.'
                    'target_mean_first_spike_earliness'
                )
            )
        else:
            self.amplifier_target_mean_first_spike_earliness = (
                amplifier_first_spike_target
            )
        if self.amplifier_target_mean_first_spike_earliness is not None:
            self.amplifier_target_mean_first_spike_earliness = float(
                self.amplifier_target_mean_first_spike_earliness
            )
            if not 0.0 <= self.amplifier_target_mean_first_spike_earliness <= 1.0:
                raise ValueError(
                    'base_cortex_settings.amplifier_settings.'
                    'target_mean_first_spike_earliness must be between 0 and 1'
                )
        self.uses_first_spike_amplifier_homeostasis = (
            self.uses_dynamic_spike_homeostasis
            or self.amplifier_first_spike_target_settings is not None
            or self.amplifier_target_mean_first_spike_earliness is not None
        )
        self.amplifier_first_spike_observation_mode = str(
            self._resolve_cortex_setting(
                amplifier_settings,
                'first_spike_observation_mode',
                cortex_id,
                default='sample_max_first_spike'
            )
        )
        valid_observation_modes = {
            'sample_max_first_spike',
            'hidden_mean_earliness',
        }
        if self.amplifier_first_spike_observation_mode not in valid_observation_modes:
            valid_modes = ', '.join(sorted(valid_observation_modes))
            raise ValueError(
                'base_cortex_settings.amplifier_settings.'
                'first_spike_observation_mode must be one of: '
                f'{valid_modes}'
            )
        self.log_amplifier_delta = float(self._resolve_cortex_setting(
            amplifier_settings,
            'log_amplifier_delta',
            cortex_id
        ))
        amplifier_lower_bound = float(self._resolve_cortex_setting(
            amplifier_settings,
            'lower_bound',
            cortex_id
        ))
        amplifier_upper_bound = float(self._resolve_cortex_setting(
            amplifier_settings,
            'upper_bound',
            cortex_id
        ))
        if amplifier_lower_bound < 0:
            raise ValueError('amplifier lower_bound must be >= 0')
        if amplifier_upper_bound <= 0:
            raise ValueError('amplifier upper_bound must be > 0')
        if amplifier_lower_bound > amplifier_upper_bound:
            raise ValueError(
                'amplifier lower_bound must be <= upper_bound'
            )
        self.log_amplifier_lower_bound = torch.tensor(
            -math.inf
            if amplifier_lower_bound == 0
            else math.log(amplifier_lower_bound)
        )
        self.log_amplifier_upper_bound = torch.tensor(
            math.log(amplifier_upper_bound)
        )
        self.amplifier_homeostasis_ema_decay = (
            self.FIXED_AMPLIFIER_HOMEOSTASIS_EMA_DECAY
        )
        self.amplifier_threshold_target = (
            self.FIXED_AMPLIFIER_THRESHOLD_TARGET
        )

        self.log_amplifier = torch.tensor(float(init_amplifier)).log()
        self.cortex_id = cortex_id
        self.learning_enabled = True
        self.homeostasis_enabled = True
        self.amplifier_homeostasis_enabled = True
        self.kernel_regulation_enabled = True
        self.kernel_norm_preservation_mode = 'none'
        self.kernel_update_preconditioner_mode = 'none'
        self.kernel_update_preconditioner = None
        self.kernel_update_sender_activity_rms = None
        self.kernel_update_preconditioner_observed_min = None
        self.kernel_update_preconditioner_observed_max = None
        self.kernel_update_preconditioner_l2_ratio = None
        self.kernel_update_class_contrast_removed_fraction = None
        self.kernel_update_class_contrast_mean_abs_max = None
        self.kernel_update_class_contrast_centered_mean_abs_max = None
        self.kernel_update_class_contrast_l2_ratio_before_restore = None
        self.kernel_update_class_contrast_l2_ratio_after_restore = None
        self.kernel_update_class_contrast_cosine = None
        self.kernel_update_class_contrast_support_fraction = None
        self.kernel_update_class_contrast_support_count_mean = None
        self.kernel_update_class_contrast_support_centerable_fraction = None
        self.kernel_update_class_contrast_support_centerable_fraction_max = None
        self.kernel_update_class_contrast_support_singleton_fraction = None
        self.kernel_update_class_contrast_support_singleton_l2_fraction = None
        self.kernel_update_class_contrast_support_singleton_l2_fraction_max = None
        self.kernel_update_class_contrast_support_full_fraction = None
        self.kernel_update_class_contrast_support_leak_fraction = None
        self.kernel_update_class_contrast_support_leak_abs_max = None
        self.kernel_update_class_contrast_signed_reconstruction_error = None
        self.kernel_update_class_contrast_signed_positive_l2_fraction = None
        self.kernel_update_class_contrast_signed_negative_l2_fraction = None
        self.kernel_update_class_contrast_total_global_scale = None
        self.kernel_update_class_contrast_total_degenerate_fallback = None
        self.kernel_update_class_contrast_total_degenerate_fallback_count = 0
        self.kernel_learning_mask = None
        self.kernel_size = kernel_size
        self.stride = stride
        self.cover_edges = bool(cover_edges)
        self.output_channel = int(output_channel)
        self.competition_group_N = (
            1 if competition_group_N is None else int(competition_group_N)
        )
        if competition_group_aggregation_stage is None:
            competition_group_aggregation_stage = 'after_pooling'
        if competition_group_aggregation_stage != 'after_pooling':
            raise ValueError(
                'competition_group_aggregation_stage is fixed to "after_pooling"; '
                'remove the field from YAML'
            )
        self.competition_group_aggregation_stage = 'after_pooling'
        if self.output_channel <= 0:
            raise ValueError('output_channel must be positive')
        if self.competition_group_N <= 0:
            raise ValueError('competition_group_N must be positive or null')
        if self.competition_group_N % temporal_receiver_phase_count != 0:
            raise ValueError(
                'temporal_receiver_phase_count must divide '
                'competition_group_N'
            )
        self.hidden_channel = self.output_channel * self.competition_group_N
        self.temporal_receiver_phase_count = temporal_receiver_phase_count
        self.temporal_receiver_phase_basis = temporal_receiver_phase_basis
        self.temporal_receiver_active_phase = 0
        self.temporal_receiver_active_fraction = (
            1.0 / float(temporal_receiver_phase_count)
        )
        self.subcortex_handoff_mode = subcortex_handoff_mode
        self.subcortex_handoff_scale = subcortex_handoff_scale
        self.latest_subcortex_handoff_waves = []
        self.pooling = CortexPooling(pooling_settings)
        self.homeostasis_batch_scale = 1.0

        input_len = (kernel_size**2) * input_channel
        self.input_nv = SpikeHolder(
            input_len,
            simulation_time=simulation_time,
            **spike_trace_settings
        )
        cortex_neuron_vectors_spec = self._prepare_neuron_vectors_spec_for_cortex(
            neuron_vectors_spec,
            cortex_id
        )
        self.hidden_nv = NeuronVector(
            self.hidden_channel, spike_trace_settings,
            simulation_time=simulation_time,
            **cortex_neuron_vectors_spec
        )
        self.output_nv = SpikeHolder(
            self.output_channel,
            simulation_time=simulation_time,
            **spike_trace_settings
        )

        if subcortexs is None:
            self.subcortexs = []
            self.subcortex_counter = 0
        else:
            self.subcortexs = subcortexs
            self.subcortex_counter = len(subcortexs)
        self.input_wave_delay_base_steps = input_wave_delay_steps
        self.input_wave_delay_per_subcortex_depth_steps = (
            input_wave_delay_per_subcortex_depth_steps
        )
        self.input_wave_delay_subcortex_depth = self._max_subcortex_depth(
            self.subcortexs
        )
        self.input_wave_delay_steps = (
            self.input_wave_delay_base_steps
            + (
                self.input_wave_delay_per_subcortex_depth_steps
                * self.input_wave_delay_subcortex_depth
            )
        )
        self._input_wave_delay_history = []
        self.input_wave_handoff_branch_activity_fraction = (
            input_wave_handoff_branch_activity_fraction
        )
        self.input_wave_handoff_closed_fraction = 0.0
        self.current_excitatory_potential = None

        self.kernel = CortexKernel(
            input_len,
            self.hidden_channel,
            self.subcortexs,
            self.kernel_size,
            self.kernel_size,
            kernel_settings['init_std'],
            kernel_settings['init_mean'],
            no_skip_links=kernel_settings['no_skip_links'],
            non_negative_weights=kernel_settings['non_negative_weights'],
            normalize_settings=normalize_settings,
            minimal_weight=kernel_settings['minimal_weight'],
            kernel=kernel
        )
        self._cached_init_input_shape = None
        self._cached_init_patch_shape = None
        self._cached_init_patch_outer_dim = None
        self._cached_init_patch_grid = None
        self.sender_mean_earliness = None
        self.input_source_mean_first_spike_earliness = None
        self.latest_input_mean_first_spike_earliness = None
        self.latest_subcortex_handoff_waves = []
        self.mean_first_spike_earliness = None
        self.hidden_mean_first_spike_earliness = None
        self.output_mean_first_spike_earliness = None
        self.optimal_first_spike_earliness = None
        self.optimal_mean_spike_earliness = None

    @staticmethod
    def _resolve_cortex_setting(settings, key, cortex_id, default=None):
        # Paper baselines use output-only homeostasis targets and rates.
        by_cortex = settings.get(f'{key}_by_cortex')
        if by_cortex is not None:
            if cortex_id in by_cortex:
                return by_cortex[cortex_id]
            if '*' in by_cortex:
                return by_cortex['*']
        return settings.get(key, default)

    @staticmethod
    def _max_subcortex_depth(subcortexs):
        if not subcortexs:
            return 0
        return 1 + max(
            BaseCortex._max_subcortex_depth(subcortex.subcortexs)
            for subcortex in subcortexs
        )

    @staticmethod
    def _normalize_source_relative_first_spike_target(settings, setting_name):
        settings = dict(settings)
        delay_factor = settings.pop('delay_factor', None)
        single_source_delay_factor = settings.pop(
            'single_source_delay_factor',
            delay_factor
        )
        multi_source_delay_factor = settings.pop(
            'multi_source_delay_factor',
            delay_factor
        )
        min_target = float(settings.pop('min_target_earliness', 0.0))
        max_target = float(settings.pop('max_target_earliness', 1.0))
        source_reference = settings.pop('source_reference', 'slowest')
        if settings:
            unknown = ', '.join(sorted(settings))
            raise ValueError(f'unknown {setting_name} setting(s): {unknown}')
        if single_source_delay_factor is None or multi_source_delay_factor is None:
            raise ValueError(
                f'{setting_name} requires delay_factor or both '
                'single_source_delay_factor and multi_source_delay_factor'
            )
        single_source_delay_factor = float(single_source_delay_factor)
        multi_source_delay_factor = float(multi_source_delay_factor)
        if single_source_delay_factor <= 0 or multi_source_delay_factor <= 0:
            raise ValueError(
                f'{setting_name} delay factors must be > 0'
            )
        if not 0.0 <= min_target <= max_target <= 1.0:
            raise ValueError(
                f'{setting_name} clamp must satisfy '
                '0 <= min_target_earliness <= max_target_earliness <= 1'
            )
        if source_reference not in {'slowest', 'mean', 'fastest'}:
            raise ValueError(
                f'{setting_name}.source_reference must be one of '
                '{"slowest", "mean", "fastest"}'
            )
        return {
            'single_source_delay_factor': single_source_delay_factor,
            'multi_source_delay_factor': multi_source_delay_factor,
            'min_target_earliness': min_target,
            'max_target_earliness': max_target,
            'source_reference': source_reference,
        }

    def _prepare_neuron_vectors_spec_for_cortex(
        self, neuron_vectors_spec, cortex_id
    ):
        spec = copy.deepcopy(neuron_vectors_spec)
        removed_keys = {
            'input_activaton',
            'top_k_used_wave_source',
            'threshold_weighting_power',
        } & set(spec)
        if removed_keys:
            removed = ', '.join(sorted(removed_keys))
            raise ValueError(
                'removed neuron_vectors_spec setting(s): '
                f'{removed}'
            )
        threshold_settings = spec.get('threshold_settings')
        if threshold_settings is None:
            return spec

        first_spike_factor = self._resolve_cortex_setting(
            threshold_settings,
            'target_mean_first_spike_factor',
            self.cortex_id,
            default=None
        )
        threshold_settings.pop('target_mean_first_spike_factor', None)
        threshold_settings.pop('target_mean_first_spike_factor_by_cortex', None)
        if first_spike_factor is not None:
            if (
                self.amplifier_target_mean_first_spike_earliness is None
                and self.amplifier_first_spike_target_settings is None
            ):
                raise ValueError(
                    'threshold_settings.target_mean_first_spike_factor requires '
                    'base_cortex_settings.amplifier_settings.'
                    'target_mean_first_spike_earliness for the same cortex'
                )
            first_spike_factor = float(first_spike_factor)
            if first_spike_factor <= 0:
                raise ValueError(
                    'threshold_settings.target_mean_first_spike_factor must be > 0'
                )
            if self.amplifier_first_spike_target_settings is None:
                target_mean_earliness = (
                    float(self.amplifier_target_mean_first_spike_earliness)
                    * first_spike_factor
                )
                if not 0.0 <= target_mean_earliness <= 1.0:
                    raise ValueError(
                        'threshold_settings.target_mean_first_spike_factor produces '
                        'a target_mean_earliness outside [0, 1]'
                    )
                threshold_settings['target_mean_earliness'] = target_mean_earliness
            else:
                self.threshold_target_mean_first_spike_factor = first_spike_factor
                threshold_settings['target_mean_earliness'] = {
                    'delay_ratio': 1.0,
                    'min_target_earliness': 0.0,
                    'max_target_earliness': 1.0,
                }

        return spec

    @property
    def shape(self):
        return [len(self.input_nv), len(self.hidden_nv)]

    def iter_cortex_tree(self):
        yield self
        for subcortex in self.subcortexs:
            yield from subcortex.iter_cortex_tree()

    def set_learning_enabled_tree(self, enabled=True, cortex_ids=None):
        cortex_ids = None if cortex_ids is None else set(cortex_ids)
        for cortex in self.iter_cortex_tree():
            if cortex_ids is None or cortex.cortex_id in cortex_ids:
                cortex.learning_enabled = bool(enabled)

    def set_homeostasis_enabled_tree(self, enabled=True, cortex_ids=None):
        cortex_ids = None if cortex_ids is None else set(cortex_ids)
        for cortex in self.iter_cortex_tree():
            if cortex_ids is None or cortex.cortex_id in cortex_ids:
                cortex.homeostasis_enabled = bool(enabled)

    def set_amplifier_homeostasis_enabled_tree(
        self, enabled=True, cortex_ids=None
    ):
        cortex_ids = None if cortex_ids is None else set(cortex_ids)
        for cortex in self.iter_cortex_tree():
            if cortex_ids is None or cortex.cortex_id in cortex_ids:
                cortex.amplifier_homeostasis_enabled = bool(enabled)

    def set_kernel_regulation_enabled_tree(self, enabled=True, cortex_ids=None):
        cortex_ids = None if cortex_ids is None else set(cortex_ids)
        for cortex in self.iter_cortex_tree():
            if cortex_ids is None or cortex.cortex_id in cortex_ids:
                cortex.kernel_regulation_enabled = bool(enabled)

    def set_kernel_norm_preservation_mode_tree(
        self, mode='none', cortex_ids=None
    ):
        if mode not in {'none', 'global_l2', 'per_output_l2'}:
            raise ValueError(
                'kernel norm preservation mode must be one of '
                'none/global_l2/per_output_l2'
            )
        cortex_ids = None if cortex_ids is None else set(cortex_ids)
        for cortex in self.iter_cortex_tree():
            if cortex_ids is None or cortex.cortex_id in cortex_ids:
                cortex.kernel_norm_preservation_mode = mode

    def set_kernel_update_preconditioner_mode_tree(
        self, mode='none', cortex_ids=None
    ):
        if mode not in {
            'none',
            'feature_row_relative',
            'feature_row_update_rms',
            'feature_row_sender_activity_rms',
            'feature_row_sender_activity_rms_class_contrast_supported',
        }:
            raise ValueError(
                'kernel update preconditioner mode must be one of '
                'none/feature_row_relative/feature_row_update_rms/'
                'feature_row_sender_activity_rms/'
                'feature_row_sender_activity_rms_class_contrast_supported'
            )
        cortex_ids = None if cortex_ids is None else set(cortex_ids)
        for cortex in self.iter_cortex_tree():
            if cortex_ids is not None and cortex.cortex_id not in cortex_ids:
                continue
            cortex.kernel_update_preconditioner_mode = mode
            cortex.kernel_update_preconditioner = None
            cortex.kernel_update_sender_activity_rms = None
            cortex.kernel_update_preconditioner_observed_min = None
            cortex.kernel_update_preconditioner_observed_max = None
            cortex.kernel_update_preconditioner_l2_ratio = None
            cortex.kernel_update_class_contrast_removed_fraction = None
            cortex.kernel_update_class_contrast_mean_abs_max = None
            cortex.kernel_update_class_contrast_centered_mean_abs_max = None
            cortex.kernel_update_class_contrast_l2_ratio_before_restore = None
            cortex.kernel_update_class_contrast_l2_ratio_after_restore = None
            cortex.kernel_update_class_contrast_cosine = None
            cortex.kernel_update_class_contrast_support_fraction = None
            cortex.kernel_update_class_contrast_support_count_mean = None
            cortex.kernel_update_class_contrast_support_centerable_fraction = None
            cortex.kernel_update_class_contrast_support_centerable_fraction_max = None
            cortex.kernel_update_class_contrast_support_singleton_fraction = None
            cortex.kernel_update_class_contrast_support_singleton_l2_fraction = None
            cortex.kernel_update_class_contrast_support_singleton_l2_fraction_max = None
            cortex.kernel_update_class_contrast_support_full_fraction = None
            cortex.kernel_update_class_contrast_support_leak_fraction = None
            cortex.kernel_update_class_contrast_support_leak_abs_max = None
            cortex.kernel_update_class_contrast_signed_reconstruction_error = None
            cortex.kernel_update_class_contrast_signed_positive_l2_fraction = None
            cortex.kernel_update_class_contrast_signed_negative_l2_fraction = None
            cortex.kernel_update_class_contrast_total_global_scale = None
            cortex.kernel_update_class_contrast_total_degenerate_fallback = None
            cortex.kernel_update_class_contrast_total_degenerate_fallback_count = 0
            if mode == 'none':
                continue
            if mode in {
                'feature_row_update_rms',
                'feature_row_sender_activity_rms',
                'feature_row_sender_activity_rms_class_contrast_supported',
            }:
                continue
            weight = cortex.kernel.weight.detach()
            mask = getattr(cortex, 'kernel_learning_mask', None)
            if mask is None:
                learned_weight = weight
                learned_rows = torch.ones(
                    weight.shape[0], dtype=torch.bool, device=weight.device
                )
            else:
                learned_mask = mask.to(dtype=torch.bool, device=weight.device)
                learned_weight = torch.where(learned_mask, weight, 0.0)
                learned_rows = learned_mask.any(dim=1)
            row_rms = torch.sqrt(torch.mean(learned_weight.square(), dim=1))
            positive = row_rms[learned_rows & (row_rms > 0)]
            reference = positive.mean() if positive.numel() else row_rms.new_tensor(1.0)
            scale = row_rms / torch.clamp(reference, min=1.0e-12)
            scale = torch.clamp(scale, min=0.1, max=10.0)
            scale = torch.where(learned_rows, scale, torch.zeros_like(scale))
            cortex.kernel_update_preconditioner = scale.unsqueeze(1)

    def set_kernel_learning_mask(self, mask):
        if mask is None:
            self.kernel_learning_mask = None
            return
        mask = torch.as_tensor(
            mask,
            dtype=self.kernel.weight.dtype,
            device=self.kernel.weight.device
        )
        if mask.shape != self.kernel.weight.shape:
            raise ValueError(
                'kernel learning mask shape must match kernel weight shape: '
                f'{tuple(mask.shape)} != {tuple(self.kernel.weight.shape)}'
            )
        self.kernel_learning_mask = mask

    def set_no_skip_links_tree(self, no_skip_links=True, cortex_ids=None):
        cortex_ids = None if cortex_ids is None else set(cortex_ids)
        for cortex in self.iter_cortex_tree():
            if cortex_ids is None or cortex.cortex_id in cortex_ids:
                cortex.kernel.no_skip_links = bool(no_skip_links)

    def set_homeostasis_batch_scale_tree(self, scale):
        scale = float(scale)
        for cortex in self.iter_cortex_tree():
            cortex.homeostasis_batch_scale = scale
            cortex.hidden_nv.homeostasis_step_scale = scale

    def begin_threshold_epoch_tree(self, epoch_index):
        for cortex in self.iter_cortex_tree():
            cortex.hidden_nv.begin_training_epoch(epoch_index)

    def adjust_label_thresholds(self, labels, num_classes, competition_group_N):
        if not self.homeostasis_enabled:
            return False
        return self.hidden_nv.adjust_s2_ncg_thresholds(
            labels,
            num_classes=num_classes,
            competition_group_N=competition_group_N
        )

    def _accumulate_sender_mean_earliness(self, sender_waves):
        sender_mean = sender_waves.detach().mean() / float(
            self.hidden_nv.simulation_time
        )
        if self.sender_mean_earliness is None:
            self.sender_mean_earliness = sender_mean
        else:
            self.sender_mean_earliness = (
                self.sender_mean_earliness.to(
                    device=sender_mean.device,
                    dtype=sender_mean.dtype
                )
                + sender_mean
            )

    @staticmethod
    def _mean_first_spike_earliness(spike_holder):
        earliness = spike_holder.earliness.detach()
        if earliness.ndim == 0:
            return earliness
        flat_earliness = earliness.reshape(earliness.shape[0], -1)
        return flat_earliness.max(dim=1).values.mean()

    def _update_input_source_mean_first_spike_earliness(self, source_means):
        source_means = torch.stack([
            torch.as_tensor(source_mean).detach()
            for source_mean in source_means
        ])
        self.input_source_mean_first_spike_earliness = source_means
        self.latest_input_mean_first_spike_earliness = (
            self.input_source_mean_first_spike_earliness.min()
        )
        self._update_optimal_spike_earliness_targets()

    def _update_mean_first_spike_earliness(self):
        self.hidden_mean_first_spike_earliness = (
            self._mean_first_spike_earliness(self.hidden_nv)
        )
        self.output_mean_first_spike_earliness = (
            self._mean_first_spike_earliness(self.output_nv)
        )
        self.mean_first_spike_earliness = (
            self.hidden_mean_first_spike_earliness
        )

    def _update_optimal_spike_earliness_targets(self):
        if self.amplifier_first_spike_target_settings is not None:
            optimal_first = self._get_source_relative_first_spike_target()
            if optimal_first is None:
                return
            self.optimal_first_spike_earliness = optimal_first
            if self.threshold_target_mean_first_spike_factor is not None:
                self.optimal_mean_spike_earliness = torch.clamp(
                    optimal_first
                    * float(self.threshold_target_mean_first_spike_factor),
                    min=0.0,
                    max=1.0
                )
            return

        if not self.uses_dynamic_spike_homeostasis:
            return
        if self.latest_input_mean_first_spike_earliness is None:
            return
        optimal_first = (
            self.latest_input_mean_first_spike_earliness
            * float(self.first_spike_delay_factor)
        )
        self.optimal_first_spike_earliness = optimal_first
        self.optimal_mean_spike_earliness = (
            optimal_first * float(self.mean_spike_delay_factor)
        )

    def aggregate_competition_groups(self, values, divisor=None):
        group_N = self.competition_group_N
        if group_N == 1:
            return values
        if values.shape[-1] % group_N != 0:
            raise ValueError(
                'competition_group_N must divide hidden channel length'
            )
        if divisor is None:
            divisor = group_N
        grouped = values.reshape(*values.shape[:-1], -1, group_N)
        return grouped.sum(dim=-1) / float(divisor)

    def _get_group_aggregation_divisor(self):
        return self.competition_group_N

    def get_group_aggregate_spike_wave(self, values=None):
        if values is None:
            values = self.hidden_nv.spike_wave
        divisor = self._get_group_aggregation_divisor()
        return self.aggregate_competition_groups(
            values,
            divisor=divisor
        )

    def _get_dominance_inhibition(self):
        dominance_inhibition_factor = float(
            getattr(self, 'dominance_inhibition_factor', 0.0)
        )
        if dominance_inhibition_factor == 0:
            return None
        if self.competition_group_N <= 1:
            return None

        # Dominance uses the same group-mean activity convention as output
        # aggregation, so its range stays comparable when group size changes.
        group_activity = self.aggregate_competition_groups(
            self.hidden_nv.spike_wave,
            divisor=self.competition_group_N
        )
        dominance = group_activity.max(dim=-1, keepdim=True).values
        dominance_inhibition_power = float(
            getattr(self, 'dominance_inhibition_power', 1.0)
        )
        if dominance_inhibition_power != 1.0:
            dominance = dominance.pow(dominance_inhibition_power)
        return dominance * dominance_inhibition_factor

    def _get_lateral_inhibition(self):
        lateral_inhibition_factor = float(
            getattr(self, 'lateral_inhibition_factor', 0.0)
        )
        if lateral_inhibition_factor == 0:
            return None

        mean_activity = torch.mean(
            self.hidden_nv.spike_wave, dim=-1, keepdim=True
        )
        lateral_inhibition_power = float(
            getattr(self, 'lateral_inhibition_power', 1.0)
        )
        if lateral_inhibition_power != 1.0:
            mean_activity = mean_activity.pow(lateral_inhibition_power)
        return mean_activity * lateral_inhibition_factor

    def _get_pooled_output_wave(self, patch_h, patch_w):
        if self.competition_group_aggregation_stage == 'after_pooling':
            pooled_hidden_wave = self.pooling.apply_output_pooling(
                self.hidden_nv.spike_wave,
                patch_h,
                patch_w
            )
            aggregation_weights = getattr(
                self, 'native_hidden_class_aggregation_weights', None
            )
            if aggregation_weights is not None:
                if (
                    aggregation_weights.ndim != 2
                    or int(aggregation_weights.shape[0])
                    != int(pooled_hidden_wave.shape[-1])
                    or int(aggregation_weights.shape[1])
                    != int(len(self.output_nv))
                ):
                    raise RuntimeError(
                        'native hidden class aggregation weights must have '
                        'shape [hidden_channels, output_classes]'
                    )
                weights = aggregation_weights.to(
                    device=pooled_hidden_wave.device,
                    dtype=pooled_hidden_wave.dtype,
                )
                self.native_hidden_class_aggregation_application_count = (
                    int(getattr(
                        self,
                        'native_hidden_class_aggregation_application_count',
                        0,
                    )) + 1
                )
                return pooled_hidden_wave @ weights
            return self.get_group_aggregate_spike_wave(pooled_hidden_wave)

        group_aggregate_spike_wave = self.get_group_aggregate_spike_wave()
        return self.pooling.apply_output_pooling(
            group_aggregate_spike_wave,
            patch_h,
            patch_w
        )

    # ================ amplifier related code ================
    @property
    def amplifier(self):
        return torch.exp(self.log_amplifier)

    def _project_sender_waves(self, sender_waves):
        """Project sender activity with optional time- or state-local weights."""
        scheduled_direct_weights = getattr(
            self,
            'direct_time_binned_weights',
            None,
        )
        scheduled_branch_offsets = getattr(
            self,
            'branch_time_binned_offsets',
            None,
        )
        state_direct_weights = getattr(
            self,
            'direct_state_residual_weights',
            None,
        )
        direct_preference_multiplier = getattr(
            self,
            'direct_class_preference_forward_multiplier',
            None,
        )
        direct_temporal_preference_multiplier = getattr(
            self,
            'direct_class_preference_forward_temporal_multiplier',
            None,
        )
        direct_preference_active_mass_matching = bool(getattr(
            self,
            'direct_class_preference_forward_active_mass_matching',
            False,
        ))
        branch_preference_multiplier = getattr(
            self,
            'branch_class_preference_forward_multiplier',
            None,
        )
        if scheduled_direct_weights is not None and state_direct_weights is not None:
            raise RuntimeError(
                'time-binned and state-conditioned direct weights are mutually exclusive'
            )
        if direct_temporal_preference_multiplier is not None and (
            direct_preference_multiplier is not None
            or scheduled_direct_weights is not None
            or scheduled_branch_offsets is not None
            or state_direct_weights is not None
        ):
            raise RuntimeError(
                'time-binned class-preference routing cannot be mixed with '
                'another direct time/state projection'
            )
        if scheduled_branch_offsets is not None and (
            scheduled_direct_weights is not None
            or state_direct_weights is not None
        ):
            raise RuntimeError(
                'time-binned branch offsets cannot be mixed with scheduled '
                'or state-conditioned direct weights'
            )
        if (
            scheduled_direct_weights is None
            and scheduled_branch_offsets is None
            and state_direct_weights is None
            and direct_preference_multiplier is None
            and direct_temporal_preference_multiplier is None
            and branch_preference_multiplier is None
        ):
            return sender_waves @ self.kernel.weight
        direct_len = len(self.input_nv)
        output_len = int(self.kernel.weight.shape[1])
        time_index = int(getattr(self, '_forward_time_index', 0))
        if scheduled_direct_weights is not None:
            if scheduled_direct_weights.ndim != 3:
                raise RuntimeError(
                    'direct_time_binned_weights must have shape '
                    '[bins, direct_inputs, outputs]'
                )
            if tuple(scheduled_direct_weights.shape[1:]) != (
                direct_len,
                output_len,
            ):
                raise RuntimeError(
                    'direct_time_binned_weights shape does not match the '
                    'cortex direct block'
                )
            simulation_time = max(1, int(getattr(
                self,
                'direct_time_binned_simulation_time',
                getattr(self, 'simulation_time', 1),
            )))
            bin_count = int(scheduled_direct_weights.shape[0])
            bin_index = min(
                bin_count - 1,
                time_index * bin_count // simulation_time,
            )
            self._forward_time_index = time_index + 1
            direct_weight = scheduled_direct_weights[bin_index].to(
                device=self.kernel.weight.device,
                dtype=self.kernel.weight.dtype,
            )
        elif scheduled_branch_offsets is not None:
            if scheduled_branch_offsets.ndim != 3:
                raise RuntimeError(
                    'branch_time_binned_offsets must have shape '
                    '[bins, branch_inputs, outputs]'
                )
            branch_len = int(self.kernel.weight.shape[0]) - direct_len
            if tuple(scheduled_branch_offsets.shape[1:]) != (
                branch_len,
                output_len,
            ):
                raise RuntimeError(
                    'branch_time_binned_offsets shape does not match the '
                    'cortex branch block'
                )
            simulation_time = max(1, int(getattr(
                self,
                'branch_time_binned_simulation_time',
                getattr(self, 'simulation_time', 1),
            )))
            bin_count = int(scheduled_branch_offsets.shape[0])
            bin_index = min(
                bin_count - 1,
                time_index * bin_count // simulation_time,
            )
            self._forward_time_index = time_index + 1
            direct_weight = self.kernel.weight[:direct_len]
        else:
            direct_weight = self.kernel.weight[:direct_len]
        unrouted_direct_weight = direct_weight
        if direct_temporal_preference_multiplier is not None:
            if (
                direct_temporal_preference_multiplier.ndim != 3
                or tuple(direct_temporal_preference_multiplier.shape[1:])
                != (direct_len, output_len)
            ):
                raise RuntimeError(
                    'time-binned class-preference multiplier must have shape '
                    '[bins, direct_inputs, outputs]'
                )
            simulation_time = max(1, int(getattr(
                self, 'simulation_time', 1
            )))
            bin_count = int(direct_temporal_preference_multiplier.shape[0])
            bin_index = min(
                bin_count - 1,
                time_index * bin_count // simulation_time,
            )
            self._forward_time_index = time_index + 1
            direct_preference_multiplier = (
                direct_temporal_preference_multiplier[bin_index]
            )
        if direct_preference_multiplier is not None:
            if tuple(direct_preference_multiplier.shape) != (
                direct_len,
                output_len,
            ):
                raise RuntimeError(
                    'direct class-preference forward multiplier shape does '
                    'not match the cortex direct block'
                )
            direct_preference_multiplier = direct_preference_multiplier.to(
                device=direct_weight.device,
                dtype=direct_weight.dtype,
            )
            cache_key = (
                int(direct_weight.data_ptr()),
                int(getattr(direct_weight, '_version', 0)),
                int(direct_preference_multiplier.data_ptr()),
                int(getattr(direct_preference_multiplier, '_version', 0)),
            )
            cache = getattr(
                self, '_direct_class_preference_forward_weight_cache', None
            )
            if cache is None or cache[0] != cache_key:
                cache = (
                    cache_key,
                    direct_weight * direct_preference_multiplier,
                )
                self._direct_class_preference_forward_weight_cache = cache
                self.class_preference_forward_last_abs_deviation = float(
                    (direct_preference_multiplier - 1.0)
                    .abs().mean().item()
                )
            direct_weight = cache[1]
            self.class_preference_forward_application_count = int(getattr(
                self, 'class_preference_forward_application_count', 0
            )) + 1
        else:
            self._direct_class_preference_forward_weight_cache = None
        direct_sender = sender_waves[..., :direct_len]
        direct_potential = direct_sender @ direct_weight
        if (
            direct_preference_active_mass_matching
            and direct_preference_multiplier is not None
        ):
            # A unit-mean map over the full sender column can still change
            # the mass seen by a sparse sample/time sender set. Match that
            # mass from the current label-free state while preserving which
            # active senders each receiver prefers.
            unweighted_mass = direct_sender.sum(dim=-1, keepdim=True)
            routed_mass = direct_sender @ direct_preference_multiplier
            valid_mass = (unweighted_mass > 0) & (routed_mass > 0)
            active_mass_scale = torch.where(
                valid_mass,
                unweighted_mass / routed_mass.clamp_min(1.0e-12),
                torch.ones_like(routed_mass),
            )
            direct_potential = direct_potential * active_mass_scale
            unsupported_mass = (unweighted_mass > 0) & (routed_mass <= 0)
            if bool(unsupported_mass.any()):
                unrouted_potential = direct_sender @ unrouted_direct_weight
                direct_potential = torch.where(
                    unsupported_mass,
                    unrouted_potential,
                    direct_potential,
                )
            matched_mass = torch.where(
                valid_mass,
                routed_mass * active_mass_scale,
                unweighted_mass.expand_as(routed_mass),
            )
            matched_ratio = torch.where(
                unweighted_mass > 0,
                matched_mass / unweighted_mass.clamp_min(1.0e-12),
                torch.ones_like(matched_mass),
            )
            valid_scale = active_mass_scale[valid_mass]
            if valid_scale.numel() > 0:
                self.class_preference_forward_active_mass_scale_mean = (
                    valid_scale.mean().detach()
                )
                self.class_preference_forward_active_mass_scale_min = (
                    valid_scale.min().detach()
                )
                self.class_preference_forward_active_mass_scale_max = (
                    valid_scale.max().detach()
                )
            else:
                one = direct_sender.new_tensor(1.0)
                self.class_preference_forward_active_mass_scale_mean = one
                self.class_preference_forward_active_mass_scale_min = one
                self.class_preference_forward_active_mass_scale_max = one
            self.class_preference_forward_active_mass_matched_ratio = (
                matched_ratio.mean().detach()
            )
            self.class_preference_forward_active_mass_valid_fraction = (
                valid_mass.to(dtype=direct_sender.dtype).mean().detach()
            )
            self.class_preference_forward_active_mass_application_count = (
                int(getattr(
                    self,
                    'class_preference_forward_active_mass_application_count',
                    0,
                )) + 1
            )
        if state_direct_weights is not None:
            if len(self.subcortexs) != 1:
                raise RuntimeError(
                    'state-conditioned direct residual requires exactly one subcortex'
                )
            if (
                state_direct_weights.ndim != 3
                or tuple(state_direct_weights.shape[1:])
                != (direct_len, output_len)
            ):
                raise RuntimeError(
                    'direct_state_residual_weights must have shape '
                    '[terms, direct_inputs, outputs]'
                )
            root_amplifier = torch.clamp(self.amplifier.detach(), min=1.0e-12)
            branch_amplifier = torch.clamp(
                self.subcortexs[0].amplifier.detach(),
                min=1.0e-12,
            )
            direct_activity = direct_sender.div(root_amplifier).mean(dim=-1)
            branch_activity = sender_waves[..., direct_len:].div(
                branch_amplifier
            ).mean(dim=-1)
            branch_state = (
                branch_activity
                - float(self.direct_state_branch_activity_mean)
            ) / float(self.direct_state_branch_activity_std)
            direct_state = (
                direct_activity
                - float(self.direct_state_direct_activity_mean)
            ) / float(self.direct_state_direct_activity_std)
            basis = self.direct_state_residual_basis
            states = []
            if basis.startswith('branch_activity_'):
                degree = {
                    'branch_activity_linear': 1,
                    'branch_activity_quadratic': 2,
                    'branch_activity_cubic': 3,
                }.get(basis)
                if degree is None:
                    raise RuntimeError(f'unsupported direct state basis: {basis}')
                states.extend(branch_state.pow(power) for power in range(1, degree + 1))
            elif basis == 'direct_activity_linear':
                states.append(direct_state)
            elif basis == 'branch_and_direct_activity_linear':
                states.extend((branch_state, direct_state))
            elif basis.startswith('branch_spatial_'):
                patch_count = int(self.direct_state_spatial_patch_count)
                grid_size = int(self.direct_state_spatial_grid_size)
                direct_channel_count = int(
                    self.direct_state_spatial_direct_channel_count
                )
                branch_channel_count = int(
                    self.direct_state_spatial_branch_channel_count
                )
                if patch_count != grid_size * grid_size:
                    raise RuntimeError(
                        'spatial direct state has inconsistent patch/grid shape'
                    )
                branch_sender = sender_waves[..., direct_len:].div(
                    branch_amplifier
                )
                expected_branch_len = patch_count * branch_channel_count
                if int(branch_sender.shape[-1]) != expected_branch_len:
                    raise RuntimeError(
                        'spatial direct state branch sender shape mismatch'
                    )
                branch_spatial = branch_sender.reshape(
                    *branch_sender.shape[:-1],
                    patch_count,
                    branch_channel_count,
                ).mean(dim=-1)
                current_spikes = self.subcortexs[0].output_nv.current_spikes
                if current_spikes is None:
                    raise RuntimeError(
                        'instantaneous spatial state requires current A-1 spikes'
                    )
                current_spikes = current_spikes.to(
                    device=branch_sender.device,
                    dtype=branch_sender.dtype,
                )
                if tuple(current_spikes.shape[-2:]) != (
                    patch_count,
                    branch_channel_count,
                ):
                    raise RuntimeError(
                        'instantaneous spatial state A-1 spike shape mismatch'
                    )
                current_spikes = current_spikes.reshape(
                    *branch_sender.shape[:-1],
                    -1,
                    patch_count,
                    branch_channel_count,
                )
                if int(current_spikes.shape[-3]) != 1:
                    raise RuntimeError(
                        'instantaneous spatial state has an unsupported '
                        'non-singleton A-1 output axis'
                    )
                current_spikes = current_spikes.squeeze(-3)
                branch_current_spatial = current_spikes.mean(dim=-1)

                def _map_spatial_state_to_direct(spatial_state):
                    mapped = torch.nn.functional.interpolate(
                        spatial_state.reshape(-1, 1, grid_size, grid_size),
                        size=(self.kernel_size, self.kernel_size),
                        mode='bilinear',
                        align_corners=True,
                    )
                    mapped = mapped.expand(
                        -1,
                        direct_channel_count,
                        -1,
                        -1,
                    )
                    return mapped.reshape(
                        *spatial_state.shape[:-1],
                        direct_len,
                    )

                def _normalize_spatial(value, mean_name, std_name):
                    mean = getattr(self, mean_name).to(
                        device=value.device,
                        dtype=value.dtype,
                    )
                    std = getattr(self, std_name).to(
                        device=value.device,
                        dtype=value.dtype,
                    )
                    return (value - mean) / std

                if basis == 'branch_spatial_cumulative_linear':
                    states.append(_map_spatial_state_to_direct(
                        _normalize_spatial(
                            branch_spatial,
                            'direct_state_branch_spatial_mean',
                            'direct_state_branch_spatial_std',
                        )
                    ))
                elif basis == 'branch_spatial_cumulative_centered_linear':
                    centered = branch_spatial - branch_spatial.mean(
                        dim=-1,
                        keepdim=True,
                    )
                    states.append(_map_spatial_state_to_direct(
                        _normalize_spatial(
                            centered,
                            'direct_state_branch_spatial_centered_mean',
                            'direct_state_branch_spatial_centered_std',
                        )
                    ))
                elif basis == 'branch_spatial_instantaneous_centered_linear':
                    centered = (
                        branch_current_spatial
                        - branch_current_spatial.mean(dim=-1, keepdim=True)
                    )
                    states.append(_map_spatial_state_to_direct(
                        _normalize_spatial(
                            centered,
                            'direct_state_branch_current_spatial_centered_mean',
                            'direct_state_branch_current_spatial_centered_std',
                        )
                    ))
                elif (
                    basis
                    == 'branch_spatial_cumulative_and_instantaneous_centered_linear'
                ):
                    cumulative_centered = (
                        branch_spatial
                        - branch_spatial.mean(dim=-1, keepdim=True)
                    )
                    current_centered = (
                        branch_current_spatial
                        - branch_current_spatial.mean(
                            dim=-1,
                            keepdim=True,
                        )
                    )
                    states.extend((
                        _map_spatial_state_to_direct(_normalize_spatial(
                            cumulative_centered,
                            'direct_state_branch_spatial_centered_mean',
                            'direct_state_branch_spatial_centered_std',
                        )),
                        _map_spatial_state_to_direct(_normalize_spatial(
                            current_centered,
                            'direct_state_branch_current_spatial_centered_mean',
                            'direct_state_branch_current_spatial_centered_std',
                        )),
                    ))
                else:
                    raise RuntimeError(f'unsupported direct state basis: {basis}')
            else:
                raise RuntimeError(f'unsupported direct state basis: {basis}')
            if len(states) != int(state_direct_weights.shape[0]):
                raise RuntimeError(
                    'direct state basis term count does not match fitted weights'
                )
            for state, state_weight in zip(states, state_direct_weights):
                if state.ndim == direct_sender.ndim:
                    state_response = (direct_sender * state) @ state_weight
                else:
                    state_response = state.unsqueeze(-1) * (
                        direct_sender @ state_weight
                    )
                direct_potential = direct_potential + state_response
        if direct_len == int(self.kernel.weight.shape[0]):
            return direct_potential
        branch_weight = self.kernel.weight[direct_len:]
        if scheduled_branch_offsets is not None:
            branch_weight = branch_weight + scheduled_branch_offsets[
                bin_index
            ].to(
                device=self.kernel.weight.device,
                dtype=self.kernel.weight.dtype,
            )
        if branch_preference_multiplier is not None:
            branch_len = int(branch_weight.shape[0])
            if tuple(branch_preference_multiplier.shape) != (
                branch_len,
                output_len,
            ):
                raise RuntimeError(
                    'branch class-preference forward multiplier shape does '
                    'not match the cortex branch block'
                )
            branch_preference_multiplier = branch_preference_multiplier.to(
                device=branch_weight.device,
                dtype=branch_weight.dtype,
            )
            cache_key = (
                int(branch_weight.data_ptr()),
                int(getattr(branch_weight, '_version', 0)),
                int(branch_preference_multiplier.data_ptr()),
                int(getattr(branch_preference_multiplier, '_version', 0)),
            )
            cache = getattr(
                self, '_branch_class_preference_forward_weight_cache', None
            )
            if cache is None or cache[0] != cache_key:
                cache = (
                    cache_key,
                    branch_weight * branch_preference_multiplier,
                )
                self._branch_class_preference_forward_weight_cache = cache
                self.class_preference_forward_last_abs_deviation = float(
                    (branch_preference_multiplier - 1.0)
                    .abs().mean().item()
                )
            branch_weight = cache[1]
            self.class_preference_forward_application_count = int(getattr(
                self, 'class_preference_forward_application_count', 0
            )) + 1
        else:
            self._branch_class_preference_forward_weight_cache = None
        branch_potential = sender_waves[..., direct_len:] @ branch_weight
        return direct_potential + branch_potential

    def _apply_input_wave_handoff_gate(self, sender_waves):
        """Retire the local sender after subcortex activity reaches a threshold."""
        threshold = getattr(
            self,
            'input_wave_handoff_branch_activity_fraction',
            None,
        )
        if threshold is None:
            self.input_wave_handoff_closed_fraction = 0.0
            return sender_waves
        if not self.subcortexs or len(sender_waves) <= 1:
            raise RuntimeError(
                'input-wave handoff requires at least one subcortex sender'
            )

        branch_wave = torch.cat(sender_waves[1:], dim=-1)
        branch_activity_fraction = (
            branch_wave.gt(0).to(dtype=branch_wave.dtype).mean(
                dim=-1,
                keepdim=True,
            )
        )
        close_direct = branch_activity_fraction >= float(threshold)
        self.input_wave_handoff_closed_fraction = (
            close_direct.detach().to(dtype=torch.float32).mean()
        )
        gated_direct = sender_waves[0] * (~close_direct).to(
            dtype=sender_waves[0].dtype
        )
        return [gated_direct, *sender_waves[1:]]

    def adjust_amplifier(self):
        if not self.homeostasis_enabled:
            return
        if not getattr(self, 'amplifier_homeostasis_enabled', True):
            return
        if self.log_amplifier_delta == 0:
            return
        if self.uses_first_spike_amplifier_homeostasis:
            self._adjust_amplifier_by_first_spike_earliness()
            return

        mean_output_threshold = self.hidden_nv.thresholds.mean()

        if not hasattr(self, 'mean_output_threshold_ema'):
            self.mean_output_threshold_ema = mean_output_threshold
        else:
            self.mean_output_threshold_ema = \
                (
                    self.amplifier_homeostasis_ema_decay
                    * self.mean_output_threshold_ema
                ) + (
                    (1.0 - self.amplifier_homeostasis_ema_decay)
                    * mean_output_threshold
                )

        threshold_error = (
            self.amplifier_threshold_target - self.mean_output_threshold_ema
        )
        self.log_amplifier += (
            self.log_amplifier_delta
            * self.homeostasis_batch_scale
            * threshold_error
        )

        self.log_amplifier = torch.clamp(
            self.log_amplifier,
            min=self.log_amplifier_lower_bound,
            max=self.log_amplifier_upper_bound
        )

    def _get_amplifier_first_spike_earliness_target(self):
        if self.amplifier_target_mean_first_spike_earliness is not None:
            return torch.as_tensor(
                self.amplifier_target_mean_first_spike_earliness,
                device=self.log_amplifier.device,
                dtype=self.log_amplifier.dtype
            )
        if self.amplifier_first_spike_target_settings is not None:
            return self._get_source_relative_first_spike_target()
        return self.optimal_first_spike_earliness

    def _get_source_relative_first_spike_target(self):
        settings = self.amplifier_first_spike_target_settings
        if settings is None or self.input_source_mean_first_spike_earliness is None:
            return None
        source_means = self.input_source_mean_first_spike_earliness.to(
            device=self.log_amplifier.device,
            dtype=self.log_amplifier.dtype
        )
        source_reference = settings['source_reference']
        if source_reference == 'slowest':
            source = source_means.min()
        elif source_reference == 'fastest':
            source = source_means.max()
        else:
            source = source_means.mean()
        delay_factor = (
            settings['multi_source_delay_factor']
            if source_means.numel() > 1
            else settings['single_source_delay_factor']
        )
        return torch.clamp(
            source * float(delay_factor),
            min=float(settings['min_target_earliness']),
            max=float(settings['max_target_earliness'])
        )

    def _get_threshold_target_mean_earliness_reference(self):
        if self.threshold_target_mean_first_spike_factor is None:
            return None
        target = self._get_amplifier_first_spike_earliness_target()
        if target is None:
            return None
        return torch.clamp(
            target * float(self.threshold_target_mean_first_spike_factor),
            min=0.0,
            max=1.0
        )

    def _get_amplifier_first_spike_observation(self):
        if self.amplifier_first_spike_observation_mode == 'sample_max_first_spike':
            return self.mean_first_spike_earliness
        if self.amplifier_first_spike_observation_mode == 'hidden_mean_earliness':
            return self.hidden_nv.earliness.detach().mean()
        raise RuntimeError(
            'unknown amplifier first-spike observation mode: '
            f'{self.amplifier_first_spike_observation_mode}'
        )

    def _adjust_amplifier_by_first_spike_earliness(self):
        target = self._get_amplifier_first_spike_earliness_target()
        observed = self._get_amplifier_first_spike_observation()
        if (
            observed is None
            or target is None
        ):
            return
        observed = observed.detach().to(
            device=self.log_amplifier.device,
            dtype=self.log_amplifier.dtype
        )
        target = target.detach().to(
            device=self.log_amplifier.device,
            dtype=self.log_amplifier.dtype
        )
        homeostasis_signal = target - observed
        if not hasattr(self, 'first_spike_homeostasis_ema'):
            self.first_spike_homeostasis_ema = homeostasis_signal
        else:
            self.first_spike_homeostasis_ema = (
                self.amplifier_homeostasis_ema_decay
                * self.first_spike_homeostasis_ema
                + (1.0 - self.amplifier_homeostasis_ema_decay)
                * homeostasis_signal
            )

        self.log_amplifier += (
            self.log_amplifier_delta
            * self.homeostasis_batch_scale
            * self.first_spike_homeostasis_ema
        )

        self.log_amplifier = torch.clamp(
            self.log_amplifier,
            min=self.log_amplifier_lower_bound,
            max=self.log_amplifier_upper_bound
        )
    # ^^^^^^^^^^^^^^^^ amplifier related code ^^^^^^^^^^^^^^^^

    def _get_input_shape(self, sample_input):
        if isinstance(sample_input, torch.Tensor):
            return tuple(sample_input.shape)
        return tuple(sample_input)

    def _get_patch_state_shape(self, input_shape):
        if self._cached_init_input_shape != input_shape:
            *outer_dims, input_channel, height, width = input_shape
            patch_h, patch_w, patch_num = self._get_patch_grid(height, width)
            self._cached_init_input_shape = input_shape
            self._cached_init_patch_grid = (patch_h, patch_w, patch_num)
            self._cached_init_patch_outer_dim = (*outer_dims, patch_num)
            self._cached_init_patch_shape = (
                *outer_dims, patch_num, input_channel, self.kernel_size, self.kernel_size
            )

        return self._cached_init_patch_outer_dim, self._cached_init_patch_shape

    def _get_patch_grid(self, height, width):
        patch_h = len(self._get_patch_starts(height))
        patch_w = len(self._get_patch_starts(width))
        patch_num = patch_h * patch_w
        return patch_h, patch_w, patch_num

    def _get_patch_starts(self, size):
        max_start = size - self.kernel_size
        if max_start < 0:
            raise ValueError('cortex kernel_size cannot exceed input spatial size')
        starts = list(range(0, max_start + 1, self.stride))
        if getattr(self, 'cover_edges', False) and starts[-1] != max_start:
            starts.append(max_start)
        return starts

    def init_state(self, sample_input, device):
        input_shape = self._get_input_shape(sample_input)
        patch_outer_dim, patch_shape = self._get_patch_state_shape(input_shape)
        patch_h, patch_w, _ = self._cached_init_patch_grid
        pooled_h, pooled_w, pooled_patch_num = self.pooling.get_pooled_grid(patch_h, patch_w)
        pooled_outer_dim = (*patch_outer_dim[:-1], pooled_patch_num)

        self.input_nv.init_state(patch_outer_dim, device)
        self.hidden_nv.init_state(patch_outer_dim, device)
        self.output_nv.init_state(pooled_outer_dim, device)
        self._input_wave_delay_history = []
        self.input_wave_handoff_closed_fraction = 0.0
        self._forward_time_index = 0
        self.current_excitatory_potential = None
        self.sender_mean_earliness = None
        self.input_source_mean_first_spike_earliness = None
        self.latest_input_mean_first_spike_earliness = None
        self.mean_first_spike_earliness = None
        self.hidden_mean_first_spike_earliness = None
        self.output_mean_first_spike_earliness = None
        self.optimal_first_spike_earliness = None
        self.optimal_mean_spike_earliness = None
        for subcortex in self.subcortexs:
            subcortex.init_state(patch_shape, device)

    def clear_state_tree(self):
        for cortex in self.iter_cortex_tree():
            cortex.input_nv.clear_state()
            cortex.hidden_nv.clear_state()
            cortex.output_nv.clear_state()
            cortex._input_wave_delay_history = []
            cortex.input_wave_handoff_closed_fraction = 0.0
            cortex._forward_time_index = 0
            cortex.current_excitatory_potential = None
            cortex.sender_mean_earliness = None
            cortex.input_source_mean_first_spike_earliness = None
            cortex.latest_input_mean_first_spike_earliness = None
            cortex.latest_subcortex_handoff_waves = []
            cortex.mean_first_spike_earliness = None
            cortex.hidden_mean_first_spike_earliness = None
            cortex.output_mean_first_spike_earliness = None
            cortex.optimal_first_spike_earliness = None
            cortex.optimal_mean_spike_earliness = None

    def assert_non_streaming_supported_tree(self):
        return None

    def extract_patches(self, input_spikes):
        """
        Extract patches from input_spikes in [..., C, H, W] format using self.kernel_size and self.stride.

        Args:
            input_spikes (Tensor): Tensor of shape [..., C, H, W]

        self:
            self.kernel_size (int): Kernel size (assumed square)
            self.stride (int): Stride size

        Returns:
            Tensor: Shape [..., patch_num, C, kernel_size, kernel_size]
        """
        *outer_dims, C, H, W = input_spikes.shape
        K = self.kernel_size
        S = self.stride
        if getattr(self, 'cover_edges', False):
            row_starts = self._get_patch_starts(H)
            col_starts = self._get_patch_starts(W)
            patches = [
                input_spikes[..., top:top + K, left:left + K]
                for top in row_starts
                for left in col_starts
            ]
            return torch.stack(patches, dim=len(outer_dims))

        unfolded_h = input_spikes.unfold(-2, K, S)    # [..., C, H_out, W, K]
        patches = unfolded_h.unfold(-2, K, S)         # [..., C, H_out, W_out, K, K]

        # get patch dims
        H_out, W_out = patches.shape[-4], patches.shape[-3]
        patch_num = H_out * W_out

        # permute [..., C, H_out, W_out, K, K] to [..., H_out, W_out, C, K, K]
        # then reshape to [..., patch_num, C, K, K]
        patches = patches.permute(*range(len(outer_dims)), -4, -3, -5, -2, -1)
        patches = patches.reshape(*outer_dims, patch_num, C, K, K)

        return patches

    def _delay_input_wave(self, input_wave):
        delay_steps = self.input_wave_delay_steps
        if delay_steps <= 0:
            return input_wave

        self._input_wave_delay_history.append(input_wave)
        if len(self._input_wave_delay_history) <= delay_steps:
            delayed_wave = torch.zeros_like(input_wave)
        else:
            delayed_wave = self._input_wave_delay_history[-delay_steps - 1]

        max_history = delay_steps + 1
        if len(self._input_wave_delay_history) > max_history:
            del self._input_wave_delay_history[:-max_history]
        return delayed_wave

    def _apply_temporal_receiver_phase_gate(self, potential):
        phase_count = self.temporal_receiver_phase_count
        if phase_count == 1:
            self.temporal_receiver_active_phase = 0
            return potential

        temporal_step = min(
            int(self.hidden_nv.temporal_step),
            self.simulation_time - 1,
        )
        if self.temporal_receiver_phase_basis == 'clock':
            active_phase = min(
                temporal_step * phase_count // self.simulation_time,
                phase_count - 1,
            )
        else:
            progress = getattr(self, 'temporal_receiver_progress', None)
            if progress is None:
                progress_sequence = getattr(
                    self,
                    'temporal_receiver_progress_sequence',
                    None,
                )
                if progress_sequence is not None:
                    progress = progress_sequence[temporal_step]
            if progress is None:
                raise RuntimeError(
                    'event-progress receiver phases require temporal '
                    'progress from the image encoder'
                )
            if progress.ndim != 1 or progress.shape[0] != potential.shape[0]:
                raise ValueError(
                    'temporal receiver progress must have one value per sample'
                )
            active_phase = torch.clamp(
                (progress * phase_count).to(torch.long),
                min=0,
                max=phase_count - 1,
            )
        phase_width = self.competition_group_N // phase_count
        receiver_in_group = torch.arange(
            self.hidden_channel,
            device=potential.device,
        ) % self.competition_group_N
        receiver_phase = receiver_in_group // phase_width
        if isinstance(active_phase, int):
            active_mask = receiver_phase == active_phase
        else:
            active_phase = active_phase.to(device=potential.device)
            receiver_shape = [1] * potential.ndim
            receiver_shape[-1] = self.hidden_channel
            phase_shape = [potential.shape[0]] + [1] * (potential.ndim - 1)
            active_mask = (
                receiver_phase.reshape(receiver_shape)
                == active_phase.reshape(phase_shape)
            )
        self.temporal_receiver_active_phase = active_phase
        return potential.masked_fill(~active_mask, 0.0)

    def _subcortex_handoff_wave(self, subcortex, cumulative_wave):
        if self.subcortex_handoff_mode == 'cumulative':
            handoff_wave = cumulative_wave
        elif self.subcortex_handoff_mode == 'onset':
            handoff_wave = self._flatten_output_wave_sequence(
                subcortex.output_nv.current_spikes
            )
        else:
            handoff_wave = self._flatten_output_wave_sequence(
                subcortex.output_nv.spike_trace
            )
        return handoff_wave * self.subcortex_handoff_scale

    def __call__(
        self, input_spikes, is_training=False, use_training_thresholds=None
    ):
        """
        Perform forward spike propagation through one hierarchical Cortex module.

        This module processes spike-based inputs in the standard BCHW format, where each
        spatial location is associated with a spike vector. The module applies spatial
        aggregation over local patches (defined by kernel size) and emits a new spike
        vector per patch.

        Args:
            input_spikes (Tensor): Input spike tensor of shape
                (batch_size, ..., in_channels, height, width)

                - batch_size: Number of input samples
                - in_channels: Number of input channels
                - height, width: Spatial dimensions of the input

            is_training (bool): Whether the model is in training mode. Default is False.

        Returns:
            Tensor: Output spike tensor of shape
                (batch_size, ..., out_channels)

                - out_channels: Number of output channels after local spatial aggregation

        Note:
            When stacked hierarchically, each Cortex module performs spatial downsampling,
            reducing spatial resolution while increasing the channel dimension. At the final
            layer, the output may be globally pooled to yield a compact representation such as:
                (batch_size, final_output_channels)
        """
        input_patches = self.extract_patches(input_spikes)
        # Shape [..., patch_num, C, kernel_size, kernel_size]

        flatten_input_patches = input_patches.contiguous().view(*input_patches.shape[:-3], -1)
        # Shape [..., patch_num, C x kernel_size x kernel_size]

        self.input_nv.stimulate_by_spike(flatten_input_patches)
        sender_waves = [self._delay_input_wave(self.input_nv.spike_wave)]
        source_mean_first_spike_earlinesses = [
            self._mean_first_spike_earliness(self.input_nv)
        ]

        self.latest_subcortex_handoff_waves = []
        for subcortex in self.subcortexs:
            subcortex_wave = subcortex(
                input_patches,
                is_training=is_training,
                use_training_thresholds=use_training_thresholds
            )
            # subcortex_wave shape [..., patch_num, subcortex_out_channel]
            handoff_wave = self._subcortex_handoff_wave(
                subcortex, subcortex_wave
            )
            sender_waves.append(handoff_wave)
            self.latest_subcortex_handoff_waves.append(handoff_wave.detach())
            source_mean_first_spike_earlinesses.append(
                subcortex.output_mean_first_spike_earliness.detach()
            )
        self._update_input_source_mean_first_spike_earliness(
            source_mean_first_spike_earlinesses
        )
        sender_waves = self._apply_input_wave_handoff_gate(sender_waves)
        sender_waves = torch.cat(sender_waves, dim=-1)
        self._accumulate_sender_mean_earliness(sender_waves)
        sender_waves = sender_waves * self.amplifier
        # sender_waves shape [..., patch_num, sender_len]

        # Contract: sender_len must equal kernel.weight.shape[0], and the matmul preserves
        # batch/patch axes while projecting the trailing sender dimension to output_len.
        potential = self._project_sender_waves(sender_waves)
        self.current_excitatory_potential = potential
        # potential shape [..., patch_num, out_channels]

        lateral_inhibition = self._get_lateral_inhibition()
        if lateral_inhibition is not None:
            potential = potential - lateral_inhibition
        dominance_inhibition = self._get_dominance_inhibition()
        if dominance_inhibition is not None:
            potential = potential - dominance_inhibition
        potential = self._apply_temporal_receiver_phase_gate(potential)

        self_learning_enabled = is_training and self.learning_enabled
        self_homeostasis_enabled = is_training and self.homeostasis_enabled
        if use_training_thresholds is None:
            use_training_thresholds = (
                self_learning_enabled or self_homeostasis_enabled
            )
        target_mean_earliness_reference = (
            self._get_threshold_target_mean_earliness_reference()
        )
        if target_mean_earliness_reference is None:
            target_mean_earliness_reference = (
                self.optimal_mean_spike_earliness
                if self.uses_dynamic_spike_homeostasis
                else self.sender_mean_earliness
            )
        self.hidden_nv.stimulate_by_potential(
            potential,
            is_training=self_homeostasis_enabled,
            top_k=self.top_k,
            target_mean_earliness_reference=target_mean_earliness_reference,
            use_training_thresholds=use_training_thresholds
        )

        # output shape [..., patch_num, out_channels]
        patch_h, patch_w, _ = self._cached_init_patch_grid
        pooled_wave = self._get_pooled_output_wave(patch_h, patch_w)
        self.output_nv.stimulate_by_wave(pooled_wave)
        self._update_mean_first_spike_earliness()
        return self.output_nv.spike_wave.reshape(*self.output_nv.spike_wave.shape[:-2], -1)

    def _flatten_output_wave_sequence(self, output_wave_sequence):
        return output_wave_sequence.reshape(*output_wave_sequence.shape[:-2], -1)

    def forward_non_streaming(
        self, input_spike_sequence, labels=None, label_usage='ignore',
        label_propagating=False, STDP_interval=None, is_training=False,
        use_training_thresholds=None
    ):
        if label_propagating:
            raise ValueError(
                'non_streaming first version supports only '
                'label_propagating=False'
            )

        accumulated = False
        input_patches_sequence = None
        if self.subcortexs:
            input_patches_sequence = self.extract_patches(input_spike_sequence)

        subcortex_wave_sequences = []
        for subcortex in self.subcortexs:
            subcortex_wave_sequence, sub_accumulated = (
                subcortex.forward_non_streaming(
                    input_patches_sequence,
                    labels=None,
                    label_usage=label_usage,
                    label_propagating=label_propagating,
                    STDP_interval=STDP_interval,
                    is_training=is_training,
                    use_training_thresholds=use_training_thresholds
                )
            )
            accumulated = sub_accumulated or accumulated
            subcortex_wave_sequences.append((subcortex, subcortex_wave_sequence))

        self_learning_enabled = is_training and self.learning_enabled
        self_homeostasis_enabled = is_training and self.homeostasis_enabled
        if use_training_thresholds is None:
            use_training_thresholds = (
                self_learning_enabled or self_homeostasis_enabled
            )
        patch_h, patch_w, _ = self._cached_init_patch_grid
        output_wave_history = []
        for time_index in range(input_spike_sequence.shape[0]):
            if input_patches_sequence is None:
                input_patches = self.extract_patches(input_spike_sequence[time_index])
            else:
                input_patches = input_patches_sequence[time_index]
            flatten_input_patches = input_patches.contiguous().view(
                *input_patches.shape[:-3],
                -1
            )
            self.input_nv.stimulate_by_spike(flatten_input_patches)

            sender_waves = [self._delay_input_wave(self.input_nv.spike_wave)]
            source_mean_first_spike_earlinesses = [
                self._mean_first_spike_earliness(self.input_nv)
            ]
            self.latest_subcortex_handoff_waves = []
            for subcortex, subcortex_wave_sequence in subcortex_wave_sequences:
                subcortex_wave = subcortex_wave_sequence[time_index]
                subcortex_wave = subcortex_wave.reshape(
                    *subcortex_wave.shape[:-1],
                    -1,
                    len(subcortex.output_nv)
                )
                subcortex.output_nv.stimulate_by_wave(subcortex_wave)
                cumulative_wave = self._flatten_output_wave_sequence(
                    subcortex.output_nv.spike_wave
                )
                handoff_wave = self._subcortex_handoff_wave(
                    subcortex, cumulative_wave
                )
                sender_waves.append(handoff_wave)
                self.latest_subcortex_handoff_waves.append(handoff_wave.detach())
                source_mean_first_spike_earlinesses.append(
                    self._mean_first_spike_earliness(subcortex.output_nv)
                )

            self._update_input_source_mean_first_spike_earliness(
                source_mean_first_spike_earlinesses
            )
            sender_waves = self._apply_input_wave_handoff_gate(sender_waves)
            sender_waves = torch.cat(sender_waves, dim=-1)
            self._accumulate_sender_mean_earliness(sender_waves)
            sender_waves = sender_waves * self.amplifier
            potential = self._project_sender_waves(sender_waves)
            self.current_excitatory_potential = potential
            lateral_inhibition = self._get_lateral_inhibition()
            if lateral_inhibition is not None:
                potential = potential - lateral_inhibition
            dominance_inhibition = self._get_dominance_inhibition()
            if dominance_inhibition is not None:
                potential = potential - dominance_inhibition
            potential = self._apply_temporal_receiver_phase_gate(potential)
            target_mean_earliness_reference = (
                self._get_threshold_target_mean_earliness_reference()
            )
            if target_mean_earliness_reference is None:
                target_mean_earliness_reference = (
                    self.optimal_mean_spike_earliness
                    if self.uses_dynamic_spike_homeostasis
                    else self.sender_mean_earliness
                )
            self.hidden_nv.stimulate_by_potential(
                potential,
                is_training=self_homeostasis_enabled,
                top_k=self.top_k,
                target_mean_earliness_reference=target_mean_earliness_reference,
                use_training_thresholds=use_training_thresholds
            )

            if is_training and STDP_interval is not None:
                accumulated = self._accumulate_STDP_self(
                    labels,
                    label_usage
                ) or accumulated

            pooled_wave = self._get_pooled_output_wave(patch_h, patch_w)
            self.output_nv.stimulate_by_wave(pooled_wave)
            self._update_mean_first_spike_earliness()
            output_wave_history.append(self._flatten_output_wave_sequence(pooled_wave))

        output_wave_sequence = torch.stack(output_wave_history, dim=0)
        self.clear_state_tree()
        return output_wave_sequence, accumulated
