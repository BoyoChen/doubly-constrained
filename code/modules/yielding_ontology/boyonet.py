import copy
import hashlib
from contextlib import nullcontext
import torch
from functools import partial
from modules.image_encoders import get_image_encoder
from modules.yielding_ontology.pca_cortex import PcaCortex as Cortex


class BoyoNet():
    FIXED_LABEL_OFF_VALUE = -1.0

    def __init__(
        self, init_cortex, cortex_spec, image_encoder_spec, remember_settings,
        competition_group_N=None,
        target_mean_earliness=0.08,
        first_spike_delay_factor=None,
        mean_spike_delay_factor=None,
        partitioned_initialization=False,
        **unknown_settings,
    ):
        removed_group_cap_settings = {
            'output_simulation_group_top_k',
            'output_simulation_group_cap_mode',
        } & set(unknown_settings)
        if removed_group_cap_settings:
            removed = ', '.join(sorted(removed_group_cap_settings))
            raise ValueError(
                'group cap settings were removed; remove from YAML: '
                f'{removed}'
            )
        if unknown_settings:
            unknown = ', '.join(sorted(unknown_settings))
            raise TypeError(f'Unexpected BoyoNet setting(s): {unknown}')
        self.device = torch.device('cpu')
        self._state_sample_input = None
        self._state_sample_shape = None
        self._state_sample_device = None
        self.target_mean_earliness = self._normalize_target_mean_earliness(
            target_mean_earliness
        )
        self.first_spike_delay_factor = self._normalize_optional_positive_factor(
            first_spike_delay_factor,
            'first_spike_delay_factor'
        )
        self.mean_spike_delay_factor = self._normalize_optional_positive_factor(
            mean_spike_delay_factor,
            'mean_spike_delay_factor'
        )
        if (
            self.first_spike_delay_factor is None
            and self.mean_spike_delay_factor is not None
        ) or (
            self.first_spike_delay_factor is not None
            and self.mean_spike_delay_factor is None
        ):
            raise ValueError(
                'first_spike_delay_factor and mean_spike_delay_factor '
                'must be set together'
            )
        self.image_encoder_target_mean_earliness = (
            self._resolve_image_encoder_target_mean_earliness(
                image_encoder_spec,
                self.target_mean_earliness
            )
        )

        image_encoder_spec = self._prepare_image_encoder_spec(image_encoder_spec)
        cortex_spec = self._prepare_cortex_spec(cortex_spec)
        self.image_encoder = get_image_encoder(**image_encoder_spec)
        self.cortex_constructor = partial(
            Cortex, simulation_time=self.image_encoder.simulation_time, **cortex_spec
        )

        if not isinstance(partitioned_initialization, bool):
            raise TypeError('partitioned_initialization must be bool')
        # Keep downstream RNG consumption independent of input tensor dimensions.
        rng_context = (torch.random.fork_rng(devices=[])
                       if partitioned_initialization else nullcontext())
        with rng_context:
            self.cortex = self.construct_init_model('A', **init_cortex)
            if partitioned_initialization:
                settings = cortex_spec['base_cortex_settings']['kernel_settings']
                self._initialize_partitioned_weights(settings, torch.initial_seed())
            for cortex in self.cortex.iter_cortex_tree():
                initialize_coupling = getattr(
                    cortex,
                    'apply_temporal_basis_shared_initialization',
                    None,
                )
                if callable(initialize_coupling):
                    initialize_coupling()
        self.cortex.fix_shape()

        root_competition_group_N = self.cortex.competition_group_N
        if competition_group_N is None:
            self.competition_group_N = root_competition_group_N
        else:
            self.competition_group_N = int(competition_group_N)
            if root_competition_group_N != 1 and (
                self.competition_group_N != root_competition_group_N
            ):
                raise ValueError(
                    'model.competition_group_N must match '
                    'init_cortex.competition_group_N when both are set'
                )
        if self.competition_group_N <= 0:
            raise ValueError('competition_group_N must be positive or null')
        if root_competition_group_N != 1:
            self.num_classes = len(self.cortex.output_nv)
        else:
            self.num_classes = self.cortex.shape[1] // self.competition_group_N
            if self.num_classes * self.competition_group_N != self.cortex.shape[1]:
                raise ValueError(
                    'output neurons must be divisible by competition_group_N'
                )
        def setup(
            self_cleaning_after_STDP, STDP_mechanism, label_off_value=None
        ):
            self.self_cleaning_after_STDP = self_cleaning_after_STDP
            if label_off_value is not None:
                label_off_value = float(label_off_value)
                if label_off_value != self.FIXED_LABEL_OFF_VALUE:
                    raise ValueError(
                        'remember_settings.label_off_value is fixed to -1.0; '
                        'remove the field from YAML'
                    )
            self.label_off_value = self.FIXED_LABEL_OFF_VALUE
            self.STDP_mechanism = STDP_mechanism

        setup(**remember_settings)
        self._validate_hinge_softmax_label_settings()
        self._validate_presynaptic_class_preference_settings()
        self._validate_native_hidden_class_aggregation_settings()

    @staticmethod
    def _normalize_optional_positive_factor(value, name):
        if value is None:
            return None
        value = float(value)
        if value <= 0:
            raise ValueError(f'{name} must be > 0')
        return value

    @staticmethod
    def _normalize_target_mean_earliness(target_mean_earliness):
        if isinstance(target_mean_earliness, dict):
            settings = dict(target_mean_earliness)
            allowed = {
                'delay_ratio',
                'fixed',
                'min_target_earliness',
                'max_target_earliness',
            }
            unknown = set(settings) - allowed
            if unknown:
                unknown_text = ', '.join(sorted(unknown))
                raise ValueError(
                    f'unknown target_mean_earliness settings: {unknown_text}'
                )
            if settings.get('fixed') is not None and settings.get('delay_ratio') is not None:
                raise ValueError(
                    'target_mean_earliness.fixed and delay_ratio are mutually exclusive'
                )
            if (
                settings.get('min_target_earliness') is None
                or settings.get('max_target_earliness') is None
            ):
                raise ValueError(
                    'target_mean_earliness must define min_target_earliness '
                    'and max_target_earliness'
                )
            min_target = float(settings['min_target_earliness'])
            max_target = float(settings['max_target_earliness'])
            if not 0.0 <= min_target <= max_target <= 1.0:
                raise ValueError(
                    'target_mean_earliness clamp must satisfy '
                    '0 <= min_target_earliness <= max_target_earliness <= 1'
                )
            if settings.get('delay_ratio') is None and min_target != max_target:
                raise ValueError(
                    'fixed target_mean_earliness requires '
                    'min_target_earliness == max_target_earliness'
                )
            if settings.get('delay_ratio') is not None:
                delay_ratio = float(settings['delay_ratio'])
                if delay_ratio <= 0:
                    raise ValueError(
                        'target_mean_earliness.delay_ratio must be > 0'
                    )
                settings['delay_ratio'] = delay_ratio
            settings['min_target_earliness'] = min_target
            settings['max_target_earliness'] = max_target
            if settings.get('fixed') is not None:
                settings['fixed'] = float(settings['fixed'])
            return settings

        target_mean_earliness = float(target_mean_earliness)
        if not 0.0 <= target_mean_earliness <= 1.0:
            raise ValueError('target_mean_earliness must be between 0 and 1')
        return target_mean_earliness

    @staticmethod
    def _resolve_image_encoder_target_mean_earliness(
        image_encoder_spec,
        target_mean_earliness
    ):
        if not isinstance(target_mean_earliness, dict):
            return target_mean_earliness
        if 'target_mean_earliness' in image_encoder_spec:
            return image_encoder_spec['target_mean_earliness']
        return 0.08

    def _prepare_image_encoder_spec(self, image_encoder_spec):
        image_encoder_spec = copy.deepcopy(image_encoder_spec)
        image_encoder_spec['target_mean_earliness'] = \
            self.image_encoder_target_mean_earliness
        return image_encoder_spec

    @staticmethod
    def _get_removed_hinge_softmax_label_settings(settings):
        removed_prefixes = (
            'temporal' + '_hinge_',
            'hinge' + '_',
            'no' + '_decision_',
        )
        removed_exact = {
            'prediction' + '_mode',
            'label' + '_signal_mode',
            'correct' + '_target_scope',
            'correct' + '_target_value',
            'correct' + '_non_selected_target_value',
            'correct' + '_off_value',
            'wrong' + '_target_value',
            'wrong' + '_predicted_value',
            'wrong' + '_predicted_top_k',
            'wrong' + '_predicted_margin_threshold',
            'wrong' + '_predicted_low_confidence_value',
            'wrong' + '_other_value',
            'wrong' + '_punishment_mode',
            'wrong' + '_temporal_margin_alpha',
            'wrong' + '_temporal_margin_min_value',
            'wrong' + '_temporal_margin_delta',
            'softmax_source',
            'true_class_responsibility_mode',
            'wrong_class_responsibility_mode',
            'responsibility_support',
            'true_class_responsibility_support',
            'wrong_class_responsibility_support',
            'wrong_nonpositive_margin_focus_gain',
            'wrong_nonpositive_margin_focus_l1_scope',
        }
        return [
            key for key in settings
            if key in removed_exact or key.startswith(removed_prefixes)
        ]

    def _get_hinge_softmax_label_settings(self):
        old_container_key = 'decision' + '_contingent_settings'
        if old_container_key in self.STDP_mechanism:
            raise ValueError(
                'old label settings container was removed; '
                'use hinge_softmax_label instead'
            )
        settings = self.STDP_mechanism.get('hinge_softmax_label', {})
        if settings is None:
            return {}
        if not isinstance(settings, dict):
            raise ValueError('hinge_softmax_label must be a mapping')
        return settings

    def _validate_hinge_softmax_label_settings(self):
        settings = self._get_hinge_softmax_label_settings()
        removed_settings = self._get_removed_hinge_softmax_label_settings(settings)
        if removed_settings:
            removed = ', '.join(sorted(removed_settings))
            raise ValueError(
                'old label setting names were removed; '
                f'use hinge_softmax_label clean names only: {removed}'
            )
        valid_responsibility_modes = {
            'softmax',
            'uniform',
            'winner_take_all',
        }
        responsibility_mode = settings.get('responsibility_mode', 'softmax')
        if responsibility_mode not in valid_responsibility_modes:
            raise ValueError(
                'responsibility_mode must be one of '
                '{"softmax", "uniform", "winner_take_all"}'
            )
        wrong_class_scope = settings.get('wrong_class_scope', 'all')
        if wrong_class_scope not in {'all', 'hardest_wrong_class'}:
            raise ValueError(
                'wrong_class_scope must be one of '
                '{"all", "hardest_wrong_class"}'
            )

    def _get_presynaptic_class_preference_settings(self):
        settings = self.STDP_mechanism.get(
            'presynaptic_class_preference', None
        )
        if settings is None:
            return None
        if not isinstance(settings, dict):
            raise ValueError(
                'presynaptic_class_preference must be a mapping'
            )
        return dict(settings)

    def _get_native_hidden_class_aggregation_settings(self):
        settings = self.STDP_mechanism.get(
            'native_hidden_class_aggregation', None
        )
        if settings is None:
            return None
        if not isinstance(settings, dict):
            raise ValueError(
                'native_hidden_class_aggregation must be a mapping'
            )
        return dict(settings)

    def _validate_native_hidden_class_aggregation_settings(self):
        settings = self._get_native_hidden_class_aggregation_settings()
        if settings is None:
            return
        allowed = {
            'enabled',
            'target_cortex_id',
            'application',
            'class_index_permutation',
        }
        unknown = set(settings) - allowed
        if unknown:
            raise ValueError(
                'native_hidden_class_aggregation contains unsupported '
                'field(s): ' + ', '.join(sorted(unknown))
            )
        if not isinstance(settings.get('enabled', False), bool):
            raise TypeError(
                'native_hidden_class_aggregation.enabled must be bool'
            )
        target_id = settings.get('target_cortex_id')
        if not isinstance(target_id, str) or not target_id:
            raise ValueError(
                'native_hidden_class_aggregation.target_cortex_id must be '
                'a non-empty cortex ID'
            )
        target = self.get_cortex_by_id(target_id)
        if target is not self.cortex:
            raise ValueError(
                'native_hidden_class_aggregation currently supports the '
                'root cortex only'
            )
        application = settings.get('application', 'evaluation_only')
        if application not in {'evaluation_only', 'train_and_evaluation'}:
            raise ValueError(
                'native_hidden_class_aggregation.application must be one '
                'of {"evaluation_only", "train_and_evaluation"}'
            )
        if (
            int(target.hidden_channel)
            != int(self.num_classes) * int(target.competition_group_N)
        ):
            raise ValueError(
                'native_hidden_class_aggregation requires hidden channels '
                'to equal num_classes * competition_group_N'
            )
        permutation = settings.get('class_index_permutation')
        if permutation is not None:
            if not isinstance(permutation, (list, tuple)):
                raise TypeError(
                    'native_hidden_class_aggregation.'
                    'class_index_permutation must be a list or tuple'
                )
            try:
                normalized = [int(value) for value in permutation]
            except (TypeError, ValueError) as error:
                raise TypeError(
                    'native_hidden_class_aggregation.'
                    'class_index_permutation entries must be integers'
                ) from error
            if any(
                isinstance(value, bool) or value != integer
                for value, integer in zip(permutation, normalized)
            ):
                raise TypeError(
                    'native_hidden_class_aggregation.'
                    'class_index_permutation entries must be integers'
                )
            if sorted(normalized) != list(range(int(self.num_classes))):
                raise ValueError(
                    'native_hidden_class_aggregation.'
                    'class_index_permutation must contain every class once'
                )

    def _ensure_native_hidden_class_aggregation_state(
        self, hidden_width, device, dtype
    ):
        expected_shape = (int(self.num_classes), int(hidden_width))
        sums = getattr(
            self, '_native_hidden_class_aggregation_sums', None
        )
        counts = getattr(
            self, '_native_hidden_class_aggregation_counts', None
        )
        if sums is None:
            self._native_hidden_class_aggregation_sums = torch.zeros(
                expected_shape, device=device, dtype=dtype
            )
            self._native_hidden_class_aggregation_counts = torch.zeros(
                int(self.num_classes), device=device, dtype=dtype
            )
            self._native_hidden_class_aggregation_state_version = 0
            return
        if tuple(sums.shape) != expected_shape:
            raise RuntimeError(
                'native hidden class aggregation width changed during '
                'training'
            )
        if sums.device != device or sums.dtype != dtype:
            self._native_hidden_class_aggregation_sums = sums.to(
                device=device, dtype=dtype
            )
            self._native_hidden_class_aggregation_counts = counts.to(
                device=device, dtype=dtype
            )

    def _normalized_native_hidden_class_preferences(self):
        sums = self._native_hidden_class_aggregation_sums
        counts = self._native_hidden_class_aggregation_counts
        class_means = sums / counts.clamp_min(1.0).unsqueeze(1)
        receiver_mass = class_means.sum(dim=0, keepdim=True)
        return torch.where(
            receiver_mass > 0,
            class_means / receiver_mass.clamp_min(1.0e-12),
            torch.zeros_like(class_means),
        )

    def _update_native_hidden_class_aggregation_state(self, labels):
        settings = self._get_native_hidden_class_aggregation_settings()
        if settings is None or not settings.get('enabled', False):
            return
        target = self.get_cortex_by_id(settings['target_cortex_id'])
        response = target.hidden_nv.earliness
        if response is None:
            raise RuntimeError(
                'native hidden class aggregation requires a completed '
                'first-pass hidden response'
            )
        response = response.detach().clamp_min(0)
        batch_size = int(response.shape[0])
        response = response.reshape(
            batch_size, -1, int(target.hidden_channel)
        ).mean(dim=1)
        self._ensure_native_hidden_class_aggregation_state(
            response.shape[1], response.device, response.dtype
        )
        labels = labels.to(device=response.device, dtype=torch.long)
        updated = False
        for class_index in range(int(self.num_classes)):
            mask = labels == class_index
            if bool(mask.any()):
                self._native_hidden_class_aggregation_sums[
                    class_index
                ].add_(response[mask].sum(dim=0))
                self._native_hidden_class_aggregation_counts[
                    class_index
                ].add_(mask.sum().to(dtype=response.dtype))
                updated = True
        if updated:
            self._native_hidden_class_aggregation_state_version = int(
                getattr(
                    self,
                    '_native_hidden_class_aggregation_state_version',
                    0,
                )
            ) + 1
        self._refresh_native_hidden_class_aggregation_metrics(
            target,
            settings,
            applied=(
                getattr(
                    target,
                    'native_hidden_class_aggregation_weights',
                    None,
                ) is not None
            ),
        )

    def _refresh_native_hidden_class_aggregation_metrics(
        self, target, settings, applied
    ):
        counts = getattr(
            self, '_native_hidden_class_aggregation_counts', None
        )
        metrics = {
            'enabled': float(settings.get('enabled', False)),
            'application_mode': float(
                settings.get('application', 'evaluation_only')
                == 'train_and_evaluation'
            ),
            'applied': float(applied),
            'state_version': float(getattr(
                self,
                '_native_hidden_class_aggregation_state_version',
                0,
            )),
            'class_seen_fraction': 0.0,
            'preference_entropy_normalized_mean': 1.0,
            'preference_top1_margin_mean': 0.0,
            'fixed_group_top1_agreement': 1.0,
            'aggregation_column_sum_min': 1.0,
            'aggregation_column_sum_max': 1.0,
            'class_permutation_identity': float(
                settings.get('class_index_permutation') is None
            ),
        }
        if counts is not None:
            metrics['class_seen_fraction'] = float(
                (counts > 0).to(dtype=counts.dtype).mean().item()
            )
            preferences = self._normalized_native_hidden_class_preferences()
            active = preferences.sum(dim=0) > 0
            sorted_preferences = preferences.sort(
                dim=0, descending=True
            ).values
            top_margin = sorted_preferences[0] - sorted_preferences[1]
            entropy = -(
                preferences
                * preferences.clamp_min(1.0e-12).log()
            ).sum(dim=0)
            if int(self.num_classes) > 1:
                entropy = entropy / torch.log(
                    entropy.new_tensor(float(self.num_classes))
                )
            if bool(active.any()):
                metrics['preference_entropy_normalized_mean'] = float(
                    entropy[active].mean().item()
                )
                metrics['preference_top1_margin_mean'] = float(
                    top_margin[active].mean().item()
                )
                fixed_assignment = torch.arange(
                    int(self.num_classes), device=preferences.device
                ).repeat_interleave(int(target.competition_group_N))
                metrics['fixed_group_top1_agreement'] = float((
                    preferences[:, active].argmax(dim=0)
                    == fixed_assignment[active]
                ).to(dtype=preferences.dtype).mean().item())
        weights = getattr(
            target, 'native_hidden_class_aggregation_weights', None
        )
        if weights is not None:
            column_sums = weights.sum(dim=0)
            metrics['aggregation_column_sum_min'] = float(
                column_sums.min().item()
            )
            metrics['aggregation_column_sum_max'] = float(
                column_sums.max().item()
            )
        self.native_hidden_class_aggregation_metrics = metrics

    def _set_native_hidden_class_aggregation(self, for_training):
        settings = self._get_native_hidden_class_aggregation_settings()
        target = self.cortex
        if settings is None:
            target.native_hidden_class_aggregation_weights = None
            return
        self._validate_native_hidden_class_aggregation_settings()
        target = self.get_cortex_by_id(settings['target_cortex_id'])
        enabled = bool(settings.get('enabled', False))
        apply_now = enabled and (
            not for_training
            or settings.get('application', 'evaluation_only')
            == 'train_and_evaluation'
        )
        counts = getattr(
            self, '_native_hidden_class_aggregation_counts', None
        )
        if (
            not apply_now
            or counts is None
            or not bool((counts > 0).all())
        ):
            target.native_hidden_class_aggregation_weights = None
            self._refresh_native_hidden_class_aggregation_metrics(
                target, settings, applied=False
            )
            return
        preferences = self._normalized_native_hidden_class_preferences()
        permutation = settings.get('class_index_permutation')
        if permutation is not None:
            permutation_tensor = torch.as_tensor(
                permutation,
                device=preferences.device,
                dtype=torch.long,
            )
            preferences = preferences[permutation_tensor]
        weights = preferences.transpose(0, 1)
        class_mass = weights.sum(dim=0, keepdim=True)
        if not bool((class_mass > 0).all()):
            target.native_hidden_class_aggregation_weights = None
            self._refresh_native_hidden_class_aggregation_metrics(
                target, settings, applied=False
            )
            return
        weights = weights / class_mass.clamp_min(1.0e-12)
        target.native_hidden_class_aggregation_weights = weights.detach()
        target.native_hidden_class_aggregation_build_count = int(getattr(
            target, 'native_hidden_class_aggregation_build_count', 0
        )) + 1
        self._refresh_native_hidden_class_aggregation_metrics(
            target, settings, applied=True
        )

    def _validate_presynaptic_class_preference_settings(self):
        settings = self._get_presynaptic_class_preference_settings()
        if settings is None:
            return
        allowed = {
            'enabled',
            'target_cortex_id',
            'source_cortex_id',
            'granularity',
            'strength',
            'application',
            'class_index_permutation',
            'wrong_receiver_class_index_permutation',
            'forward_application',
            'forward_strength',
            'forward_class_index_permutation',
            'forward_time_bin_count',
            'forward_time_bin_permutation',
            'forward_active_mass_matching',
            'sample_timing_emphasis',
            'sample_timing_strength',
        }
        unknown = set(settings) - allowed
        if unknown:
            raise ValueError(
                'presynaptic_class_preference contains unsupported field(s): '
                + ', '.join(sorted(unknown))
            )
        if not isinstance(settings.get('enabled', False), bool):
            raise TypeError(
                'presynaptic_class_preference.enabled must be bool'
            )
        for name in ('target_cortex_id', 'source_cortex_id'):
            value = settings.get(name)
            if not isinstance(value, str) or not value:
                raise ValueError(
                    f'presynaptic_class_preference.{name} must be a '
                    'non-empty cortex ID'
                )
        granularity = settings.get('granularity', 'flattened')
        if granularity not in {'flattened', 'channel_shared'}:
            raise ValueError(
                'presynaptic_class_preference.granularity must be one of '
                '{"flattened", "channel_shared"}'
            )
        strength = float(settings.get('strength', 1.0))
        if not 0.0 <= strength <= 1.0:
            raise ValueError(
                'presynaptic_class_preference.strength must be in [0, 1]'
            )
        application = settings.get('application', 'sample_label')
        if application not in {
            'none',
            'sample_label',
            'sample_label_correct_receiver',
            'sample_label_wrong_receiver',
            'sample_label_semantic_split',
            'receiver_class',
        }:
            raise ValueError(
                'presynaptic_class_preference.application must be one of '
                '{"none", "sample_label", '
                '"sample_label_correct_receiver", '
                '"sample_label_wrong_receiver", '
                '"sample_label_semantic_split", "receiver_class"}'
            )
        forward_application = settings.get('forward_application', 'none')
        if forward_application not in {'none', 'receiver_class'}:
            raise ValueError(
                'presynaptic_class_preference.forward_application must be '
                'one of {"none", "receiver_class"}'
            )
        forward_strength = float(settings.get('forward_strength', 1.0))
        if not 0.0 <= forward_strength <= 1.0:
            raise ValueError(
                'presynaptic_class_preference.forward_strength must be in '
                '[0, 1]'
            )
        forward_active_mass_matching = settings.get(
            'forward_active_mass_matching', False
        )
        if not isinstance(forward_active_mass_matching, bool):
            raise TypeError(
                'presynaptic_class_preference.'
                'forward_active_mass_matching must be bool'
            )
        if (
            forward_active_mass_matching
            and forward_application != 'receiver_class'
        ):
            raise ValueError(
                'forward_active_mass_matching requires '
                'forward_application receiver_class'
            )
        if (
            forward_application != 'none'
            and settings.get('source_cortex_id') != 'input'
        ):
            raise ValueError(
                'presynaptic class-preference forward routing currently '
                'supports source_cortex_id input only'
            )
        forward_time_bin_count = settings.get('forward_time_bin_count', 1)
        if (
            isinstance(forward_time_bin_count, bool)
            or not isinstance(forward_time_bin_count, int)
            or forward_time_bin_count < 1
        ):
            raise ValueError(
                'presynaptic_class_preference.forward_time_bin_count must '
                'be a positive integer'
            )
        if forward_time_bin_count > int(self.image_encoder.simulation_time):
            raise ValueError(
                'presynaptic_class_preference.forward_time_bin_count cannot '
                'exceed simulation_time'
            )
        if forward_time_bin_count > 1 and (
            forward_application != 'receiver_class'
            or settings.get('source_cortex_id') != 'input'
            or granularity != 'flattened'
        ):
            raise ValueError(
                'time-binned class-preference forward routing requires '
                'receiver_class, input source, and flattened granularity'
            )
        forward_time_bin_permutation = settings.get(
            'forward_time_bin_permutation'
        )
        if forward_time_bin_permutation is not None:
            if forward_time_bin_count <= 1:
                raise ValueError(
                    'forward_time_bin_permutation requires '
                    'forward_time_bin_count > 1'
                )
            if not isinstance(forward_time_bin_permutation, (list, tuple)):
                raise TypeError(
                    'forward_time_bin_permutation must be a list or tuple'
                )
            if len(forward_time_bin_permutation) != forward_time_bin_count:
                raise ValueError(
                    'forward_time_bin_permutation must contain one entry '
                    'per time bin'
                )
            if any(isinstance(value, bool) for value in forward_time_bin_permutation):
                raise TypeError(
                    'forward_time_bin_permutation entries must be integers'
                )
            normalized_time_permutation = [
                int(value) for value in forward_time_bin_permutation
            ]
            if any(
                value != normalized
                for value, normalized in zip(
                    forward_time_bin_permutation,
                    normalized_time_permutation,
                )
            ):
                raise TypeError(
                    'forward_time_bin_permutation entries must be integers'
                )
            if sorted(normalized_time_permutation) != list(
                range(forward_time_bin_count)
            ):
                raise ValueError(
                    'forward_time_bin_permutation must be a permutation of '
                    'all time-bin indices'
                )
        sample_timing_emphasis = settings.get(
            'sample_timing_emphasis', 'none'
        )
        if sample_timing_emphasis not in {'none', 'early', 'late'}:
            raise ValueError(
                'presynaptic_class_preference.sample_timing_emphasis '
                'must be one of {"none", "early", "late"}'
            )
        sample_timing_strength = float(settings.get(
            'sample_timing_strength', 1.0
        ))
        if not 0.0 <= sample_timing_strength <= 1.0:
            raise ValueError(
                'presynaptic_class_preference.sample_timing_strength '
                'must be in [0, 1]'
            )
        if (
            sample_timing_emphasis != 'none'
            and application not in {
                'sample_label',
                'sample_label_correct_receiver',
                'sample_label_wrong_receiver',
                'sample_label_semantic_split',
            }
        ):
            raise ValueError(
                'presynaptic_class_preference.sample_timing_emphasis '
                'requires a sample-label application'
            )
        for permutation_name in (
            'class_index_permutation',
            'wrong_receiver_class_index_permutation',
            'forward_class_index_permutation',
        ):
            permutation = settings.get(permutation_name)
            if permutation is None:
                continue
            if not isinstance(permutation, (list, tuple)):
                raise TypeError(
                    f'presynaptic_class_preference.{permutation_name} '
                    'must be a list or tuple'
                )
            if len(permutation) != int(self.num_classes):
                raise ValueError(
                    f'presynaptic_class_preference.{permutation_name} '
                    'must contain one entry per class'
                )
            if any(isinstance(value, bool) for value in permutation):
                raise TypeError(
                    f'presynaptic_class_preference.{permutation_name} '
                    'entries must be integer class indices'
                )
            try:
                normalized_permutation = [int(value) for value in permutation]
            except (TypeError, ValueError) as error:
                raise TypeError(
                    f'presynaptic_class_preference.{permutation_name} '
                    'entries must be integer class indices'
                ) from error
            if any(
                value != normalized
                for value, normalized in zip(
                    permutation, normalized_permutation
                )
            ):
                raise TypeError(
                    f'presynaptic_class_preference.{permutation_name} '
                    'entries must be integer class indices'
                )
            if sorted(normalized_permutation) != list(range(self.num_classes)):
                raise ValueError(
                    f'presynaptic_class_preference.{permutation_name} '
                    'must be a permutation of all class indices'
                )
        if (
            settings.get('wrong_receiver_class_index_permutation') is not None
            and application != 'sample_label_semantic_split'
        ):
            raise ValueError(
                'presynaptic_class_preference.'
                'wrong_receiver_class_index_permutation requires '
                'application sample_label_semantic_split'
            )
        if (
            settings.get('forward_class_index_permutation') is not None
            and forward_application != 'receiver_class'
        ):
            raise ValueError(
                'presynaptic_class_preference.'
                'forward_class_index_permutation requires '
                'forward_application receiver_class'
            )

    def _class_preference_response(self, response_holder, granularity):
        response = response_holder.earliness
        if response is None:
            raise RuntimeError(
                'presynaptic class preference requires a completed '
                'first-pass source response'
            )
        response = response.detach().clamp_min(0)
        if response.ndim < 2:
            raise RuntimeError(
                'presynaptic class preference source response must include '
                'batch and channel dimensions'
            )
        batch_size = response.shape[0]
        channel_count = response.shape[-1]
        positioned = response.reshape(batch_size, -1, channel_count)
        if granularity == 'channel_shared':
            return positioned.mean(dim=1), positioned.shape[1]
        return positioned.reshape(batch_size, -1), positioned.shape[1]

    @staticmethod
    def _class_preference_sample_timing_factor(
        response, response_holder, emphasis, strength
    ):
        active = response > 0
        active_float = active.to(dtype=response.dtype)
        active_count = active_float.sum(dim=1, keepdim=True)
        if emphasis == 'none' or strength == 0.0:
            factor = torch.ones_like(response)
        else:
            if emphasis == 'early':
                timing_score = response
            else:
                minimum_earliness = 1.0 / float(
                    response_holder.simulation_time
                )
                timing_score = active_float * (
                    1.0 + minimum_earliness - response
                )
            score_mean = timing_score.sum(
                dim=1, keepdim=True
            ) / active_count.clamp_min(1.0)
            normalized_score = torch.where(
                active,
                timing_score / score_mean.clamp_min(1.0e-12),
                torch.ones_like(response),
            )
            factor = 1.0 + strength * (normalized_score - 1.0)
            factor = torch.where(
                active_count > 0,
                factor,
                torch.ones_like(factor),
            )

        valid = active_count.squeeze(1) > 0
        unweighted = (
            (response * active_float).sum(dim=1)
            / active_count.squeeze(1).clamp_min(1.0)
        )
        timing_mass = (factor * active_float).sum(dim=1)
        weighted = (
            (response * factor * active_float).sum(dim=1)
            / timing_mass.clamp_min(1.0e-12)
        )
        if bool(valid.any()):
            unweighted_mean = unweighted[valid].mean()
            weighted_mean = weighted[valid].mean()
        else:
            unweighted_mean = response.new_tensor(0.0)
            weighted_mean = response.new_tensor(0.0)
        metrics = {
            'sample_timing_mode': {
                'none': 0.0,
                'early': 1.0,
                'late': 2.0,
            }[emphasis],
            'sample_timing_strength': float(
                strength if emphasis != 'none' else 0.0
            ),
            'sample_timing_active_fraction': float(
                active_float.mean().item()
            ),
            'sample_timing_factor_mean': float(factor.mean().item()),
            'sample_timing_factor_q10': float(
                torch.quantile(factor, 0.10).item()
            ),
            'sample_timing_factor_q90': float(
                torch.quantile(factor, 0.90).item()
            ),
            'sample_timing_unweighted_earliness': float(
                unweighted_mean.item()
            ),
            'sample_timing_weighted_earliness': float(
                weighted_mean.item()
            ),
            'sample_timing_earliness_shift': float(
                (weighted_mean - unweighted_mean).item()
            ),
        }
        return factor, active, metrics

    def _ensure_class_preference_state(
        self, feature_count, granularity, device, dtype,
        temporal_bin_count=1,
    ):
        expected_shape = (int(self.num_classes), int(feature_count))
        sums = getattr(self, '_class_preference_sums', None)
        counts = getattr(self, '_class_preference_counts', None)
        state_granularity = getattr(
            self, '_class_preference_granularity', None
        )
        if sums is None:
            self._class_preference_sums = torch.zeros(
                expected_shape, device=device, dtype=dtype
            )
            self._class_preference_counts = torch.zeros(
                int(self.num_classes), device=device, dtype=dtype
            )
            self._class_preference_granularity = granularity
            self._class_preference_state_version = 0
        elif tuple(sums.shape) != expected_shape or state_granularity != granularity:
            raise RuntimeError(
                'presynaptic class preference shape or granularity changed '
                'during training'
            )
        elif sums.device != device or sums.dtype != dtype:
            self._class_preference_sums = sums.to(
                device=device, dtype=dtype
            )
            self._class_preference_counts = counts.to(
                device=device, dtype=dtype
            )
        temporal_bin_count = int(temporal_bin_count)
        if temporal_bin_count <= 1:
            return
        expected_temporal_shape = (
            int(self.num_classes), temporal_bin_count, int(feature_count)
        )
        temporal_sums = getattr(
            self, '_class_preference_temporal_sums', None
        )
        state_bin_count = getattr(
            self, '_class_preference_temporal_bin_count', None
        )
        if temporal_sums is None:
            self._class_preference_temporal_sums = torch.zeros(
                expected_temporal_shape, device=device, dtype=dtype
            )
            self._class_preference_temporal_bin_count = temporal_bin_count
        elif (
            tuple(temporal_sums.shape) != expected_temporal_shape
            or state_bin_count != temporal_bin_count
        ):
            raise RuntimeError(
                'presynaptic temporal class preference shape or bin count '
                'changed during training'
            )
        elif temporal_sums.device != device or temporal_sums.dtype != dtype:
            self._class_preference_temporal_sums = temporal_sums.to(
                device=device, dtype=dtype
            )

    def _normalized_class_preferences(self):
        sums = self._class_preference_sums
        counts = self._class_preference_counts
        class_means = sums / counts.clamp_min(1.0).unsqueeze(1)
        mass = class_means.sum(dim=0, keepdim=True)
        uniform = torch.full_like(
            class_means, 1.0 / float(self.num_classes)
        )
        return torch.where(
            mass > 0,
            class_means / mass.clamp_min(1.0e-12),
            uniform,
        )

    def _normalized_class_temporal_preferences(self):
        sums = self._class_preference_temporal_sums
        counts = self._class_preference_counts
        class_means = sums / counts.clamp_min(1.0).view(-1, 1, 1)
        mass = class_means.sum(dim=0, keepdim=True)
        uniform = torch.full_like(
            class_means, 1.0 / float(self.num_classes)
        )
        return torch.where(
            mass > 0,
            class_means / mass.clamp_min(1.0e-12),
            uniform,
        )

    def _update_class_preference_state(
        self, response, labels, temporal_bin_count=1, simulation_time=None
    ):
        labels = labels.to(device=response.device, dtype=torch.long)
        temporal_bin_count = int(temporal_bin_count)
        response_bins = None
        if temporal_bin_count > 1:
            simulation_time = int(simulation_time)
            first_step = torch.floor(
                (1.0 - response.clamp(max=1.0)) * simulation_time
            ).to(dtype=torch.long)
            first_step = first_step.clamp(0, simulation_time - 1)
            response_bins = (
                first_step * temporal_bin_count // simulation_time
            ).clamp(0, temporal_bin_count - 1)
        updated = False
        for class_index in range(int(self.num_classes)):
            mask = labels == class_index
            if bool(mask.any()):
                class_response = response[mask]
                self._class_preference_sums[class_index].add_(
                    class_response.sum(dim=0)
                )
                if response_bins is not None:
                    class_bins = response_bins[mask]
                    for bin_index in range(temporal_bin_count):
                        self._class_preference_temporal_sums[
                            class_index, bin_index
                        ].add_((
                            class_response
                            * (class_bins == bin_index).to(
                                dtype=class_response.dtype
                            )
                        ).sum(dim=0))
                self._class_preference_counts[class_index].add_(
                    mask.sum().to(dtype=response.dtype)
                )
                updated = True
        if updated:
            self._class_preference_state_version = int(getattr(
                self, '_class_preference_state_version', 0
            )) + 1

    def _clear_presynaptic_class_preference_forward_projection(self):
        for cortex in self.cortex.iter_cortex_tree():
            cortex.direct_class_preference_forward_multiplier = None
            cortex.direct_class_preference_forward_temporal_multiplier = None
            cortex.direct_class_preference_forward_active_mass_matching = False
            cortex.branch_class_preference_forward_multiplier = None
            cortex._direct_class_preference_forward_weight_cache = None
            cortex._branch_class_preference_forward_weight_cache = None
            cortex._class_preference_forward_projection_state_version = None
        self._class_preference_forward_projection_active = False

    def _set_presynaptic_class_preference_forward_projection(self):
        """Apply a label-free, train-state-only preference to native current."""
        settings = self._get_presynaptic_class_preference_settings()
        if (
            settings is None
            or not settings.get('enabled', False)
            or settings.get('forward_application', 'none') == 'none'
        ):
            if getattr(
                self, '_class_preference_forward_projection_active', False
            ):
                self._clear_presynaptic_class_preference_forward_projection()
            return
        self._validate_presynaptic_class_preference_settings()
        self._class_preference_forward_projection_active = True
        target = self.get_cortex_by_id(settings['target_cortex_id'])
        state_version = int(getattr(
            self, '_class_preference_state_version', 0
        ))
        if getattr(
            target,
            '_class_preference_forward_projection_state_version',
            None,
        ) == state_version:
            return
        if not hasattr(self, '_class_preference_sums'):
            target.direct_class_preference_forward_multiplier = None
            target.direct_class_preference_forward_temporal_multiplier = None
            target.direct_class_preference_forward_active_mass_matching = False
            target.branch_class_preference_forward_multiplier = None
            target._class_preference_forward_projection_state_version = (
                state_version
            )
            return

        temporal_bin_count = int(settings.get('forward_time_bin_count', 1))
        if temporal_bin_count > 1:
            if not hasattr(self, '_class_preference_temporal_sums'):
                target.direct_class_preference_forward_multiplier = None
                target.direct_class_preference_forward_temporal_multiplier = None
                target.direct_class_preference_forward_active_mass_matching = False
                target._class_preference_forward_projection_state_version = (
                    state_version
                )
                return
            preferences = self._normalized_class_temporal_preferences()
            time_permutation = settings.get('forward_time_bin_permutation')
            if time_permutation is not None:
                time_permutation_tensor = torch.as_tensor(
                    time_permutation,
                    device=preferences.device,
                    dtype=torch.long,
                )
                preferences = preferences[:, time_permutation_tensor, :]
        else:
            preferences = self._normalized_class_preferences()
        direct_len = int(target.kernel.input_len)
        receiver_len = int(target.kernel.weight.shape[1])
        granularity = settings.get('granularity', 'flattened')
        if granularity == 'channel_shared':
            if direct_len % int(preferences.shape[1]) != 0:
                raise RuntimeError(
                    'class-preference forward routing cannot expand the '
                    'channel-shared preference to the target direct block'
                )
            position_count = direct_len // int(preferences.shape[1])
            preferences = (
                preferences.unsqueeze(1)
                .expand(-1, position_count, -1)
                .reshape(self.num_classes, -1)
            )
        feature_dim = 2 if temporal_bin_count > 1 else 1
        if int(preferences.shape[feature_dim]) != direct_len:
            raise RuntimeError(
                'class-preference forward routing does not match the target '
                'direct-input width'
            )
        receiver_classes = self._get_output_label_classes(
            preferences.device, output_len=receiver_len
        )
        preference_classes = receiver_classes
        permutation = settings.get('forward_class_index_permutation')
        if permutation is not None:
            permutation_tensor = torch.as_tensor(
                permutation,
                device=preferences.device,
                dtype=torch.long,
            )
            preference_classes = permutation_tensor[receiver_classes]
        if temporal_bin_count > 1:
            class_multiplier = preferences.permute(1, 2, 0) * float(
                self.num_classes
            )
        else:
            class_multiplier = preferences.transpose(0, 1) * float(
                self.num_classes
            )
        strength = float(settings.get('forward_strength', 1.0))
        class_multiplier = 1.0 + strength * (class_multiplier - 1.0)
        multiplier_mean = class_multiplier.mean(
            dim=1 if temporal_bin_count > 1 else 0,
            keepdim=True,
        )
        class_multiplier = torch.where(
            multiplier_mean > 0,
            class_multiplier / multiplier_mean.clamp_min(1.0e-12),
            torch.ones_like(class_multiplier),
        )
        if temporal_bin_count > 1:
            multiplier = class_multiplier[:, :, preference_classes].detach()
            expected_multiplier_shape = (
                temporal_bin_count, direct_len, receiver_len
            )
        else:
            multiplier = class_multiplier[:, preference_classes].detach()
            expected_multiplier_shape = (direct_len, receiver_len)
        if tuple(multiplier.shape) != expected_multiplier_shape:
            raise RuntimeError(
                'class-preference forward multiplier has an invalid shape'
            )
        target.direct_class_preference_forward_multiplier = (
            None if temporal_bin_count > 1 else multiplier
        )
        target.direct_class_preference_forward_temporal_multiplier = (
            multiplier if temporal_bin_count > 1 else None
        )
        target.direct_class_preference_forward_active_mass_matching = bool(
            settings.get('forward_active_mass_matching', False)
        )
        target.branch_class_preference_forward_multiplier = None
        target._class_preference_forward_projection_state_version = (
            state_version
        )
        target.class_preference_forward_build_count = int(getattr(
            target, 'class_preference_forward_build_count', 0
        )) + 1
        target.class_preference_forward_multiplier_mean = float(
            class_multiplier.mean().item()
        )
        target.class_preference_forward_multiplier_q10 = float(
            torch.quantile(class_multiplier, 0.10).item()
        )
        target.class_preference_forward_multiplier_q90 = float(
            torch.quantile(class_multiplier, 0.90).item()
        )
        target.class_preference_forward_multiplier_min = float(
            class_multiplier.min().item()
        )
        target.class_preference_forward_multiplier_max = float(
            class_multiplier.max().item()
        )
        target.class_preference_forward_multiplier_abs_deviation = float(
            (class_multiplier - 1.0).abs().mean().item()
        )
        target.class_preference_forward_time_bin_count = float(
            temporal_bin_count
        )
        target.class_preference_forward_time_permutation_identity = float(
            settings.get('forward_time_bin_permutation') is None
            or all(
                int(value) == index
                for index, value in enumerate(
                    settings.get('forward_time_bin_permutation', [])
                )
            )
        )

    def _presynaptic_class_preference_weight_metrics(self):
        settings = self._get_presynaptic_class_preference_settings()
        if (
            settings is None
            or not settings.get('enabled', False)
            or not hasattr(self, '_class_preference_sums')
        ):
            return {}
        target = self.get_cortex_by_id(settings['target_cortex_id'])
        source_id = settings['source_cortex_id']
        if source_id != 'input':
            self.get_cortex_by_id(source_id)
        granularity = settings.get('granularity', 'flattened')
        preferences = self._normalized_class_preferences()
        source_offset = getattr(
            self, '_class_preference_source_offset', None
        )
        source_width = getattr(
            self, '_class_preference_source_width', None
        )
        if source_offset is None or source_width is None:
            return {}
        if granularity == 'channel_shared':
            position_count = source_width // preferences.shape[1]
            preferences = (
                preferences.unsqueeze(1)
                .expand(-1, position_count, -1)
                .reshape(self.num_classes, -1)
            )
        branch_weights = target.kernel.weight[
            source_offset:source_offset + source_width, :
        ].detach().abs()
        preference_weight_mass = preferences @ branch_weights
        receiver_classes = self._get_output_label_classes(
            branch_weights.device, output_len=branch_weights.shape[1]
        )
        matrix = torch.stack([
            preference_weight_mass[:, receiver_classes == class_index].mean(
                dim=1
            )
            for class_index in range(self.num_classes)
        ], dim=1)
        matrix = matrix / matrix.sum(
            dim=0, keepdim=True
        ).clamp_min(1.0e-12)
        diagonal = matrix.diagonal()
        off_diagonal_mean = (
            (matrix.sum() - diagonal.sum())
            / float(self.num_classes * (self.num_classes - 1))
        )
        entropy = -(
            matrix * matrix.clamp_min(1.0e-12).log()
        ).sum(dim=0) / torch.log(
            matrix.new_tensor(float(self.num_classes))
        )
        metrics = {
            'weight_class_diagonal_mean': float(diagonal.mean().item()),
            'weight_class_off_diagonal_mean': float(
                off_diagonal_mean.item()
            ),
            'weight_class_diagonal_margin': float(
                (diagonal.mean() - off_diagonal_mean).item()
            ),
            'weight_class_entropy_normalized_mean': float(
                entropy.mean().item()
            ),
            'weight_class_top1_agreement': float((
                matrix.argmax(dim=0)
                == torch.arange(
                    self.num_classes, device=matrix.device
                )
            ).to(dtype=matrix.dtype).mean().item()),
        }
        for receiver_class in range(self.num_classes):
            for sender_class in range(self.num_classes):
                metrics[
                    'weight_class_matrix/'
                    f'receiver_{receiver_class}/sender_{sender_class}'
                ] = float(matrix[sender_class, receiver_class].item())
        return metrics

    def _set_presynaptic_class_preference_modulation(
        self, labels, settings
    ):
        self._validate_presynaptic_class_preference_settings()
        target = self.get_cortex_by_id(settings['target_cortex_id'])
        source_id = settings['source_cortex_id']
        source_is_target_input = source_id == 'input'
        if source_is_target_input:
            source = None
            response_holder = target.input_nv
        else:
            source = self.get_cortex_by_id(source_id)
            response_holder = source.output_nv
            if source not in target.subcortexs:
                raise ValueError(
                    'presynaptic class preference source must be the target '
                    'input or a direct subcortex of the target cortex'
                )
        granularity = settings.get('granularity', 'flattened')
        response, position_count = self._class_preference_response(
            response_holder, granularity
        )
        self._ensure_class_preference_state(
            response.shape[1],
            granularity,
            response.device,
            response.dtype,
            temporal_bin_count=int(settings.get(
                'forward_time_bin_count', 1
            )),
        )

        normalized = self._normalized_class_preferences()
        labels = labels.to(device=response.device, dtype=torch.long)
        permutation = settings.get('class_index_permutation')
        permutation_tensor = None
        if permutation is not None:
            permutation_tensor = torch.as_tensor(
                permutation, device=response.device, dtype=torch.long
            )
        wrong_receiver_permutation = settings.get(
            'wrong_receiver_class_index_permutation'
        )
        wrong_receiver_permutation_tensor = None
        if wrong_receiver_permutation is not None:
            wrong_receiver_permutation_tensor = torch.as_tensor(
                wrong_receiver_permutation,
                device=response.device,
                dtype=torch.long,
            )
        application = settings.get('application', 'sample_label')
        strength = float(settings.get('strength', 1.0))
        sender_width = int(target.kernel.weight.shape[0])
        receiver_width = int(target.kernel.weight.shape[1])
        branch_offset = int(target.kernel.input_len)
        found_source = source_is_target_input
        if source_is_target_input:
            source_offset = 0
            source_width = branch_offset
        for branch in target.subcortexs:
            branch_width = int(branch.output_nv.earliness.reshape(
                response.shape[0], -1
            ).shape[1])
            if branch is source:
                source_offset = branch_offset
                source_width = branch_width
                found_source = True
            branch_offset += branch_width
        if not found_source or branch_offset != sender_width:
            raise RuntimeError(
                'presynaptic class preference could not map source branch '
                'onto target sender rows'
            )
        self._class_preference_source_offset = int(source_offset)
        self._class_preference_source_width = int(source_width)

        timing_metrics = {
            'sample_timing_mode': 0.0,
            'sample_timing_strength': 0.0,
            'sample_timing_active_fraction': float(
                (response > 0).to(dtype=response.dtype).mean().item()
            ),
            'sample_timing_factor_mean': 1.0,
            'sample_timing_factor_q10': 1.0,
            'sample_timing_factor_q90': 1.0,
            'sample_timing_unweighted_earliness': 0.0,
            'sample_timing_weighted_earliness': 0.0,
            'sample_timing_earliness_shift': 0.0,
            'sample_timing_active_mass_ratio': 1.0,
        }
        if application in {
            'sample_label',
            'sample_label_correct_receiver',
            'sample_label_wrong_receiver',
            'sample_label_semantic_split',
        }:
            preference_labels = labels
            if permutation_tensor is not None:
                preference_labels = permutation_tensor[labels]
            multiplier = normalized[preference_labels] * float(
                self.num_classes
            )
            multiplier = 1.0 + strength * (multiplier - 1.0)
            multiplier_mean = multiplier.mean(dim=1, keepdim=True)
            multiplier = torch.where(
                multiplier_mean > 0,
                multiplier / multiplier_mean.clamp_min(1.0e-12),
                torch.ones_like(multiplier),
            )
            base_multiplier = multiplier
            sample_timing_emphasis = settings.get(
                'sample_timing_emphasis', 'none'
            )
            sample_timing_strength = float(settings.get(
                'sample_timing_strength', 1.0
            ))
            timing_factor, active_sender, timing_metrics = (
                self._class_preference_sample_timing_factor(
                    response,
                    response_holder,
                    sample_timing_emphasis,
                    sample_timing_strength,
                )
            )
            multiplier = base_multiplier * timing_factor
            active_float = active_sender.to(dtype=response.dtype)
            base_active_mass = (
                base_multiplier * active_float
            ).sum(dim=1, keepdim=True)
            modulated_active_mass = (
                multiplier * active_float
            ).sum(dim=1, keepdim=True)
            active_mass_scale = torch.where(
                modulated_active_mass > 0,
                base_active_mass
                / modulated_active_mass.clamp_min(1.0e-12),
                torch.ones_like(modulated_active_mass),
            )
            multiplier = torch.where(
                active_sender,
                multiplier * active_mass_scale,
                base_multiplier,
            )
            matched_active_mass = (
                multiplier * active_float
            ).sum(dim=1, keepdim=True)
            valid_active_mass = base_active_mass.squeeze(1) > 0
            if bool(valid_active_mass.any()):
                active_mass_ratio = (
                    matched_active_mass.squeeze(1)[valid_active_mass]
                    / base_active_mass.squeeze(1)[valid_active_mass]
                ).mean()
            else:
                active_mass_ratio = response.new_tensor(1.0)
            timing_metrics['sample_timing_active_mass_ratio'] = float(
                active_mass_ratio.item()
            )
            if granularity == 'channel_shared':
                multiplier = (
                    multiplier.unsqueeze(1)
                    .expand(-1, position_count, -1)
                    .reshape(response.shape[0], -1)
                )
            if source_width != multiplier.shape[1]:
                raise RuntimeError(
                    'presynaptic class preference multiplier does not '
                    'match the source branch width'
                )
            full_multiplier = torch.ones(
                response.shape[0],
                sender_width,
                device=response.device,
                dtype=response.dtype,
            )
            full_multiplier[
                :, source_offset:source_offset + source_width
            ] = multiplier
            if application == 'sample_label':
                target.presynaptic_sender_modulation = full_multiplier
            else:
                receiver_classes = self._get_output_label_classes(
                    response.device, output_len=receiver_width
                )
                receiver_gate = (
                    receiver_classes.unsqueeze(0) == labels.unsqueeze(1)
                )
                if application == 'sample_label_wrong_receiver':
                    receiver_gate = ~receiver_gate
                target.presynaptic_selective_sender_modulation = (
                    full_multiplier
                )
                target.presynaptic_selective_receiver_gate = (
                    receiver_gate.to(dtype=response.dtype)
                )
                if application == 'sample_label_semantic_split':
                    wrong_preference_classes = receiver_classes
                    if wrong_receiver_permutation_tensor is not None:
                        wrong_preference_classes = (
                            wrong_receiver_permutation_tensor[receiver_classes]
                        )
                    wrong_multiplier = normalized[
                        wrong_preference_classes
                    ].transpose(0, 1)
                    wrong_multiplier = wrong_multiplier * float(
                        self.num_classes
                    )
                    wrong_multiplier = 1.0 + strength * (
                        wrong_multiplier - 1.0
                    )
                    wrong_multiplier_mean = wrong_multiplier.mean(
                        dim=0, keepdim=True
                    )
                    wrong_multiplier = torch.where(
                        wrong_multiplier_mean > 0,
                        wrong_multiplier
                        / wrong_multiplier_mean.clamp_min(1.0e-12),
                        torch.ones_like(wrong_multiplier),
                    )
                    if granularity == 'channel_shared':
                        wrong_multiplier = (
                            wrong_multiplier.unsqueeze(0)
                            .expand(position_count, -1, -1)
                            .reshape(-1, receiver_width)
                        )
                    if tuple(wrong_multiplier.shape) != (
                        source_width, receiver_width
                    ):
                        raise RuntimeError(
                            'semantic-split wrong-receiver multiplier does '
                            'not match source branch and receiver widths'
                        )
                    full_wrong_multiplier = torch.ones(
                        sender_width,
                        receiver_width,
                        device=response.device,
                        dtype=response.dtype,
                    )
                    full_wrong_multiplier[
                        source_offset:source_offset + source_width, :
                    ] = wrong_multiplier
                    target.presynaptic_selective_receiver_modulation = (
                        full_wrong_multiplier
                    )
                    target.presynaptic_selective_receiver_modulation_gate = (
                        (~receiver_gate).to(dtype=response.dtype)
                    )
                    timing_metrics.update({
                        'wrong_receiver_multiplier_mean': float(
                            wrong_multiplier.mean().item()
                        ),
                        'wrong_receiver_multiplier_q10': float(
                            torch.quantile(wrong_multiplier, 0.10).item()
                        ),
                        'wrong_receiver_multiplier_q90': float(
                            torch.quantile(wrong_multiplier, 0.90).item()
                        ),
                    })
            branch_multiplier = multiplier
        elif application == 'receiver_class':
            receiver_classes = self._get_output_label_classes(
                response.device, output_len=receiver_width
            )
            preference_classes = receiver_classes
            if permutation_tensor is not None:
                preference_classes = permutation_tensor[receiver_classes]
            multiplier = normalized[preference_classes].transpose(0, 1)
            multiplier = multiplier * float(self.num_classes)
            multiplier = 1.0 + strength * (multiplier - 1.0)
            multiplier_mean = multiplier.mean(dim=0, keepdim=True)
            multiplier = torch.where(
                multiplier_mean > 0,
                multiplier / multiplier_mean.clamp_min(1.0e-12),
                torch.ones_like(multiplier),
            )
            if granularity == 'channel_shared':
                multiplier = (
                    multiplier.unsqueeze(0)
                    .expand(position_count, -1, -1)
                    .reshape(-1, receiver_width)
                )
            if tuple(multiplier.shape) != (source_width, receiver_width):
                raise RuntimeError(
                    'presynaptic receiver multiplier does not match the '
                    'source branch and target receiver widths'
                )
            full_multiplier = torch.ones(
                sender_width,
                receiver_width,
                device=response.device,
                dtype=response.dtype,
            )
            full_multiplier[
                source_offset:source_offset + source_width, :
            ] = multiplier
            target.presynaptic_receiver_modulation = full_multiplier
            branch_multiplier = multiplier
        else:
            branch_multiplier = torch.ones(
                source_width,
                device=response.device,
                dtype=response.dtype,
            )
            timing_metrics = {
                'sample_timing_mode': 0.0,
                'sample_timing_strength': 0.0,
                'sample_timing_active_fraction': float(
                    (response > 0).to(dtype=response.dtype).mean().item()
                ),
                'sample_timing_factor_mean': 1.0,
                'sample_timing_factor_q10': 1.0,
                'sample_timing_factor_q90': 1.0,
                'sample_timing_unweighted_earliness': 0.0,
                'sample_timing_weighted_earliness': 0.0,
                'sample_timing_earliness_shift': 0.0,
                'sample_timing_active_mass_ratio': 1.0,
            }

        preference_learning_enabled = bool(
            target.learning_enabled
            if source_is_target_input
            else source.learning_enabled
        )
        if preference_learning_enabled:
            self._update_class_preference_state(
                response,
                labels,
                temporal_bin_count=int(settings.get(
                    'forward_time_bin_count', 1
                )),
                simulation_time=int(response_holder.simulation_time),
            )
            if settings.get('forward_application', 'none') != 'none':
                # Keep the causal projection selected before this batch fixed
                # throughout its second pass. The next sequence rebuilds it
                # from the newly accumulated training-only state.
                target._class_preference_forward_projection_state_version = (
                    int(getattr(
                        self, '_class_preference_state_version', 0
                    ))
                )
        updated = self._normalized_class_preferences()
        sorted_preferences = updated.sort(dim=0, descending=True).values
        entropy = -(
            updated * updated.clamp_min(1.0e-12).log()
        ).sum(dim=0)
        if int(self.num_classes) > 1:
            entropy = entropy / torch.log(
                entropy.new_tensor(float(self.num_classes))
            )
        self.presynaptic_class_preference_metrics = {
            'enabled': 1.0,
            'preference_learning_enabled': float(
                preference_learning_enabled
            ),
            'source_mode': float(source_is_target_input),
            'granularity': (
                0.0 if granularity == 'flattened' else 1.0
            ),
            'application_mode': {
                'none': 0.0,
                'sample_label': 1.0,
                'receiver_class': 2.0,
                'sample_label_correct_receiver': 3.0,
                'sample_label_wrong_receiver': 4.0,
                'sample_label_semantic_split': 5.0,
            }[application],
            'class_lookup_identity': float(
                permutation is None
                or all(
                    int(value) == class_index
                    for class_index, value in enumerate(permutation)
                )
            ),
            'class_lookup_fixed_point_fraction': float(
                1.0 if permutation is None else sum(
                    int(value) == class_index
                    for class_index, value in enumerate(permutation)
                ) / float(self.num_classes)
            ),
            'wrong_receiver_class_lookup_identity': float(
                wrong_receiver_permutation is None
                or all(
                    int(value) == class_index
                    for class_index, value in enumerate(
                        wrong_receiver_permutation
                    )
                )
            ),
            'wrong_receiver_class_lookup_fixed_point_fraction': float(
                1.0 if wrong_receiver_permutation is None else sum(
                    int(value) == class_index
                    for class_index, value in enumerate(
                        wrong_receiver_permutation
                    )
                ) / float(self.num_classes)
            ),
            'class_seen_fraction': float(
                (self._class_preference_counts > 0)
                .to(dtype=response.dtype).mean().item()
            ),
            'preference_entropy_normalized_mean': float(entropy.mean().item()),
            'preference_top1_margin_mean': float(
                (sorted_preferences[0] - sorted_preferences[1]).mean().item()
            ),
            'multiplier_mean': float(branch_multiplier.mean().item()),
            'multiplier_q10': float(
                torch.quantile(branch_multiplier, 0.10).item()
            ),
            'multiplier_q90': float(
                torch.quantile(branch_multiplier, 0.90).item()
            ),
            'multiplier_min': float(branch_multiplier.min().item()),
            'multiplier_max': float(branch_multiplier.max().item()),
            **timing_metrics,
        }

    def _clear_presynaptic_sender_modulation(self):
        for cortex in self.cortex.iter_cortex_tree():
            cortex.presynaptic_sender_modulation = None
            cortex.presynaptic_receiver_modulation = None
            cortex.presynaptic_selective_sender_modulation = None
            cortex.presynaptic_selective_receiver_gate = None
            cortex.presynaptic_selective_receiver_modulation = None
            cortex.presynaptic_selective_receiver_modulation_gate = None

    def _prepare_cortex_spec(self, cortex_spec):
        cortex_spec = copy.deepcopy(cortex_spec)
        base_cortex_settings = cortex_spec.setdefault('base_cortex_settings', {})
        if self.first_spike_delay_factor is not None:
            base_cortex_settings['first_spike_delay_factor'] = (
                self.first_spike_delay_factor
            )
            base_cortex_settings['mean_spike_delay_factor'] = (
                self.mean_spike_delay_factor
            )
        threshold_settings = cortex_spec.setdefault(
            'neuron_vectors_spec', {}
        ).setdefault('threshold_settings', {})
        threshold_settings['target_mean_earliness'] = \
            self.target_mean_earliness
        return cortex_spec

    @staticmethod
    def _earliness_from_spike_wave_sum(spike_wave_sum, simulation_time):
        if simulation_time <= 0:
            raise ValueError('simulation_time must be positive')
        return spike_wave_sum / float(simulation_time)

    def _earliness_from_wave_sum(self, spike_wave_sum):
        return self._earliness_from_spike_wave_sum(
            spike_wave_sum,
            self.image_encoder.simulation_time
        )

    @classmethod
    def score_output_neurons(cls, spike_train):
        return spike_train.sum(dim=-1) / float(spike_train.shape[-1])

    def score_output_spike_train(self, spike_train):
        neuron_scores = self.score_output_neurons(spike_train)
        return self.score_output_neuron_scores(neuron_scores)

    def _get_streamed_output_neuron_scores(self, images, is_training=False):
        earliness_scores = None
        simulation_time = self.image_encoder.simulation_time

        for time_step, spikes in enumerate(self.image_encoder(images)):
            output_wave = self.forward_cortex(
                spikes,
                is_training=is_training
            )

            if earliness_scores is None:
                earliness_scores = torch.zeros_like(output_wave)
            earliness_scores += output_wave / float(simulation_time)

        return earliness_scores

    def _get_input_spike_sequence(self, images):
        spike_sequence = torch.stack(list(self.image_encoder(images)), dim=0)
        progress_sequence = getattr(
            self.image_encoder,
            'temporal_progress_sequence',
            None,
        )
        if progress_sequence is not None:
            self.cortex.temporal_receiver_progress_sequence = (
                progress_sequence.transpose(0, 1)
            )
            self.cortex.temporal_receiver_progress = None
        return spike_sequence

    def _get_non_streaming_output_neuron_scores(self, images, is_training=False):
        input_spike_sequence = self._get_input_spike_sequence(images)
        output_wave_sequence, _ = self.forward_cortex_non_streaming(
            input_spike_sequence,
            labels=None,
            label_usage='ignore',
            label_propagating=False,
            STDP_interval=None,
            is_training=is_training
        )
        return (
            output_wave_sequence.sum(dim=0)
            / float(self.image_encoder.simulation_time)
        )

    def score_output_neuron_scores(self, neuron_scores):
        batch_size = neuron_scores.shape[0]
        class_scores = neuron_scores.reshape(batch_size, -1)
        if class_scores.shape[-1] != self.num_classes:
            raise ValueError(
                'cortex output must contain exactly one score per class; '
                f'got {class_scores.shape[-1]} scores for '
                f'{self.num_classes} classes'
            )
        return class_scores

    @staticmethod
    def _mean_cortex_channels(values):
        batch_size = values.shape[0]
        neuron_num = values.shape[-1]
        return values.reshape(batch_size, -1, neuron_num).mean(dim=1)

    @staticmethod
    def _select_label_aligned_decision_field(
        decision_snapshot,
        output_key,
        receiver_key,
        output_len,
        context
    ):
        output_values = decision_snapshot[output_key]
        if output_values is not None and output_values.shape[-1] == output_len:
            return output_values

        receiver_values = decision_snapshot[receiver_key]
        if receiver_values is not None and receiver_values.shape[-1] == output_len:
            return receiver_values

        raise ValueError(
            f'{context} requires first-spike decision fields aligned to '
            f'output_len={output_len}'
        )

    def _get_classwise_max_earliness(self, labels, earliness, output_len):
        output_label_classes = self._get_output_label_classes(
            labels.device,
            output_len=output_len
        )
        class_ids = torch.arange(self.num_classes, device=labels.device)
        class_member_mask = (
            output_label_classes.view(1, 1, -1)
            == class_ids.view(1, -1, 1)
        )
        class_earliness = earliness.unsqueeze(1).masked_fill(
            ~class_member_mask,
            float('-inf')
        ).amax(dim=-1)
        return torch.where(
            torch.isfinite(class_earliness),
            class_earliness,
            torch.zeros_like(class_earliness)
        )

    def get_hinge_temporal_margin_stats(self, images, labels):
        self._validate_hinge_softmax_label_settings()
        output_len = self.cortex.shape[1]
        decision_snapshot = self._get_first_spike_decision_snapshot(
            images,
            collect_output_first_spike_times=True,
            collect_receiver_first_spike_times=True
        )
        first_spike_times = self._select_label_aligned_decision_field(
            decision_snapshot,
            'output_first_spike_times',
            'receiver_first_spike_times',
            output_len,
            'hinge temporal diagnostics'
        )
        first_spike_times = first_spike_times.to(device=labels.device)
        has_spike = first_spike_times >= 0
        first_spike_earliness = (
            (
                float(self.image_encoder.simulation_time)
                - first_spike_times.clamp_min(0).to(dtype=torch.float32)
            )
            / float(self.image_encoder.simulation_time)
        )
        first_spike_earliness = torch.where(
            has_spike,
            first_spike_earliness,
            torch.zeros_like(first_spike_earliness)
        )

        class_earliness = self._get_classwise_max_earliness(
            labels,
            first_spike_earliness,
            output_len
        )
        class_ids = torch.arange(self.num_classes, device=labels.device)
        true_class_mask = labels.unsqueeze(1) == class_ids.unsqueeze(0)
        T_correct = class_earliness.gather(1, labels.unsqueeze(1))
        if self.num_classes > 1:
            T_wrong = class_earliness.masked_fill(
                true_class_mask,
                float('-inf')
            ).amax(dim=1, keepdim=True)
            T_wrong = torch.where(
                torch.isfinite(T_wrong),
                T_wrong,
                torch.zeros_like(T_wrong)
            )
        else:
            T_wrong = torch.zeros_like(T_correct)

        per_class_count = torch.bincount(
            labels,
            minlength=self.num_classes
        ).to(dtype=torch.float64)
        per_class_T_correct_sum = torch.zeros(
            self.num_classes,
            device=labels.device,
            dtype=torch.float64
        )
        per_class_T_wrong_sum = torch.zeros_like(per_class_T_correct_sum)
        per_class_T_correct_sum.scatter_add_(
            0,
            labels,
            T_correct.squeeze(1).to(dtype=torch.float64)
        )
        per_class_T_wrong_sum.scatter_add_(
            0,
            labels,
            T_wrong.squeeze(1).to(dtype=torch.float64)
        )

        output_group_neuron_stats = {}
        group_width = self.num_classes * self.competition_group_N
        if output_len % group_width == 0:
            repeat_count = output_len // group_width
            grouped_earliness = first_spike_earliness.reshape(
                first_spike_earliness.shape[0],
                repeat_count,
                self.num_classes,
                self.competition_group_N
            )
            output_group_neuron_stats = {
                'slot_count': int(
                    grouped_earliness.shape[0] * grouped_earliness.shape[1]
                ),
                'active_count': (
                    grouped_earliness > 0
                ).sum(dim=(0, 1)).detach().to('cpu', dtype=torch.float64),
                'earliness_sum': grouped_earliness.sum(
                    dim=(0, 1)
                ).detach().to('cpu', dtype=torch.float64),
            }

        return {
            'T_correct_sum': float(T_correct.sum().item()),
            'T_wrong_sum': float(T_wrong.sum().item()),
            'count': int(labels.shape[0]),
            'T_gap_values': (
                T_correct - T_wrong
            ).squeeze(1).detach().to('cpu'),
            'per_class_count': per_class_count.detach().to('cpu'),
            'per_class_T_correct_sum': per_class_T_correct_sum.detach().to('cpu'),
            'per_class_T_wrong_sum': per_class_T_wrong_sum.detach().to('cpu'),
            'output_group_neuron_stats': output_group_neuron_stats,
        }

    def _get_margin_label_signal(
        self, decision_snapshot, labels, true_mask, output_len, settings
    ):
        margin = float(settings.get('margin', 0.1))
        if margin < 0.0:
            raise ValueError('margin must be >= 0')
        correct_margin = float(settings.get('correct_margin', margin))
        wrong_margin = float(settings.get('wrong_margin', margin))
        if correct_margin < 0.0 or wrong_margin < 0.0:
            raise ValueError('correct_margin and wrong_margin must be >= 0')
        label_scope = settings.get('label_scope', 'neuron')
        if label_scope not in {'neuron', 'class_aggregate'}:
            raise ValueError(
                'label_scope must be one of '
                '{"neuron", "class_aggregate"}'
            )
        positive_scale = float(settings.get('positive_scale', 1.0))
        negative_scale = float(settings.get('negative_scale', 1.0))
        if positive_scale < 0.0 or negative_scale < 0.0:
            raise ValueError(
                'positive_scale and negative_scale must be >= 0'
            )
        wrong_background_value = float(
            settings.get('wrong_background_value', 0.0)
        )
        if wrong_background_value < 0.0:
            raise ValueError('wrong_background_value must be >= 0')
        correct_background_value = float(
            settings.get('correct_background_value', 0.0)
        )
        if correct_background_value < 0.0:
            raise ValueError('correct_background_value must be >= 0')
        fixed_desired_earliness = self._get_fixed_desired_earliness(
            settings,
            device=labels.device,
            dtype=torch.float32
        )

        first_spike_times = self._select_label_aligned_decision_field(
            decision_snapshot,
            'output_first_spike_times',
            'receiver_first_spike_times',
            output_len,
            'hinge label signal'
        )
        signal_dtype = torch.float32
        if true_mask.is_floating_point():
            signal_dtype = true_mask.dtype
        first_spike_times = first_spike_times.to(device=labels.device)
        has_spike = first_spike_times >= 0
        first_spike_earliness = (
            (
                float(self.image_encoder.simulation_time)
                - first_spike_times.clamp_min(0).to(dtype=signal_dtype)
            )
            / float(self.image_encoder.simulation_time)
        )
        first_spike_earliness = torch.where(
            has_spike,
            first_spike_earliness,
            torch.zeros_like(first_spike_earliness)
        )

        class_earliness = self._get_classwise_max_earliness(
            labels,
            first_spike_earliness,
            output_len
        )
        class_ids = torch.arange(self.num_classes, device=labels.device)
        true_class_mask = labels.unsqueeze(1) == class_ids.unsqueeze(0)
        T_correct = class_earliness.gather(1, labels.unsqueeze(1))
        if self.num_classes > 1:
            T_wrong = class_earliness.masked_fill(
                true_class_mask,
                float('-inf')
            ).amax(dim=1, keepdim=True)
            T_wrong = torch.where(
                torch.isfinite(T_wrong),
                T_wrong,
                torch.zeros_like(T_wrong)
            )
        else:
            T_wrong = torch.zeros_like(T_correct)

        true_neuron_mask = true_mask.bool()
        label_signal = torch.zeros_like(true_mask, dtype=signal_dtype)

        if label_scope == 'neuron':
            if fixed_desired_earliness is None:
                correct_values = (
                    T_wrong + correct_margin - first_spike_earliness
                ).clamp_min(0.0) * positive_scale
                wrong_values = -(
                    first_spike_earliness - (T_correct - wrong_margin)
                ).clamp_min(0.0) * negative_scale
            else:
                wrong_cutoff = (
                    fixed_desired_earliness - wrong_margin
                ).clamp_min(0.0)
                correct_values = (
                    fixed_desired_earliness - first_spike_earliness
                ).clamp_min(0.0) * positive_scale
                wrong_values = -(
                    first_spike_earliness - wrong_cutoff
                ).clamp_min(0.0) * negative_scale
        else:
            output_label_classes = self._get_output_label_classes(
                labels.device,
                output_len=output_len
            )
            neuron_class_earliness = class_earliness.gather(
                1,
                output_label_classes.unsqueeze(0).expand(labels.shape[0], -1)
            )
            if fixed_desired_earliness is None:
                correct_values = (
                    T_wrong + correct_margin - T_correct
                ).clamp_min(0.0).expand_as(label_signal) * positive_scale
                wrong_values = -(
                    neuron_class_earliness - (T_correct - wrong_margin)
                ).clamp_min(0.0) * negative_scale
            else:
                wrong_cutoff = (
                    fixed_desired_earliness - wrong_margin
                ).clamp_min(0.0)
                correct_values = (
                    fixed_desired_earliness - T_correct
                ).clamp_min(0.0).expand_as(label_signal) * positive_scale
                wrong_values = -(
                    neuron_class_earliness - wrong_cutoff
                ).clamp_min(0.0) * negative_scale

        if correct_background_value > 0.0:
            correct_background = torch.full_like(
                correct_values,
                correct_background_value
            )
            correct_values = torch.maximum(correct_values, correct_background)

        label_signal = torch.where(
            true_neuron_mask,
            correct_values,
            wrong_values
        )
        label_signal = label_signal * self._get_hinge_softmax_responsibility(
            decision_snapshot,
            labels,
            output_len,
            settings,
            signal_dtype
        )
        if wrong_background_value > 0.0:
            wrong_background = torch.full_like(
                label_signal,
                -wrong_background_value
            )
            label_signal = torch.where(
                true_neuron_mask,
                label_signal,
                torch.minimum(label_signal, wrong_background)
            )
        wrong_class_scope = settings.get('wrong_class_scope', 'all')
        if wrong_class_scope == 'hardest_wrong_class':
            output_label_classes = self._get_output_label_classes(
                labels.device,
                output_len=output_len
            )
            if self.num_classes > 1:
                hardest_wrong_class = class_earliness.masked_fill(
                    true_class_mask,
                    float('-inf')
                ).argmax(dim=1, keepdim=True)
                hardest_wrong_mask = (
                    output_label_classes.unsqueeze(0)
                    == hardest_wrong_class
                )
            else:
                hardest_wrong_mask = torch.zeros_like(true_neuron_mask)
            selected_class_mask = true_neuron_mask | hardest_wrong_mask
            label_signal = torch.where(
                selected_class_mask,
                label_signal,
                torch.zeros_like(label_signal)
            )
        return label_signal

    def _get_fixed_desired_earliness(self, settings, device, dtype):
        fixed_desired_time = settings.get('fixed_desired_time')
        if fixed_desired_time is None:
            return None
        fixed_desired_time = float(fixed_desired_time)
        simulation_time = float(self.image_encoder.simulation_time)
        if not 0.0 <= fixed_desired_time <= simulation_time:
            raise ValueError(
                'fixed_desired_time must be in timesteps between 0 and '
                'image_encoder.simulation_time'
            )
        fixed_desired_earliness = (
            simulation_time - fixed_desired_time
        ) / simulation_time
        return torch.as_tensor(
            fixed_desired_earliness,
            device=device,
            dtype=dtype
        )

    def _get_hinge_softmax_responsibility(
        self, decision_snapshot, labels, output_len, settings, signal_dtype
    ):
        if 'softmax_source' in settings:
            raise ValueError(
                'hinge_softmax_label.softmax_source was removed; '
                'receiver timing is the fixed source'
            )
        first_spike_times = self._select_label_aligned_decision_field(
            decision_snapshot,
            'output_first_spike_times',
            'receiver_first_spike_times',
            output_len,
            'hinge softmax'
        )

        first_spike_times = first_spike_times.to(device=labels.device)
        has_spike = first_spike_times >= 0
        first_spike_earliness = (
            (
                float(self.image_encoder.simulation_time)
                - first_spike_times.clamp_min(0).to(dtype=signal_dtype)
            )
            / float(self.image_encoder.simulation_time)
        )
        first_spike_earliness = torch.where(
            has_spike,
            first_spike_earliness,
            torch.zeros_like(first_spike_earliness)
        )

        group_width = self.num_classes * self.competition_group_N
        if output_len % group_width != 0:
            raise ValueError(
                'hinge softmax requires output_len to be a '
                'multiple of num_classes * competition_group_N'
            )
        grouped_earliness = first_spike_earliness.reshape(
            first_spike_earliness.shape[0],
            -1,
            self.num_classes,
            self.competition_group_N
        )
        grouped_has_spike = has_spike.reshape(grouped_earliness.shape)

        def get_mode_factor(mode):
            if mode == 'uniform':
                return torch.full_like(
                    grouped_earliness,
                    1.0 / float(self.competition_group_N)
                )
            if mode == 'winner_take_all':
                winner_index = grouped_earliness.argmax(dim=-1, keepdim=True)
                factor = torch.zeros_like(grouped_earliness).scatter(
                    -1,
                    winner_index,
                    1.0
                )
                return factor * grouped_has_spike.any(
                    dim=-1,
                    keepdim=True
                ).to(dtype=factor.dtype)
            if mode != 'softmax':
                raise ValueError(
                    'responsibility mode must be one of '
                    '{"softmax", "uniform", "winner_take_all"}'
                )

            temperature = float(settings.get('softmax_temperature', 0.1))
            if temperature <= 0.0:
                raise ValueError('softmax_temperature must be > 0')
            scores = grouped_earliness / temperature
            max_scores = scores.amax(dim=-1, keepdim=True)
            shifted_scores = scores - max_scores
            exp_scores = torch.exp(shifted_scores)
            denominator = exp_scores.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            return exp_scores / denominator

        responsibility_mode = settings.get('responsibility_mode', 'softmax')
        return get_mode_factor(
            responsibility_mode,
        ).reshape(
            first_spike_earliness.shape
        )

    def _get_softmax_threshold_label_signal(
        self, decision_snapshot, true_mask, output_len, settings
    ):
        temperature = float(settings.get('softmax_threshold_temperature', 1.0))
        if temperature <= 0.0:
            raise ValueError('softmax_threshold_temperature must be > 0')
        correct_baseline = float(
            settings.get('softmax_threshold_correct_baseline', 0.0)
        )
        if correct_baseline < 0.0:
            raise ValueError('softmax_threshold_correct_baseline must be >= 0')
        positive_scale = float(settings.get('softmax_threshold_positive_scale', 1.0))
        negative_scale = float(settings.get('softmax_threshold_negative_scale', 1.0))
        if positive_scale < 0.0 or negative_scale < 0.0:
            raise ValueError(
                'softmax_threshold_positive_scale and '
                'softmax_threshold_negative_scale must be >= 0'
        )

        softmax_settings = dict(settings)
        softmax_settings['softmax_temperature'] = temperature
        factor = self._get_hinge_softmax_responsibility(
            decision_snapshot,
            true_mask,
            output_len,
            softmax_settings,
            torch.float32
        )
        wrong_baseline = 1.0 / float(self.competition_group_N)
        correct_values = (
            factor - correct_baseline
        ).clamp_min(0.0) * positive_scale
        wrong_values = -(
            factor - wrong_baseline
        ).clamp_min(0.0) * negative_scale
        return torch.where(true_mask.bool(), correct_values, wrong_values)

    def init_state(self, image_shape, device):
        if self._state_sample_shape != tuple(image_shape) or self._state_sample_device != device:
            self._state_sample_input = torch.empty(image_shape, device=device)
            self._state_sample_shape = tuple(image_shape)
            self._state_sample_device = device
        self.cortex.init_state(self._state_sample_input, device)

    def _initialize_partitioned_weights(self, settings, seed):
        """Pair same-shaped branch blocks despite changes to raw-input fan-in."""
        for cortex in self.cortex.iter_cortex_tree():
            kernel = cortex.kernel
            for block_name, block in (
                ('direct', kernel.weight[:kernel.input_len]),
                ('branch', kernel.weight[kernel.input_len:]),
            ):
                if not block.numel():
                    continue
                key = f'{seed}:{cortex.cortex_id}:{block_name}'.encode('utf-8')
                block_seed = int.from_bytes(hashlib.sha256(key).digest()[:8], 'little')
                generator = torch.Generator(device=block.device).manual_seed(block_seed)
                block.copy_(torch.randn(block.shape, generator=generator,
                                        device=block.device, dtype=block.dtype)
                            * settings['init_std'] + settings['init_mean'])
            if kernel.has_subcortexs and kernel.no_skip_links:
                kernel.weight[:kernel.input_len, :kernel.output_len] = 0

    def construct_init_model(
        self, name,
        kernel_size, stride,
        input_channel, output_channel,
        fix_shape=False, subcortexs=None, init_amplifier=1, pooling_settings=None,
        competition_group_N=None,
        competition_group_aggregation_stage=None,
        cover_edges=False,
    ):
        subcortexs = [] if subcortexs is None else subcortexs
        constructed_subcortexs = []
        for i, subcortex in enumerate(subcortexs, 1):
            sub_name = f'{name}-{i}'
            constructed_subcortexs.append(
                self.construct_init_model(sub_name, **subcortex)
            )
        new_cortex = self.cortex_constructor(
            cortex_id=name,
            kernel_size=kernel_size,
            stride=stride,
            input_channel=input_channel,
            output_channel=output_channel,
            subcortexs=constructed_subcortexs,
            init_amplifier=init_amplifier,
            pooling_settings=pooling_settings,
            competition_group_N=competition_group_N,
            competition_group_aggregation_stage=competition_group_aggregation_stage,
            cover_edges=cover_edges,
        )
        if fix_shape:
            new_cortex.fix_shape()
        return new_cortex

    def sleep(self):
        self.cortex.self_cleaning()
        self.cortex.splitting(self.cortex_constructor)
        return

    def get_cortex_by_id(self, cortex_id):
        for cortex in self.cortex.iter_cortex_tree():
            if cortex.cortex_id == cortex_id:
                return cortex
        raise ValueError(f'unknown cortex_id for post-load transform: {cortex_id}')

    def pca_yield_cortex(self, cortex_id='A', **settings):
        cortex = self.get_cortex_by_id(cortex_id)
        return cortex.yield_pca_subcortex(self.cortex_constructor, **settings)

    def patch_pca_yield_cortex(self, cortex_id='A', **settings):
        cortex = self.get_cortex_by_id(cortex_id)
        return cortex.yield_patch_pca_subcortex(
            self.cortex_constructor,
            **settings
        )

    def patch_mean_pca_yield_cortex(self, cortex_id='A', **settings):
        cortex = self.get_cortex_by_id(cortex_id)
        return cortex.yield_patch_mean_pca_subcortex(
            self.cortex_constructor,
            **settings
        )

    def patch_tensor_cp_yield_cortex(self, cortex_id='A', **settings):
        cortex = self.get_cortex_by_id(cortex_id)
        return cortex.yield_patch_tensor_cp_subcortex(
            self.cortex_constructor,
            **settings
        )

    def patch_tucker_yield_cortex(self, cortex_id='A', **settings):
        cortex = self.get_cortex_by_id(cortex_id)
        return cortex.yield_patch_tucker_subcortex(
            self.cortex_constructor,
            **settings
        )

    def receiver_grouped_patch_pca_yield_cortex(self, cortex_id='A', **settings):
        cortex = self.get_cortex_by_id(cortex_id)
        return cortex.yield_receiver_grouped_patch_pca_subcortex(
            self.cortex_constructor,
            **settings
        )

    def forward_cortex(
        self, spikes, is_training=False, use_training_thresholds=None
    ):
        self._set_presynaptic_class_preference_forward_projection()
        current_progress = getattr(
            self.image_encoder,
            'current_temporal_progress',
            None,
        )
        if current_progress is not None:
            self.cortex.temporal_receiver_progress = current_progress
        return self.cortex(
            spikes,
            is_training=is_training,
            use_training_thresholds=use_training_thresholds
        )

    def forward_cortex_non_streaming(
        self, spike_sequence, labels=None, label_usage='ignore',
        label_propagating=False, STDP_interval=None, is_training=False,
        use_training_thresholds=None
    ):
        return self.cortex.forward_non_streaming(
            spike_sequence,
            labels=labels,
            label_usage=label_usage,
            label_propagating=label_propagating,
            STDP_interval=STDP_interval,
            is_training=is_training,
            use_training_thresholds=use_training_thresholds
        )

    def _get_simulation_forward_mode(self, simulation_forward_mode=None):
        mode = simulation_forward_mode
        if mode is None:
            mode = getattr(self, 'simulation_forward_mode', 'streaming')
        if mode not in {'streaming', 'non_streaming'}:
            raise ValueError(
                'simulation_forward_mode must be one of '
                '{"streaming", "non_streaming"}'
            )
        return mode

    def _get_channel_label_classes(self, device):
        return torch.arange(
            self.num_classes,
            device=device
        ).repeat_interleave(self.competition_group_N)

    def _get_output_label_classes(self, device, output_len=None):
        output_label_classes = self._get_channel_label_classes(device)
        if output_len is None:
            return output_label_classes

        if output_len % output_label_classes.shape[0] != 0:
            raise ValueError('output length must be a multiple of class group count')
        return output_label_classes.repeat(output_len // output_label_classes.shape[0])

    def _get_output_target_mask(self, labels, dtype=torch.float32, output_len=None):
        output_label_classes = self._get_output_label_classes(
            labels.device,
            output_len=output_len
        )
        return (
            labels.unsqueeze(1) == output_label_classes.unsqueeze(0)
        ).to(dtype)

    def _get_onehot_label(self, labels, output_len=None):
        onehot_label = self._get_output_target_mask(
            labels,
            dtype=torch.float32,
            output_len=output_len
        )
        reversed_label = self.label_off_value * (1.0 - onehot_label)
        return onehot_label + reversed_label

    @staticmethod
    def _new_first_spike_decision_snapshot():
        return {
            'class_scores': None,
            'output_neuron_scores': None,
            'receiver_neuron_scores': None,
            'output_first_spike_times': None,
            'receiver_first_spike_times': None,
            'decision_times': None,
            '_decided': None,
        }

    def _update_first_spike_decision_snapshot(
        self,
        decision_snapshot,
        time_step,
        output_wave,
        receiver_wave,
        collect_output_first_spike_times=False,
        collect_receiver_first_spike_times=False
    ):
        step_class_scores = self.score_output_neuron_scores(output_wave)
        step_decided = step_class_scores.amax(dim=1) > 0

        if decision_snapshot['class_scores'] is None:
            decision_snapshot['class_scores'] = torch.zeros_like(step_class_scores)
            decision_snapshot['output_neuron_scores'] = torch.zeros_like(output_wave)
            decision_snapshot['receiver_neuron_scores'] = torch.zeros_like(receiver_wave)
            if collect_output_first_spike_times:
                decision_snapshot['output_first_spike_times'] = torch.full(
                    output_wave.shape,
                    -1,
                    device=output_wave.device,
                    dtype=torch.long
                )
            if collect_receiver_first_spike_times:
                decision_snapshot['receiver_first_spike_times'] = torch.full(
                    receiver_wave.shape,
                    -1,
                    device=receiver_wave.device,
                    dtype=torch.long
                )
            decision_snapshot['decision_times'] = torch.full(
                step_decided.shape,
                -1,
                device=step_decided.device,
                dtype=torch.long
            )
            decision_snapshot['_decided'] = torch.zeros_like(step_decided)

        if collect_output_first_spike_times:
            first_spike_times = decision_snapshot['output_first_spike_times']
            newly_spiked = (output_wave > 0) & (first_spike_times < 0)
            decision_snapshot['output_first_spike_times'] = torch.where(
                newly_spiked,
                torch.full_like(first_spike_times, time_step),
                first_spike_times
            )

        if collect_receiver_first_spike_times:
            first_spike_times = decision_snapshot['receiver_first_spike_times']
            newly_spiked = (receiver_wave > 0) & (first_spike_times < 0)
            decision_snapshot['receiver_first_spike_times'] = torch.where(
                newly_spiked,
                torch.full_like(first_spike_times, time_step),
                first_spike_times
            )

        decided = decision_snapshot['_decided']
        newly_decided = step_decided & ~decided
        if newly_decided.any():
            decision_snapshot['class_scores'][newly_decided] = (
                step_class_scores[newly_decided]
            )
            decision_snapshot['output_neuron_scores'][newly_decided] = (
                output_wave[newly_decided]
            )
            decision_snapshot['receiver_neuron_scores'][newly_decided] = (
                receiver_wave[newly_decided]
            )
            decision_snapshot['decision_times'][newly_decided] = time_step
            decision_snapshot['_decided'] = decided | newly_decided

    @staticmethod
    def _finalize_first_spike_decision_snapshot(decision_snapshot):
        if decision_snapshot['class_scores'] is None:
            raise RuntimeError('image encoder emitted no spikes')
        return {
            key: value
            for key, value in decision_snapshot.items()
            if not key.startswith('_')
        }

    def _get_first_spike_decision_snapshot(
        self,
        images,
        collect_output_first_spike_times=False,
        collect_receiver_first_spike_times=False,
        use_training_thresholds=False
    ):
        self.init_state(images.shape, images.device)
        decision_snapshot = self._new_first_spike_decision_snapshot()

        for time_step, spikes in enumerate(self.image_encoder(images)):
            output_wave = self.forward_cortex(
                spikes,
                is_training=False,
                use_training_thresholds=use_training_thresholds
            )
            receiver_wave = self._mean_cortex_channels(
                self.cortex.hidden_nv.spike_wave
            )
            self._update_first_spike_decision_snapshot(
                decision_snapshot,
                time_step,
                output_wave,
                receiver_wave,
                collect_output_first_spike_times=collect_output_first_spike_times,
                collect_receiver_first_spike_times=collect_receiver_first_spike_times
            )

            if (
                bool(decision_snapshot['_decided'].all())
                and not collect_output_first_spike_times
            ):
                break

        return self._finalize_first_spike_decision_snapshot(decision_snapshot)

    def _get_first_spike_class_scores(self, images):
        return self._get_first_spike_decision_snapshot(images)['class_scores']

    def _get_hinge_softmax_label_from_snapshot(
        self, decision_snapshot, labels, return_snapshot=False
    ):
        settings = self._get_hinge_softmax_label_settings()
        self._validate_hinge_softmax_label_settings()
        class_scores = decision_snapshot['class_scores']

        output_len = self.cortex.shape[1]
        true_mask = self._get_output_target_mask(
            labels,
            dtype=torch.float32,
            output_len=output_len
        )

        uses_softmax_threshold = any(
            key.startswith('softmax_threshold_') for key in settings
        )
        if uses_softmax_threshold:
            label_signal = self._get_softmax_threshold_label_signal(
                decision_snapshot,
                true_mask,
                output_len,
                settings
            )
        else:
            label_signal = self._get_margin_label_signal(
                decision_snapshot,
                labels,
                true_mask,
                output_len,
                settings
            )
        if return_snapshot:
            return {
                'label_signal': label_signal,
                'decision_times': decision_snapshot['decision_times'],
                'class_scores': class_scores,
            }
        return label_signal

    def _get_hinge_softmax_label(
        self, images, labels, simulation_forward_mode, return_snapshot=False,
        use_training_thresholds=True
    ):
        settings = self._get_hinge_softmax_label_settings()
        self._validate_hinge_softmax_label_settings()
        if simulation_forward_mode != 'streaming':
            raise ValueError('hinge label signal requires streaming simulation')

        decision_snapshot = self._get_first_spike_decision_snapshot(
            images,
            collect_output_first_spike_times=True,
            collect_receiver_first_spike_times=True,
            use_training_thresholds=use_training_thresholds
        )
        return self._get_hinge_softmax_label_from_snapshot(
            decision_snapshot,
            labels,
            return_snapshot=return_snapshot
        )

    @staticmethod
    def _get_decision_time_gated_label(
        label_signal, decision_times, time_step, settings
    ):
        gate_mode = settings.get('decision_time_label_gate', 'none')
        if gate_mode == 'none':
            return label_signal
        if gate_mode != 'after_first_decision':
            raise ValueError(
                'decision_time_label_gate must be one of '
                '{"none", "after_first_decision"}'
            )
        if decision_times is None:
            raise ValueError(
                'decision_time_label_gate requires a first-spike decision snapshot'
            )

        start_offset = int(settings.get('decision_time_gate_start_offset', 0))
        end_offset = settings.get('decision_time_gate_end_offset', None)
        if end_offset is not None:
            end_offset = int(end_offset)
            if end_offset < start_offset:
                raise ValueError(
                    'decision_time_gate_end_offset must be >= '
                    'decision_time_gate_start_offset'
                )

        has_decision = decision_times >= 0
        active = ~has_decision | (time_step >= decision_times + start_offset)
        if end_offset is not None:
            active = active & (
                ~has_decision | (time_step <= decision_times + end_offset)
            )
        return label_signal * active.to(label_signal.dtype).unsqueeze(1)

    def _get_supervised_STDP_settings(self):
        settings = self.STDP_mechanism
        label_usage = settings.get('label_usage', 'mask')
        if label_usage not in {
            'mask',
            'signed_ltp',
        }:
            raise ValueError(
                'supervised STDP output learning only supports '
                'label_usage in {"mask", "signed_ltp"}'
            )
        label_propagating = bool(settings.get('label_propagating', False))
        return label_usage, label_propagating

    def _validate_eligibility_trace_decision_mode(
        self,
        STDP_interval,
        simulation_forward_mode,
        label_propagating,
        hinge_softmax_label_settings
    ):
        if STDP_interval not in (None, 1):
            raise ValueError(
                'eligibility_trace_decision_contingent supports only '
                'STDP_interval=None or 1'
            )
        if simulation_forward_mode != 'streaming':
            raise ValueError(
                'eligibility_trace_decision_contingent requires '
                'simulation_forward_mode="streaming"'
            )
        if label_propagating:
            raise ValueError(
                'eligibility_trace_decision_contingent supports only '
                'label_propagating=False'
            )
        if hinge_softmax_label_settings.get('decision_time_label_gate', 'none') != 'none':
            raise ValueError(
                'eligibility_trace_decision_contingent does not support '
                'decision_time_label_gate'
            )
        terms_dtype = hinge_softmax_label_settings.get(
            'eligibility_terms_dtype', 'float32'
        )
        if terms_dtype not in {'float16', 'float32'}:
            raise ValueError(
                'eligibility_terms_dtype must be one of {"float16", "float32"}'
            )
    def remember(
        self, images, labels, STDP_interval, stdp_update_mode='online',
        simulation_forward_mode=None
    ):
        if stdp_update_mode not in {
            'online',
            'after_simulation',
            'two_pass_decision_contingent',
            'eligibility_trace_decision_contingent',
        }:
            raise ValueError(
                'stdp_update_mode must be one of '
                '{"online", "after_simulation", '
                '"two_pass_decision_contingent", '
                '"eligibility_trace_decision_contingent"}'
            )
        simulation_forward_mode = self._get_simulation_forward_mode(
            simulation_forward_mode
        )
        uses_accumulated_STDP = stdp_update_mode in {
            'after_simulation',
            'two_pass_decision_contingent',
            'eligibility_trace_decision_contingent',
        }
        if (
            simulation_forward_mode == 'non_streaming'
            and not uses_accumulated_STDP
        ):
            raise ValueError(
                f'{simulation_forward_mode} simulation_forward_mode requires '
                'an accumulated STDP update mode'
            )

        self.init_state(images.shape, images.device)
        self._clear_presynaptic_sender_modulation()
        self._set_native_hidden_class_aggregation(for_training=True)
        label_usage, label_propagating = self._get_supervised_STDP_settings()
        if simulation_forward_mode == 'non_streaming':
            if label_propagating:
                raise ValueError(
                    'non_streaming first version supports only '
                    'label_propagating=False'
                )
            if STDP_interval not in (None, 1):
                raise ValueError(
                    'non_streaming first version supports only '
                    'STDP_interval=None or 1'
                )
            self.cortex.assert_non_streaming_supported_tree()
        hinge_softmax_label_settings = {}
        decision_times = None
        uses_eligibility_trace = (
            stdp_update_mode == 'eligibility_trace_decision_contingent'
        )
        if (
            stdp_update_mode == 'two_pass_decision_contingent'
            and STDP_interval is not None
        ):
            if (
                self.STDP_mechanism
                    .get('hinge_softmax_label', {})
                    .get('decision_time_label_gate', 'none') != 'none'
                and simulation_forward_mode != 'streaming'
            ):
                raise ValueError(
                    'decision_time_label_gate is only supported with '
                    'simulation_forward_mode="streaming"'
                )
            decision_info = self._get_hinge_softmax_label(
                images,
                labels,
                simulation_forward_mode=simulation_forward_mode,
                return_snapshot=True
            )
            label_signal = decision_info['label_signal']
            decision_times = decision_info['decision_times']
            self._update_native_hidden_class_aggregation_state(labels)
            hinge_softmax_label_settings = self._get_hinge_softmax_label_settings()
            class_preference_settings = (
                self._get_presynaptic_class_preference_settings()
            )
            if class_preference_settings is not None:
                if class_preference_settings.get('enabled', False):
                    self._set_presynaptic_class_preference_modulation(
                        labels,
                        class_preference_settings,
                    )
                else:
                    self.presynaptic_class_preference_metrics = {
                        'enabled': 0.0,
                        'preference_learning_enabled': 0.0,
                        'granularity': -1.0,
                        'class_seen_fraction': 0.0,
                        'preference_entropy_normalized_mean': 1.0,
                        'preference_top1_margin_mean': 0.0,
                        'multiplier_mean': 1.0,
                        'multiplier_q10': 1.0,
                        'multiplier_q90': 1.0,
                        'multiplier_min': 1.0,
                        'multiplier_max': 1.0,
                    }
            self.init_state(images.shape, images.device)
        elif uses_eligibility_trace and STDP_interval is not None:
            hinge_softmax_label_settings = self._get_hinge_softmax_label_settings()
            self._validate_eligibility_trace_decision_mode(
                STDP_interval,
                simulation_forward_mode,
                label_propagating,
                hinge_softmax_label_settings
            )
            label_signal = None
        else:
            label_signal = self._get_onehot_label(labels)
        STDP_flag = False
        steps_since_last_stdp = 0
        self.cortex.set_homeostasis_batch_scale_tree(images.shape[0])
        if uses_accumulated_STDP:
            self.cortex.clear_accumulated_STDP_tree()
        eligibility_decision_snapshot = None
        experience_observers = [
            (cortex, cortex.neuron_experience)
            for cortex in self.cortex.iter_cortex_tree()
            if getattr(cortex, 'neuron_experience', None) is not None
        ] if STDP_interval is not None else []
        for cortex, observer in experience_observers:
            observer.samples += int(images.shape[0])
            observer.batches += 1

        try:
            if simulation_forward_mode == 'streaming':
                for time_step, spikes in enumerate(self.image_encoder(images)):
                    steps_since_last_stdp += 1
                    output_wave = self.forward_cortex(spikes, is_training=True)
                    for cortex, observer in experience_observers:
                        observer.fire(cortex.hidden_nv.current_spikes)

                    if STDP_interval is not None and steps_since_last_stdp >= STDP_interval:
                        if uses_eligibility_trace:
                            if eligibility_decision_snapshot is None:
                                eligibility_decision_snapshot = (
                                    self._new_first_spike_decision_snapshot()
                                )
                            receiver_wave = self._mean_cortex_channels(
                                self.cortex.hidden_nv.spike_wave
                            )
                            self._update_first_spike_decision_snapshot(
                                eligibility_decision_snapshot,
                                time_step,
                                output_wave,
                                receiver_wave,
                                collect_output_first_spike_times=True,
                                collect_receiver_first_spike_times=True
                            )
                            STDP_flag = self.cortex.accumulate_subcortex_STDP(
                                None,
                                label_usage=label_usage,
                                label_propagating=False
                            ) or STDP_flag
                            terms_dtype_name = hinge_softmax_label_settings.get(
                                'eligibility_terms_dtype', 'float32'
                            )
                            terms_dtype = getattr(torch, terms_dtype_name)
                            STDP_flag = (
                                self.cortex.accumulate_batch_resolved_STDP_self(
                                    terms_dtype=terms_dtype
                                )
                                or STDP_flag
                            )
                        else:
                            step_label_signal = self._get_decision_time_gated_label(
                                label_signal,
                                decision_times,
                                time_step,
                                hinge_softmax_label_settings
                            )
                            if stdp_update_mode == 'online':
                                STDP_flag = self.cortex.STDP(
                                    step_label_signal,
                                    label_usage=label_usage,
                                    label_propagating=label_propagating
                                ) or STDP_flag
                            else:
                                STDP_flag = self.cortex.accumulate_STDP(
                                    step_label_signal,
                                    label_usage=label_usage,
                                    label_propagating=label_propagating
                                ) or STDP_flag
                        steps_since_last_stdp = 0
            else:
                input_spike_sequence = self._get_input_spike_sequence(images)
                _, STDP_flag = self.forward_cortex_non_streaming(
                    input_spike_sequence,
                    labels=label_signal,
                    label_usage=label_usage,
                    label_propagating=label_propagating,
                    STDP_interval=STDP_interval,
                    is_training=True
                )

            if uses_eligibility_trace and STDP_interval is not None:
                decision_snapshot = self._finalize_first_spike_decision_snapshot(
                    eligibility_decision_snapshot
                )
                label_signal = self._get_hinge_softmax_label_from_snapshot(
                    decision_snapshot,
                    labels
                )
                STDP_flag = self.cortex.commit_batch_resolved_STDP_self(
                    label_signal,
                    label_usage
                ) or STDP_flag

            if STDP_interval is not None:
                self.cortex.adjust_label_thresholds(
                    labels,
                    num_classes=self.num_classes,
                    competition_group_N=self.competition_group_N
                )

            if STDP_flag and uses_accumulated_STDP:
                STDP_flag = self.cortex.apply_accumulated_STDP_tree()

            if STDP_flag:
                self.cortex.adjust_amplifier_tree()

            if STDP_flag and self.self_cleaning_after_STDP:
                self.cortex.self_cleaning()
        finally:
            self._clear_presynaptic_sender_modulation()
            settings = self._get_presynaptic_class_preference_settings()
            if (
                settings is not None
                and settings.get('enabled', False)
                and settings.get('forward_application', 'none') != 'none'
            ):
                target = self.get_cortex_by_id(
                    settings['target_cortex_id']
                )
                target._class_preference_forward_projection_state_version = (
                    None
                )
            if uses_accumulated_STDP:
                self.cortex.clear_accumulated_STDP_tree()
            if simulation_forward_mode == 'non_streaming':
                self.cortex.clear_state_tree()
            self.cortex.set_homeostasis_batch_scale_tree(1.0)

        return

    def __call__(
        self, images, is_training=False, simulation_forward_mode=None
    ):
        self.init_state(images.shape, images.device)
        self._set_native_hidden_class_aggregation(
            for_training=bool(is_training)
        )
        simulation_forward_mode = self._get_simulation_forward_mode(
            simulation_forward_mode
        )
        try:
            if simulation_forward_mode == 'streaming':
                neuron_scores = self._get_streamed_output_neuron_scores(
                    images,
                    is_training=is_training
                )
            else:
                self.cortex.assert_non_streaming_supported_tree()
                neuron_scores = self._get_non_streaming_output_neuron_scores(
                    images,
                    is_training=is_training
                )
            return self.score_output_neuron_scores(neuron_scores)
        finally:
            if simulation_forward_mode == 'non_streaming':
                self.cortex.clear_state_tree()
