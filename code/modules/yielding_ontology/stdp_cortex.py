import math
import torch
from modules.yielding_ontology.base_cortex import BaseCortex


class StdpCortex(BaseCortex):
    SIGNED_CLASS_CONTRAST_MODES = {
        'feature_row_sender_activity_rms_class_contrast_supported',
    }

    def __init__(
        self, STDP_cortex_settings, **kargs
    ):
        super().__init__(**kargs)

        self.propagated_label_update_mix = None
        self.propagated_label_update_mode = 'linear'
        self._accumulated_STDP_label_free_update = None
        self._accumulated_STDP_propagated_update = None
        self.propagated_label_update_raw_l2_ratio = None
        self.propagated_label_update_cosine = None
        self.propagated_label_update_normalized_l2_ratio = None
        self.propagated_label_update_mixed_l2_ratio = None
        self.propagated_label_update_orthogonal_l2_ratio = None
        self.propagated_label_update_label_free_projection = None
        self.propagated_label_update_degenerate_fallback = None
        self.propagated_label_update_degenerate_fallback_count = 0
        self.propagated_label_update_batch_count = 0

        def setup(
            init_P_rate,
            init_D_P_rate_ratio,
            learning_rate_decrease_base,
            oja_decay_rate=0.0,
            oja_cortex_ids=None,
            stdp_decay=None,
            temporal_basis_update_coupling=None,
            spatial_update_diffusion=None,
            spatial_weight_prox=None,
        ):
            self.base_potentiation_rate = init_P_rate
            self.potentiation_rate = init_P_rate
            self.base_DP_ratio = self._validate_DP_ratio(init_D_P_rate_ratio)
            self.DP_ratio = self.base_DP_ratio
            self.learning_rate_decrease_base = learning_rate_decrease_base
            if oja_cortex_ids is not None and self.cortex_id not in set(oja_cortex_ids):
                oja_decay_rate = 0.0
            self.oja_decay_rate = float(oja_decay_rate)
            self._configure_stdp_decay(stdp_decay)
            self._configure_temporal_basis_update_coupling(
                temporal_basis_update_coupling
            )
            self._configure_spatial_update_diffusion(
                spatial_update_diffusion
            )
            self._configure_spatial_weight_prox(spatial_weight_prox)

        setup(**STDP_cortex_settings)

    def _configure_temporal_basis_update_coupling(self, settings):
        settings = {} if settings is None else dict(settings)
        unknown = set(settings) - {
            'block_count', 'specific_update_scale',
            'shared_initialization', 'cortex_ids',
        }
        if unknown:
            raise ValueError(
                'STDP_cortex_settings.temporal_basis_update_coupling has '
                'unsupported field(s): ' + ', '.join(sorted(unknown))
            )

        configured_block_count = settings.get('block_count', 1)
        if isinstance(configured_block_count, bool):
            raise ValueError('temporal basis coupling block_count must be an integer')
        try:
            block_count = int(configured_block_count)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                'temporal basis coupling block_count must be an integer'
            ) from exc
        if block_count < 1 or block_count != configured_block_count:
            raise ValueError('temporal basis coupling block_count must be >= 1')

        specific_scale = float(settings.get('specific_update_scale', 1.0))
        if not math.isfinite(specific_scale) or not 0.0 <= specific_scale <= 1.0:
            raise ValueError(
                'temporal basis coupling specific_update_scale must be '
                'finite and in [0, 1]'
            )
        shared_initialization = settings.get('shared_initialization', False)
        if not isinstance(shared_initialization, bool):
            raise ValueError(
                'temporal basis coupling shared_initialization must be bool'
            )
        cortex_ids = settings.get('cortex_ids')
        if cortex_ids is not None:
            if (
                not isinstance(cortex_ids, (list, tuple, set))
                or not cortex_ids
                or any(not isinstance(item, str) or not item for item in cortex_ids)
            ):
                raise ValueError(
                    'temporal basis coupling cortex_ids must be a non-empty '
                    'sequence of non-empty cortex IDs'
                )
            cortex_ids = tuple(dict.fromkeys(cortex_ids))

        active = block_count > 1 and (
            cortex_ids is None or self.cortex_id in cortex_ids
        )
        if not active:
            block_count = 1
            specific_scale = 1.0
            shared_initialization = False
        elif self.kernel.input_len % block_count != 0:
            raise ValueError(
                'direct input rows must be divisible by temporal basis '
                f'block_count: {self.kernel.input_len} vs {block_count}'
            )

        self.temporal_basis_update_block_count = block_count
        self.temporal_basis_specific_update_scale = specific_scale
        self.temporal_basis_shared_initialization = shared_initialization
        self.temporal_basis_update_cortex_ids = cortex_ids

    def _configure_spatial_update_diffusion(self, settings):
        settings = {} if settings is None else dict(settings)
        unknown = set(settings) - {
            'strength', 'input_height', 'input_width', 'cortex_ids',
        }
        if unknown:
            raise ValueError(
                'STDP_cortex_settings.spatial_update_diffusion has '
                'unsupported field(s): ' + ', '.join(sorted(unknown))
            )

        strength = float(settings.get('strength', 0.0))
        if not math.isfinite(strength) or not 0.0 <= strength <= 1.0:
            raise ValueError(
                'spatial update diffusion strength must be finite and in [0, 1]'
            )

        dimensions = {}
        for name in ('input_height', 'input_width'):
            configured = settings.get(name, 1)
            if isinstance(configured, bool):
                raise ValueError(f'spatial update diffusion {name} must be an integer')
            try:
                parsed = int(configured)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f'spatial update diffusion {name} must be an integer'
                ) from exc
            if parsed < 1 or parsed != configured:
                raise ValueError(
                    f'spatial update diffusion {name} must be positive'
                )
            dimensions[name] = parsed

        cortex_ids = settings.get('cortex_ids')
        if cortex_ids is not None:
            if (
                not isinstance(cortex_ids, (list, tuple, set))
                or not cortex_ids
                or any(not isinstance(item, str) or not item for item in cortex_ids)
            ):
                raise ValueError(
                    'spatial update diffusion cortex_ids must be a non-empty '
                    'sequence of non-empty cortex IDs'
                )
            cortex_ids = tuple(dict.fromkeys(cortex_ids))

        selected = cortex_ids is None or self.cortex_id in cortex_ids
        spatial_area = dimensions['input_height'] * dimensions['input_width']
        if selected and self.kernel.input_len % spatial_area != 0:
            raise ValueError(
                'direct input rows must be divisible by the configured spatial '
                f'area: {self.kernel.input_len} vs {spatial_area}'
            )
        if not selected:
            strength = 0.0
            dimensions = {'input_height': 1, 'input_width': 1}
            spatial_area = 1

        self.spatial_update_diffusion_strength = strength
        self.spatial_update_diffusion_input_height = dimensions['input_height']
        self.spatial_update_diffusion_input_width = dimensions['input_width']
        self.spatial_update_diffusion_channels = (
            self.kernel.input_len // spatial_area
        )
        self.spatial_update_diffusion_cortex_ids = cortex_ids

    def _diffuse_spatial_update(self, weight_update):
        strength = float(getattr(
            self, 'spatial_update_diffusion_strength', 0.0
        ))
        height = int(getattr(
            self, 'spatial_update_diffusion_input_height', 1
        ))
        width = int(getattr(
            self, 'spatial_update_diffusion_input_width', 1
        ))
        if strength == 0.0 or height * width <= 1:
            return weight_update
        if weight_update.shape != self.kernel.weight.shape:
            raise RuntimeError(
                'spatial update diffusion requires a full kernel-shaped update'
            )
        spatial_area = height * width
        if self.kernel.input_len % spatial_area != 0:
            raise RuntimeError(
                'direct input rows no longer match the configured spatial area'
            )

        direct = weight_update[:self.kernel.input_len].reshape(
            self.kernel.input_len // spatial_area,
            height,
            width,
            self.kernel.output_len,
        )
        diffused = direct.clone()
        rate = strength / 4.0
        vertical_flux = rate * (direct[:, 1:, :, :] - direct[:, :-1, :, :])
        diffused[:, :-1, :, :] += vertical_flux
        diffused[:, 1:, :, :] -= vertical_flux
        horizontal_flux = rate * (direct[:, :, 1:, :] - direct[:, :, :-1, :])
        diffused[:, :, :-1, :] += horizontal_flux
        diffused[:, :, 1:, :] -= horizontal_flux
        direct.copy_(diffused)
        return weight_update

    def _configure_spatial_weight_prox(self, settings):
        settings = {} if settings is None else dict(settings)
        unknown = set(settings) - {
            'strength', 'input_height', 'input_width', 'cortex_ids',
        }
        if unknown:
            raise ValueError(
                'STDP_cortex_settings.spatial_weight_prox has unsupported '
                'field(s): ' + ', '.join(sorted(unknown))
            )

        strength = float(settings.get('strength', 0.0))
        if not math.isfinite(strength) or not 0.0 <= strength <= 1.0:
            raise ValueError(
                'spatial weight prox strength must be finite and in [0, 1]'
            )

        dimensions = {}
        for name in ('input_height', 'input_width'):
            configured = settings.get(name, 1)
            if isinstance(configured, bool):
                raise ValueError(f'spatial weight prox {name} must be an integer')
            try:
                parsed = int(configured)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f'spatial weight prox {name} must be an integer'
                ) from exc
            if parsed < 1 or parsed != configured:
                raise ValueError(
                    f'spatial weight prox {name} must be positive'
                )
            dimensions[name] = parsed

        cortex_ids = settings.get('cortex_ids')
        if cortex_ids is not None:
            if (
                not isinstance(cortex_ids, (list, tuple, set))
                or not cortex_ids
                or any(not isinstance(item, str) or not item for item in cortex_ids)
            ):
                raise ValueError(
                    'spatial weight prox cortex_ids must be a non-empty '
                    'sequence of non-empty cortex IDs'
                )
            cortex_ids = tuple(dict.fromkeys(cortex_ids))

        selected = cortex_ids is None or self.cortex_id in cortex_ids
        spatial_area = dimensions['input_height'] * dimensions['input_width']
        if selected and self.kernel.input_len % spatial_area != 0:
            raise ValueError(
                'direct input rows must be divisible by the configured spatial '
                f'area: {self.kernel.input_len} vs {spatial_area}'
            )
        if not selected:
            strength = 0.0
            dimensions = {'input_height': 1, 'input_width': 1}
            spatial_area = 1

        self.spatial_weight_prox_strength = strength
        self.spatial_weight_prox_input_height = dimensions['input_height']
        self.spatial_weight_prox_input_width = dimensions['input_width']
        self.spatial_weight_prox_channels = self.kernel.input_len // spatial_area
        self.spatial_weight_prox_cortex_ids = cortex_ids
        self.spatial_weight_prox_application_count = 0

    def apply_spatial_weight_prox(self):
        strength = float(getattr(self, 'spatial_weight_prox_strength', 0.0))
        height = int(getattr(self, 'spatial_weight_prox_input_height', 1))
        width = int(getattr(self, 'spatial_weight_prox_input_width', 1))
        if strength == 0.0 or height * width <= 1:
            return False
        if not bool(getattr(self, 'learning_enabled', True)):
            return False

        spatial_area = height * width
        if self.kernel.input_len % spatial_area != 0:
            raise RuntimeError(
                'direct input rows no longer match the configured spatial area'
            )
        with torch.no_grad():
            direct = self.kernel.weight[:self.kernel.input_len].reshape(
                self.kernel.input_len // spatial_area,
                height,
                width,
                self.kernel.output_len,
            )
            diffused = direct.clone()
            rate = strength / 4.0
            vertical_flux = rate * (direct[:, 1:, :, :] - direct[:, :-1, :, :])
            diffused[:, :-1, :, :] += vertical_flux
            diffused[:, 1:, :, :] -= vertical_flux
            horizontal_flux = rate * (direct[:, :, 1:, :] - direct[:, :, :-1, :])
            diffused[:, :, :-1, :] += horizontal_flux
            diffused[:, :, 1:, :] -= horizontal_flux
            direct.copy_(diffused)
        self.spatial_weight_prox_application_count += 1
        return True

    def apply_temporal_basis_shared_initialization(self):
        block_count = int(getattr(
            self, 'temporal_basis_update_block_count', 1
        ))
        if (
            block_count <= 1
            or not bool(getattr(
                self, 'temporal_basis_shared_initialization', False
            ))
        ):
            return
        rows_per_block = self.kernel.input_len // block_count
        direct = self.kernel.weight[:self.kernel.input_len].reshape(
            block_count,
            rows_per_block,
            self.kernel.output_len,
        )
        shared = direct[0].clone()
        direct.copy_(shared.unsqueeze(0).expand_as(direct))

    def _couple_temporal_basis_update(self, weight_update):
        block_count = int(getattr(
            self, 'temporal_basis_update_block_count', 1
        ))
        specific_scale = float(getattr(
            self, 'temporal_basis_specific_update_scale', 1.0
        ))
        if block_count <= 1 or specific_scale == 1.0:
            return weight_update
        if weight_update.shape != self.kernel.weight.shape:
            raise RuntimeError(
                'temporal basis coupling requires a full kernel-shaped update'
            )
        rows_per_block = self.kernel.input_len // block_count
        direct = weight_update[:self.kernel.input_len].reshape(
            block_count,
            rows_per_block,
            self.kernel.output_len,
        )
        shared = direct.mean(dim=0, keepdim=True)
        direct.copy_(shared + specific_scale * (direct - shared))
        return weight_update

    @staticmethod
    def _validate_DP_ratio(ratio):
        ratio = float(ratio)
        if not math.isfinite(ratio) or ratio < 0:
            raise ValueError('DP_ratio must be finite and nonnegative')
        return ratio

    @property
    def depression_rate(self):
        return self.potentiation_rate * self.DP_ratio

    def __setstate__(self, state):
        # Existing saved models stored independent P and D rates. Migrate once
        # at load, so current code never retains a second mutable source of D.
        state = dict(state)
        old_d = state.pop('depression_rate', None)
        old_base_d = state.pop('base_depression_rate', None)
        if 'DP_ratio' not in state:
            p = state['potentiation_rate']
            if p != 0:
                state['DP_ratio'] = old_d / p
            elif old_d == 0:
                base_p = state.get('base_potentiation_rate', 0)
                state['DP_ratio'] = old_base_d / base_p if base_p and old_base_d is not None else 0.0
            else:
                raise ValueError('Cannot migrate nonzero depression with zero potentiation')
        if 'base_DP_ratio' not in state:
            base_p = state.get('base_potentiation_rate', 0)
            state['base_DP_ratio'] = old_base_d / base_p if base_p and old_base_d is not None else state['DP_ratio']
        state['DP_ratio'] = self._validate_DP_ratio(state['DP_ratio'])
        state['base_DP_ratio'] = self._validate_DP_ratio(state['base_DP_ratio'])
        self.__dict__.update(state)

    def reset_DP_ratio_tree(self):
        for cortex in self.iter_cortex_tree():
            cortex.DP_ratio = cortex.base_DP_ratio

    def set_DP_ratio_tree(self, ratio, cortex_ids=None):
        ratio = self._validate_DP_ratio(ratio)
        cortex_ids = None if cortex_ids is None else set(cortex_ids)
        cortexes = list(self.iter_cortex_tree())
        if cortex_ids is not None:
            unknown = cortex_ids - {c.cortex_id for c in cortexes}
            if unknown:
                raise ValueError(f'Unknown DP_ratio cortex IDs: {sorted(unknown)}')
        for cortex in cortexes:
            if cortex_ids is None or cortex.cortex_id in cortex_ids:
                cortex.DP_ratio = ratio

    @staticmethod
    def _validate_propagated_label_update_mix(mix):
        if mix is None:
            return None
        mix = float(mix)
        if not math.isfinite(mix) or not 0.0 <= mix <= 1.0:
            raise ValueError(
                'propagated_label_update_mix must be finite and in [0, 1]'
            )
        return mix

    @staticmethod
    def _validate_propagated_label_update_mode(mode):
        mode = str(mode)
        if mode not in {'linear', 'orthogonal'}:
            raise ValueError(
                'propagated_label_update_mode must be linear or orthogonal'
            )
        return mode

    def set_propagated_label_update_mode_tree(
        self, mode='linear', cortex_ids=None
    ):
        mode = self._validate_propagated_label_update_mode(mode)
        cortex_ids = None if cortex_ids is None else set(cortex_ids)
        cortexes = list(self.iter_cortex_tree())
        if cortex_ids is not None:
            unknown = cortex_ids - {c.cortex_id for c in cortexes}
            if unknown:
                raise ValueError(
                    'Unknown propagated-label mode cortex IDs: '
                    f'{sorted(unknown)}'
                )
        for cortex in cortexes:
            if cortex_ids is None or cortex.cortex_id in cortex_ids:
                cortex.propagated_label_update_mode = mode

    def set_propagated_label_update_mix_tree(self, mix=None, cortex_ids=None):
        mix = self._validate_propagated_label_update_mix(mix)
        cortex_ids = None if cortex_ids is None else set(cortex_ids)
        cortexes = list(self.iter_cortex_tree())
        if cortex_ids is not None:
            unknown = cortex_ids - {c.cortex_id for c in cortexes}
            if unknown:
                raise ValueError(
                    'Unknown propagated-label mix cortex IDs: '
                    f'{sorted(unknown)}'
                )
        for cortex in cortexes:
            if cortex_ids is None or cortex.cortex_id in cortex_ids:
                cortex.propagated_label_update_mix = mix
                cortex.propagated_label_update_raw_l2_ratio = None
                cortex.propagated_label_update_cosine = None
                cortex.propagated_label_update_normalized_l2_ratio = None
                cortex.propagated_label_update_mixed_l2_ratio = None
                cortex.propagated_label_update_orthogonal_l2_ratio = None
                cortex.propagated_label_update_label_free_projection = None
                cortex.propagated_label_update_degenerate_fallback = None
                cortex.propagated_label_update_degenerate_fallback_count = 0
                cortex.propagated_label_update_batch_count = 0

    def _configure_stdp_decay(self, settings):
        settings = {} if settings is None else dict(settings)
        unknown = set(settings) - {
            'mode', 'reference', 'exponent', 'warmup_updates', 'cortex_ids',
        }
        if unknown:
            raise ValueError(
                'STDP_cortex_settings.stdp_decay has unsupported field(s): '
                + ', '.join(sorted(unknown))
            )
        configured_mode = settings.get('mode', 'none')
        if configured_mode not in {
            'none', 'global_update_count', 'receiver_relative_stdp_l1',
        }:
            raise ValueError(f'unknown STDP decay mode: {configured_mode}')
        cortex_ids = settings.get('cortex_ids')
        if cortex_ids is not None:
            if (
                not isinstance(cortex_ids, (list, tuple, set))
                or not cortex_ids
                or any(not isinstance(item, str) or not item for item in cortex_ids)
            ):
                raise ValueError(
                    'stdp_decay.cortex_ids must be a non-empty sequence of '
                    'non-empty cortex IDs'
                )
            cortex_ids = tuple(dict.fromkeys(cortex_ids))
            if configured_mode == 'none':
                raise ValueError(
                    'stdp_decay.cortex_ids requires a non-none decay mode'
                )
        mode = (
            configured_mode
            if cortex_ids is None or self.cortex_id in cortex_ids
            else 'none'
        )
        reference = settings.get('reference')
        if configured_mode == 'none':
            if reference is not None:
                raise ValueError('stdp_decay.reference requires a non-none mode')
            reference = 1.0
        else:
            if reference is None:
                raise ValueError(
                    'stdp_decay.reference is required for mode '
                    f'{configured_mode}'
                )
            reference = float(reference)
            if not math.isfinite(reference) or reference <= 0:
                raise ValueError('stdp_decay.reference must be finite and > 0')

        exponent = float(settings.get('exponent', 1.0))
        if not math.isfinite(exponent) or exponent <= 0:
            raise ValueError('stdp_decay.exponent must be finite and > 0')
        warmup_updates = settings.get('warmup_updates', 0)
        if isinstance(warmup_updates, bool):
            raise ValueError('stdp_decay.warmup_updates must be an integer >= 0')
        try:
            parsed_warmup_updates = int(warmup_updates)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                'stdp_decay.warmup_updates must be an integer >= 0'
            ) from exc
        if parsed_warmup_updates < 0 or parsed_warmup_updates != warmup_updates:
            raise ValueError('stdp_decay.warmup_updates must be an integer >= 0')

        self.stdp_decay_mode = mode
        self.stdp_decay_configured_mode = configured_mode
        self.stdp_decay_cortex_ids = cortex_ids
        self.stdp_decay_reference = float(reference)
        self.stdp_decay_exponent = exponent
        self.stdp_decay_warmup_updates = parsed_warmup_updates
        self.stdp_decay_applied_update_count = 0
        self.stdp_decay_support = None
        self.stdp_decay_initial_effective_l1 = None
        self.stdp_decay_cumulative_stdp_l1 = None
        self.stdp_decay_last_multiplier = None

    def _ensure_stdp_decay_configured(self):
        # Checkpoints written before STDP decay was introduced do not carry
        # these attributes.  Their historical behaviour is exactly mode=none.
        if not hasattr(self, 'stdp_decay_mode'):
            self._configure_stdp_decay(None)
            return
        if not hasattr(self, 'stdp_decay_exponent'):
            self.stdp_decay_exponent = 1.0
        if not hasattr(self, 'stdp_decay_warmup_updates'):
            self.stdp_decay_warmup_updates = 0
        if not hasattr(self, 'stdp_decay_configured_mode'):
            self.stdp_decay_configured_mode = self.stdp_decay_mode
        if not hasattr(self, 'stdp_decay_cortex_ids'):
            self.stdp_decay_cortex_ids = None

    def _get_stdp_decay_support(self, weight):
        mask = getattr(self, 'kernel_learning_mask', None)
        support = (
            torch.ones_like(weight, dtype=torch.bool)
            if mask is None
            else mask.to(dtype=torch.bool, device=weight.device)
        )
        if self.kernel.has_subcortexs and self.kernel.no_skip_links:
            support = support.clone()
            support[:self.kernel.input_len, :self.kernel.output_len] = False
        return support

    def _ensure_stdp_decay_receiver_state(self, weight):
        support = self._get_stdp_decay_support(weight)
        if self.stdp_decay_support is None:
            initial_l1 = torch.where(support, weight, 0.).abs().sum(0)
            if bool((initial_l1 <= 0).any()):
                raise ValueError(
                    'receiver_relative_stdp_l1 requires positive initial '
                    'effective incoming-weight L1 for every receiver'
                )
            self.stdp_decay_support = support.clone()
            self.stdp_decay_initial_effective_l1 = initial_l1.detach().clone()
            self.stdp_decay_cumulative_stdp_l1 = torch.zeros_like(initial_l1)
        elif not torch.equal(self.stdp_decay_support, support):
            raise ValueError('STDP decay effective learning support changed after initialization')

    def get_stdp_decay_age(self):
        self._ensure_stdp_decay_configured()
        if self.stdp_decay_mode == 'receiver_relative_stdp_l1':
            if self.stdp_decay_cumulative_stdp_l1 is None:
                return torch.zeros(
                    self.kernel.weight.shape[1],
                    dtype=self.kernel.weight.dtype,
                    device=self.kernel.weight.device,
                )
            return (
                self.stdp_decay_cumulative_stdp_l1
                / self.stdp_decay_initial_effective_l1
            )
        if self.stdp_decay_mode == 'global_update_count':
            return float(self.stdp_decay_applied_update_count)
        return 0.0

    def get_stdp_decay_multiplier(self):
        age = self.get_stdp_decay_age()
        if self.stdp_decay_mode == 'none':
            return 1.0
        if self.stdp_decay_applied_update_count < self.stdp_decay_warmup_updates:
            return torch.ones_like(age) if torch.is_tensor(age) else 1.0
        return (1.0 + age / self.stdp_decay_reference) ** (
            -self.stdp_decay_exponent
        )

    def _apply_stdp_decay(self, weight_update):
        self._ensure_stdp_decay_configured()
        if self.stdp_decay_mode == 'none':
            return weight_update
        if self.stdp_decay_mode == 'receiver_relative_stdp_l1':
            self._ensure_stdp_decay_receiver_state(self.kernel.weight)
            multiplier = self.get_stdp_decay_multiplier().to(
                dtype=weight_update.dtype,
                device=weight_update.device,
            )
            self.stdp_decay_last_multiplier = multiplier.detach().clone()
            return weight_update * multiplier.unsqueeze(0)
        multiplier = float(self.get_stdp_decay_multiplier())
        self.stdp_decay_last_multiplier = multiplier
        return weight_update * multiplier

    def _record_stdp_decay_update(self, weight_before, weight_after):
        self._ensure_stdp_decay_configured()
        if self.stdp_decay_mode == 'none':
            return
        if self.stdp_decay_mode == 'receiver_relative_stdp_l1':
            self._ensure_stdp_decay_receiver_state(weight_before)
            delta = weight_after - weight_before
            delta_l1 = torch.where(
                self.stdp_decay_support,
                delta,
                torch.zeros_like(delta),
            ).abs().sum(0)
            self.stdp_decay_cumulative_stdp_l1.add_(delta_l1)
        self.stdp_decay_applied_update_count += 1

    def set_STDP_rate_scale_tree(self, scale=1.0, cortex_ids=None):
        cortex_ids = None if cortex_ids is None else set(cortex_ids)
        scale = float(scale)
        for cortex in self.iter_cortex_tree():
            if cortex_ids is not None and cortex.cortex_id not in cortex_ids:
                continue
            cortex.potentiation_rate = cortex.base_potentiation_rate * scale

    def _get_sender_activity(self):
        sender_spikes = [self.input_nv.current_spikes]
        sender_spike_trace = [self.input_nv.spike_trace]
        for subcortex in self.subcortexs:
            sub_spikes, sub_spike_trace = subcortex.output_nv.get_activity()
            sender_spikes.append(sub_spikes)
            sender_spike_trace.append(sub_spike_trace)

        sender_spikes = torch.cat(sender_spikes, dim=-1)
        sender_spike_trace = torch.cat(sender_spike_trace, dim=-1)

        modulation = getattr(
            self, 'presynaptic_sender_modulation', None
        )
        if modulation is not None:
            if (
                modulation.ndim != 2
                or modulation.shape[0] != sender_spikes.shape[0]
                or modulation.shape[1] != sender_spikes.shape[-1]
            ):
                raise RuntimeError(
                    'presynaptic sender modulation must have shape '
                    '[batch, sender_neurons]'
                )
            expanded = modulation.to(
                device=sender_spikes.device,
                dtype=sender_spikes.dtype,
            )
            while expanded.ndim < sender_spikes.ndim:
                expanded = expanded.unsqueeze(1)
            sender_spikes = sender_spikes * expanded
            sender_spike_trace = sender_spike_trace * expanded
            self.presynaptic_modulation_application_count = int(getattr(
                self, 'presynaptic_modulation_application_count', 0
            )) + 1
            self.presynaptic_modulation_last_abs_deviation = float(
                (modulation - 1.0).abs().mean().item()
            )
            self.presynaptic_sender_modulation_application_count = int(
                getattr(
                    self,
                    'presynaptic_sender_modulation_application_count',
                    0,
                )
            ) + 1
            self.presynaptic_sender_modulation_last_abs_deviation = float(
                (modulation - 1.0).abs().mean().item()
            )

        return sender_spikes, sender_spike_trace

    def _get_modulated_receiver_activity(self, labels, label_usage):
        receiver_spikes = self.hidden_nv.current_spikes
        receiver_spike_trace = self.hidden_nv.spike_trace

        if labels is None:
            return receiver_spikes, receiver_spike_trace

        if (
            labels.ndim == 2
            and receiver_spikes.ndim == 3
            and labels.shape[-1] == receiver_spikes.shape[-2] * receiver_spikes.shape[-1]
        ):
            labels = labels.reshape(
                labels.shape[0],
                receiver_spikes.shape[-2],
                receiver_spikes.shape[-1]
            )

        if labels.ndim == 2:
            num_middle_dims = receiver_spikes.ndim - 2
            for _ in range(num_middle_dims):
                labels = labels.unsqueeze(1)

        if label_usage == 'ignore':
            return receiver_spikes, receiver_spike_trace
        if label_usage == 'mask':
            return receiver_spikes * labels, receiver_spike_trace * labels
        if label_usage == 'signed_ltp':
            return receiver_spikes * labels, torch.zeros_like(receiver_spike_trace)
        if label_usage == 'mask_only_LTP':
            return receiver_spikes * labels, receiver_spike_trace
        raise ValueError(
            'Fast-path STDP only supports label_usage in '
            "{'mask', 'signed_ltp', 'mask_only_LTP', 'ignore'}."
        )

    def _get_selective_presynaptic_update_terms(
        self,
        sender_spikes,
        sender_spike_trace,
        receiver_spikes,
        receiver_spike_trace,
    ):
        selective_modulation = getattr(
            self, 'presynaptic_selective_sender_modulation', None
        )
        selective_gate = getattr(
            self, 'presynaptic_selective_receiver_gate', None
        )
        if selective_modulation is None and selective_gate is None:
            return None
        if selective_modulation is None or selective_gate is None:
            raise RuntimeError(
                'selective presynaptic modulation requires both sender '
                'modulation and receiver gate'
            )
        batch_size = sender_spikes.shape[0]
        if tuple(selective_modulation.shape) != (
            batch_size, sender_spikes.shape[-1]
        ):
            raise RuntimeError(
                'selective sender modulation must have shape '
                '[batch, sender_neurons]'
            )
        if tuple(selective_gate.shape) != (
            batch_size, receiver_spikes.shape[-1]
        ):
            raise RuntimeError(
                'selective receiver gate must have shape '
                '[batch, receiver_neurons]'
            )
        sender_delta = selective_modulation.to(
            device=sender_spikes.device,
            dtype=sender_spikes.dtype,
        ) - 1.0
        receiver_gate = selective_gate.to(
            device=receiver_spikes.device,
            dtype=receiver_spikes.dtype,
        )
        potentiation_delta = torch.einsum(
            'bmy,bmz->yz',
            sender_spike_trace * sender_delta.unsqueeze(1),
            receiver_spikes * receiver_gate.unsqueeze(1),
        )
        depression_delta = torch.einsum(
            'bmy,bmz->yz',
            sender_spikes * sender_delta.unsqueeze(1),
            receiver_spike_trace * receiver_gate.unsqueeze(1),
        )
        gate_fraction = float(receiver_gate.mean().item())
        deviation = float(
            sender_delta.abs().mean().item() * gate_fraction
        )
        self.presynaptic_modulation_application_count = int(getattr(
            self, 'presynaptic_modulation_application_count', 0
        )) + 1
        self.presynaptic_modulation_last_abs_deviation = deviation
        self.presynaptic_selective_modulation_application_count = int(
            getattr(
                self,
                'presynaptic_selective_modulation_application_count',
                0,
            )
        ) + 1
        self.presynaptic_selective_modulation_last_abs_deviation = deviation
        self.presynaptic_selective_modulation_last_gate_fraction = (
            gate_fraction
        )
        return potentiation_delta, depression_delta

    def _get_selective_presynaptic_receiver_update_terms(
        self,
        sender_spikes,
        sender_spike_trace,
        receiver_spikes,
        receiver_spike_trace,
    ):
        selective_modulation = getattr(
            self, 'presynaptic_selective_receiver_modulation', None
        )
        selective_gate = getattr(
            self, 'presynaptic_selective_receiver_modulation_gate', None
        )
        if selective_modulation is None and selective_gate is None:
            return None
        if selective_modulation is None or selective_gate is None:
            raise RuntimeError(
                'selective presynaptic receiver modulation requires both '
                'sender-by-receiver modulation and receiver gate'
            )
        batch_size = sender_spikes.shape[0]
        expected_modulation_shape = (
            sender_spikes.shape[-1], receiver_spikes.shape[-1]
        )
        if tuple(selective_modulation.shape) != expected_modulation_shape:
            raise RuntimeError(
                'selective sender-by-receiver modulation must have shape '
                '[sender_neurons, receiver_neurons]'
            )
        if tuple(selective_gate.shape) != (
            batch_size, receiver_spikes.shape[-1]
        ):
            raise RuntimeError(
                'selective sender-by-receiver gate must have shape '
                '[batch, receiver_neurons]'
            )
        sender_receiver_delta = selective_modulation.to(
            device=sender_spikes.device,
            dtype=sender_spikes.dtype,
        ) - 1.0
        receiver_gate = selective_gate.to(
            device=receiver_spikes.device,
            dtype=receiver_spikes.dtype,
        )
        potentiation_delta = torch.einsum(
            'bmy,bmz,yz,bz->yz',
            sender_spike_trace,
            receiver_spikes,
            sender_receiver_delta,
            receiver_gate,
        )
        depression_delta = torch.einsum(
            'bmy,bmz,yz,bz->yz',
            sender_spikes,
            receiver_spike_trace,
            sender_receiver_delta,
            receiver_gate,
        )
        gate_fraction = float(receiver_gate.mean().item())
        receiver_gate_fraction = receiver_gate.mean(dim=0)
        deviation = float((
            sender_receiver_delta.abs()
            * receiver_gate_fraction.unsqueeze(0)
        ).mean().item())
        self.presynaptic_modulation_application_count = int(getattr(
            self, 'presynaptic_modulation_application_count', 0
        )) + 1
        self.presynaptic_modulation_last_abs_deviation = deviation
        self.presynaptic_selective_receiver_modulation_application_count = int(
            getattr(
                self,
                'presynaptic_selective_receiver_modulation_application_count',
                0,
            )
        ) + 1
        self.presynaptic_selective_receiver_modulation_last_abs_deviation = (
            deviation
        )
        self.presynaptic_selective_receiver_modulation_last_gate_fraction = (
            gate_fraction
        )
        return potentiation_delta, depression_delta

    def _get_fast_weight_update(self, labels, label_usage):
        sender_spikes, sender_spike_trace = self._get_sender_activity()
        receiver_spikes, receiver_spike_trace = self._get_modulated_receiver_activity(
            labels, label_usage
        )

        # Contract: reshape both sides to [batch, locations, neurons] before einsum so
        # the reduced result matches self.kernel.weight with shape [sender_neurons, receiver_neurons].
        batch_size = sender_spikes.shape[0]
        sender_spikes = sender_spikes.reshape(batch_size, -1, sender_spikes.shape[-1])
        sender_spike_trace = sender_spike_trace.reshape(
            batch_size, -1, sender_spike_trace.shape[-1]
        )
        preconditioner_mode = getattr(
            self, 'kernel_update_preconditioner_mode', 'none'
        )
        if preconditioner_mode in {
            'feature_row_sender_activity_rms',
            *self.SIGNED_CLASS_CONTRAST_MODES,
        }:
            sender_activity_rms = torch.sqrt(
                torch.mean(
                    0.5 * (
                        sender_spikes.square()
                        + sender_spike_trace.square()
                    ),
                    dim=(0, 1),
                )
            )
            self.kernel_update_sender_activity_rms = sender_activity_rms
        receiver_spikes = receiver_spikes.reshape(batch_size, -1, receiver_spikes.shape[-1])
        receiver_spike_trace = receiver_spike_trace.reshape(
            batch_size, -1, receiver_spike_trace.shape[-1]
        )
        potentiation_matrix = torch.einsum(
            'bmy,bmz->yz', sender_spike_trace, receiver_spikes
        )
        depression_matrix = torch.einsum(
            'bmy,bmz->yz', sender_spikes, receiver_spike_trace
        )
        selective_terms = self._get_selective_presynaptic_update_terms(
            sender_spikes,
            sender_spike_trace,
            receiver_spikes,
            receiver_spike_trace,
        )
        if selective_terms is not None:
            potentiation_delta, depression_delta = selective_terms
            potentiation_matrix = potentiation_matrix + potentiation_delta
            depression_matrix = depression_matrix + depression_delta
        selective_receiver_terms = (
            self._get_selective_presynaptic_receiver_update_terms(
                sender_spikes,
                sender_spike_trace,
                receiver_spikes,
                receiver_spike_trace,
            )
        )
        if selective_receiver_terms is not None:
            potentiation_delta, depression_delta = selective_receiver_terms
            potentiation_matrix = potentiation_matrix + potentiation_delta
            depression_matrix = depression_matrix + depression_delta
        observer = getattr(self, 'neuron_experience', None)
        if observer is not None:
            observer.stdp_terms(
                self.potentiation_rate * potentiation_matrix,
                self.depression_rate * depression_matrix,
            )
        weight_update = (
            self.potentiation_rate * potentiation_matrix
            - self.depression_rate * depression_matrix
        )
        if self.oja_decay_rate != 0.0:
            receiver_activity = receiver_spikes.abs().sum(dim=(0, 1))
            oja_matrix = self.kernel.weight * receiver_activity.unsqueeze(0)
            oja_update = (
                self.potentiation_rate
                * self.oja_decay_rate
                * oja_matrix
            )
            weight_update -= oja_update
        return weight_update

    def _get_batch_resolved_STDP_terms(self, terms_dtype=None):
        sender_spikes, sender_spike_trace = self._get_sender_activity()
        receiver_spikes = self.hidden_nv.current_spikes
        receiver_spike_trace = self.hidden_nv.spike_trace

        batch_size = sender_spikes.shape[0]
        sender_spikes = sender_spikes.reshape(batch_size, -1, sender_spikes.shape[-1])
        sender_spike_trace = sender_spike_trace.reshape(
            batch_size, -1, sender_spike_trace.shape[-1]
        )
        receiver_spikes = receiver_spikes.reshape(
            batch_size, -1, receiver_spikes.shape[-1]
        )
        receiver_spike_trace = receiver_spike_trace.reshape(
            batch_size, -1, receiver_spike_trace.shape[-1]
        )
        if terms_dtype is not None:
            sender_spikes = sender_spikes.to(dtype=terms_dtype)
            sender_spike_trace = sender_spike_trace.to(dtype=terms_dtype)
            receiver_spikes = receiver_spikes.to(dtype=terms_dtype)
            receiver_spike_trace = receiver_spike_trace.to(dtype=terms_dtype)

        potentiation_terms = torch.einsum(
            'bmy,bmz->byz', sender_spike_trace, receiver_spikes
        )
        depression_terms = torch.einsum(
            'bmy,bmz->byz', sender_spikes, receiver_spike_trace
        )
        receiver_activity_terms = receiver_spikes.abs().sum(dim=1)
        return potentiation_terms, depression_terms, receiver_activity_terms

    def _prepare_batch_resolved_labels(self, labels, terms, label_usage):
        if labels is None or label_usage == 'ignore':
            return None, None
        if labels.ndim != 2:
            raise ValueError(
                'Batch-resolved eligibility labels must have shape '
                '[batch, receiver_neurons]'
            )
        if labels.shape[0] != terms.shape[0] or labels.shape[1] != terms.shape[2]:
            raise ValueError(
                'Batch-resolved eligibility labels must match '
                '[batch, receiver_neurons]'
            )
        if label_usage == 'mask':
            labels = labels.to(dtype=terms.dtype)
            return labels, labels
        if label_usage == 'signed_ltp':
            labels = labels.to(dtype=terms.dtype)
            return labels, torch.zeros_like(labels)
        if label_usage == 'mask_only_LTP':
            labels = labels.to(dtype=terms.dtype)
            return labels, torch.ones_like(labels)
        raise ValueError(
            'Batch-resolved eligibility only supports label_usage in '
            "{'mask', 'signed_ltp', 'mask_only_LTP', 'ignore'}."
        )

    def _get_weight_update_from_batch_resolved_terms(
        self, labels, label_usage, potentiation_terms, depression_terms,
        receiver_activity_terms
    ):
        potentiation_labels, depression_labels = (
            self._prepare_batch_resolved_labels(
                labels,
                potentiation_terms,
                label_usage
            )
        )
        if potentiation_labels is None:
            potentiation_matrix = potentiation_terms.sum(dim=0)
            depression_matrix = depression_terms.sum(dim=0)
            receiver_activity = receiver_activity_terms.sum(dim=0)
        else:
            potentiation_matrix = torch.einsum(
                'byz,bz->yz', potentiation_terms, potentiation_labels
            )
            depression_matrix = torch.einsum(
                'byz,bz->yz', depression_terms, depression_labels
            )
            receiver_activity = torch.einsum(
                'bz,bz->z',
                receiver_activity_terms,
                potentiation_labels.abs()
            )

        weight_update = (
            self.potentiation_rate * potentiation_matrix
            - self.depression_rate * depression_matrix
        )
        if self.oja_decay_rate != 0.0:
            oja_matrix = self.kernel.weight * receiver_activity.unsqueeze(0)
            weight_update -= (
                self.potentiation_rate
                * self.oja_decay_rate
                * oja_matrix
            )
        return weight_update

    def _add_accumulated_STDP_update(self, weight_update):
        if not hasattr(self, '_accumulated_STDP_update'):
            self._accumulated_STDP_update = None
        weight_update = self._apply_kernel_update_preconditioner(weight_update)
        weight_update = self._apply_kernel_learning_mask(weight_update)
        self._add_prepared_accumulated_STDP_update(weight_update)

    @staticmethod
    def _accumulate_update_tensor(previous, weight_update):
        if previous is None:
            return weight_update.clone()
        previous += weight_update
        return previous

    def _add_propagated_label_update_candidates(
        self, label_free_update, propagated_update
    ):
        self._accumulated_STDP_label_free_update = (
            self._accumulate_update_tensor(
                getattr(
                    self,
                    '_accumulated_STDP_label_free_update',
                    None,
                ),
                label_free_update,
            )
        )
        self._accumulated_STDP_propagated_update = (
            self._accumulate_update_tensor(
                getattr(
                    self,
                    '_accumulated_STDP_propagated_update',
                    None,
                ),
                propagated_update,
            )
        )

    def _mix_propagated_label_updates(
        self, label_free_update, propagated_update
    ):
        mix = self._validate_propagated_label_update_mix(
            getattr(self, 'propagated_label_update_mix', None)
        )
        mode = self._validate_propagated_label_update_mode(
            getattr(self, 'propagated_label_update_mode', 'linear')
        )
        if mix is None:
            raise RuntimeError('propagated-label update mixing is not enabled')

        label_free_l2 = torch.linalg.vector_norm(label_free_update)
        propagated_l2 = torch.linalg.vector_norm(propagated_update)
        eps = label_free_update.new_tensor(1.0e-12)
        label_free_nonzero = bool(label_free_l2 > eps)
        propagated_nonzero = bool(propagated_l2 > eps)
        fallback = not (label_free_nonzero and propagated_nonzero)

        if fallback:
            mixed_update = label_free_update
            raw_l2_ratio = 0.0
            cosine = 0.0
            normalized_l2_ratio = 0.0
            mixed_l2_ratio = 1.0 if label_free_nonzero else 0.0
            orthogonal_l2_ratio = 0.0
            label_free_projection = 1.0 if label_free_nonzero else 0.0
        else:
            normalized_propagated = (
                propagated_update * (label_free_l2 / propagated_l2)
            )
            cosine = float(torch.nn.functional.cosine_similarity(
                label_free_update.reshape(1, -1),
                propagated_update.reshape(1, -1),
            ).item())
            projection = (
                torch.sum(normalized_propagated * label_free_update)
                / (label_free_l2 * label_free_l2)
            )
            orthogonal_update = (
                normalized_propagated - projection * label_free_update
            )
            orthogonal_l2 = torch.linalg.vector_norm(orthogonal_update)
            orthogonal_l2_ratio = float(
                (orthogonal_l2 / label_free_l2).item()
            )
            if mode == 'linear':
                candidate = (
                    (1.0 - mix) * label_free_update
                    + mix * normalized_propagated
                )
            elif bool(orthogonal_l2 <= eps):
                fallback = True
                candidate = label_free_update
            else:
                normalized_orthogonal = (
                    orthogonal_update * (label_free_l2 / orthogonal_l2)
                )
                label_free_scale = math.sqrt(max(0.0, 1.0 - mix * mix))
                candidate = (
                    label_free_scale * label_free_update
                    + mix * normalized_orthogonal
                )
            candidate_l2 = torch.linalg.vector_norm(candidate)
            if bool(candidate_l2 <= eps):
                fallback = True
                mixed_update = label_free_update
            else:
                mixed_update = candidate * (label_free_l2 / candidate_l2)
            raw_l2_ratio = float((propagated_l2 / label_free_l2).item())
            normalized_l2_ratio = float(
                (
                    torch.linalg.vector_norm(normalized_propagated)
                    / label_free_l2
                ).item()
            )
            mixed_l2_ratio = float(
                (torch.linalg.vector_norm(mixed_update) / label_free_l2).item()
            )
            label_free_projection = float(
                (
                    torch.sum(mixed_update * label_free_update)
                    / (label_free_l2 * label_free_l2)
                ).item()
            )

        self.propagated_label_update_raw_l2_ratio = raw_l2_ratio
        self.propagated_label_update_cosine = cosine
        self.propagated_label_update_normalized_l2_ratio = normalized_l2_ratio
        self.propagated_label_update_mixed_l2_ratio = mixed_l2_ratio
        self.propagated_label_update_orthogonal_l2_ratio = (
            orthogonal_l2_ratio
        )
        self.propagated_label_update_label_free_projection = (
            label_free_projection
        )
        self.propagated_label_update_degenerate_fallback = float(fallback)
        self.propagated_label_update_degenerate_fallback_count += int(fallback)
        self.propagated_label_update_batch_count += 1
        return mixed_update

    def _get_propagated_label_update_candidates(self, labels, label_usage):
        if labels is None:
            raise ValueError(
                'propagated_label_update_mix requires propagated labels'
            )
        mode = getattr(self, 'kernel_update_preconditioner_mode', 'none')
        if mode != 'none':
            raise ValueError(
                'propagated_label_update_mix currently requires '
                'kernel_update_preconditioner mode none'
            )

        observer = getattr(self, 'neuron_experience', None)
        if observer is not None:
            self.neuron_experience = None
        try:
            label_free_update = self._get_fast_weight_update(None, label_usage)
            propagated_update = self._get_fast_weight_update(labels, label_usage)
        finally:
            if observer is not None:
                self.neuron_experience = observer
        return (
            self._apply_kernel_learning_mask(label_free_update),
            self._apply_kernel_learning_mask(propagated_update),
        )

    def _add_prepared_accumulated_STDP_update(self, weight_update):
        if self._accumulated_STDP_update is None:
            self._accumulated_STDP_update = weight_update.clone()
        else:
            self._accumulated_STDP_update += weight_update

    def _balance_kernel_update_rows(
        self, weight_update, row_measure, return_scale=False
    ):
        mask = getattr(self, 'kernel_learning_mask', None)
        if mask is None:
            learned_mask = torch.ones_like(weight_update, dtype=torch.bool)
        else:
            learned_mask = mask.to(
                dtype=torch.bool,
                device=weight_update.device,
            )
        if row_measure.ndim != 1 or row_measure.shape[0] != weight_update.shape[0]:
            raise ValueError(
                'kernel update row measure must match the sender dimension: '
                f'{tuple(row_measure.shape)} != {(weight_update.shape[0],)}'
            )
        row_measure = row_measure.to(
            dtype=weight_update.dtype,
            device=weight_update.device,
        )
        learned_update = torch.where(
            learned_mask,
            weight_update,
            torch.zeros_like(weight_update),
        )
        learned_rows = learned_mask.any(dim=1)
        active_rows = learned_rows & (row_measure > 0)
        positive = row_measure[active_rows]
        if not positive.numel():
            scale = learned_rows.to(dtype=weight_update.dtype)
            return (learned_update, scale) if return_scale else learned_update

        reference = positive.mean()
        scale = torch.sqrt(
            reference / torch.clamp(row_measure, min=1.0e-12)
        )
        scale = torch.clamp(scale, min=0.25, max=4.0)
        scale = torch.where(
            active_rows,
            scale,
            torch.zeros_like(scale),
        )
        conditioned = learned_update * scale.unsqueeze(1)
        raw_l2 = torch.linalg.vector_norm(learned_update)
        conditioned_l2 = torch.linalg.vector_norm(conditioned)
        if conditioned_l2 > 0:
            scale = scale * (raw_l2 / conditioned_l2)
        effective = scale[active_rows]
        current_min = float(effective.min().item())
        current_max = float(effective.max().item())
        previous_min = getattr(
            self,
            'kernel_update_preconditioner_observed_min',
            None,
        )
        previous_max = getattr(
            self,
            'kernel_update_preconditioner_observed_max',
            None,
        )
        self.kernel_update_preconditioner_observed_min = (
            current_min
            if previous_min is None
            else min(previous_min, current_min)
        )
        self.kernel_update_preconditioner_observed_max = (
            current_max
            if previous_max is None
            else max(previous_max, current_max)
        )
        conditioned = learned_update * scale.unsqueeze(1)
        self.kernel_update_preconditioner_l2_ratio = float(
            (
                torch.linalg.vector_norm(conditioned)
                / torch.clamp(raw_l2, min=1.0e-12)
            ).item()
        )
        return (conditioned, scale) if return_scale else conditioned

    def _apply_kernel_update_preconditioner(self, weight_update):
        mode = getattr(self, 'kernel_update_preconditioner_mode', 'none')
        if mode == 'feature_row_update_rms':
            mask = getattr(self, 'kernel_learning_mask', None)
            learned_update = (
                weight_update
                if mask is None
                else torch.where(
                    mask.to(dtype=torch.bool, device=weight_update.device),
                    weight_update,
                    torch.zeros_like(weight_update),
                )
            )
            row_rms = torch.sqrt(torch.mean(learned_update.square(), dim=1))
            return self._balance_kernel_update_rows(weight_update, row_rms)
        if mode in {
            'feature_row_sender_activity_rms',
        }:
            row_rms = getattr(
                self,
                'kernel_update_sender_activity_rms',
                None,
            )
            if row_rms is None:
                raise RuntimeError(
                    'sender-activity preconditioning requires a fast-path '
                    'sender activity snapshot'
                )
            weight_update = self._balance_kernel_update_rows(
                weight_update,
                row_rms,
            )
            return weight_update
        if mode in self.SIGNED_CLASS_CONTRAST_MODES:
            raise RuntimeError(
                'signed class contrast requires the fast signed label '
                'components, not a pre-combined update'
            )
        preconditioner = getattr(self, 'kernel_update_preconditioner', None)
        if preconditioner is not None:
            weight_update = weight_update * preconditioner.to(
                dtype=weight_update.dtype,
                device=weight_update.device,
            )
        return weight_update

    def _project_kernel_update_class_contrast(
        self, weight_update, record_metrics=True
    ):
        group_N = int(getattr(self, 'competition_group_N', 1))
        output_len = weight_update.shape[1]
        if group_N <= 1 or output_len % group_N != 0:
            raise ValueError(
                'class-contrast update projection requires output neurons '
                'grouped by competition_group_N'
            )
        num_classes = output_len // group_N
        if num_classes <= 1:
            raise ValueError(
                'class-contrast update projection requires at least two classes'
            )

        grouped = weight_update.reshape(
            weight_update.shape[0],
            num_classes,
            group_N,
        )
        support = grouped != 0
        support_count = support.sum(dim=1, keepdim=True)
        # Center only slots with at least two eligible classes. A singleton
        # cannot be both nonzero and zero-sum without writing into an
        # unsupported class, so pass it through unchanged.
        common = grouped.sum(dim=1, keepdim=True) / torch.clamp(
            support_count,
            min=1,
        )
        centered = torch.where(
            support,
            grouped - common,
            torch.zeros_like(grouped),
        )
        centerable = support_count > 1
        centerable_mask = centerable.expand_as(grouped)
        centered_centerable = torch.where(
            centerable_mask,
            centered,
            torch.zeros_like(grouped),
        )
        projected = torch.where(
            centerable_mask,
            centered_centerable,
            grouped,
        )
        projected_before_total = projected
        projected_update = projected_before_total.reshape_as(weight_update)

        reference_l2 = torch.linalg.vector_norm(weight_update)
        projected_l2 = torch.linalg.vector_norm(projected_update)
        total_degenerate_fallback = False
        total_global_scale = weight_update.new_tensor(1.0)
        if reference_l2 > 0 and projected_l2 == 0:
            # A pure class-common update has no direction left for the
            # requested norm restore. Preserve the raw update only for this
            # exact-zero boundary.
            total_degenerate_fallback = True
            projected = grouped
            projected_update = weight_update
            projected_l2 = reference_l2
        elif projected_l2 > 0:
            total_global_scale = reference_l2 / projected_l2
            projected = projected_before_total * total_global_scale
            projected_update = projected.reshape_as(weight_update)

        removed_l2 = torch.linalg.vector_norm(
            grouped - projected_before_total
        )
        if reference_l2 > 0:
            removed_fraction = float(
                (removed_l2 / reference_l2).item()
            )
            l2_ratio_before_restore = float(
                (projected_l2 / reference_l2).item()
            )
        else:
            removed_fraction = 0.0
            l2_ratio_before_restore = 1.0

        if not record_metrics:
            return projected_update

        restored_l2 = torch.linalg.vector_norm(projected_update)
        self.kernel_update_class_contrast_removed_fraction = removed_fraction
        self.kernel_update_class_contrast_l2_ratio_before_restore = (
            l2_ratio_before_restore
        )
        if reference_l2 > 0:
            self.kernel_update_class_contrast_l2_ratio_after_restore = float(
                (restored_l2 / reference_l2).item()
            )
            self.kernel_update_class_contrast_cosine = float(
                torch.nn.functional.cosine_similarity(
                    weight_update.reshape(1, -1),
                    projected_update.reshape(1, -1),
                ).item()
            )
        else:
            self.kernel_update_class_contrast_l2_ratio_after_restore = 1.0
            self.kernel_update_class_contrast_cosine = 1.0
        restored_grouped = projected_update.reshape(
            projected_update.shape[0],
            num_classes,
            group_N,
        )
        self.kernel_update_class_contrast_mean_abs_max = float(
            restored_grouped.mean(dim=1).abs().max().item()
        )
        centerable_slots = centerable.squeeze(1)
        supported_mean = restored_grouped.sum(dim=1) / torch.clamp(
            support_count.squeeze(1),
            min=1,
        )
        self.kernel_update_class_contrast_centered_mean_abs_max = float(
            (
                supported_mean.masked_select(centerable_slots).abs().max()
                if centerable_slots.any()
                else restored_grouped.new_tensor(0.0)
            ).item()
        )
        projected_nonzero = restored_grouped != 0
        unsupported = ~support
        unsupported_count = unsupported.sum()
        leaked = projected_nonzero & unsupported
        self.kernel_update_class_contrast_support_fraction = float(
            support.to(dtype=weight_update.dtype).mean().item()
        )
        self.kernel_update_class_contrast_support_count_mean = float(
            support_count.to(dtype=weight_update.dtype).mean().item()
        )
        centerable_fraction = float(
            centerable.to(dtype=weight_update.dtype).mean().item()
        )
        self.kernel_update_class_contrast_support_centerable_fraction = (
            centerable_fraction
        )
        previous_centerable_max = getattr(
            self,
            'kernel_update_class_contrast_support_centerable_fraction_max',
            None,
        )
        self.kernel_update_class_contrast_support_centerable_fraction_max = (
            centerable_fraction
            if previous_centerable_max is None
            else max(previous_centerable_max, centerable_fraction)
        )
        self.kernel_update_class_contrast_support_singleton_fraction = float(
            (support_count == 1)
            .to(dtype=weight_update.dtype)
            .mean()
            .item()
        )
        singleton_update = torch.where(
            (support_count == 1).expand_as(grouped),
            grouped,
            torch.zeros_like(grouped),
        )
        singleton_l2_fraction = float(
            (
                torch.linalg.vector_norm(singleton_update)
                / torch.clamp(reference_l2, min=1.0e-12)
            ).item()
        )
        self.kernel_update_class_contrast_support_singleton_l2_fraction = (
            singleton_l2_fraction
        )
        previous_singleton_l2_max = getattr(
            self,
            'kernel_update_class_contrast_support_singleton_l2_fraction_max',
            None,
        )
        self.kernel_update_class_contrast_support_singleton_l2_fraction_max = (
            singleton_l2_fraction
            if previous_singleton_l2_max is None
            else max(previous_singleton_l2_max, singleton_l2_fraction)
        )
        self.kernel_update_class_contrast_support_full_fraction = float(
            (support_count == num_classes)
            .to(dtype=weight_update.dtype)
            .mean()
            .item()
        )
        self.kernel_update_class_contrast_support_leak_fraction = float(
            (
                leaked.sum().to(dtype=weight_update.dtype)
                / torch.clamp(unsupported_count, min=1)
            ).item()
        )
        self.kernel_update_class_contrast_support_leak_abs_max = float(
            (
                restored_grouped.masked_select(unsupported).abs().max()
                if unsupported_count > 0
                else restored_grouped.new_tensor(0.0)
            ).item()
        )
        self.kernel_update_class_contrast_total_global_scale = float(
            total_global_scale.item()
        )
        self.kernel_update_class_contrast_total_degenerate_fallback = float(
            total_degenerate_fallback
        )
        self.kernel_update_class_contrast_total_degenerate_fallback_count += int(
            total_degenerate_fallback
        )
        return projected_update

    def _get_signed_class_contrast_update(self, labels, label_usage):
        mode = getattr(self, 'kernel_update_preconditioner_mode', 'none')
        if mode not in self.SIGNED_CLASS_CONTRAST_MODES:
            raise RuntimeError('signed class contrast mode is not enabled')
        if labels is None or label_usage != 'signed_ltp':
            raise ValueError(
                'signed class contrast requires signed_ltp labels'
            )

        positive_update = self._get_fast_weight_update(
            torch.clamp(labels, min=0),
            label_usage,
        )
        negative_update = self._get_fast_weight_update(
            torch.clamp(labels, max=0),
            label_usage,
        )
        raw_update = positive_update + negative_update
        row_rms = getattr(
            self,
            'kernel_update_sender_activity_rms',
            None,
        )
        if row_rms is None:
            raise RuntimeError(
                'signed class contrast requires a sender activity snapshot'
            )
        balanced_update, row_scale = self._balance_kernel_update_rows(
            raw_update,
            row_rms,
            return_scale=True,
        )
        learned_mask = getattr(self, 'kernel_learning_mask', None)
        if learned_mask is None:
            learned_mask = torch.ones_like(raw_update, dtype=torch.bool)
        else:
            learned_mask = learned_mask.to(
                dtype=torch.bool,
                device=raw_update.device,
            )
        positive_update = torch.where(
            learned_mask,
            positive_update,
            torch.zeros_like(positive_update),
        ) * row_scale.unsqueeze(1)
        negative_update = torch.where(
            learned_mask,
            negative_update,
            torch.zeros_like(negative_update),
        ) * row_scale.unsqueeze(1)

        reference_l2 = torch.linalg.vector_norm(balanced_update)
        reference_denom = torch.clamp(reference_l2, min=1.0e-12)
        reconstruction_error = torch.linalg.vector_norm(
            positive_update + negative_update - balanced_update
        ) / reference_denom
        self.kernel_update_class_contrast_signed_reconstruction_error = float(
            reconstruction_error.item()
        )
        self.kernel_update_class_contrast_signed_positive_l2_fraction = float(
            (
                torch.linalg.vector_norm(positive_update)
                / reference_denom
            ).item()
        )
        self.kernel_update_class_contrast_signed_negative_l2_fraction = float(
            (
                torch.linalg.vector_norm(negative_update)
                / reference_denom
            ).item()
        )

        projected_update = self._project_kernel_update_class_contrast(
            balanced_update
        )
        return projected_update

    def _apply_kernel_learning_mask(self, weight_update):
        receiver_modulation = getattr(
            self, 'presynaptic_receiver_modulation', None
        )
        if receiver_modulation is not None:
            if (
                receiver_modulation.ndim != 2
                or receiver_modulation.shape != weight_update.shape
            ):
                raise RuntimeError(
                    'presynaptic receiver modulation must match the '
                    'STDP update shape [sender_neurons, receiver_neurons]'
                )
            receiver_modulation = receiver_modulation.to(
                dtype=weight_update.dtype,
                device=weight_update.device,
            )
            weight_update = weight_update * receiver_modulation
            deviation = float(
                (receiver_modulation - 1.0).abs().mean().item()
            )
            self.presynaptic_modulation_application_count = int(getattr(
                self, 'presynaptic_modulation_application_count', 0
            )) + 1
            self.presynaptic_modulation_last_abs_deviation = deviation
            self.presynaptic_receiver_modulation_application_count = int(
                getattr(
                    self,
                    'presynaptic_receiver_modulation_application_count',
                    0,
                )
            ) + 1
            self.presynaptic_receiver_modulation_last_abs_deviation = (
                deviation
            )
        mask = getattr(self, 'kernel_learning_mask', None)
        if mask is None:
            return weight_update
        if mask.shape != weight_update.shape:
            raise ValueError(
                'kernel learning mask shape must match STDP update shape: '
                f'{tuple(mask.shape)} != {tuple(weight_update.shape)}'
            )
        return weight_update * mask.to(
            dtype=weight_update.dtype,
            device=weight_update.device
        )

    def _regulate_kernel_after_learning(self):
        if not getattr(self, 'kernel_regulation_enabled', True):
            return
        regulate = (
            self.kernel.normalize_kernel
            if getattr(self, 'kernel_normalization_only', False)
            else self.kernel.kernel_regulation
        )
        mask = getattr(self, 'kernel_learning_mask', None)
        if mask is None:
            regulate()
            return

        protected_weight = self.kernel.weight.clone()
        regulate()
        mask = mask.to(dtype=torch.bool, device=self.kernel.weight.device)
        self.kernel.weight = torch.where(mask, self.kernel.weight, protected_weight)

    def _preserve_kernel_norm_after_learning(self, weight_before):
        mode = getattr(self, 'kernel_norm_preservation_mode', 'none')
        if mode == 'none':
            return

        weight_after = self.kernel.weight
        mask = getattr(self, 'kernel_learning_mask', None)
        if mask is None:
            mask = torch.ones_like(weight_after, dtype=torch.bool)
        else:
            mask = mask.to(dtype=torch.bool, device=weight_after.device)
        before_learned = torch.where(mask, weight_before, 0.0)
        after_learned = torch.where(mask, weight_after, 0.0)

        if mode == 'global_l2':
            target_norm = torch.linalg.vector_norm(before_learned)
            current_norm = torch.linalg.vector_norm(after_learned)
        elif mode == 'per_output_l2':
            target_norm = torch.linalg.vector_norm(
                before_learned, dim=0, keepdim=True
            )
            current_norm = torch.linalg.vector_norm(
                after_learned, dim=0, keepdim=True
            )
        else:
            raise ValueError(f'unknown kernel norm preservation mode: {mode}')

        scale = torch.where(
            current_norm > 0,
            target_norm / torch.clamp(current_norm, min=1e-12),
            torch.ones_like(current_norm),
        )
        preserved = after_learned * scale
        self.kernel.weight = torch.where(mask, preserved, weight_after)

    def _iter_subcortex_labels(self, labels, label_propagating):
        if len(self.subcortexs) == 0:
            return

        if labels is None or label_propagating is False:
            labels_for_senders = None
        else:
            labels_for_senders = labels @ self.kernel.weight.T / self.kernel.shape_ratio

        i = self.shape[0]
        for subcortex in self.subcortexs:
            emitted_feature_len = (
                subcortex.output_nv.current_spikes.shape[-2]
                * len(subcortex.output_nv)
            )
            next_i = i + emitted_feature_len
            sub_labels = None if labels_for_senders is None else labels_for_senders[..., i:next_i]
            if sub_labels is not None and sub_labels.shape[-1] != subcortex.shape[1]:
                sub_labels = sub_labels.reshape(
                    *sub_labels.shape[:-1],
                    -1,
                    subcortex.shape[1]
                ).mean(dim=-2)
            yield subcortex, sub_labels
            i = next_i

    def _propagate_STDP_to_subcortexs(self, labels, label_usage, label_propagating):
        applied = False
        for subcortex, sub_labels in self._iter_subcortex_labels(
            labels,
            label_propagating
        ):
            applied = subcortex.STDP(
                sub_labels,
                label_usage,
                label_propagating
            ) or applied
        return applied

    def STDP_self(self, labels, label_usage):
        if not self.learning_enabled:
            return False

        mix = getattr(self, 'propagated_label_update_mix', None)
        mode = getattr(self, 'kernel_update_preconditioner_mode', 'none')
        if mix is not None:
            label_free_update, propagated_update = (
                self._get_propagated_label_update_candidates(
                    labels,
                    label_usage,
                )
            )
            weight_update = self._mix_propagated_label_updates(
                label_free_update,
                propagated_update,
            )
        elif mode in self.SIGNED_CLASS_CONTRAST_MODES:
            weight_update = self._get_signed_class_contrast_update(
                labels,
                label_usage,
            )
        else:
            weight_update = self._apply_kernel_learning_mask(
                self._apply_kernel_update_preconditioner(
                    self._get_fast_weight_update(labels, label_usage)
                )
            )
        weight_update = self._apply_stdp_decay(weight_update)
        weight_update = self._couple_temporal_basis_update(weight_update)
        weight_update = self._diffuse_spatial_update(weight_update)
        weight_before = self.kernel.weight.clone()
        self.kernel.weight += weight_update
        self._record_stdp_decay_update(weight_before, self.kernel.weight)
        self._preserve_kernel_norm_after_learning(weight_before)
        self._regulate_kernel_after_learning()
        return True

    def STDP(self, labels, label_usage, label_propagating):
        # self._adjust_learning_rate()
        propagated = self._propagate_STDP_to_subcortexs(
            labels,
            label_usage,
            label_propagating
        )
        return self.STDP_self(labels, label_usage) or propagated

    def _accumulate_STDP_self(self, labels, label_usage):
        if not self.learning_enabled:
            return False

        mix = getattr(self, 'propagated_label_update_mix', None)
        mode = getattr(self, 'kernel_update_preconditioner_mode', 'none')
        if mix is not None:
            label_free_update, propagated_update = (
                self._get_propagated_label_update_candidates(
                    labels,
                    label_usage,
                )
            )
            self._add_propagated_label_update_candidates(
                label_free_update,
                propagated_update,
            )
        elif mode in self.SIGNED_CLASS_CONTRAST_MODES:
            weight_update = self._get_signed_class_contrast_update(
                labels,
                label_usage,
            )
            self._add_prepared_accumulated_STDP_update(weight_update)
        else:
            weight_update = self._get_fast_weight_update(labels, label_usage)
            self._add_accumulated_STDP_update(weight_update)
        return True

    def _accumulate_batch_resolved_STDP_self(self, terms_dtype=None):
        if not self.learning_enabled:
            return False

        (
            potentiation_terms,
            depression_terms,
            receiver_activity_terms
        ) = self._get_batch_resolved_STDP_terms(terms_dtype=terms_dtype)

        if not hasattr(self, '_accumulated_STDP_potentiation_terms'):
            self._accumulated_STDP_potentiation_terms = None
            self._accumulated_STDP_depression_terms = None
            self._accumulated_STDP_receiver_activity_terms = None

        if self._accumulated_STDP_potentiation_terms is None:
            self._accumulated_STDP_potentiation_terms = potentiation_terms
            self._accumulated_STDP_depression_terms = depression_terms
            self._accumulated_STDP_receiver_activity_terms = (
                receiver_activity_terms
            )
        else:
            self._accumulated_STDP_potentiation_terms += potentiation_terms
            self._accumulated_STDP_depression_terms += depression_terms
            self._accumulated_STDP_receiver_activity_terms += receiver_activity_terms
        return True

    def accumulate_batch_resolved_STDP_self(self, terms_dtype=None):
        return self._accumulate_batch_resolved_STDP_self(
            terms_dtype=terms_dtype
        )

    def commit_batch_resolved_STDP_self(self, labels, label_usage):
        potentiation_terms = getattr(
            self,
            '_accumulated_STDP_potentiation_terms',
            None
        )
        if potentiation_terms is None:
            return False

        weight_update = self._get_weight_update_from_batch_resolved_terms(
            labels,
            label_usage,
            potentiation_terms,
            self._accumulated_STDP_depression_terms,
            self._accumulated_STDP_receiver_activity_terms
        )
        self._add_accumulated_STDP_update(weight_update)
        self._accumulated_STDP_potentiation_terms = None
        self._accumulated_STDP_depression_terms = None
        self._accumulated_STDP_receiver_activity_terms = None
        return True

    def accumulate_subcortex_STDP(self, labels, label_usage, label_propagating):
        accumulated = False
        for subcortex, sub_labels in self._iter_subcortex_labels(
            labels,
            label_propagating
        ):
            accumulated = subcortex.accumulate_STDP(
                sub_labels,
                label_usage,
                label_propagating
            ) or accumulated
        return accumulated

    def accumulate_STDP(self, labels, label_usage, label_propagating):
        accumulated = self.accumulate_subcortex_STDP(
            labels,
            label_usage,
            label_propagating
        )
        return self._accumulate_STDP_self(labels, label_usage) or accumulated

    def clear_accumulated_STDP_tree(self):
        for cortex in self.iter_cortex_tree():
            cortex._accumulated_STDP_update = None
            cortex._accumulated_STDP_label_free_update = None
            cortex._accumulated_STDP_propagated_update = None
            cortex._accumulated_STDP_potentiation_terms = None
            cortex._accumulated_STDP_depression_terms = None
            cortex._accumulated_STDP_receiver_activity_terms = None

    def apply_accumulated_STDP_tree(self):
        applied = False
        for cortex in self.iter_cortex_tree():
            weight_update = getattr(cortex, '_accumulated_STDP_update', None)
            label_free_update = getattr(
                cortex,
                '_accumulated_STDP_label_free_update',
                None,
            )
            propagated_update = getattr(
                cortex,
                '_accumulated_STDP_propagated_update',
                None,
            )
            if label_free_update is not None or propagated_update is not None:
                if weight_update is not None:
                    raise RuntimeError(
                        'mixed and regular accumulated STDP updates cannot '
                        'target the same cortex in one batch'
                    )
                if label_free_update is None or propagated_update is None:
                    raise RuntimeError(
                        'propagated-label update candidates must be paired'
                    )
                weight_update = cortex._mix_propagated_label_updates(
                    label_free_update,
                    propagated_update,
                )
            if weight_update is None:
                continue
            weight_update = cortex._apply_kernel_learning_mask(weight_update)
            weight_update = cortex._apply_stdp_decay(weight_update)
            weight_update = cortex._couple_temporal_basis_update(weight_update)
            weight_update = cortex._diffuse_spatial_update(weight_update)
            weight_before = cortex.kernel.weight.clone()
            cortex.kernel.weight += weight_update
            cortex._record_stdp_decay_update(weight_before, cortex.kernel.weight)
            observer = getattr(cortex, 'neuron_experience', None)
            after_stdp = (
                observer.before_regulation(weight_before, cortex.kernel.weight)
                if observer is not None else None
            )
            cortex._preserve_kernel_norm_after_learning(weight_before)
            cortex._regulate_kernel_after_learning()
            if observer is not None:
                observer.after_regulation(weight_before, after_stdp, cortex.kernel.weight)
            cortex._accumulated_STDP_update = None
            cortex._accumulated_STDP_label_free_update = None
            cortex._accumulated_STDP_propagated_update = None
            applied = True
        return applied

    def subcortex_STDP(self, labels, label_usage, label_propagating):
        self._propagate_STDP_to_subcortexs(labels, label_usage, label_propagating)

    def adjust_amplifier_tree(self):
        for cortex in self.iter_cortex_tree():
            cortex.adjust_amplifier()
