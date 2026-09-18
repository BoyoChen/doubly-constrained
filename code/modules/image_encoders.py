import math

import torch
import torch.nn.functional as F


class RankOrderEncoder():
    def __init__(
        self, simulation_time, start_threshold=0.95, end_threshold=0.05,
        target_mean_earliness=None, end_score_search_steps=24
    ):
        if simulation_time <= 0:
            raise ValueError('simulation_time must be positive')
        if end_threshold > start_threshold:
            raise ValueError('end_threshold must be <= start_threshold')
        if end_score_search_steps <= 0:
            raise ValueError('end_score_search_steps must be positive')
        self.simulation_time = simulation_time
        self.start_threshold = start_threshold
        self.end_threshold = end_threshold
        self.end_score_search_steps = int(end_score_search_steps)
        self.target_mean_earliness = None
        if target_mean_earliness is not None:
            self.target_mean_earliness = float(target_mean_earliness)
            if not 0.0 <= self.target_mean_earliness <= 1.0:
                raise ValueError('target_mean_earliness must be between 0 and 1')

    def _relative_earliness_from_end_score(self, images, end_scores, max_scores):
        epsilon = 1.0e-12
        relative_earliness = (
            (images - end_scores) / (max_scores - end_scores).clamp_min(epsilon)
        ).clamp(0.0, 1.0)
        return torch.where(
            max_scores > 0,
            relative_earliness,
            torch.zeros_like(relative_earliness)
        )

    def _end_score_upper_bound(self, max_scores):
        return torch.where(
            max_scores > 0,
            max_scores * (1.0 - 1.0e-6),
            torch.zeros_like(max_scores)
        )

    def _normalize_sample_earliness(self, clipped_images):
        reduce_dims = tuple(range(1, clipped_images.ndim))
        max_scores = clipped_images.amax(dim=reduce_dims, keepdim=True)
        zero_end_scores = torch.zeros_like(max_scores)
        base_earliness = self._relative_earliness_from_end_score(
            clipped_images,
            zero_end_scores,
            max_scores
        )
        if self.target_mean_earliness is None:
            end_score_upper_bound = self._end_score_upper_bound(max_scores)
            fixed_end_scores = torch.minimum(
                torch.full_like(max_scores, float(self.end_threshold)),
                end_score_upper_bound
            )
            return self._relative_earliness_from_end_score(
                clipped_images,
                fixed_end_scores,
                max_scores
            )

        target = torch.full_like(max_scores, self.target_mean_earliness)
        base_mean = base_earliness.mean(dim=reduce_dims, keepdim=True)
        search_mask = (max_scores > 0) & (base_mean > target)
        end_low = torch.zeros_like(max_scores)
        end_high = max_scores

        for _ in range(self.end_score_search_steps):
            end_mid = (end_low + end_high) * 0.5
            mid_earliness = self._relative_earliness_from_end_score(
                clipped_images,
                end_mid,
                max_scores
            )
            mid_mean = mid_earliness.mean(dim=reduce_dims, keepdim=True)
            mean_too_high = mid_mean > target
            end_low = torch.where(search_mask & mean_too_high, end_mid, end_low)
            end_high = torch.where(search_mask & (~mean_too_high), end_mid, end_high)

        tuned_end_scores = torch.minimum(
            (end_low + end_high) * 0.5,
            self._end_score_upper_bound(max_scores)
        )
        tuned_earliness = self._relative_earliness_from_end_score(
            clipped_images,
            tuned_end_scores,
            max_scores
        )
        return torch.where(
            search_mask,
            tuned_earliness,
            base_earliness
        )

    def __call__(self, images):
        clipped_images = images.clamp(0.0, 1.0)
        earliness = self._normalize_sample_earliness(clipped_images)
        active_pixels = earliness > 0
        fired_pixels = torch.zeros_like(active_pixels, dtype=torch.bool)
        thresholds = torch.linspace(
            1.0,
            0.0,
            steps=self.simulation_time,
            device=images.device,
            dtype=images.dtype,
        )
        for threshold in thresholds:
            spikes = active_pixels & (~fired_pixels) & (earliness >= threshold)
            fired_pixels = fired_pixels | spikes
            yield spikes.to(images.dtype)


class TemporalBinRankOrderEncoder():
    def __init__(self, simulation_time, temporal_bins, channels_per_bin=2):
        if simulation_time <= 0:
            raise ValueError('simulation_time must be positive')
        if temporal_bins <= 0:
            raise ValueError('temporal_bins must be positive')
        if channels_per_bin <= 0:
            raise ValueError('channels_per_bin must be positive')
        self.simulation_time = int(simulation_time)
        self.temporal_bins = int(temporal_bins)
        self.channels_per_bin = int(channels_per_bin)

    def _bin_bounds(self, bin_index):
        start = int(bin_index * self.simulation_time // self.temporal_bins)
        end = int((bin_index + 1) * self.simulation_time // self.temporal_bins)
        return start, max(end, start + 1)

    def __call__(self, images):
        if images.ndim < 2:
            raise ValueError('images must have a channel dimension')
        expected_channels = self.temporal_bins * self.channels_per_bin
        if images.shape[1] != expected_channels:
            raise ValueError(
                'temporal-bin encoder expected '
                f'{expected_channels} channels, got {images.shape[1]}'
            )

        clipped_images = images.clamp(0.0, 1.0)
        active_pixels = clipped_images > 0
        fired_pixels = torch.zeros_like(active_pixels, dtype=torch.bool)

        for time_step in range(self.simulation_time):
            bin_index = min(
                time_step * self.temporal_bins // self.simulation_time,
                self.temporal_bins - 1
            )
            bin_start, bin_end = self._bin_bounds(bin_index)
            bin_steps = max(bin_end - bin_start, 1)
            local_step = time_step - bin_start
            if bin_steps == 1:
                threshold = 0.0
            else:
                threshold = 1.0 - float(local_step) / float(bin_steps - 1)

            channel_start = bin_index * self.channels_per_bin
            channel_end = channel_start + self.channels_per_bin
            spikes = torch.zeros_like(clipped_images, dtype=torch.bool)
            current_slice = (
                active_pixels[:, channel_start:channel_end]
                & (~fired_pixels[:, channel_start:channel_end])
                & (clipped_images[:, channel_start:channel_end] >= threshold)
            )
            spikes[:, channel_start:channel_end] = current_slice
            fired_pixels = fired_pixels | spikes
            yield spikes.to(images.dtype)


class DirectEventBinEncoder():
    def __init__(
        self, simulation_time, temporal_bins, channels_per_bin=2,
        binarize=True
    ):
        if simulation_time <= 0:
            raise ValueError('simulation_time must be positive')
        if temporal_bins <= 0:
            raise ValueError('temporal_bins must be positive')
        if channels_per_bin <= 0:
            raise ValueError('channels_per_bin must be positive')
        if int(simulation_time) != int(temporal_bins):
            raise ValueError(
                'direct-event-bin encoder requires simulation_time == temporal_bins'
            )
        self.simulation_time = int(simulation_time)
        self.temporal_bins = int(temporal_bins)
        self.channels_per_bin = int(channels_per_bin)
        self.binarize = bool(binarize)

    def __call__(self, images):
        if images.ndim < 2:
            raise ValueError('images must have a channel dimension')
        expected_channels = self.temporal_bins * self.channels_per_bin
        if images.shape[1] != expected_channels:
            raise ValueError(
                'direct-event-bin encoder expected '
                f'{expected_channels} channels, got {images.shape[1]}'
            )

        event_bins = images.clamp_min(0.0).reshape(
            images.shape[0],
            self.temporal_bins,
            self.channels_per_bin,
            *images.shape[2:],
        )
        for bin_index in range(self.temporal_bins):
            spikes = event_bins[:, bin_index]
            if self.binarize:
                spikes = spikes > 0
            yield spikes.to(images.dtype)


class EventTimeSurfaceEncoder():
    def __init__(
        self, simulation_time, temporal_bins, channels_per_bin=2,
        surface_decay=0.8
    ):
        if simulation_time <= 0:
            raise ValueError('simulation_time must be positive')
        if temporal_bins <= 0:
            raise ValueError('temporal_bins must be positive')
        if channels_per_bin <= 0:
            raise ValueError('channels_per_bin must be positive')
        if int(simulation_time) != int(temporal_bins):
            raise ValueError(
                'event-time-surface encoder requires '
                'simulation_time == temporal_bins'
            )
        if not 0.0 <= float(surface_decay) < 1.0:
            raise ValueError('surface_decay must be in [0, 1)')
        self.simulation_time = int(simulation_time)
        self.temporal_bins = int(temporal_bins)
        self.channels_per_bin = int(channels_per_bin)
        self.surface_decay = float(surface_decay)

    def __call__(self, images):
        if images.ndim < 2:
            raise ValueError('images must have a channel dimension')
        expected_channels = self.temporal_bins * self.channels_per_bin
        if images.shape[1] != expected_channels:
            raise ValueError(
                'event-time-surface encoder expected '
                f'{expected_channels} channels, got {images.shape[1]}'
            )

        event_bins = images.reshape(
            images.shape[0],
            self.temporal_bins,
            self.channels_per_bin,
            *images.shape[2:],
        )
        surface = torch.zeros_like(event_bins[:, 0])
        for bin_index in range(self.temporal_bins):
            current_events = (event_bins[:, bin_index] > 0).to(images.dtype)
            surface = torch.maximum(
                current_events,
                surface * self.surface_decay,
            )
            yield surface


class EventTemporalFeatureEncoder():
    def __init__(
        self, simulation_time, temporal_bins, channels_per_bin=2,
        feature_mode='signed_change', history_decay=0.8,
        partition_segments=1, temporal_basis_contrast=0.5,
        temporal_ramp_direction='late', temporal_basis_direction='forward',
        temporal_basis_count=2,
        temporal_basis_permutation_multiplier=1,
        temporal_basis_permutation_offset=0,
    ):
        if simulation_time <= 0:
            raise ValueError('simulation_time must be positive')
        if temporal_bins <= 0:
            raise ValueError('temporal_bins must be positive')
        if channels_per_bin <= 0:
            raise ValueError('channels_per_bin must be positive')
        if int(simulation_time) != int(temporal_bins):
            raise ValueError(
                'event-temporal-feature encoder requires '
                'simulation_time == temporal_bins'
            )
        if feature_mode not in (
            'signed_change',
            'signed_rise_only',
            'signed_fall_only',
            'current_only',
            'duplicated_current',
            'duplicated_rise',
            'duplicated_fall',
            'current_plus_motion',
            'phase_gated_signed_change',
            'spatial_partitioned_signed_change',
            'dense_temporal_basis_signed_change',
            'temporal_ramp_signed_change',
        ):
            raise ValueError(
                'feature_mode must be signed_change, signed_rise_only, '
                'signed_fall_only, current_only, duplicated_current, '
                'duplicated_rise, duplicated_fall, current_plus_motion, '
                'phase_gated_signed_change, spatial_partitioned_signed_change, '
                'dense_temporal_basis_signed_change, or '
                'temporal_ramp_signed_change'
            )
        if not 0.0 <= float(history_decay) < 1.0:
            raise ValueError('history_decay must be in [0, 1)')
        if not 0.0 <= float(temporal_basis_contrast) <= 1.0:
            raise ValueError('temporal_basis_contrast must be in [0, 1]')
        if temporal_ramp_direction not in ('early', 'late'):
            raise ValueError("temporal_ramp_direction must be 'early' or 'late'")
        if temporal_basis_direction not in ('forward', 'reverse'):
            raise ValueError(
                "temporal_basis_direction must be 'forward' or 'reverse'"
            )
        if isinstance(temporal_basis_count, bool) or int(temporal_basis_count) != temporal_basis_count:
            raise ValueError('temporal_basis_count must be an integer')
        parsed_basis_count = int(temporal_basis_count)
        if parsed_basis_count < 2:
            raise ValueError('temporal_basis_count must be at least 2')
        if (
            feature_mode != 'dense_temporal_basis_signed_change'
            and parsed_basis_count != 2
        ):
            raise ValueError(
                'temporal_basis_count is only configurable for '
                'dense_temporal_basis_signed_change'
            )
        for name, value in (
            ('temporal_basis_permutation_multiplier',
             temporal_basis_permutation_multiplier),
            ('temporal_basis_permutation_offset', temporal_basis_permutation_offset),
        ):
            if isinstance(value, bool) or int(value) != value:
                raise ValueError(f'{name} must be an integer')
        parsed_basis_multiplier = int(temporal_basis_permutation_multiplier)
        parsed_basis_offset = int(temporal_basis_permutation_offset)
        if int(temporal_bins) == 1:
            if parsed_basis_multiplier != 1 or parsed_basis_offset != 0:
                raise ValueError(
                    'single-bin temporal basis requires permutation multiplier=1 '
                    'and offset=0'
                )
        elif (
            parsed_basis_multiplier <= 0
            or parsed_basis_multiplier >= int(temporal_bins)
            or math.gcd(parsed_basis_multiplier, int(temporal_bins)) != 1
        ):
            raise ValueError(
                'temporal_basis_permutation_multiplier must be in '
                '[1, temporal_bins) and coprime with temporal_bins'
            )
        if not 0 <= parsed_basis_offset < int(temporal_bins):
            raise ValueError(
                'temporal_basis_permutation_offset must be in '
                '[0, temporal_bins)'
            )
        if isinstance(partition_segments, bool):
            raise ValueError('partition_segments must be a positive integer')
        try:
            parsed_partition_segments = int(partition_segments)
        except (TypeError, ValueError) as exc:
            raise ValueError('partition_segments must be a positive integer') from exc
        if (
            parsed_partition_segments <= 0
            or parsed_partition_segments != partition_segments
        ):
            raise ValueError('partition_segments must be a positive integer')
        partition_modes = {
            'phase_gated_signed_change',
            'spatial_partitioned_signed_change',
        }
        if feature_mode in partition_modes:
            if parsed_partition_segments <= 1:
                raise ValueError(
                    'partitioned signed-change modes require '
                    'partition_segments > 1'
                )
            if temporal_bins % parsed_partition_segments != 0:
                raise ValueError(
                    'temporal_bins must be divisible by partition_segments'
                )
        elif parsed_partition_segments != 1:
            raise ValueError(
                'partition_segments is only valid for partitioned '
                'signed-change modes'
            )
        self.simulation_time = int(simulation_time)
        self.temporal_bins = int(temporal_bins)
        self.channels_per_bin = int(channels_per_bin)
        self.feature_mode = feature_mode
        self.history_decay = float(history_decay)
        self.partition_segments = parsed_partition_segments
        self.temporal_basis_contrast = float(temporal_basis_contrast)
        self.temporal_ramp_direction = temporal_ramp_direction
        self.temporal_basis_direction = temporal_basis_direction
        self.temporal_basis_count = parsed_basis_count
        self.temporal_basis_permutation_multiplier = parsed_basis_multiplier
        self.temporal_basis_permutation_offset = parsed_basis_offset

    def _temporal_coordinate(self, bin_index):
        if self.temporal_bins == 1:
            return 0.5
        return bin_index / (self.temporal_bins - 1)

    def _apply_temporal_coordinate(self, features, bin_index):
        tau = self._temporal_coordinate(bin_index)
        contrast = self.temporal_basis_contrast
        if self.feature_mode == 'dense_temporal_basis_signed_change':
            permuted_bin_index = (
                self.temporal_basis_permutation_multiplier * bin_index
                + self.temporal_basis_permutation_offset
            ) % self.temporal_bins
            tau = self._temporal_coordinate(permuted_bin_index)
            if self.temporal_basis_direction == 'reverse':
                tau = 1.0 - tau
            if self.temporal_basis_count == 2:
                early_coefficient = 1.0 + contrast * (1.0 - 2.0 * tau)
                late_coefficient = 1.0 - contrast * (1.0 - 2.0 * tau)
                return torch.cat(
                    (
                        features * early_coefficient,
                        features * late_coefficient,
                    ),
                    dim=1,
                )
            degree = self.temporal_basis_count - 1
            coefficients = [
                (1.0 - contrast)
                + contrast
                * self.temporal_basis_count
                * math.comb(degree, index)
                * tau ** index
                * (1.0 - tau) ** (degree - index)
                for index in range(self.temporal_basis_count)
            ]
            return torch.cat(
                tuple(features * coefficient for coefficient in coefficients),
                dim=1,
            )

        ramp = 2.0 * tau - 1.0
        if self.temporal_ramp_direction == 'early':
            ramp = -ramp
        return features * (1.0 + contrast * ramp)

    def _partition_signed_features(self, features, bin_index):
        segments = self.partition_segments
        if self.feature_mode == 'phase_gated_signed_change':
            partition = min(
                bin_index * segments // self.temporal_bins,
                segments - 1,
            )
            output = features.new_zeros(
                features.shape[0],
                features.shape[1] * segments,
                *features.shape[2:],
            )
            start = partition * features.shape[1]
            output[:, start:start + features.shape[1]] = features
            return output

        height, width = features.shape[-2:]
        rows = torch.arange(height, device=features.device).view(height, 1)
        columns = torch.arange(width, device=features.device).view(1, width)
        assignment = (rows + columns) % segments
        return torch.cat([
            features * (assignment == partition)
            for partition in range(segments)
        ], dim=1)

    def __call__(self, images):
        if images.ndim != 4:
            raise ValueError(
                'event-temporal-feature encoder expects [batch, channel, height, width]'
            )
        expected_channels = self.temporal_bins * self.channels_per_bin
        if images.shape[1] != expected_channels:
            raise ValueError(
                'event-temporal-feature encoder expected '
                f'{expected_channels} channels, got {images.shape[1]}'
            )

        event_bins = images.reshape(
            images.shape[0],
            self.temporal_bins,
            self.channels_per_bin,
            *images.shape[2:],
        )
        event_mass = (event_bins > 0).to(images.dtype).flatten(2).sum(dim=2)
        cumulative_event_mass = event_mass.cumsum(dim=1)
        total_event_mass = cumulative_event_mass[:, -1:].clamp_min(1.0)
        self.temporal_progress_sequence = (
            cumulative_event_mass / total_event_mass
        )
        surface = torch.zeros_like(event_bins[:, 0])
        neighbor_kernel = None
        if self.feature_mode == 'current_plus_motion':
            neighbor_kernel = images.new_ones(
                self.channels_per_bin, 1, 3, 3
            )
            neighbor_kernel[:, :, 1, 1] = 0.0

        for bin_index in range(self.temporal_bins):
            self.current_temporal_progress = (
                self.temporal_progress_sequence[:, bin_index]
            )
            current_events = (event_bins[:, bin_index] > 0).to(images.dtype)
            decayed_history = surface * self.history_decay
            if self.feature_mode.startswith('signed_') or self.feature_mode in (
                'duplicated_rise', 'duplicated_fall',
                'phase_gated_signed_change',
                'spatial_partitioned_signed_change',
                'dense_temporal_basis_signed_change',
                'temporal_ramp_signed_change',
            ):
                rising = (current_events - decayed_history).clamp_min(0.0)
                falling = (decayed_history - current_events).clamp_min(0.0)
                if self.feature_mode == 'signed_rise_only':
                    falling = torch.zeros_like(falling)
                elif self.feature_mode == 'signed_fall_only':
                    rising = torch.zeros_like(rising)
                if self.feature_mode == 'duplicated_rise':
                    spikes = torch.cat((rising, rising), dim=1)
                elif self.feature_mode == 'duplicated_fall':
                    spikes = torch.cat((falling, falling), dim=1)
                else:
                    spikes = torch.cat((rising, falling), dim=1)
                    if self.feature_mode in (
                        'phase_gated_signed_change',
                        'spatial_partitioned_signed_change',
                    ):
                        spikes = self._partition_signed_features(
                            spikes,
                            bin_index,
                        )
                    elif self.feature_mode in (
                        'dense_temporal_basis_signed_change',
                        'temporal_ramp_signed_change',
                    ):
                        spikes = self._apply_temporal_coordinate(
                            spikes,
                            bin_index,
                        )
            elif self.feature_mode == 'current_only':
                spikes = torch.cat(
                    (current_events, torch.zeros_like(current_events)), dim=1
                )
            elif self.feature_mode == 'duplicated_current':
                spikes = torch.cat((current_events, current_events), dim=1)
            else:
                neighbor_history = F.conv2d(
                    decayed_history,
                    neighbor_kernel,
                    padding=1,
                    groups=self.channels_per_bin,
                ).clamp(0.0, 1.0)
                local_motion = current_events * neighbor_history
                spikes = torch.cat((current_events, local_motion), dim=1)
            surface = torch.maximum(current_events, decayed_history)
            yield spikes


def _filter_args(args, allowed_keys):
    return {
        key: value
        for key, value in args.items()
        if key in allowed_keys
    }


def get_image_encoder(encoder, **args):
    if encoder in ('rank_order', 'first_spike'):
        return RankOrderEncoder(
            **_filter_args(
                args,
                {
                    'simulation_time',
                    'start_threshold',
                    'end_threshold',
                    'target_mean_earliness',
                    'end_score_search_steps',
                }
            )
        )
    if encoder in ('temporal_bin_rank_order', 'temporal_bins'):
        return TemporalBinRankOrderEncoder(
            **_filter_args(
                args,
                {
                    'simulation_time',
                    'temporal_bins',
                    'channels_per_bin',
                }
            )
        )
    if encoder in ('direct_event_bins', 'event_bins'):
        return DirectEventBinEncoder(
            **_filter_args(
                args,
                {
                    'simulation_time',
                    'temporal_bins',
                    'channels_per_bin',
                    'binarize',
                }
            )
        )
    if encoder in ('event_time_surface', 'time_surface'):
        return EventTimeSurfaceEncoder(
            **_filter_args(
                args,
                {
                    'simulation_time',
                    'temporal_bins',
                    'channels_per_bin',
                    'surface_decay',
                }
            )
        )
    if encoder in ('event_temporal_features', 'temporal_features'):
        return EventTemporalFeatureEncoder(
            **_filter_args(
                args,
                {
                    'simulation_time',
                    'temporal_bins',
                    'channels_per_bin',
                    'feature_mode',
                    'history_decay',
                    'partition_segments',
                    'temporal_basis_contrast',
                    'temporal_basis_direction',
                    'temporal_basis_count',
                    'temporal_basis_permutation_multiplier',
                    'temporal_basis_permutation_offset',
                    'temporal_ramp_direction',
                }
            )
        )
    raise ValueError(f"Unsupported encoder type: {encoder}")
