"""Scale-preserving calibration and normalization-aware structural receiver fitting.

These operations belong to the unlabeled structural initialization, not to the
online STDP learner. No validation/test targets or class-specific gains are used.
"""
import math
import torch


def calibrate_normalization_scale(cortex, target_mean_column_l1=1.0):
    target = float(target_mean_column_l1)
    if not math.isfinite(target) or target <= 0:
        raise ValueError('target_mean_column_l1 must be finite and positive')
    if float(cortex.oja_decay_rate) != 0:
        raise ValueError('normalization scale calibration currently requires additive STDP without Oja')
    weight = cortex.kernel.weight
    before = float(weight.abs().sum(0).mean())
    if not math.isfinite(before) or before <= 0:
        raise ValueError('normalization scale calibration requires nonzero finite weights')
    scale = target / before
    amplifier_before = float(cortex.amplifier)
    with torch.no_grad():
        weight.mul_(scale)
        cortex.log_amplifier.sub_(math.log(scale))
    # Sender traces are unamplified; additive STDP must follow the weight gauge.
    for name in ('base_potentiation_rate', 'potentiation_rate'):
        setattr(cortex, name, getattr(cortex, name) * scale)
    return {
        'kernel_scale': scale, 'stdp_rate_scale': scale,
        'mean_column_l1_before': before,
        'mean_column_l1_after': float(weight.abs().sum(0).mean()),
        'amplifier_before': amplifier_before,
        'amplifier_after': float(cortex.amplifier),
        'effective_drive_ratio': scale * float(cortex.amplifier) / amplifier_before,
    }


def fit_normalized_receiver_mixing(
    branch,
    response,
    target,
    mixing,
    row_weights,
    ridge,
    steps=100,
    learning_rate=0.01,
    native_onset_mode='none',
    source_onset=None,
    source_pre_onset=None,
    model_inhibition=None,
    native_threshold=None,
    native_recurrence_mode='fixed_surrogate',
    native_sequence_batch_sizes=None,
    native_lateral_inhibition_factor=0.0,
    native_lateral_inhibition_power=1.0,
    native_dominance_inhibition_factor=0.0,
    native_dominance_inhibition_power=1.0,
    native_competition_group_N=1,
    native_top_k=None,
):
    """Fit receiver mixing with a column-L1 constraint inside the objective.

    The common budget is the unconstrained fit's mean column L1. A later global
    kernel/amplifier gauge can move this budget to one without changing responses.
    Tucker's inner factor remains fixed. This is constrained receiver refitting,
    not joint optimization of all Tucker factors or supervised network training.
    """
    steps = int(steps)
    learning_rate = float(learning_rate)
    if not 1 <= steps <= 1000 or not 0 < learning_rate <= 1:
        raise ValueError('invalid normalization-aware solver budget')
    native_onset_mode = str(native_onset_mode).lower()
    if native_onset_mode not in {
        'none', 'positive_only', 'pre_onset_only', 'balanced',
    }:
        raise ValueError(
            'native_onset_mode must be none, positive_only, '
            'pre_onset_only, or balanced'
        )
    native_recurrence_mode = str(native_recurrence_mode).lower()
    if native_recurrence_mode not in {'fixed_surrogate', 'hard_unroll'}:
        raise ValueError(
            'native_recurrence_mode must be fixed_surrogate or hard_unroll'
        )
    if native_onset_mode == 'none' and native_recurrence_mode != 'fixed_surrogate':
        raise ValueError('hard native recurrence requires an active onset mode')
    branch, response, target = (x.detach().float() for x in (branch, response, target))
    initial = mixing.detach().float()
    budget = (branch @ initial).abs().sum(0).mean().detach()
    if not torch.isfinite(budget) or budget <= 0:
        raise ValueError('normalization-aware refit needs positive finite budget')
    identity = torch.eye(initial.shape[0], device=initial.device)
    weights = row_weights.detach().float()[:, None]
    reference_energy = (response.square() * weights).sum() / response.shape[1]
    penalty = float(ridge) * reference_energy.clamp_min(1e-12)
    target_energy = (target.square() * weights).sum().clamp_min(1e-12)
    onset_mask = pre_onset_mask = fixed_inhibition = None
    threshold = None
    sequence_batch_sizes = None
    sequence_time_steps = None
    if native_onset_mode != 'none':
        if any(value is None for value in (
            source_onset, source_pre_onset, model_inhibition,
            native_threshold,
        )):
            raise ValueError(
                'native onset fitting requires onset/pre-onset masks, '
                'model inhibition, and native threshold'
            )
        onset_mask = source_onset.detach().to(
            device=response.device, dtype=torch.bool,
        )
        pre_onset_mask = source_pre_onset.detach().to(
            device=response.device, dtype=torch.bool,
        )
        fixed_inhibition = model_inhibition.detach().to(
            device=response.device, dtype=response.dtype,
        )
        threshold = torch.as_tensor(
            native_threshold, device=response.device, dtype=response.dtype,
        )
        if threshold.numel() not in {1, response.shape[1]}:
            raise ValueError('native threshold must be scalar or per receiver')
        threshold = threshold.reshape(1, -1)
        if bool((threshold <= 0).any()) or not bool(torch.isfinite(threshold).all()):
            raise ValueError('native threshold must be finite and positive')
        expected_shape = response.shape
        for name, value in (
            ('source_onset', onset_mask),
            ('source_pre_onset', pre_onset_mask),
            ('model_inhibition', fixed_inhibition),
        ):
            if value.shape != expected_shape:
                raise ValueError(
                    f'{name} shape {tuple(value.shape)} does not match '
                    f'response shape {tuple(expected_shape)}'
                )
        if native_onset_mode in {'positive_only', 'balanced'} and not bool(onset_mask.any()):
            raise ValueError('native onset fitting found no source first-onset targets')
        if native_onset_mode in {'pre_onset_only', 'balanced'} and not bool(pre_onset_mask.any()):
            raise ValueError('native onset fitting found no source pre-onset targets')
        if native_recurrence_mode == 'hard_unroll':
            if not native_sequence_batch_sizes:
                raise ValueError('hard native recurrence requires replay batch sizes')
            sequence_batch_sizes = [
                int(value) for value in native_sequence_batch_sizes
            ]
            if any(value <= 0 for value in sequence_batch_sizes):
                raise ValueError('native replay batch sizes must be positive')
            sample_count = sum(sequence_batch_sizes)
            if response.shape[0] % sample_count != 0:
                raise ValueError(
                    'native replay rows must equal samples times simulation time'
                )
            sequence_time_steps = response.shape[0] // sample_count
            if sequence_time_steps <= 0:
                raise ValueError('native replay has no simulation steps')
            native_lateral_inhibition_factor = float(
                native_lateral_inhibition_factor
            )
            native_lateral_inhibition_power = float(
                native_lateral_inhibition_power
            )
            native_dominance_inhibition_factor = float(
                native_dominance_inhibition_factor
            )
            native_dominance_inhibition_power = float(
                native_dominance_inhibition_power
            )
            if (
                native_lateral_inhibition_power <= 0
                or native_dominance_inhibition_power <= 0
            ):
                raise ValueError('native inhibition powers must be positive')
            native_competition_group_N = int(native_competition_group_N)
            if (
                native_competition_group_N <= 0
                or response.shape[1] % native_competition_group_N != 0
            ):
                raise ValueError(
                    'native competition group size must divide receiver count'
                )
            if native_top_k is not None:
                native_top_k = int(native_top_k)
                if native_top_k <= 0:
                    raise ValueError('native top_k must be positive or null')

    # Compute gradients through the actual normalization, including its sign
    # dependent denominator. Fit loss uses the original unlabeled response rows.
    def normalized(value):
        norms = (branch @ value).abs().sum(0)
        if bool((norms <= 1e-12).any()):
            raise ValueError('normalization-aware fit generated a zero receiver')
        return value * (budget / norms)[None, :]
    def _hard_unrolled_native_loss(excitatory):
        positive_sum = excitatory.new_zeros(())
        pre_onset_sum = excitatory.new_zeros(())
        positive_count = excitatory.new_zeros(())
        pre_onset_count = excitatory.new_zeros(())
        offset = 0
        receiver_count = int(excitatory.shape[1])
        for batch_size in sequence_batch_sizes:
            row_count = sequence_time_steps * batch_size
            stop = offset + row_count
            response_sequence = excitatory[offset:stop].reshape(
                sequence_time_steps, batch_size, receiver_count,
            )
            onset_sequence = onset_mask[offset:stop].reshape(
                sequence_time_steps, batch_size, receiver_count,
            )
            pre_onset_sequence = pre_onset_mask[offset:stop].reshape(
                sequence_time_steps, batch_size, receiver_count,
            )
            previous_wave = torch.zeros(
                (batch_size, receiver_count),
                device=excitatory.device,
                dtype=torch.bool,
            )
            for time_index in range(sequence_time_steps):
                previous_float = previous_wave.to(dtype=excitatory.dtype)
                potential = response_sequence[time_index]
                if native_lateral_inhibition_factor != 0:
                    lateral = previous_float.mean(dim=-1, keepdim=True)
                    if native_lateral_inhibition_power != 1.0:
                        lateral = lateral.pow(native_lateral_inhibition_power)
                    potential = (
                        potential
                        - lateral * native_lateral_inhibition_factor
                    )
                if (
                    native_dominance_inhibition_factor != 0
                    and native_competition_group_N > 1
                ):
                    grouped = previous_float.reshape(
                        batch_size, -1, native_competition_group_N,
                    ).mean(dim=-1)
                    dominance = grouped.max(dim=-1, keepdim=True).values
                    if native_dominance_inhibition_power != 1.0:
                        dominance = dominance.pow(
                            native_dominance_inhibition_power
                        )
                    potential = (
                        potential
                        - dominance * native_dominance_inhibition_factor
                    )
                normalized_net = potential / threshold
                eligible = ~previous_wave
                positive_targets = onset_sequence[time_index] & eligible
                pre_onset_targets = (
                    pre_onset_sequence[time_index] & eligible
                )
                if bool(positive_targets.any()):
                    positive_error = torch.relu(
                        1.0 - normalized_net[positive_targets]
                    )
                    positive_sum = positive_sum + positive_error.square().sum()
                    positive_count = (
                        positive_count
                        + positive_targets.sum().to(excitatory.dtype)
                    )
                if bool(pre_onset_targets.any()):
                    pre_onset_error = torch.relu(
                        normalized_net[pre_onset_targets] - 1.0
                    )
                    pre_onset_sum = (
                        pre_onset_sum + pre_onset_error.square().sum()
                    )
                    pre_onset_count = (
                        pre_onset_count
                        + pre_onset_targets.sum().to(excitatory.dtype)
                    )
                candidate = (normalized_net >= 1.0) & eligible
                if native_top_k is not None and native_top_k < receiver_count:
                    scores = normalized_net.masked_fill(~candidate, -torch.inf)
                    top_values, top_indices = torch.topk(
                        scores,
                        native_top_k,
                        dim=-1,
                        sorted=False,
                    )
                    selected = torch.zeros_like(candidate)
                    selected.scatter_(-1, top_indices, torch.isfinite(top_values))
                    candidate = selected & candidate
                previous_wave = previous_wave | candidate
            offset = stop
        if offset != excitatory.shape[0]:
            raise RuntimeError('hard native recurrence did not consume all rows')
        return (
            positive_sum / positive_count.clamp_min(1.0),
            pre_onset_sum / pre_onset_count.clamp_min(1.0),
            positive_count,
            pre_onset_count,
        )

    def loss_components(value):
        effective = normalized(value)
        modeled_response = response @ effective
        residual = modeled_response - target
        base_loss = (
            (residual.square() * weights).sum()
            + penalty * (effective - identity).square().sum()
        ) / target_energy
        positive_loss = pre_onset_loss = base_loss.new_zeros(())
        positive_count = pre_onset_count = base_loss.new_zeros(())
        if native_onset_mode != 'none':
            if native_recurrence_mode == 'hard_unroll':
                (
                    positive_loss,
                    pre_onset_loss,
                    positive_count,
                    pre_onset_count,
                ) = _hard_unrolled_native_loss(modeled_response)
            else:
                normalized_net = (
                    response @ effective - fixed_inhibition
                ) / threshold
                if native_onset_mode in {'positive_only', 'balanced'}:
                    positive_loss = torch.relu(
                        1.0 - normalized_net[onset_mask]
                    ).square().mean()
                    positive_count = onset_mask.sum().to(base_loss.dtype)
                if native_onset_mode in {'pre_onset_only', 'balanced'}:
                    pre_onset_loss = torch.relu(
                        normalized_net[pre_onset_mask] - 1.0
                    ).square().mean()
                    pre_onset_count = pre_onset_mask.sum().to(base_loss.dtype)
        return (
            base_loss,
            effective,
            positive_loss,
            pre_onset_loss,
            positive_count,
            pre_onset_count,
        )

    with torch.no_grad():
        (
            initial_base,
            _,
            initial_positive,
            initial_pre_onset,
            initial_positive_count,
            initial_pre_onset_count,
        ) = loss_components(initial)
        if (
            native_onset_mode in {'positive_only', 'balanced'}
            and initial_positive_count <= 0
        ):
            raise ValueError('native recurrence found no eligible onset targets')
        if (
            native_onset_mode in {'pre_onset_only', 'balanced'}
            and initial_pre_onset_count <= 0
        ):
            raise ValueError('native recurrence found no eligible pre-onset targets')
        positive_scale = initial_base / initial_positive.clamp_min(1.0e-12)
        pre_onset_scale = initial_base / initial_pre_onset.clamp_min(1.0e-12)

    def objective(value):
        (
            base_loss,
            effective,
            positive_loss,
            pre_onset_loss,
            positive_count,
            pre_onset_count,
        ) = loss_components(value)
        native_loss = base_loss.new_zeros(())
        if native_onset_mode == 'positive_only':
            native_loss = positive_scale * positive_loss
        elif native_onset_mode == 'pre_onset_only':
            native_loss = pre_onset_scale * pre_onset_loss
        elif native_onset_mode == 'balanced':
            native_loss = 0.5 * (
                positive_scale * positive_loss
                + pre_onset_scale * pre_onset_loss
            )
        return (
            base_loss + native_loss,
            effective,
            base_loss,
            positive_loss,
            pre_onset_loss,
            native_loss,
            positive_count,
            pre_onset_count,
        )
    with torch.enable_grad():
        parameter = torch.nn.Parameter(initial.clone())
        optimizer = torch.optim.Adam([parameter], lr=learning_rate)
        best_loss = float('inf'); best = None; trace = []
        for step in range(steps + 1):
            (
                loss,
                effective,
                base_loss,
                positive_loss,
                pre_onset_loss,
                native_loss,
                positive_count,
                pre_onset_count,
            ) = objective(parameter)
            if not torch.isfinite(loss):
                raise ValueError('normalization-aware fit objective is nonfinite')
            number = float(loss.detach())
            if number < best_loss:
                best_loss, best = number, effective.detach().clone()
            if step in {0, 1, 10, steps}:
                trace.append({
                    'step': step,
                    'objective': number,
                    'base_objective': float(base_loss.detach()),
                    'native_objective': float(native_loss.detach()),
                    'positive_onset_violation': float(positive_loss.detach()),
                    'pre_onset_violation': float(pre_onset_loss.detach()),
                })
            if step == steps:
                break
            optimizer.zero_grad(); loss.backward(); optimizer.step()
    with torch.no_grad():
        projected = normalized(initial)
        fitted = branch @ best
        output = response @ best
        (
            final_objective,
            _,
            final_base,
            final_positive,
            final_pre_onset,
            final_native,
            final_positive_count,
            final_pre_onset_count,
        ) = objective(best)
        audit = {
            'steps': steps, 'learning_rate': learning_rate,
            'common_column_l1_budget': float(budget),
            'unconstrained_mse': float((response @ initial - target).square().mean()),
            'project_after_fit_mse': float((response @ projected - target).square().mean()),
            'constraint_aware_mse': float((output - target).square().mean()),
            'best_objective': best_loss,
            'column_budget_max_relative_error': float((fitted.abs().sum(0)/budget-1).abs().max()),
            'native_onset_mode_code': {
                'none': 0.0,
                'positive_only': 1.0,
                'pre_onset_only': 2.0,
                'balanced': 3.0,
            }[native_onset_mode],
            'native_recurrence_mode_code': {
                'fixed_surrogate': 0.0,
                'hard_unroll': 1.0,
            }[native_recurrence_mode],
            'native_onset_positive_count': float(
                0 if onset_mask is None else onset_mask.sum()
            ),
            'native_pre_onset_count': float(
                0 if pre_onset_mask is None else pre_onset_mask.sum()
            ),
            'native_positive_scale': float(
                positive_scale
                if native_onset_mode in {'positive_only', 'balanced'}
                else 0.0
            ),
            'native_pre_onset_scale': float(
                pre_onset_scale
                if native_onset_mode in {'pre_onset_only', 'balanced'}
                else 0.0
            ),
            'native_initial_positive_violation': float(initial_positive),
            'native_initial_pre_onset_violation': float(initial_pre_onset),
            'native_initial_effective_positive_count': float(
                initial_positive_count
            ),
            'native_initial_effective_pre_onset_count': float(
                initial_pre_onset_count
            ),
            'native_final_positive_violation': float(final_positive),
            'native_final_pre_onset_violation': float(final_pre_onset),
            'native_final_effective_positive_count': float(
                final_positive_count
            ),
            'native_final_effective_pre_onset_count': float(
                final_pre_onset_count
            ),
            'native_final_base_objective': float(final_base),
            'native_final_objective': float(final_native),
            'native_final_total_objective': float(final_objective),
            'trace': trace,
        }
    return best, fitted, output, audit
