import numpy as np
import torch
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from PIL import Image
from modules.utils import remove_nan
from io import BytesIO


# avoid PIL.Image.DecompressionBombError when hidden neurons increased.
Image.MAX_IMAGE_PIXELS = 1000000000
MAX_PLOT_PIXELS_PER_SIDE = 2048
MAX_FIGURE_INCHES_PER_SIDE = 20


def get_cortex_top_maps(cortex, target_index=None):
    if target_index is None:
        target_index = torch.arange(cortex.kernel.weight.shape[-1]).to(
            cortex.kernel.weight.device
        )

    input_len, hidden_num = cortex.shape
    valid = (target_index >= 0) & (target_index < hidden_num)
    target_index_safe = target_index.clamp(0, hidden_num - 1)
    target_kernel = cortex.kernel.weight.T[target_index_safe] * valid.unsqueeze(1)

    if not cortex.subcortexs:
        return target_kernel.reshape(target_kernel.shape[0], -1, cortex.kernel_size, cortex.kernel_size)

    threshold_parts = [torch.ones(input_len).to(cortex.kernel.weight.device)]
    subcortex_target_indices = []
    i = input_len
    for subcortex in cortex.subcortexs:
        patch_h, patch_w, _ = subcortex._get_patch_grid(
            cortex.kernel_size,
            cortex.kernel_size
        )
        _, _, patch_num = subcortex.pooling.get_pooled_grid(patch_h, patch_w)
        output_feature_len = patch_num * len(subcortex.output_nv)
        sub_target_index = target_index_safe - i
        threshold_parts.append(subcortex.hidden_nv.thresholds.repeat(patch_num))
        subcortex_target_indices.append(sub_target_index % subcortex.shape[1])
        i += output_feature_len

    thresholds = torch.concat(threshold_parts)
    thres_weighted_target_kerrnel = thresholds * target_kernel
    top_map_index = thres_weighted_target_kerrnel.argmax(axis=1)
    subcortex_top_maps = []
    for subcortex, sub_target_index in zip(cortex.subcortexs, subcortex_target_indices):
        subcortex_top_maps.append(get_cortex_top_maps(subcortex, sub_target_index))

    return torch.stack(subcortex_top_maps).sum(0)


def get_cortex_dict(cortex_obj):
    cortex_dict = {
        cortex_obj.cortex_id: cortex_obj
    }
    for i, subcortex in enumerate(cortex_obj.subcortexs):
        cortex_dict.update(
            get_cortex_dict(subcortex)
        )

    return cortex_dict


def _get_cortex_charge_status(cortex):
    if cortex._cached_init_patch_grid is None:
        raise ValueError('cortex state must be initialized before charge status is read')

    patch_h, patch_w, _ = cortex._cached_init_patch_grid
    potential = cortex.hidden_nv.current_potential
    if potential is None:
        raise ValueError('cortex potential must be initialized before charge status is read')

    charge_status = torch.where(
        cortex.hidden_nv.spike_wave > 0,
        torch.zeros_like(potential),
        potential
    )
    charge_status = cortex.aggregate_competition_groups(charge_status)
    charge_status = cortex.pooling.apply_output_pooling(charge_status, patch_h, patch_w)
    return charge_status.reshape(*charge_status.shape[:-2], -1)


def _get_cortex_output_activity(cortex):
    spikes = cortex.output_nv.current_spikes.reshape(*cortex.output_nv.current_spikes.shape[:-2], -1)
    charge_status = _get_cortex_charge_status(cortex)
    return spikes, charge_status


def _get_cortex_hidden_activity(cortex):
    return cortex.hidden_nv.current_spikes.reshape(*cortex.hidden_nv.current_spikes.shape[:-2], -1)


def _get_cortex_output_earliness(cortex):
    return cortex.output_nv.earliness.reshape(*cortex.output_nv.earliness.shape[:-2], -1)


def _get_cortex_hidden_earliness(cortex):
    return cortex.hidden_nv.earliness.reshape(*cortex.hidden_nv.earliness.shape[:-2], -1)


def get_spike_train_dict(
    model, images, cortex_dict, rate_only=False, return_emitted_rate=False,
    return_earliness=False
):
    """Collect post-pooling emitted trains and charge status without center-patch sampling."""
    device = getattr(
        model,
        'device',
        torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    )
    images = images.to(device)
    model.init_state(images.shape, device)

    if not rate_only:
        output_spike_trains = defaultdict(list)
        hidden_spike_trains = defaultdict(list)
        charge_status_trains = defaultdict(list)
    if rate_only or return_emitted_rate:
        spike_count = {}

    for spikes in model.image_encoder(images):
        model.forward_cortex(spikes, is_training=False)

        for cortex_name, cortex in cortex_dict.items():
            cortex_spikes, cortex_charge_status = _get_cortex_output_activity(cortex)
            if not rate_only:
                output_spike_trains[cortex_name].append(cortex_spikes)
                hidden_spike_trains[cortex_name].append(_get_cortex_hidden_activity(cortex))
                charge_status_trains[cortex_name].append(cortex_charge_status)
            if rate_only or return_emitted_rate:
                if cortex_name not in spike_count:
                    spike_count[cortex_name] = cortex_spikes
                else:
                    spike_count[cortex_name] += cortex_spikes

    if rate_only:
        spike_rate = {
            cortex_name: count / model.image_encoder.simulation_time
            for cortex_name, count in spike_count.items()
        }
        return spike_rate, {}

    for cortex_name in cortex_dict:
        output_spike_trains[cortex_name] = torch.stack(output_spike_trains[cortex_name], axis=-1)
        hidden_spike_trains[cortex_name] = torch.stack(hidden_spike_trains[cortex_name], axis=-1)
        charge_status_trains[cortex_name] = torch.stack(charge_status_trains[cortex_name], axis=-1)

    if return_earliness:
        output_earliness = {
            cortex_name: _get_cortex_output_earliness(cortex)
            for cortex_name, cortex in cortex_dict.items()
        }
        hidden_earliness = {
            cortex_name: _get_cortex_hidden_earliness(cortex)
            for cortex_name, cortex in cortex_dict.items()
        }

    if return_emitted_rate:
        spike_rate = {
            cortex_name: count / model.image_encoder.simulation_time
            for cortex_name, count in spike_count.items()
        }
        if return_earliness:
            return (
                output_spike_trains, hidden_spike_trains, charge_status_trains,
                output_earliness, hidden_earliness, spike_rate
            )
        return output_spike_trains, charge_status_trains, spike_rate

    if return_earliness:
        return (
            output_spike_trains, hidden_spike_trains, charge_status_trains,
            output_earliness, hidden_earliness
        )

    return output_spike_trains, hidden_spike_trains, charge_status_trains


def _plt_to_np_array(plt, dpi=100):
    # matplotlib default dpi = 100, set it lower to speed up when the image is big.

    buf = BytesIO()
    plt.savefig(buf, format='jpeg', dpi=dpi)
    buf.seek(0)

    image = Image.open(buf)
    image_np_array = np.array(image)

    buf.close()

    return image_np_array


def _downsample_heatmap_inputs(matrix, dot_mask=None, max_side=MAX_PLOT_PIXELS_PER_SIDE):
    matrix = np.asarray(matrix)
    dot_mask = None if dot_mask is None else np.asarray(dot_mask)

    if matrix.ndim != 2:
        return matrix, dot_mask

    row_step = max(1, int(np.ceil(matrix.shape[0] / max_side)))
    col_step = max(1, int(np.ceil(matrix.shape[1] / max_side)))

    if row_step == 1 and col_step == 1:
        return matrix, dot_mask

    downsampled_matrix = matrix[::row_step, ::col_step]
    downsampled_mask = None if dot_mask is None else dot_mask[::row_step, ::col_step]
    return downsampled_matrix, downsampled_mask


def _summarize_temporal_spike_stats(spike_train_batch, model, earliness=None):
    time_steps = spike_train_batch.shape[-1]
    flat_spike_train = spike_train_batch.reshape(spike_train_batch.shape[0], -1, time_steps)
    spike_mask = flat_spike_train > 0
    has_spike = spike_mask.any(dim=-1)
    if earliness is None:
        flat_spike_wave_train = flat_spike_train.cumsum(dim=-1).clamp(max=1)
        earliness = flat_spike_wave_train.sum(dim=-1) / float(time_steps)
    else:
        earliness = earliness.reshape(
            earliness.shape[0],
            -1
        )
    first_spike_earliness = earliness.max(dim=1).values

    return {
        'no spike ratio': (~has_spike).to(torch.float32).mean().detach().cpu(),
        'mean first spike earliness': first_spike_earliness.mean().detach().cpu(),
        'mean earliness': earliness.mean().detach().cpu(),
    }


def _flatten_spike_train_for_heatmap(spike_train):
    return spike_train.reshape(-1, spike_train.shape[-1])


def _format_label_for_title(label):
    if torch.is_tensor(label) and label.numel() == 1:
        return label.item()
    return label


def analysis_spike_activity(
    sample_data_dict, model, showing_length=100, make_plots=True,
    plot_phases=('train', 'valid')
):
    cortex_dict = get_cortex_dict(model.cortex)
    spike_train_image_dict = defaultdict(list)
    cortex_activity_stats_dict = defaultdict(lambda: defaultdict(list))
    plot_phases = set(plot_phases)
    for phase, (images, labels) in sample_data_dict.items():
        should_plot_phase = make_plots and phase in plot_phases
        image_spike_train = torch.stack(list(model.image_encoder(images)), dim=-1)
        image_activity_stats = _summarize_temporal_spike_stats(image_spike_train, model)
        for stat_name, value in image_activity_stats.items():
            cortex_activity_stats_dict['image_encoder'][stat_name].append(value)
        if should_plot_phase:
            figure_name = f'[{phase}]spike train/image_encoder'
            for spike_train, label in zip(image_spike_train, labels):
                flat_spike_train = _flatten_spike_train_for_heatmap(spike_train)
                label_value = _format_label_for_title(label)
                spike_train_image_dict[figure_name].append(
                    draw_heat_map(
                        flat_spike_train[:, :showing_length].cpu().numpy(),
                        length=10,
                        title=f'image_encoder, label: {label_value}',
                        dot_mask=flat_spike_train[:, :showing_length].cpu().numpy(),
                    )
                )
        (
            output_spike_train_dict,
            hidden_spike_train_dict,
            charge_status_train_dict,
            output_earliness_dict,
            hidden_earliness_dict
        ) = get_spike_train_dict(
            model, images, cortex_dict, return_earliness=True
        )
        for cortex_name, spike_train_batch in output_spike_train_dict.items():
            output_stats = _summarize_temporal_spike_stats(
                spike_train_batch,
                model,
                output_earliness_dict[cortex_name]
            )
            hidden_stats = _summarize_temporal_spike_stats(
                hidden_spike_train_dict[cortex_name],
                model,
                hidden_earliness_dict[cortex_name]
            )
            cortex_activity_stats_dict[cortex_name]['mean output earliness'].append(
                output_stats['mean earliness']
            )
            cortex_activity_stats_dict[cortex_name]['mean hidden earliness'].append(
                hidden_stats['mean earliness']
            )
            cortex_activity_stats_dict[cortex_name]['output no spike ratio'].append(
                output_stats['no spike ratio']
            )
            cortex_activity_stats_dict[cortex_name]['output mean first spike earliness'].append(
                output_stats['mean first spike earliness']
            )
            cortex_activity_stats_dict[cortex_name]['hidden no spike ratio'].append(
                hidden_stats['no spike ratio']
            )
            cortex_activity_stats_dict[cortex_name]['hidden mean first spike earliness'].append(
                hidden_stats['mean first spike earliness']
            )
            if not should_plot_phase:
                continue
            figure_name = f'[{phase}]spike train/{cortex_name}'
            for spike_train, charge_status_train, label in zip(
                spike_train_batch, charge_status_train_dict[cortex_name], labels
            ):
                flat_spike_train = _flatten_spike_train_for_heatmap(spike_train)
                flat_charge_status_train = _flatten_spike_train_for_heatmap(charge_status_train)
                label_value = _format_label_for_title(label)
                spike_train_image_dict[figure_name].append(
                    draw_heat_map(
                        flat_charge_status_train[:, :showing_length].cpu().numpy(),
                        length=10,
                        title=f'{cortex_name}, label: {label_value}',
                        dot_mask=flat_spike_train[:, :showing_length].cpu().numpy(),
                    )
                )

    for figure_name in spike_train_image_dict:
        spike_train_image_dict[figure_name] = np.stack(spike_train_image_dict[figure_name])
    for cortex_name, stats in cortex_activity_stats_dict.items():
        cortex_activity_stats_dict[cortex_name] = {
            stat_name: np.mean(values)
            for stat_name, values in stats.items()
        }
    return spike_train_image_dict, cortex_activity_stats_dict


def draw_singular_value_spectrum(W, cortex_id, max_rank=20, relative=True, show=False):
    """
    Plot the singular value spectrum of a weight matrix.

    Args:
        W: Weight matrix of shape [N_pre, N_post].
    """
    if W.ndim != 2:
        raise ValueError(f'W must be 2D, got shape {tuple(W.shape)}')

    # Compute singular values in descending order
    singular_value = torch.linalg.svdvals(W.detach().float().cpu())

    if max_rank is not None:
        singular_value = singular_value[:max_rank]

    if relative and singular_value.numel() > 0 and singular_value[0] > 0:
        singular_value = singular_value / singular_value[0]

    x = torch.arange(1, singular_value.numel() + 1)

    plt.figure()
    plt.plot(x.numpy(), singular_value.numpy(), marker='o', linewidth=1)
    plt.xlabel('Rank')
    plt.ylabel(r'Relative Singular Value (normalized by $\sigma_{max}$)')
    plt.title(f'Singular value spectrum {cortex_id}')

    plt.grid(True, alpha=0.3)

    if show:
        plt.show()
    RGB_matrix = _plt_to_np_array(plt)
    plt.close()
    return RGB_matrix


def draw_heat_map(matrix, length=6, title=None, show=False, h_lines=[], dot_mask=None):
    matrix, dot_mask = _downsample_heatmap_inputs(matrix, dot_mask=dot_mask)
    WH_ratio = matrix.shape[0] / matrix.shape[1]
    figsize = np.array([1, max(WH_ratio, 0.2)]) * length

    space_for_notation = length/10
    figsize[0] += space_for_notation
    figsize = np.clip(figsize, 1, MAX_FIGURE_INCHES_PER_SIDE)

    plt.figure(figsize=figsize)
    if title is not None:
        plt.title(title)
    matrix = remove_nan(matrix, 0)
    norm = colors.TwoSlopeNorm(vmin=min(-1, matrix.min()), vcenter=0, vmax=max(1, matrix.max()))
    plt.pcolormesh(matrix, cmap='coolwarm', norm=norm)
    plt.colorbar()

    ax = plt.gca()
    for line in h_lines:
        ax.hlines(line, xmin=0, xmax=matrix.shape[1], colors='k', linewidth=1)

    if dot_mask is not None:
        y_idx, x_idx = np.where(dot_mask > 0)
        ax.scatter(x_idx + 0.5, y_idx + 0.5, color='black', marker='o', s=50)

    ax.invert_yaxis()

    if show:
        plt.show()
    RGB_matrix = _plt_to_np_array(plt)
    plt.close()
    return RGB_matrix


def draw_4D_tensor_in_2D(tensor_4D, show=False, title=None):
    """
    tensor_4D: (N, C, W, H) torch.Tensor or np.ndarray
    show: bool, whether to call plt.show()
    title: optional figure title
    """
    # ---- to numpy ----
    if isinstance(tensor_4D, torch.Tensor):
        tensor_4D = tensor_4D.detach().cpu().numpy()
    tensor_4D = np.asarray(tensor_4D)
    N, C, W, H = tensor_4D.shape

    # ---- fix color scale ----
    vmax = float(np.max(np.abs(tensor_4D)))
    vmax = 1.0 if vmax == 0 else vmax

    # ---- build mosaic_2D: (N,C,W,H) -> (N*W, C*H) ----
    mosaic_2D = tensor_4D.transpose(0, 2, 1, 3).reshape(N * W, C * H)

    # ---- figure size: depends on N, C ----
    fig_w = max(6.0, 2.4 * C)
    fig_h = max(6.0, 2.0 * N)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(
        mosaic_2D,
        cmap='bwr',
        vmin=-vmax,
        vmax=vmax,
        interpolation='nearest',
        origin='upper',
    )

    # ---- grid lines between blocks (fixed on) ----
    for i in range(1, N):
        ax.axhline(i * W - 0.5, color='k', linewidth=0.8)
    for j in range(1, C):
        ax.axvline(j * H - 0.5, color='k', linewidth=0.8)

    # ---- TOP axis: 1..C ----
    x_centers = [j * H + (H - 1) / 2 for j in range(C)]
    ax.set_xticks(x_centers)
    ax.set_xticklabels([str(j + 1) for j in range(C)])
    ax.xaxis.set_ticks_position('top')
    ax.tick_params(axis='x', labeltop=True, labelbottom=False, length=0)

    # ---- LEFT axis: 1..N ----
    y_centers = [i * W + (W - 1) / 2 for i in range(N)]
    ax.set_yticks(y_centers)
    ax.set_yticklabels([str(i + 1) for i in range(N)])
    ax.tick_params(axis='y', length=0)

    # ---- bounds cleanup ----
    ax.set_xlim(-0.5, C * H - 0.5)
    ax.set_ylim(N * W - 0.5, -0.5)

    if title:
        ax.set_title(title, pad=20)

    fig.colorbar(im, ax=ax, shrink=0.8, pad=0.01)

    if show:
        plt.show()

    RGB_matrix = _plt_to_np_array(plt)
    plt.close()
    return RGB_matrix


def draw_bar_chart(D, hlines=[], bottom=None, title=None, labels=None, show=False):
    keys, values = zip(*D.items())
    bars = plt.bar(keys, values, bottom=bottom)
    plt.xticks(rotation=45)
    plt.ylim(0.1, 1)
    for hline in hlines:
        plt.axhline(y=hline, c='r', ls='--', lw=2)
    if labels is not None:
        plt.bar_label(bars, labels=labels, label_type='center', fontsize=15)
    if title is not None:
        plt.title(title)
    if show:
        plt.show()
    RGB_matrix = _plt_to_np_array(plt)
    plt.close()
    return RGB_matrix


def draw_distribution_chart(T, bins, title=None, xlabel=None, ylabel=None, show=False):
    plt.hist(T.flatten(), bins=bins)
    if title is not None:
        plt.title(title)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    if show:
        plt.show()
    RGB_matrix = _plt_to_np_array(plt)
    plt.close()
    return RGB_matrix


def get_nested_shape(cortex):
    input_len, output_len = cortex.shape
    kernel_shape = cortex.kernel.weight.shape
    subcortex_shapes = [
        get_nested_shape(subcortex)
        for subcortex in cortex.subcortexs
    ]
    return {
        'cortex_id': cortex.cortex_id,
        'input_len': input_len,
        'output_len': output_len,
        'kernel_shape': kernel_shape,
        'amplifier': cortex.amplifier,
        'p_rate': cortex.potentiation_rate,
        'subcortexs': subcortex_shapes
    }
