import copy
import torch
import numpy as np
from sklearn.decomposition import PCA as sklearn_PCA
# to avoid centering trick,
# we implement our own version based on sklearn.decomposition.TruncatedSVD
from sklearn.decomposition import TruncatedSVD, NMF
from sklearn.model_selection import LeavePOut
from modules.yielding_ontology.stdp_cortex import StdpCortex
from modules.yielding_ontology.cortex_pooling import CortexPooling
from modules.utils import move_to_device


class PcaCortex(StdpCortex):
    def __init__(
        self, PCA_cortex_settings, **kargs
    ):
        super().__init__(**kargs)

        def setup(
            new_cortex_split_ratio, regularize_afterward,
            PCA_settings, weight_refund,
            refund_with_amplifier, split_only_tail
        ):
            self.shape_fixed = False
            self.new_cortex_split_ratio = new_cortex_split_ratio
            self.regularize_afterward = regularize_afterward
            self.PCA_settings = PCA_settings
            self.weight_refund = weight_refund
            self.refund_with_amplifier = refund_with_amplifier
            self.split_only_tail = split_only_tail

        setup(**PCA_cortex_settings)

    def fix_shape(self):
        self.shape_fixed = True

    def self_cleaning(self):
        input_len, output_len = self.shape
        removed_receiver_indices = []
        simplified_kernel = self.kernel.weight[:, :output_len].clone()
        i = len(self.hidden_nv)
        for subcortex in self.subcortexs:
            subcortex_inlen, subcortex_outlen = subcortex.shape
            sub_removed_indices, sub_simplified_kernel = subcortex.self_cleaning()
            removed_receiver_indices.append(sub_removed_indices+i)

            if self.weight_refund:
                if len(sub_removed_indices) > 0:
                    refund_weight = torch.einsum(
                        'Sr, rR -> SR',
                        self.kernel.weight[:, sub_removed_indices+i],
                        sub_simplified_kernel[sub_removed_indices, :]
                    )
                    self.kernel.weight[:, :output_len] += refund_weight
                    simplified_kernel += refund_weight

                # Keep only the indices that are not removed
                all_indices = torch.arange(subcortex_inlen, device=self.kernel.weight.device)
                kept_indices_mask = ~torch.isin(all_indices, sub_removed_indices)
                sub_kept_indices = all_indices[kept_indices_mask]

                simplified_sub_weight = torch.einsum(
                    'Sk, kR -> SR',
                    self.kernel.weight[:, sub_kept_indices+i],
                    sub_simplified_kernel[sub_kept_indices, :]
                )
                simplified_kernel += simplified_sub_weight

            i += subcortex_inlen

        # remove small threshold input (if shape is not fixed).
        if not self.shape_fixed:
            removed_indices = self.input_nv.delete_small_threshold_neurons()
            self._remove_kernel_indices(dim=0, indices=removed_indices)
        else:
            removed_indices = torch.tensor([])

        # remove small threshold subcortex input neurons
        if len(removed_receiver_indices) > 0:
            self._remove_kernel_indices(dim=1, indices=torch.cat(removed_receiver_indices))
        # if any subcortex has no input neuron:
        self._remove_dead_subcortexs()
        self.kernel.input_len = len(self.input_nv)
        self.kernel.output_len = len(self.hidden_nv)
        self.kernel.has_subcortexs = len(self.subcortexs) > 0

        if self.refund_with_amplifier:
            simplified_kernel *= self.amplifier

        return removed_indices, simplified_kernel

    def _remove_dead_subcortexs(self):
        i = 0
        while i < len(self.subcortexs):
            sub_inlen, sub_outlen = self.subcortexs[i].shape
            if sub_inlen == 0:
                self.subcortexs.pop(i)
            else:
                i += 1

    def _remove_kernel_indices(self, dim, indices):
        mask = torch.ones(
            self.kernel.weight.shape[dim],
            dtype=torch.bool,
            device=self.kernel.weight.device
        )
        mask[indices] = False
        if dim == 0:
            self.kernel.weight = self.kernel.weight[mask, :]
        elif dim == 1:
            self.kernel.weight = self.kernel.weight[:, mask]
        else:
            raise ValueError(f"Invalid dim {dim}, expected 0 or 1")

    def normalizing_PCA(PCA_func):
        def wrap(self, kernel, normalizing, **kwargs):
            S_remaining, D_remaining, amplifier = PCA_func(self, kernel, **kwargs)

            if normalizing:
                epsilon = 1.0e-12
                S_L1_norm = abs(S_remaining).sum(axis=1).mean() + epsilon
                D_L1_norm = abs(D_remaining).sum(axis=1).mean() + epsilon
                S_remaining /= S_L1_norm
                D_remaining /= D_L1_norm
                amplifier *= S_L1_norm * D_L1_norm

            return S_remaining, D_remaining, amplifier
        return wrap

    def transposing_PCA(PCA_func):
        def wrap(self, kernel, transpose, **kwargs):
            if transpose:
                transposed_kernel = kernel.transpose()
            else:
                transposed_kernel = kernel

            S_remaining, D_remaining = PCA_func(self, transposed_kernel, **kwargs)

            if transpose:
                tmp = S_remaining.transpose()
                S_remaining = D_remaining.transpose()
                D_remaining = tmp

            return S_remaining, D_remaining
        return wrap

    def leave_P_out_PCA(PCA_func):
        def wrap(self, kernel, leave_out_num, **kwargs):
            if leave_out_num == 0:
                S_remaining, D_remaining = PCA_func(self, kernel, **kwargs)
                return S_remaining, D_remaining, 1

            lpo = LeavePOut(leave_out_num)
            S_remainings = []
            D_remainings = []
            split_count = 0
            for kept_index, left_out_index in lpo.split(np.arange(kernel.shape[-1])):
                split_count += 1
                masked_kernel = np.copy(kernel)
                masked_kernel[:, left_out_index] = 0.0
                S_remaining, D_remaining = PCA_func(self, masked_kernel, **kwargs)
                S_remainings.append(S_remaining)
                D_remainings.append(D_remaining)

            amplifier = 1.0 / max(split_count, 1)
            return np.concatenate(S_remainings, axis=1), np.concatenate(D_remainings, axis=0), amplifier
        return wrap

    @normalizing_PCA
    @leave_P_out_PCA
    @transposing_PCA
    def PCA(self, kernel, remained_variance, method):
        # method in {'PCA', 'randomized_PCA', 'non_centering_PCA', 'truncated_SVD', 'PNMF'}
        if method == 'PCA':
            if remained_variance == 1:
                # parameter for sklearn.decomposition.PCA
                # https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
                remained_variance = None
            pca = sklearn_PCA(svd_solver='full', n_components=remained_variance)
            S_remaining = pca.fit_transform(kernel)
            D_remaining = pca.components_
        elif method == 'randomized_PCA':
            component_count = int(remained_variance)
            if component_count <= 0:
                raise ValueError('randomized_PCA remained_variance must be a positive component count')
            pca = sklearn_PCA(
                svd_solver='randomized',
                n_components=component_count,
                random_state=722,
            )
            S_remaining = pca.fit_transform(kernel)
            D_remaining = pca.components_
        elif method == 'non_centering_PCA':
            sender_num, receiver_num = kernel.shape
            svd = TruncatedSVD(n_oversamples=receiver_num, n_components=receiver_num)
            S_full = svd.fit_transform(kernel)
            D_full = svd.components_

            # calculate how many components to keep
            remaining_components_num = np.searchsorted(
                np.cumsum(svd.explained_variance_ratio_), remained_variance
            )+1

            S_remaining = S_full[:, :remaining_components_num]
            D_remaining = D_full[:remaining_components_num, :]
        elif method == 'truncated_SVD':
            component_count = int(remained_variance)
            if component_count <= 0:
                raise ValueError('truncated_SVD remained_variance must be a positive component count')
            svd = TruncatedSVD(
                n_components=component_count,
                random_state=722,
            )
            S_remaining = svd.fit_transform(kernel)
            D_remaining = svd.components_
        elif method == 'PNMF':
            if remained_variance == 1:
                # parameter for sklearn.decomposition.PCA
                # https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
                remained_variance = None

            kernel_pos = np.maximum(kernel, 0)
            kernel_neg = np.maximum(-kernel, 0)

            model_pos = NMF(n_components=remained_variance, init='random', random_state=722)
            W_pos = model_pos.fit_transform(kernel_pos)
            H_pos = model_pos.components_

            model_neg = NMF(n_components=remained_variance, init='random', random_state=722)
            W_neg = model_neg.fit_transform(kernel_neg)
            H_neg = -model_neg.components_

            S_remaining = np.concatenate([W_pos, W_neg], axis=1)
            D_remaining = np.concatenate([H_pos, H_neg], axis=0)

        return S_remaining, D_remaining

    def splitting(self, cortex_constructor):
        # recursive into subcortex first to prevent infinite splitting!
        for subcortex in self.subcortexs:
            subcortex.splitting(cortex_constructor)

        if self.split_only_tail and len(self.subcortexs) > 0:
            return

        self.yield_pca_subcortex(cortex_constructor)

    def yield_pca_subcortex(
        self,
        cortex_constructor,
        PCA_settings=None,
        new_cortex_split_ratio=None,
        regularize_afterward=None,
        pooling_settings=None,
        root_direct_mode=None,
        root_projection_scale=1.0,
        root_learning_mask=None,
    ):
        if self.subcortexs:
            raise ValueError(
                'pca_yield currently supports only a target cortex without '
                'existing subcortexs'
            )

        h, w = self.shape
        if h < 2 or w < 2:
            raise ValueError('pca_yield target cortex is too small to split')

        pca_settings = copy.deepcopy(
            self.PCA_settings if PCA_settings is None else PCA_settings
        )
        split_ratio = (
            self.new_cortex_split_ratio
            if new_cortex_split_ratio is None
            else float(new_cortex_split_ratio)
        )
        should_regularize = (
            self.regularize_afterward
            if regularize_afterward is None
            else bool(regularize_afterward)
        )
        root_direct_mode = self._get_root_direct_mode(
            root_direct_mode,
            should_regularize
        )
        root_projection_scale = self._get_root_projection_scale(
            root_projection_scale
        )
        if pooling_settings is None:
            pooling_settings = {'mode': 'max', 'kernel_size': 'global'}

        source_weight = self.kernel.weight[:h, :w].detach()
        source_device = source_weight.device
        source_dtype = source_weight.dtype
        source_kernel = source_weight.cpu().numpy()
        new_cortex_kernel, right_matrix, amplifier = self.PCA(
            source_kernel,
            **pca_settings
        )

        new_cortex_kernel *= split_ratio

        summarized_new_kernel = (
            np.einsum('in,no->io', new_cortex_kernel, right_matrix)
            * amplifier
        )
        if root_direct_mode == 'source_passthrough':
            residual_kernel = source_kernel.copy()
        else:
            residual_kernel = source_kernel - summarized_new_kernel
        new_root_kernel = np.concatenate(
            [residual_kernel, right_matrix * root_projection_scale],
            axis=0
        )

        self.kernel.weight = torch.as_tensor(
            new_root_kernel,
            dtype=source_dtype,
            device=source_device
        )
        self.kernel.has_subcortexs = True
        self.kernel.output_len = len(self.hidden_nv)
        if should_regularize:
            self.kernel.kernel_regulation()
        self._apply_root_learning_mask(
            root_learning_mask,
            direct_input_len=h
        )

        input_channel = h // (self.kernel_size ** 2)
        if input_channel * (self.kernel_size ** 2) != h:
            raise ValueError(
                'pca_yield cannot infer input_channel from target cortex shape'
            )
        output_channel = int(new_cortex_kernel.shape[1])
        if output_channel <= 0:
            raise ValueError('pca_yield produced no PCA components')
        subcortex_kernel = torch.as_tensor(
            new_cortex_kernel,
            dtype=source_dtype,
            device=source_device
        )

        self.subcortex_counter += 1
        new_cortex = cortex_constructor(
            cortex_id=f'{self.cortex_id}-{self.subcortex_counter}',
            kernel_size=self.kernel_size,
            stride=self.stride,
            input_channel=input_channel,
            output_channel=output_channel,
            kernel=subcortex_kernel,
            init_amplifier=float(amplifier),
            pooling_settings=pooling_settings,
            competition_group_N=1,
        )
        new_cortex = move_to_device(new_cortex, source_device)
        self.subcortexs.append(new_cortex)
        self.kernel.has_subcortexs = len(self.subcortexs) > 0
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
        return new_cortex

    def yield_patch_pca_subcortex(
        self,
        cortex_constructor,
        local_kernel_size,
        local_stride,
        PCA_settings=None,
        new_cortex_split_ratio=None,
        regularize_afterward=None,
        pooling_settings=None,
        root_direct_mode=None,
        root_projection_scale=1.0,
        root_learning_mask=None,
    ):
        if self.subcortexs:
            raise ValueError(
                'patch_pca_yield currently supports only a target cortex '
                'without existing subcortexs'
            )

        h, w = self.shape
        input_channel = h // (self.kernel_size ** 2)
        if input_channel * (self.kernel_size ** 2) != h:
            raise ValueError(
                'patch_pca_yield cannot infer input_channel from target cortex shape'
            )

        local_kernel_size = int(local_kernel_size)
        local_stride = int(local_stride)
        if local_kernel_size <= 0:
            raise ValueError('patch_pca_yield local_kernel_size must be positive')
        if local_stride <= 0:
            raise ValueError('patch_pca_yield local_stride must be positive')
        if local_kernel_size > self.kernel_size:
            raise ValueError(
                'patch_pca_yield local_kernel_size cannot exceed target kernel_size'
            )

        patch_h = (self.kernel_size - local_kernel_size) // local_stride + 1
        patch_w = (self.kernel_size - local_kernel_size) // local_stride + 1
        patch_num = patch_h * patch_w
        if patch_num <= 0:
            raise ValueError('patch_pca_yield produced no local patch positions')

        pca_settings = copy.deepcopy(
            self.PCA_settings if PCA_settings is None else PCA_settings
        )
        split_ratio = (
            self.new_cortex_split_ratio
            if new_cortex_split_ratio is None
            else float(new_cortex_split_ratio)
        )
        should_regularize = (
            self.regularize_afterward
            if regularize_afterward is None
            else bool(regularize_afterward)
        )
        root_direct_mode = self._get_root_direct_mode(
            root_direct_mode,
            should_regularize
        )
        root_projection_scale = self._get_root_projection_scale(
            root_projection_scale
        )
        if pooling_settings is None:
            pooling_settings = {'mode': 'max', 'kernel_size': 1}
        pooling = CortexPooling(pooling_settings)

        source_weight = self.kernel.weight[:h, :w].detach()
        source_device = source_weight.device
        source_dtype = source_weight.dtype
        source_kernel = source_weight.cpu().numpy().reshape(
            input_channel,
            self.kernel_size,
            self.kernel_size,
            w,
        )

        coverage = np.zeros(
            (input_channel, self.kernel_size, self.kernel_size, 1),
            dtype=source_kernel.dtype,
        )
        patch_slices = []
        for row in range(patch_h):
            top = row * local_stride
            bottom = top + local_kernel_size
            for col in range(patch_w):
                left = col * local_stride
                right = left + local_kernel_size
                patch_slices.append((top, bottom, left, right))
                coverage[:, top:bottom, left:right, :] += 1.0
        coverage = np.maximum(coverage, 1.0)

        patch_columns = []
        for top, bottom, left, right in patch_slices:
            weighted_patch = (
                source_kernel[:, top:bottom, left:right, :]
                / coverage[:, top:bottom, left:right, :]
            )
            patch_columns.append(
                weighted_patch.reshape(
                    input_channel * local_kernel_size * local_kernel_size,
                    w,
                )
            )
        patch_matrix = np.concatenate(patch_columns, axis=1)

        new_cortex_kernel, right_matrix, amplifier = self.PCA(
            patch_matrix,
            **pca_settings
        )
        new_cortex_kernel *= split_ratio

        output_channel = int(new_cortex_kernel.shape[1])
        if output_channel <= 0:
            raise ValueError('patch_pca_yield produced no PCA components')

        right_by_patch = right_matrix.reshape(
            output_channel,
            patch_num,
            w,
        ).transpose(1, 0, 2)

        reconstructed = np.zeros_like(source_kernel)
        for patch_index, (top, bottom, left, right) in enumerate(patch_slices):
            patch_reconstruction = (
                np.einsum(
                    'ic,co->io',
                    new_cortex_kernel,
                    right_by_patch[patch_index],
                )
                * amplifier
            ).reshape(input_channel, local_kernel_size, local_kernel_size, w)
            reconstructed[:, top:bottom, left:right, :] += patch_reconstruction

        if root_direct_mode == 'source_passthrough':
            residual_kernel = source_kernel.reshape(h, w).copy()
        else:
            residual_kernel = (source_kernel - reconstructed).reshape(h, w)
        pooled_right_by_patch = self._pool_patch_root_projection(
            right_by_patch,
            patch_h,
            patch_w,
            pooling,
        )
        new_root_kernel = np.concatenate(
            [
                residual_kernel,
                pooled_right_by_patch.reshape(
                    pooled_right_by_patch.shape[0] * output_channel,
                    w,
                ) * root_projection_scale,
            ],
            axis=0
        )

        self.kernel.weight = torch.as_tensor(
            new_root_kernel,
            dtype=source_dtype,
            device=source_device
        )
        self.kernel.has_subcortexs = True
        self.kernel.output_len = len(self.hidden_nv)
        if should_regularize:
            self.kernel.kernel_regulation()
        self._apply_root_learning_mask(
            root_learning_mask,
            direct_input_len=h
        )

        subcortex_kernel = torch.as_tensor(
            new_cortex_kernel,
            dtype=source_dtype,
            device=source_device
        )
        self.subcortex_counter += 1
        new_cortex = cortex_constructor(
            cortex_id=f'{self.cortex_id}-{self.subcortex_counter}',
            kernel_size=local_kernel_size,
            stride=local_stride,
            input_channel=input_channel,
            output_channel=output_channel,
            kernel=subcortex_kernel,
            init_amplifier=float(amplifier),
            pooling_settings=pooling_settings,
            competition_group_N=1,
        )
        new_cortex = move_to_device(new_cortex, source_device)
        self.subcortexs.append(new_cortex)
        self.kernel.has_subcortexs = len(self.subcortexs) > 0
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
        return new_cortex

    def _prepare_patch_yield_tensor(
        self,
        local_kernel_size,
        local_stride,
        transform_name,
        cover_edges=False,
    ):
        if self.subcortexs:
            raise ValueError(
                f'{transform_name} currently supports only a target cortex '
                'without existing subcortexs'
            )

        h, w = self.shape
        input_channel = h // (self.kernel_size ** 2)
        if input_channel * (self.kernel_size ** 2) != h:
            raise ValueError(
                f'{transform_name} cannot infer input_channel from target '
                'cortex shape'
            )

        local_kernel_size = int(local_kernel_size)
        local_stride = int(local_stride)
        if local_kernel_size <= 0:
            raise ValueError(f'{transform_name} local_kernel_size must be positive')
        if local_stride <= 0:
            raise ValueError(f'{transform_name} local_stride must be positive')
        if local_kernel_size > self.kernel_size:
            raise ValueError(
                f'{transform_name} local_kernel_size cannot exceed target '
                'kernel_size'
            )

        max_start = self.kernel_size - local_kernel_size

        def patch_starts():
            starts = list(range(0, max_start + 1, local_stride))
            if cover_edges and starts[-1] != max_start:
                starts.append(max_start)
            return starts

        row_starts = patch_starts()
        col_starts = patch_starts()
        patch_h = len(row_starts)
        patch_w = len(col_starts)
        patch_num = patch_h * patch_w
        if patch_num <= 0:
            raise ValueError(f'{transform_name} produced no local patch positions')

        source_weight = self.kernel.weight[:h, :w].detach()
        source_device = source_weight.device
        source_dtype = source_weight.dtype
        source_kernel = source_weight.cpu().numpy().reshape(
            input_channel,
            self.kernel_size,
            self.kernel_size,
            w,
        )

        coverage = np.zeros(
            (input_channel, self.kernel_size, self.kernel_size, 1),
            dtype=source_kernel.dtype,
        )
        patch_slices = []
        for top in row_starts:
            bottom = top + local_kernel_size
            for left in col_starts:
                right = left + local_kernel_size
                patch_slices.append((top, bottom, left, right))
                coverage[:, top:bottom, left:right, :] += 1.0
        coverage = np.maximum(coverage, 1.0)

        patch_columns = []
        for top, bottom, left, right in patch_slices:
            weighted_patch = (
                source_kernel[:, top:bottom, left:right, :]
                / coverage[:, top:bottom, left:right, :]
            )
            patch_columns.append(
                weighted_patch.reshape(
                    input_channel * local_kernel_size * local_kernel_size,
                    w,
                )
            )

        return {
            'h': h,
            'w': w,
            'input_channel': input_channel,
            'source_device': source_device,
            'source_dtype': source_dtype,
            'source_kernel': source_kernel,
            'coverage': coverage,
            'patch_h': patch_h,
            'patch_w': patch_w,
            'patch_num': patch_num,
            'patch_slices': patch_slices,
            'patch_tensor': np.stack(patch_columns, axis=0),
        }

    def _finalize_patch_factor_yield(
        self,
        cortex_constructor,
        local_kernel_size,
        local_stride,
        patch_state,
        new_cortex_kernel,
        branch_projection_by_patch,
        transform_name,
        regularize_afterward=None,
        pooling_settings=None,
        pooling_projection_mode='legacy_mean',
        root_direct_mode=None,
        root_projection_scale=1.0,
        root_learning_mask=None,
        new_cortex_init_amplifier=1.0,
        cover_edges=False,
    ):
        h = patch_state['h']
        w = patch_state['w']
        input_channel = patch_state['input_channel']
        source_device = patch_state['source_device']
        source_dtype = patch_state['source_dtype']
        source_kernel = patch_state['source_kernel']
        patch_h = patch_state['patch_h']
        patch_w = patch_state['patch_w']
        patch_slices = patch_state['patch_slices']

        output_channel = int(new_cortex_kernel.shape[1])
        if output_channel <= 0:
            raise ValueError(f'{transform_name} produced no components')
        expected_branch_shape = (
            patch_state['patch_num'],
            output_channel,
            w,
        )
        if branch_projection_by_patch.shape != expected_branch_shape:
            raise ValueError(
                f'{transform_name} branch projection shape must be '
                f'{expected_branch_shape}, got {branch_projection_by_patch.shape}'
            )

        should_regularize = (
            self.regularize_afterward
            if regularize_afterward is None
            else bool(regularize_afterward)
        )
        root_direct_mode = self._get_root_direct_mode(
            root_direct_mode,
            should_regularize
        )
        root_projection_scale = self._get_root_projection_scale(
            root_projection_scale
        )
        if pooling_settings is None:
            pooling_settings = {'mode': 'mean', 'kernel_size': 1}
        pooling = CortexPooling(pooling_settings)

        if root_direct_mode == 'source_passthrough':
            residual_kernel = source_kernel.reshape(h, w).copy()
        elif root_direct_mode == 'factorized_only':
            residual_kernel = np.zeros((h, w), dtype=source_kernel.dtype)
        else:
            reconstructed = np.zeros_like(source_kernel)
            for patch_index, (top, bottom, left, right) in enumerate(patch_slices):
                patch_reconstruction = np.einsum(
                    'ic,co->io',
                    new_cortex_kernel,
                    branch_projection_by_patch[patch_index],
                ).reshape(input_channel, local_kernel_size, local_kernel_size, w)
                reconstructed[:, top:bottom, left:right, :] += patch_reconstruction
            residual_kernel = (source_kernel - reconstructed).reshape(h, w)

        pooled_branch_projection = self._pool_patch_root_projection(
            branch_projection_by_patch,
            patch_h,
            patch_w,
            pooling,
            projection_mode=pooling_projection_mode,
        )
        new_root_kernel = np.concatenate(
            [
                residual_kernel,
                pooled_branch_projection.reshape(
                    pooled_branch_projection.shape[0] * output_channel,
                    w,
                ) * root_projection_scale,
            ],
            axis=0
        )

        self.kernel.weight = torch.as_tensor(
            new_root_kernel,
            dtype=source_dtype,
            device=source_device
        )
        self.kernel.has_subcortexs = True
        self.kernel.output_len = len(self.hidden_nv)
        if should_regularize:
            self.kernel.kernel_regulation()
        self._apply_root_learning_mask(
            root_learning_mask,
            direct_input_len=h
        )

        subcortex_kernel = torch.as_tensor(
            new_cortex_kernel,
            dtype=source_dtype,
            device=source_device
        )
        self.subcortex_counter += 1
        new_cortex = cortex_constructor(
            cortex_id=f'{self.cortex_id}-{self.subcortex_counter}',
            kernel_size=local_kernel_size,
            stride=local_stride,
            input_channel=input_channel,
            output_channel=output_channel,
            kernel=subcortex_kernel,
            init_amplifier=float(new_cortex_init_amplifier),
            pooling_settings=pooling_settings,
            competition_group_N=1,
            cover_edges=bool(cover_edges),
        )
        new_cortex = move_to_device(new_cortex, source_device)
        self.subcortexs.append(new_cortex)
        self.kernel.has_subcortexs = len(self.subcortexs) > 0
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
        return new_cortex

    def _patchify_kernel_with_state(
        self,
        kernel,
        patch_state,
        local_kernel_size,
    ):
        input_channel = patch_state['input_channel']
        receiver_count = patch_state['w']
        coverage = patch_state['coverage']
        patch_columns = []
        for top, bottom, left, right in patch_state['patch_slices']:
            weighted_patch = (
                kernel[:, top:bottom, left:right, :]
                / coverage[:, top:bottom, left:right, :]
            )
            patch_columns.append(
                weighted_patch.reshape(
                    input_channel * local_kernel_size * local_kernel_size,
                    receiver_count,
                )
            )
        return np.stack(patch_columns, axis=0)

    def _patch_center_base_kernel(
        self,
        source_kernel,
        center_mode,
        transform_name,
    ):
        center_mode = 'none' if center_mode is None else str(center_mode)
        if center_mode == 'none':
            return np.zeros_like(source_kernel)
        if center_mode == 'global':
            return np.full_like(source_kernel, source_kernel.mean())
        if center_mode == 'receiver':
            return np.broadcast_to(
                source_kernel.mean(axis=(0, 1, 2), keepdims=True),
                source_kernel.shape,
            ).copy()
        if center_mode == 'pixel':
            return np.broadcast_to(
                source_kernel.mean(axis=3, keepdims=True),
                source_kernel.shape,
            ).copy()
        if center_mode == 'two_way':
            global_mean = source_kernel.mean(keepdims=True)
            pixel_mean = source_kernel.mean(axis=3, keepdims=True)
            receiver_mean = source_kernel.mean(axis=(0, 1, 2), keepdims=True)
            return (pixel_mean + receiver_mean - global_mean).copy()
        raise ValueError(
            f'{transform_name} unsupported center_mode: {center_mode!r}'
        )

    def _unfold_patch_tensor(self, patch_tensor, mode):
        return np.moveaxis(patch_tensor, mode, 0).reshape(
            patch_tensor.shape[mode],
            -1,
        )

    def _patch_mode_dot(self, patch_tensor, matrix, mode):
        result = np.tensordot(matrix, patch_tensor, axes=(1, mode))
        return np.moveaxis(result, 0, mode)

    def _leading_left_singular_vectors(
        self,
        matrix,
        component_count,
        random_state,
        transform_name,
        full_rank_basis_mode='identity',
    ):
        component_count = int(component_count)
        max_component_count = min(matrix.shape)
        if component_count <= 0:
            raise ValueError(f'{transform_name} rank values must be positive')
        if component_count > max_component_count:
            raise ValueError(
                f'{transform_name} requested rank {component_count} exceeds '
                f'matrix rank limit {max_component_count}'
            )
        if component_count == matrix.shape[0]:
            full_rank_basis_mode = (
                'identity'
                if full_rank_basis_mode is None
                else str(full_rank_basis_mode).lower()
            )
            if full_rank_basis_mode in {'identity', 'none'}:
                return np.eye(matrix.shape[0], dtype=matrix.dtype)
            if full_rank_basis_mode in {'covariance_eigh', 'eigh'}:
                matrix_64 = matrix.astype(np.float64, copy=False)
                covariance = matrix_64 @ matrix_64.T
                _, eigenvectors = np.linalg.eigh(covariance)
                eigenvectors = eigenvectors[:, ::-1]
                for component_index in range(eigenvectors.shape[1]):
                    component = eigenvectors[:, component_index]
                    pivot = int(np.argmax(np.abs(component)))
                    if component[pivot] < 0.0:
                        eigenvectors[:, component_index] *= -1.0
                return eigenvectors.astype(matrix.dtype)
            raise ValueError(
                f'{transform_name} unsupported full-rank basis mode: '
                f'{full_rank_basis_mode!r}'
            )

        svd = TruncatedSVD(
            n_components=component_count,
            random_state=int(random_state),
        )
        transformed = svd.fit_transform(matrix)
        singular_values = np.maximum(svd.singular_values_, 1.0e-12)
        return transformed / singular_values[None, :]

    def _batched_rank1_svd(
        self,
        matrices,
        iteration_count,
        random_state,
        transform_name,
        epsilon=1.0e-12,
    ):
        if matrices.ndim != 3:
            raise ValueError(
                f'{transform_name} batched rank-1 input must be 3D'
            )
        iteration_count = int(iteration_count)
        if iteration_count <= 0:
            raise ValueError(
                f'{transform_name} rank1_iteration_count must be positive'
            )
        epsilon = float(epsilon)
        if epsilon <= 0.0:
            raise ValueError(
                f'{transform_name} rank-1 epsilon must be positive'
            )

        batch_count, _, column_count = matrices.shape
        generator = np.random.RandomState(int(random_state))
        right = generator.standard_normal(
            (batch_count, column_count)
        ).astype(matrices.dtype)
        right /= np.maximum(
            np.linalg.norm(right, axis=1, keepdims=True),
            epsilon,
        )
        transposed = np.swapaxes(matrices, 1, 2)
        for _ in range(iteration_count):
            left = np.matmul(matrices, right[:, :, None])[:, :, 0]
            left /= np.maximum(
                np.linalg.norm(left, axis=1, keepdims=True),
                epsilon,
            )
            right = np.matmul(transposed, left[:, :, None])[:, :, 0]
            right /= np.maximum(
                np.linalg.norm(right, axis=1, keepdims=True),
                epsilon,
            )

        left_scaled = np.matmul(matrices, right[:, :, None])[:, :, 0]
        pivots = np.argmax(np.abs(right), axis=1)
        signs = np.sign(right[np.arange(batch_count), pivots])
        signs[signs == 0.0] = 1.0
        left_scaled *= signs[:, None]
        right *= signs[:, None]

        matrix_energy = np.sum(
            np.square(matrices),
            axis=(1, 2),
            dtype=np.float64,
        )
        rank1_energy = np.sum(
            np.square(left_scaled),
            axis=1,
            dtype=np.float64,
        )
        captured_energy_fraction = rank1_energy / np.maximum(
            matrix_energy,
            epsilon,
        )
        return left_scaled, right, captured_energy_fraction

    def _patch_tucker_decomposition(
        self,
        patch_tensor,
        rank_patch,
        rank_local,
        rank_receiver,
        iteration_count,
        random_state,
        transform_name,
        full_rank_local_basis_mode='identity',
    ):
        ranks = (
            int(rank_patch),
            int(rank_local),
            int(rank_receiver),
        )
        for mode, rank in enumerate(ranks):
            if rank > patch_tensor.shape[mode]:
                raise ValueError(
                    f'{transform_name} rank for mode {mode} exceeds tensor '
                    f'dimension {patch_tensor.shape[mode]}'
                )

        factors = []
        for mode, rank in enumerate(ranks):
            unfolded = self._unfold_patch_tensor(patch_tensor, mode)
            factors.append(
                self._leading_left_singular_vectors(
                    unfolded,
                    rank,
                    int(random_state) + mode,
                    transform_name,
                    full_rank_basis_mode=(
                        full_rank_local_basis_mode
                        if mode == 1
                        else 'identity'
                    ),
                )
            )

        for iteration_index in range(int(iteration_count)):
            for mode, rank in enumerate(ranks):
                projected = patch_tensor
                for other_mode in range(3):
                    if other_mode == mode:
                        continue
                    projected = self._patch_mode_dot(
                        projected,
                        factors[other_mode].T,
                        other_mode,
                    )
                unfolded = self._unfold_patch_tensor(projected, mode)
                factors[mode] = self._leading_left_singular_vectors(
                    unfolded,
                    rank,
                    int(random_state) + 10 + iteration_index * 3 + mode,
                    transform_name,
                    full_rank_basis_mode=(
                        full_rank_local_basis_mode
                        if mode == 1
                        else 'identity'
                    ),
                )

        core = patch_tensor
        for mode, factor in enumerate(factors):
            core = self._patch_mode_dot(core, factor.T, mode)
        return core, factors

    def _patch_tucker_to_factor_branch(self, core, factors):
        patch_factor, local_factor, receiver_factor = factors
        branch_projection_by_patch = np.einsum(
            'pa,abc,oc->pbo',
            patch_factor,
            core,
            receiver_factor,
            optimize=True,
        )
        return local_factor, branch_projection_by_patch

    def _pooling_window_patch_groups(
        self,
        patch_state,
        pooling_settings,
        transform_name,
    ):
        pooling = CortexPooling(pooling_settings)
        if pooling.settings is None:
            raise ValueError(
                f'{transform_name} pooling-aware factorization requires '
                'pooling_settings'
            )

        patch_h = int(patch_state['patch_h'])
        patch_w = int(patch_state['patch_w'])
        kernel, stride = pooling.get_kernel_stride(patch_h, patch_w)
        pooled_h, pooled_w, _ = pooling.get_pooled_grid(patch_h, patch_w)
        patch_coverage = np.zeros(patch_h * patch_w, dtype=np.int64)
        groups = []
        for row in range(pooled_h):
            top = row * stride[0]
            bottom = top + kernel[0]
            for col in range(pooled_w):
                left = col * stride[1]
                right = left + kernel[1]
                indices = []
                for patch_row in range(top, bottom):
                    for patch_col in range(left, right):
                        indices.append(patch_row * patch_w + patch_col)
                indices = np.asarray(indices, dtype=np.int64)
                patch_coverage[indices] += 1
                groups.append(indices)

        if np.any(patch_coverage > 1):
            raise ValueError(
                f'{transform_name} pooling-aware factorization currently '
                'supports only non-overlapping pooling windows'
            )
        return groups

    def _pooling_window_mean_patch_tensor(
        self,
        patch_tensor,
        patch_state,
        pooling_settings,
        transform_name,
    ):
        groups = self._pooling_window_patch_groups(
            patch_state,
            pooling_settings,
            transform_name,
        )
        pooled_tensor = np.stack(
            [
                patch_tensor[group_indices].mean(axis=0)
                for group_indices in groups
            ],
            axis=0,
        )
        return pooled_tensor, groups

    @staticmethod
    def _expand_pooling_window_branch_projection(
        pooled_branch_projection,
        patch_num,
        groups,
    ):
        expanded = np.zeros(
            (
                int(patch_num),
                pooled_branch_projection.shape[1],
                pooled_branch_projection.shape[2],
            ),
            dtype=pooled_branch_projection.dtype,
        )
        for pooled_index, group_indices in enumerate(groups):
            expanded[group_indices] = pooled_branch_projection[pooled_index]
        return expanded

    def _pooling_aware_als_factor_branch(
        self,
        pooled_tensor,
        rank_local,
        iteration_count,
        random_state,
        transform_name,
        ridge=1.0e-6,
    ):
        pooled_num, patch_dim, receiver_count = pooled_tensor.shape
        component_count = int(rank_local)
        if component_count <= 0:
            raise ValueError(f'{transform_name} rank_local must be positive')
        if component_count > patch_dim:
            raise ValueError(
                f'{transform_name} rank_local cannot exceed local patch '
                f'dimension {patch_dim}'
            )
        iteration_count = int(iteration_count)
        if iteration_count < 0:
            raise ValueError(
                f'{transform_name} pooling-aware ALS iterations must be >= 0'
            )
        ridge = float(ridge)
        if ridge < 0.0:
            raise ValueError(
                f'{transform_name} pooling-aware ALS ridge must be >= 0'
            )

        matrix = pooled_tensor.transpose(1, 0, 2).reshape(
            patch_dim,
            pooled_num * receiver_count,
        )
        local_factor = self._leading_left_singular_vectors(
            matrix,
            component_count,
            int(random_state),
            transform_name,
            full_rank_basis_mode='identity',
        ).astype(np.float64, copy=False)
        pooled_tensor64 = pooled_tensor.astype(np.float64, copy=False)
        identity = np.eye(component_count, dtype=np.float64)

        def regularized_solve(matrix, rhs):
            scale = float(np.trace(matrix)) / max(component_count, 1)
            scale = max(scale, 1.0)
            return np.linalg.solve(matrix + ridge * scale * identity, rhs)

        def solve_branch_projection(kernel_matrix):
            gram = kernel_matrix.T @ kernel_matrix
            rhs = np.einsum(
                'lc,plo->pco',
                kernel_matrix,
                pooled_tensor64,
                optimize=True,
            )
            rhs = rhs.transpose(1, 0, 2).reshape(
                component_count,
                pooled_num * receiver_count,
            )
            solved = regularized_solve(gram, rhs)
            return solved.reshape(
                component_count,
                pooled_num,
                receiver_count,
            ).transpose(1, 0, 2)

        branch_projection = solve_branch_projection(local_factor)
        for _ in range(iteration_count):
            branch_gram = np.einsum(
                'pco,pdo->cd',
                branch_projection,
                branch_projection,
                optimize=True,
            )
            branch_rhs = np.einsum(
                'plo,pco->lc',
                pooled_tensor64,
                branch_projection,
                optimize=True,
            )
            local_factor = regularized_solve(
                branch_gram.T,
                branch_rhs.T,
            ).T
            norms = np.maximum(
                np.linalg.norm(local_factor, axis=0),
                1.0e-12,
            )
            local_factor = local_factor / norms[None, :]
            branch_projection = branch_projection * norms[None, :, None]
            branch_projection = solve_branch_projection(local_factor)

        return (
            local_factor.astype(pooled_tensor.dtype, copy=False),
            branch_projection.astype(pooled_tensor.dtype, copy=False),
        )

    def _pooling_phase_demultiplex_factor_branch(
        self,
        patch_tensor,
        patch_state,
        pooling_settings,
        component_count,
        random_state,
        transform_name,
    ):
        groups = self._pooling_window_patch_groups(
            patch_state,
            pooling_settings,
            transform_name,
        )
        pooling = CortexPooling(pooling_settings)
        kernel, stride = pooling.get_kernel_stride(
            int(patch_state['patch_h']),
            int(patch_state['patch_w']),
        )
        if kernel != stride:
            raise ValueError(
                f'{transform_name} phase-demultiplex factorization requires '
                'non-overlapping pooling windows'
            )
        phase_count = int(kernel[0] * kernel[1])
        component_count = int(component_count)
        if component_count < phase_count:
            raise ValueError(
                f'{transform_name} phase-demultiplex rank_local must be at '
                f'least {phase_count}'
            )

        patch_dim = int(patch_tensor.shape[1])
        receiver_count = int(patch_tensor.shape[2])
        channel_phase = (
            np.arange(component_count, dtype=np.int64) * phase_count
        ) // component_count
        new_cortex_kernel = np.zeros(
            (patch_dim, component_count),
            dtype=patch_tensor.dtype,
        )
        branch_projection_by_patch = np.zeros(
            (
                int(patch_state['patch_num']),
                component_count,
                receiver_count,
            ),
            dtype=patch_tensor.dtype,
        )

        for phase_index in range(phase_count):
            phase_channels = np.flatnonzero(channel_phase == phase_index)
            phase_tensor = np.stack(
                [patch_tensor[group[phase_index]] for group in groups],
                axis=0,
            )
            phase_matrix = phase_tensor.transpose(1, 0, 2).reshape(
                patch_dim,
                len(groups) * receiver_count,
            )
            phase_basis = self._leading_left_singular_vectors(
                phase_matrix,
                len(phase_channels),
                int(random_state) + phase_index,
                transform_name,
                full_rank_basis_mode='identity',
            ).astype(patch_tensor.dtype, copy=False)
            phase_projection = np.einsum(
                'lc,plo->pco',
                phase_basis,
                phase_tensor,
                optimize=True,
            )
            new_cortex_kernel[:, phase_channels] = phase_basis
            for pooled_index, group in enumerate(groups):
                branch_projection_by_patch[
                    group[phase_index], phase_channels, :
                ] = phase_projection[pooled_index] * phase_count

        return new_cortex_kernel, branch_projection_by_patch

    def _pooling_aware_factor_branch(
        self,
        patch_tensor,
        patch_state,
        pooling_settings,
        rank_patch,
        rank_local,
        rank_receiver,
        iteration_count,
        random_state,
        transform_name,
        mode,
        full_rank_local_basis_mode='identity',
        als_iteration_count=None,
        als_ridge=1.0e-6,
    ):
        mode = str(mode).lower()
        if mode in {'phase_demux_svd', 'phase_demultiplex_svd'}:
            return self._pooling_phase_demultiplex_factor_branch(
                patch_tensor,
                patch_state,
                pooling_settings,
                rank_local,
                random_state,
                transform_name,
            )
        pooled_tensor, groups = self._pooling_window_mean_patch_tensor(
            patch_tensor,
            patch_state,
            pooling_settings,
            transform_name,
        )

        if mode in {'window_mean_svd', 'window_mean_tucker'}:
            core, factors = self._patch_tucker_decomposition(
                pooled_tensor,
                rank_patch,
                rank_local,
                rank_receiver,
                iteration_count,
                random_state,
                transform_name,
                full_rank_local_basis_mode=full_rank_local_basis_mode,
            )
            new_cortex_kernel, pooled_branch_projection = (
                self._patch_tucker_to_factor_branch(core, factors)
            )
        elif mode in {'shared_window_als', 'window_mean_als'}:
            if int(rank_patch) != int(pooled_tensor.shape[0]):
                raise ValueError(
                    f'{transform_name} pooling-aware ALS requires rank_patch '
                    f'to equal pooled patch count {pooled_tensor.shape[0]}'
                )
            if int(rank_receiver) != int(pooled_tensor.shape[2]):
                raise ValueError(
                    f'{transform_name} pooling-aware ALS requires '
                    f'rank_receiver to equal receiver count '
                    f'{pooled_tensor.shape[2]}'
                )
            if als_iteration_count is None:
                als_iteration_count = iteration_count
            new_cortex_kernel, pooled_branch_projection = (
                self._pooling_aware_als_factor_branch(
                    pooled_tensor,
                    rank_local,
                    als_iteration_count,
                    random_state,
                    transform_name,
                    ridge=als_ridge,
                )
            )
        else:
            raise ValueError(
                f'{transform_name} unsupported pooling-aware factorization '
                f'mode: {mode!r}'
            )

        branch_projection_by_patch = (
            self._expand_pooling_window_branch_projection(
                pooled_branch_projection,
                patch_state['patch_num'],
                groups,
            )
        )
        return new_cortex_kernel, branch_projection_by_patch

    def _pooling_aware_matrix_svd_factor_branch(
        self,
        patch_tensor,
        patch_state,
        pooling_settings,
        component_count,
        random_state,
        transform_name,
    ):
        pooled_tensor, groups = self._pooling_window_mean_patch_tensor(
            patch_tensor,
            patch_state,
            pooling_settings,
            transform_name,
        )
        new_cortex_kernel, pooled_branch_projection = (
            self._patch_matrix_svd_factor_branch(
                pooled_tensor,
                component_count,
                random_state,
                transform_name,
            )
        )
        branch_projection_by_patch = (
            self._expand_pooling_window_branch_projection(
                pooled_branch_projection,
                patch_state['patch_num'],
                groups,
            )
        )
        return new_cortex_kernel, branch_projection_by_patch

    def _patch_grouped_tucker_to_factor_branch(
        self,
        patch_tensor,
        rank_patch,
        rank_local,
        rank_receiver,
        group_count,
        iteration_count,
        random_state,
        transform_name,
        full_rank_local_basis_mode='identity',
    ):
        receiver_count = int(patch_tensor.shape[2])
        group_slices = self._receiver_group_slices(
            receiver_count,
            transform_name,
            group_count=group_count,
        )
        group_results = []
        local_kernels = []
        total_component_count = 0
        for group_index, (group_start, group_end) in enumerate(group_slices):
            group_tensor = patch_tensor[:, :, group_start:group_end]
            core, factors = self._patch_tucker_decomposition(
                group_tensor,
                rank_patch,
                rank_local,
                rank_receiver,
                iteration_count,
                int(random_state) + (group_index + 1) * 100,
                transform_name,
                full_rank_local_basis_mode=full_rank_local_basis_mode,
            )
            group_kernel, group_branch = (
                self._patch_tucker_to_factor_branch(core, factors)
            )
            group_component_count = int(group_kernel.shape[1])
            group_results.append(
                (
                    group_start,
                    group_end,
                    total_component_count,
                    group_component_count,
                    group_branch,
                )
            )
            local_kernels.append(group_kernel)
            total_component_count += group_component_count

        new_cortex_kernel = np.concatenate(local_kernels, axis=1)
        branch_projection_by_patch = np.zeros(
            (
                patch_tensor.shape[0],
                total_component_count,
                receiver_count,
            ),
            dtype=patch_tensor.dtype,
        )
        for (
            group_start,
            group_end,
            component_start,
            group_component_count,
            group_branch,
        ) in group_results:
            component_end = component_start + group_component_count
            branch_projection_by_patch[
                :,
                component_start:component_end,
                group_start:group_end,
            ] = group_branch
        return new_cortex_kernel, branch_projection_by_patch

    def _apply_local_polarity_split(
        self,
        new_cortex_kernel,
        branch_projection_by_patch,
        polarity_split,
        component_count,
        transform_name,
        duplicate_kernel_scales=None,
        duplicate_branch_weights=None,
    ):
        if polarity_split is None:
            return new_cortex_kernel, branch_projection_by_patch
        duplicate_mode = False
        if isinstance(polarity_split, str):
            polarity_split = polarity_split.lower()
            if polarity_split in {'none', 'false', 'off'}:
                return new_cortex_kernel, branch_projection_by_patch
            duplicate_mode = polarity_split in {
                'duplicate',
                'duplicate_nonnegative',
            }
            if (
                polarity_split not in {'posneg', 'true', 'on'}
                and not duplicate_mode
            ):
                raise ValueError(
                    f'{transform_name} unsupported polarity_split: '
                    f'{polarity_split!r}'
                )
        elif not bool(polarity_split):
            return new_cortex_kernel, branch_projection_by_patch

        output_channel = int(new_cortex_kernel.shape[1])
        if component_count is None:
            split_count = output_channel
        else:
            split_count = int(component_count)
        if split_count <= 0 or split_count > output_channel:
            raise ValueError(
                f'{transform_name} polarity_split_component_count must be in '
                f'[1, {output_channel}], got {split_count}'
            )

        split_kernel = new_cortex_kernel[:, :split_count]
        split_branch = branch_projection_by_patch[:, :split_count, :]
        kept_kernel = new_cortex_kernel[:, split_count:]
        kept_branch = branch_projection_by_patch[:, split_count:, :]
        if duplicate_mode:
            if np.min(split_kernel) < -1.0e-12:
                raise ValueError(
                    f'{transform_name} duplicate_nonnegative polarity split '
                    'requires a nonnegative split kernel'
                )
            kernel_scales = np.asarray(
                duplicate_kernel_scales,
                dtype=np.float64,
            )
            branch_weights = np.asarray(
                duplicate_branch_weights,
                dtype=np.float64,
            )
            if (
                kernel_scales.ndim != 1
                or branch_weights.ndim != 1
                or kernel_scales.size < 2
                or kernel_scales.size != branch_weights.size
            ):
                raise ValueError(
                    f'{transform_name} duplicate polarity split requires '
                    'matching one-dimensional kernel scales and branch '
                    'weights with at least two entries'
                )
            if (
                not np.all(np.isfinite(kernel_scales))
                or np.any(kernel_scales <= 0.0)
            ):
                raise ValueError(
                    f'{transform_name} duplicate kernel scales must be '
                    'finite and positive'
                )
            if (
                not np.all(np.isfinite(branch_weights))
                or np.any(branch_weights < 0.0)
                or not np.isclose(
                    np.sum(branch_weights),
                    1.0,
                    rtol=0.0,
                    atol=1.0e-9,
                )
            ):
                raise ValueError(
                    f'{transform_name} duplicate branch weights must be '
                    'finite, nonnegative, and sum to one'
                )
            kernel_blocks = [
                split_kernel * np.asarray(scale, dtype=split_kernel.dtype)
                for scale in kernel_scales
            ]
            branch_blocks = [
                split_branch
                * np.asarray(weight / scale, dtype=split_branch.dtype)
                for scale, weight in zip(kernel_scales, branch_weights)
            ]
            return (
                np.concatenate(kernel_blocks + [kept_kernel], axis=1),
                np.concatenate(branch_blocks + [kept_branch], axis=1),
            )

        split_positive = np.maximum(split_kernel, 0.0)
        split_negative = np.maximum(-split_kernel, 0.0)
        new_cortex_kernel = np.concatenate(
            [split_positive, split_negative, kept_kernel],
            axis=1,
        )
        branch_projection_by_patch = np.concatenate(
            [split_branch, -split_branch, kept_branch],
            axis=1,
        )
        return new_cortex_kernel, branch_projection_by_patch

    def _receiver_group_slices(
        self,
        receiver_count,
        transform_name,
        group_count=None,
    ):
        group_count = (
            int(self.output_channel)
            if group_count is None
            else int(group_count)
        )
        if group_count <= 0 or receiver_count % group_count != 0:
            raise ValueError(
                f'{transform_name} cannot split {receiver_count} receivers '
                f'into {group_count} equal groups'
            )
        group_width = receiver_count // group_count
        return [
            (index * group_width, (index + 1) * group_width)
            for index in range(group_count)
        ]

    def _precondition_patch_tensor_receiver_groups(
        self,
        patch_tensor,
        group_weights,
        transform_name,
    ):
        if group_weights is None:
            return patch_tensor, None

        weights = np.asarray(group_weights, dtype=np.float64)
        if weights.ndim != 1 or weights.size == 0:
            raise ValueError(
                f'{transform_name} factorization_receiver_objective_group_weights '
                'must be a non-empty one-dimensional sequence'
            )
        if not np.all(np.isfinite(weights)) or np.any(weights <= 0.0):
            raise ValueError(
                f'{transform_name} factorization_receiver_objective_group_weights '
                'must contain only finite positive values'
            )

        weighted = patch_tensor.copy()
        group_slices = self._receiver_group_slices(
            int(weighted.shape[2]),
            transform_name,
            group_count=int(weights.size),
        )
        for weight, (start, end) in zip(weights, group_slices):
            weighted[:, :, start:end] *= float(weight)
        return weighted, weights

    def _rms_norm(self, values, axis):
        return np.sqrt(np.mean(np.square(values), axis=axis))

    def _append_post_load_transform_audit(self, audit):
        audits = getattr(self, 'post_load_transform_audits', None)
        if audits is None:
            audits = []
            self.post_load_transform_audits = audits
        audits.append(audit)

    def _append_factorization_reconstruction_audit(
        self,
        target,
        new_cortex_kernel,
        branch_projection_by_patch,
        transform_name,
        stage,
    ):
        reconstructed = np.einsum(
            'lc,pcr->plr',
            new_cortex_kernel,
            branch_projection_by_patch,
            optimize=True,
        )
        residual = reconstructed - target
        target_energy = float(np.sum(np.square(target), dtype=np.float64))
        reconstructed_energy = float(np.sum(
            np.square(reconstructed),
            dtype=np.float64,
        ))
        residual_energy = float(np.sum(
            np.square(residual),
            dtype=np.float64,
        ))
        cross = float(np.sum(target * reconstructed, dtype=np.float64))
        epsilon = 1.0e-24
        relative_error = np.sqrt(residual_energy / max(target_energy, epsilon))
        cosine = cross / np.sqrt(max(
            target_energy * reconstructed_energy,
            epsilon,
        ))
        self._append_post_load_transform_audit({
            'prefix': (
                f'factorization/reconstruction/{self.cortex_id}/{stage}'
            ),
            'text': {
                'transform': transform_name,
                'stage': stage,
            },
            'scalars': {
                'relative_error_l2': float(relative_error),
                'cosine': float(cosine),
                'target_rms': float(np.sqrt(target_energy / target.size)),
                'reconstructed_rms': float(np.sqrt(
                    reconstructed_energy / reconstructed.size
                )),
                'residual_rms': float(np.sqrt(
                    residual_energy / residual.size
                )),
            },
        })

    @staticmethod
    def _quantile_summary(values):
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        values = values[np.isfinite(values)]
        if values.size == 0:
            return {
                'count': 0.0,
                'min': 0.0,
                'q10': 0.0,
                'median': 0.0,
                'q90': 0.0,
                'max': 0.0,
            }
        return {
            'count': float(values.size),
            'min': float(np.min(values)),
            'q10': float(np.quantile(values, 0.10)),
            'median': float(np.median(values)),
            'q90': float(np.quantile(values, 0.90)),
            'max': float(np.max(values)),
        }

    @staticmethod
    def _prefixed_summary(prefix, values):
        return {
            f'{prefix}_{name}': value
            for name, value in PcaCortex._quantile_summary(values).items()
        }

    def _component_norm_gauge_groups(
        self,
        total_component_count,
        component_count,
        structure,
        transform_name,
    ):
        structure = (
            'independent' if structure is None else str(structure).lower()
        )
        if structure in {'independent', 'component', 'components'}:
            count = (
                total_component_count
                if component_count is None
                else int(component_count)
            )
            if count <= 0 or count > total_component_count:
                raise ValueError(
                    f'{transform_name} '
                    'factorization_component_norm_gauge_component_count '
                    f'must be in [1, {total_component_count}], got {count}'
                )
            return [np.asarray([index], dtype=np.int64) for index in range(count)]

        if structure in {'polarity_pairs', 'posneg_pairs'}:
            pair_count = (
                total_component_count // 2
                if component_count is None
                else int(component_count)
            )
            if pair_count <= 0 or 2 * pair_count > total_component_count:
                raise ValueError(
                    f'{transform_name} polarity-pair norm gauge requires '
                    'component_count pairs with 2*component_count <= total '
                    f'components; got {pair_count} for {total_component_count}'
                )
            return [
                np.asarray([index, index + pair_count], dtype=np.int64)
                for index in range(pair_count)
            ]

        raise ValueError(
            f'{transform_name} unsupported '
            'factorization_component_norm_gauge_structure: '
            f'{structure!r}'
        )

    def _apply_component_norm_gauge(
        self,
        new_cortex_kernel,
        branch_projection_by_patch,
        mode,
        transform_name,
        component_count=None,
        structure='independent',
        min_scale=0.25,
        max_scale=4.0,
        epsilon=1.0e-12,
    ):
        mode = 'none' if mode is None else str(mode).lower()
        if mode in {'none', 'off', 'false'}:
            return new_cortex_kernel, branch_projection_by_patch

        min_scale = float(min_scale)
        max_scale = float(max_scale)
        epsilon = float(epsilon)
        if (
            not np.isfinite(min_scale)
            or not np.isfinite(max_scale)
            or min_scale <= 0.0
            or max_scale < min_scale
        ):
            raise ValueError(
                f'{transform_name} component norm gauge bounds must satisfy '
                f'0 < min <= max, got {min_scale} and {max_scale}'
            )
        if epsilon <= 0.0:
            raise ValueError(
                f'{transform_name} '
                'factorization_component_norm_gauge_epsilon must be positive'
            )

        total_component_count = int(new_cortex_kernel.shape[1])
        groups = self._component_norm_gauge_groups(
            total_component_count,
            component_count,
            structure,
            transform_name,
        )
        kernel_norms = np.asarray(
            [
                self._rms_norm(new_cortex_kernel[:, group], axis=None)
                for group in groups
            ],
            dtype=np.float64,
        )
        branch_norms = np.asarray(
            [
                self._rms_norm(branch_projection_by_patch[:, group, :], axis=None)
                for group in groups
            ],
            dtype=np.float64,
        )
        kernel_l1 = np.asarray(
            [
                np.sum(np.abs(new_cortex_kernel[:, group]))
                / float(len(group))
                for group in groups
            ],
            dtype=np.float64,
        )
        active = (kernel_norms > epsilon) & (branch_norms > epsilon)
        if not np.any(active):
            group_scales = np.ones(len(groups), dtype=np.float64)
        else:
            kernel_target = float(np.median(kernel_norms[active]))
            branch_target = float(np.median(branch_norms[active]))
            if mode in {
                'audit',
                'audit_only',
                'identity',
            }:
                raw_scales = np.ones_like(kernel_norms)
            elif mode in {
                'kernel_median',
                'kernel_rms_median',
                'equalize_kernel',
            }:
                raw_scales = kernel_target / np.maximum(kernel_norms, epsilon)
            elif mode in {
                'branch_median',
                'branch_rms_median',
                'equalize_branch',
            }:
                raw_scales = branch_norms / max(branch_target, epsilon)
            elif mode in {
                'balanced_rms',
                'kernel_branch_balanced',
                'balanced',
            }:
                kernel_ratio = kernel_norms / max(kernel_target, epsilon)
                branch_ratio = branch_norms / max(branch_target, epsilon)
                raw_scales = np.sqrt(
                    np.maximum(branch_ratio, epsilon)
                    / np.maximum(kernel_ratio, epsilon)
                )
            elif mode in {
                'kernel_l1_unit',
                'normalization_l1',
            }:
                if any(len(group) != 1 for group in groups):
                    raise ValueError(
                        f'{transform_name} {mode} requires independent '
                        'single-component gauge groups'
                    )
                # Post-normalization constrains each A-1 receiver column to
                # unit incoming L1. Apply that constraint while the Tucker
                # factors are still being constructed, before the downstream
                # receiver refit observes the resulting native spike response.
                raw_scales = np.reciprocal(
                    np.maximum(kernel_l1, epsilon)
                )
            else:
                raise ValueError(
                    f'{transform_name} unsupported '
                    'factorization_component_norm_gauge_mode: '
                    f'{mode!r}'
                )
            group_scales = np.where(active, raw_scales, 1.0)
            group_scales = np.clip(group_scales, min_scale, max_scale)

        product_before = np.einsum(
            'lc,pcr->plr',
            new_cortex_kernel,
            branch_projection_by_patch,
            optimize=True,
        )
        gauged_kernel = new_cortex_kernel.copy()
        gauged_branch = branch_projection_by_patch.copy()
        for group, scale in zip(groups, group_scales):
            scale = np.asarray(scale, dtype=gauged_kernel.dtype)
            gauged_kernel[:, group] *= scale
            gauged_branch[:, group, :] /= scale
        product_after = np.einsum(
            'lc,pcr->plr',
            gauged_kernel,
            gauged_branch,
            optimize=True,
        )
        product_l2 = float(np.linalg.norm(product_before))
        product_delta_l2 = float(np.linalg.norm(product_after - product_before))
        if product_l2 <= 0.0:
            product_relative_delta = 0.0
        else:
            product_relative_delta = product_delta_l2 / product_l2

        gauged_kernel_norms = np.asarray(
            [
                self._rms_norm(gauged_kernel[:, group], axis=None)
                for group in groups
            ],
            dtype=np.float64,
        )
        gauged_branch_norms = np.asarray(
            [
                self._rms_norm(gauged_branch[:, group, :], axis=None)
                for group in groups
            ],
            dtype=np.float64,
        )
        gauged_kernel_l1 = np.asarray(
            [
                np.sum(np.abs(gauged_kernel[:, group]))
                / float(len(group))
                for group in groups
            ],
            dtype=np.float64,
        )
        clipped_fraction = float(np.mean(
            (group_scales <= min_scale) | (group_scales >= max_scale)
        ))
        scalars = {
            'enabled': 1.0,
            'product_relative_delta_l2': product_relative_delta,
            'clipped_fraction': clipped_fraction,
            'group_count': float(len(groups)),
            'min_scale_bound': min_scale,
            'max_scale_bound': max_scale,
        }
        scalars.update(self._prefixed_summary('scale', group_scales))
        scalars.update(self._prefixed_summary('kernel_rms_before', kernel_norms))
        scalars.update(self._prefixed_summary('branch_rms_before', branch_norms))
        scalars.update(self._prefixed_summary('kernel_rms_after', gauged_kernel_norms))
        scalars.update(self._prefixed_summary('branch_rms_after', gauged_branch_norms))
        scalars.update(self._prefixed_summary('kernel_l1_before', kernel_l1))
        scalars.update(self._prefixed_summary('kernel_l1_after', gauged_kernel_l1))
        if mode in {'kernel_l1_unit', 'normalization_l1'}:
            scalars['kernel_l1_unit_max_abs_error'] = float(
                np.max(np.abs(gauged_kernel_l1[active] - 1.0))
            ) if np.any(active) else 0.0
        self._append_post_load_transform_audit({
            'prefix': (
                f'factorization/component_norm_gauge/{self.cortex_id}'
            ),
            'text': {
                'transform': transform_name,
                'mode': mode,
                'structure': structure,
                'component_count': component_count,
            },
            'scalars': scalars,
        })
        return gauged_kernel, gauged_branch

    def _balance_branch_projection(
        self,
        branch_projection_by_patch,
        source_kernel,
        balance_mode,
        transform_name,
        epsilon=1.0e-12,
    ):
        balance_mode = 'none' if balance_mode is None else str(balance_mode)
        balance_mode = balance_mode.lower()
        if balance_mode in {'none', 'off', 'false'}:
            return branch_projection_by_patch
        epsilon = float(epsilon)
        if epsilon <= 0.0:
            raise ValueError(
                f'{transform_name} receiver_projection_balance_epsilon must be '
                'positive'
            )

        balanced = branch_projection_by_patch.copy()
        receiver_count = int(balanced.shape[2])
        if balance_mode in {'class_mean', 'source_class'}:
            group_slices = self._receiver_group_slices(
                receiver_count,
                transform_name,
            )
            current_norms = np.asarray(
                [
                    self._rms_norm(balanced[:, :, start:end], axis=None)
                    for start, end in group_slices
                ],
                dtype=balanced.dtype,
            )
            if balance_mode == 'class_mean':
                target_norms = np.full_like(
                    current_norms,
                    np.maximum(current_norms.mean(), epsilon),
                )
            else:
                source_norms = np.asarray(
                    [
                        self._rms_norm(source_kernel[:, :, :, start:end], axis=None)
                        for start, end in group_slices
                    ],
                    dtype=balanced.dtype,
                )
                source_norms = np.maximum(source_norms, epsilon)
                target_norms = (
                    source_norms
                    * np.maximum(current_norms.mean(), epsilon)
                    / np.maximum(source_norms.mean(), epsilon)
                )
            scales = target_norms / np.maximum(current_norms, epsilon)
            for scale, (start, end) in zip(scales, group_slices):
                balanced[:, :, start:end] *= scale
            return balanced

        if balance_mode in {'column_mean', 'source_column'}:
            current_norms = self._rms_norm(balanced, axis=(0, 1))
            if balance_mode == 'column_mean':
                target_norms = np.full_like(
                    current_norms,
                    np.maximum(current_norms.mean(), epsilon),
                )
            else:
                source_norms = self._rms_norm(source_kernel, axis=(0, 1, 2))
                source_norms = np.maximum(source_norms, epsilon)
                target_norms = (
                    source_norms
                    * np.maximum(current_norms.mean(), epsilon)
                    / np.maximum(source_norms.mean(), epsilon)
                )
            scales = target_norms / np.maximum(current_norms, epsilon)
            balanced *= scales[None, None, :]
            return balanced

        raise ValueError(
            f'{transform_name} unsupported receiver_projection_balance_mode: '
            f'{balance_mode!r}'
        )

    def _calibrate_branch_projection_from_source_response(
        self,
        new_cortex_kernel,
        branch_projection_by_patch,
        target_patch_response,
        calibration_mode,
        transform_name,
        ridge=1.0e-3,
        blend=1.0,
        epsilon=1.0e-12,
        group_count=None,
    ):
        calibration_mode = (
            'none' if calibration_mode is None else str(calibration_mode)
        ).lower()
        if calibration_mode in {'none', 'off', 'false'}:
            return branch_projection_by_patch

        ridge = float(ridge)
        blend = float(blend)
        epsilon = float(epsilon)
        if ridge <= 0.0:
            raise ValueError(
                f'{transform_name} source_response_calibration_ridge must be '
                'positive'
            )
        if blend < 0.0 or blend > 1.0:
            raise ValueError(
                f'{transform_name} source_response_calibration_blend must be '
                f'in [0, 1], got {blend}'
            )
        if epsilon <= 0.0:
            raise ValueError(
                f'{transform_name} source_response_calibration_epsilon must '
                'be positive'
            )

        reconstructed_response = np.einsum(
            'lc,pcr->plr',
            new_cortex_kernel,
            branch_projection_by_patch,
            optimize=True,
        )
        if reconstructed_response.shape != target_patch_response.shape:
            raise ValueError(
                f'{transform_name} source-response target shape must be '
                f'{reconstructed_response.shape}, got '
                f'{target_patch_response.shape}'
            )

        calibrated = branch_projection_by_patch.copy()
        receiver_count = int(calibrated.shape[2])
        if calibration_mode in {'column', 'column_gain'}:
            response_energy = np.sum(
                np.square(reconstructed_response),
                axis=(0, 1),
                dtype=np.float64,
            )
            response_alignment = np.sum(
                reconstructed_response * target_patch_response,
                axis=(0, 1),
                dtype=np.float64,
            )
            positive_energy = response_energy[response_energy > epsilon]
            reference_energy = (
                float(positive_energy.mean())
                if positive_energy.size > 0
                else epsilon
            )
            penalty = ridge * max(reference_energy, epsilon)
            gains = (
                response_alignment + penalty
            ) / np.maximum(response_energy + penalty, epsilon)
            gains = 1.0 + blend * (gains - 1.0)
            if not np.all(np.isfinite(gains)):
                raise ValueError(
                    f'{transform_name} source-response column calibration '
                    'produced non-finite gains'
                )
            calibrated *= gains.astype(calibrated.dtype)[None, None, :]
            return calibrated

        if calibration_mode in {'class', 'class_matrix'}:
            group_slices = self._receiver_group_slices(
                receiver_count,
                transform_name,
                group_count=group_count,
            )
            for start, end in group_slices:
                group_width = end - start
                reconstructed_group = reconstructed_response[
                    :, :, start:end
                ].reshape(-1, group_width).astype(np.float64, copy=False)
                target_group = target_patch_response[
                    :, :, start:end
                ].reshape(-1, group_width).astype(np.float64, copy=False)
                gram = reconstructed_group.T @ reconstructed_group
                cross = reconstructed_group.T @ target_group
                identity = np.eye(group_width, dtype=np.float64)
                reference_energy = max(
                    float(np.trace(gram)) / group_width,
                    epsilon,
                )
                penalty = ridge * reference_energy
                mixing = np.linalg.solve(
                    gram + penalty * identity,
                    cross + penalty * identity,
                )
                mixing = identity + blend * (mixing - identity)
                if not np.all(np.isfinite(mixing)):
                    raise ValueError(
                        f'{transform_name} source-response class calibration '
                        'produced a non-finite mixing matrix'
                    )
                calibrated[:, :, start:end] = np.einsum(
                    'pci,ij->pcj',
                    calibrated[:, :, start:end],
                    mixing.astype(calibrated.dtype),
                    optimize=True,
                )
            return calibrated

        if calibration_mode in {'global', 'global_matrix'}:
            reconstructed_flat = reconstructed_response.reshape(
                -1, receiver_count
            ).astype(np.float64, copy=False)
            target_flat = target_patch_response.reshape(
                -1, receiver_count
            ).astype(np.float64, copy=False)
            gram = reconstructed_flat.T @ reconstructed_flat
            cross = reconstructed_flat.T @ target_flat
            identity = np.eye(receiver_count, dtype=np.float64)
            reference_energy = max(
                float(np.trace(gram)) / receiver_count,
                epsilon,
            )
            penalty = ridge * reference_energy
            mixing = np.linalg.solve(
                gram + penalty * identity,
                cross + penalty * identity,
            )
            mixing = identity + blend * (mixing - identity)
            if not np.all(np.isfinite(mixing)):
                raise ValueError(
                    f'{transform_name} source-response global calibration '
                    'produced a non-finite mixing matrix'
                )
            return np.einsum(
                'pci,ij->pcj',
                calibrated,
                mixing.astype(calibrated.dtype),
                optimize=True,
            )

        raise ValueError(
            f'{transform_name} unsupported source_response_calibration_mode: '
            f'{calibration_mode!r}'
        )

    def _select_factorization_components(
        self,
        new_cortex_kernel,
        branch_projection_by_patch,
        selection_mode,
        keep_count,
        component_count,
        transform_name,
        selection_structure='polarity_pairs',
        receiver_group_count=None,
        epsilon=1.0e-12,
    ):
        selection_mode = (
            'none' if selection_mode is None else str(selection_mode)
        ).lower()
        if selection_mode in {'none', 'off', 'false'}:
            return new_cortex_kernel, branch_projection_by_patch
        supported_modes = {
            'branch_energy',
            'receiver_class_selectivity',
            'energy_selectivity',
        }
        if selection_mode not in supported_modes:
            raise ValueError(
                f'{transform_name} unsupported '
                'factorization_component_selection_mode: '
                f'{selection_mode!r}'
            )

        selection_structure = (
            'polarity_pairs'
            if selection_structure is None
            else str(selection_structure).lower()
        )
        supported_structures = {'polarity_pairs', 'single'}
        if selection_structure not in supported_structures:
            raise ValueError(
                f'{transform_name} unsupported '
                'factorization_component_selection_structure: '
                f'{selection_structure!r}'
            )

        total_component_count = int(new_cortex_kernel.shape[1])
        if component_count is None:
            raise ValueError(
                f'{transform_name} component selection requires '
                'factorization_component_selection_component_count'
            )
        if keep_count is None:
            raise ValueError(
                f'{transform_name} component selection requires '
                'factorization_component_selection_keep_count'
            )
        component_count = int(component_count)
        keep_count = int(keep_count)
        selected_block_count = (
            2 if selection_structure == 'polarity_pairs' else 1
        )
        if (
            component_count <= 0
            or selected_block_count * component_count > total_component_count
        ):
            raise ValueError(
                f'{transform_name} component selection requires '
                f'{selected_block_count} block(s) of size {component_count} '
                f'within {total_component_count} components'
            )
        if keep_count <= 0 or keep_count > component_count:
            raise ValueError(
                f'{transform_name} factorization_component_selection_keep_count '
                f'must be in [1, {component_count}], got {keep_count}'
            )
        epsilon = float(epsilon)
        if not np.isfinite(epsilon) or epsilon <= 0.0:
            raise ValueError(
                f'{transform_name} factorization_component_selection_epsilon '
                'must be finite and positive'
            )

        if selection_structure == 'polarity_pairs':
            scored_branch = np.stack(
                [
                    branch_projection_by_patch[:, :component_count, :],
                    branch_projection_by_patch[
                        :, component_count:2 * component_count, :
                    ],
                ],
                axis=0,
            )
        else:
            scored_branch = branch_projection_by_patch[
                None, :, :component_count, :
            ]
        total_energy = np.sum(
            np.square(scored_branch),
            axis=(0, 1, 3),
            dtype=np.float64,
        )
        if selection_mode == 'branch_energy':
            scores = total_energy
        else:
            group_slices = self._receiver_group_slices(
                int(branch_projection_by_patch.shape[2]),
                transform_name,
                group_count=receiver_group_count,
            )
            group_energy = np.stack(
                [
                    np.sum(
                        np.square(scored_branch[:, :, :, start:end]),
                        axis=(0, 1, 3),
                        dtype=np.float64,
                    )
                    for start, end in group_slices
                ],
                axis=1,
            )
            selectivity = np.max(group_energy, axis=1) / np.maximum(
                total_energy,
                epsilon,
            )
            scores = (
                selectivity
                if selection_mode == 'receiver_class_selectivity'
                else total_energy * selectivity
            )
        if not np.all(np.isfinite(scores)):
            raise ValueError(
                f'{transform_name} component selection produced non-finite '
                'scores'
            )

        selected = np.argsort(-scores, kind='stable')[:keep_count]
        selected = np.sort(selected)
        kept_start = selected_block_count * component_count
        kernel_blocks = [new_cortex_kernel[:, selected]]
        branch_blocks = [branch_projection_by_patch[:, selected, :]]
        if selection_structure == 'polarity_pairs':
            kernel_blocks.append(
                new_cortex_kernel[:, component_count + selected]
            )
            branch_blocks.append(
                branch_projection_by_patch[
                    :, component_count + selected, :
                ]
            )
        kernel_blocks.append(new_cortex_kernel[:, kept_start:])
        branch_blocks.append(branch_projection_by_patch[:, kept_start:, :])
        new_cortex_kernel = np.concatenate(kernel_blocks, axis=1)
        branch_projection_by_patch = np.concatenate(branch_blocks, axis=1)
        return new_cortex_kernel, branch_projection_by_patch

    def _apply_component_receiver_objective_gauge(
        self,
        new_cortex_kernel,
        branch_projection_by_patch,
        group_weights,
        exponent,
        transform_name,
        component_count=None,
        min_scale=0.5,
        max_scale=2.0,
        epsilon=1.0e-12,
    ):
        if group_weights is None:
            return new_cortex_kernel, branch_projection_by_patch

        weights = np.asarray(group_weights, dtype=np.float64)
        if weights.ndim != 1 or weights.size == 0:
            raise ValueError(
                f'{transform_name} '
                'factorization_component_gauge_receiver_group_weights must '
                'be a non-empty one-dimensional sequence'
            )
        if not np.all(np.isfinite(weights)) or np.any(weights <= 0.0):
            raise ValueError(
                f'{transform_name} '
                'factorization_component_gauge_receiver_group_weights must '
                'contain only finite positive values'
            )

        exponent = float(exponent)
        min_scale = float(min_scale)
        max_scale = float(max_scale)
        epsilon = float(epsilon)
        if not np.isfinite(exponent):
            raise ValueError(
                f'{transform_name} factorization_component_gauge_exponent '
                'must be finite'
            )
        if (
            not np.isfinite(min_scale)
            or not np.isfinite(max_scale)
            or min_scale <= 0.0
            or max_scale < min_scale
        ):
            raise ValueError(
                f'{transform_name} component gauge scale bounds must satisfy '
                f'0 < min <= max, got {min_scale} and {max_scale}'
            )
        if epsilon <= 0.0:
            raise ValueError(
                f'{transform_name} factorization_component_gauge_epsilon '
                'must be positive'
            )

        total_component_count = int(new_cortex_kernel.shape[1])
        if component_count is None:
            component_count = total_component_count
        component_count = int(component_count)
        if component_count <= 0 or component_count > total_component_count:
            raise ValueError(
                f'{transform_name} '
                'factorization_component_gauge_component_count must be in '
                f'[1, {total_component_count}], got {component_count}'
            )

        group_slices = self._receiver_group_slices(
            int(branch_projection_by_patch.shape[2]),
            transform_name,
            group_count=int(weights.size),
        )
        group_energy = np.asarray(
            [
                self._rms_norm(
                    branch_projection_by_patch[
                        :, :component_count, start:end
                    ],
                    axis=(0, 2),
                )
                for start, end in group_slices
            ],
            dtype=np.float64,
        ).T
        energy_sum = group_energy.sum(axis=1)
        active = energy_sum > epsilon
        log_scales = np.zeros(component_count, dtype=np.float64)
        if np.any(active):
            support = group_energy[active] / energy_sum[active, None]
            log_priority = support @ np.log(weights)
            log_priority -= float(log_priority.mean())
            log_scales[active] = exponent * log_priority
        component_scales = np.clip(
            np.exp(log_scales),
            min_scale,
            max_scale,
        ).astype(new_cortex_kernel.dtype)

        gauged_kernel = new_cortex_kernel.copy()
        gauged_branch = branch_projection_by_patch.copy()
        gauged_kernel[:, :component_count] *= component_scales[None, :]
        gauged_branch[:, :component_count, :] /= component_scales[
            None, :, None
        ]
        return gauged_kernel, gauged_branch

    def _append_source_response_residual_components(
        self,
        new_cortex_kernel,
        branch_projection_by_patch,
        target_patch_response,
        residual_mode,
        transform_name,
        component_count=0,
        residual_scale=1.0,
        group_indices=None,
        group_count=None,
        polarity_split=True,
        kernel_scale=1.0,
        basis_normalization='left_scaled',
        random_state=722,
    ):
        residual_mode = (
            'none' if residual_mode is None else str(residual_mode)
        ).lower()
        if residual_mode in {'none', 'off', 'false'}:
            return new_cortex_kernel, branch_projection_by_patch

        component_count = int(component_count)
        residual_scale = float(residual_scale)
        if component_count < 0:
            raise ValueError(
                f'{transform_name} source_response_residual_component_count '
                'must be non-negative'
            )
        if component_count == 0 or residual_scale == 0.0:
            return new_cortex_kernel, branch_projection_by_patch

        kernel_scale = float(kernel_scale)
        if kernel_scale <= 0.0:
            raise ValueError(
                f'{transform_name} source_response_residual_kernel_scale must '
                'be positive'
            )

        current_response = np.einsum(
            'lc,pcr->plr',
            new_cortex_kernel,
            branch_projection_by_patch,
            optimize=True,
        )
        if current_response.shape != target_patch_response.shape:
            raise ValueError(
                f'{transform_name} source-response residual target shape must '
                f'be {current_response.shape}, got '
                f'{target_patch_response.shape}'
            )

        if residual_mode in {'target_minus_current', 'target-current'}:
            residual = target_patch_response - current_response
        elif residual_mode in {'current_minus_target', 'current-target'}:
            residual = current_response - target_patch_response
        else:
            raise ValueError(
                f'{transform_name} unsupported source_response_residual_mode: '
                f'{residual_mode!r}'
            )
        residual = residual * residual_scale

        if group_indices is not None:
            receiver_count = int(residual.shape[2])
            group_slices = self._receiver_group_slices(
                receiver_count,
                transform_name,
                group_count=group_count,
            )
            if isinstance(group_indices, (str, int)):
                group_indices = [group_indices]
            selected = set()
            for index in group_indices:
                index = int(index)
                if index < 0 or index >= len(group_slices):
                    raise ValueError(
                        f'{transform_name} source_response_residual_group_indices '
                        f'contains invalid group {index}; expected [0, '
                        f'{len(group_slices) - 1}]'
                    )
                selected.add(index)
            masked = np.zeros_like(residual)
            for index in selected:
                start, end = group_slices[index]
                masked[:, :, start:end] = residual[:, :, start:end]
            residual = masked

        if not np.all(np.isfinite(residual)):
            raise ValueError(
                f'{transform_name} source-response residual produced '
                'non-finite values'
            )

        residual_kernel, residual_branch = self._patch_matrix_svd_factor_branch(
            residual,
            component_count,
            int(random_state) + 2000,
            transform_name,
        )
        basis_normalization = (
            'left_scaled'
            if basis_normalization is None
            else str(basis_normalization).lower()
        )
        if basis_normalization in {'left_scaled', 'legacy'}:
            pass
        elif basis_normalization in {'orthonormal', 'unit_norm'}:
            component_norms = np.sqrt(
                np.sum(
                    np.square(residual_kernel),
                    axis=0,
                    dtype=np.float64,
                )
            )
            active_components = component_norms > 1.0e-12
            normalized_kernel = np.zeros_like(residual_kernel)
            normalized_kernel[:, active_components] = (
                residual_kernel[:, active_components]
                / component_norms[active_components][None, :]
            )
            residual_branch = residual_branch.copy()
            residual_branch[:, active_components, :] *= (
                component_norms[active_components][None, :, None]
            )
            residual_branch[:, ~active_components, :] = 0.0
            residual_kernel = normalized_kernel
        else:
            raise ValueError(
                f'{transform_name} unsupported '
                'source_response_residual_basis_normalization: '
                f'{basis_normalization!r}'
            )
        residual_kernel = residual_kernel * kernel_scale
        residual_branch = residual_branch / kernel_scale
        residual_kernel, residual_branch = self._apply_local_polarity_split(
            residual_kernel,
            residual_branch,
            polarity_split,
            component_count,
            transform_name,
        )

        new_cortex_kernel = np.concatenate(
            [new_cortex_kernel, residual_kernel],
            axis=1,
        )
        branch_projection_by_patch = np.concatenate(
            [branch_projection_by_patch, residual_branch],
            axis=1,
        )
        return new_cortex_kernel, branch_projection_by_patch

    def _patch_matrix_svd_factor_branch(
        self,
        patch_tensor,
        component_count,
        random_state,
        transform_name,
    ):
        patch_num, patch_dim, receiver_count = patch_tensor.shape
        component_count = int(component_count)
        if component_count <= 0:
            raise ValueError(
                f'{transform_name} dc_component_count must be positive'
            )
        if component_count > patch_dim:
            raise ValueError(
                f'{transform_name} dc_component_count cannot exceed '
                f'local patch dimension {patch_dim}'
            )

        matrix = patch_tensor.transpose(1, 0, 2).reshape(
            patch_dim,
            patch_num * receiver_count,
        )
        if component_count == patch_dim:
            left_matrix = np.eye(patch_dim, dtype=matrix.dtype)
            right_matrix = matrix
            branch_projection_by_patch = right_matrix.reshape(
                component_count,
                patch_num,
                receiver_count,
            ).transpose(1, 0, 2)
            return left_matrix, branch_projection_by_patch

        svd = TruncatedSVD(
            n_components=component_count,
            random_state=int(random_state),
        )
        left_matrix = svd.fit_transform(matrix)
        right_matrix = svd.components_
        branch_projection_by_patch = right_matrix.reshape(
            component_count,
            patch_num,
            receiver_count,
        ).transpose(1, 0, 2)
        return left_matrix, branch_projection_by_patch

    def yield_patch_tucker_subcortex(
        self,
        cortex_constructor,
        local_kernel_size,
        local_stride,
        rank_patch,
        rank_local,
        rank_receiver,
        factorization_receiver_group_count=None,
        factorization_receiver_objective_group_weights=None,
        factorization_full_rank_local_basis_mode='identity',
        iteration_count=8,
        center_mode='none',
        dc_component_count=0,
        cover_edges=False,
        new_cortex_kernel_scale=1.0,
        random_state=722,
        new_cortex_split_ratio=None,
        regularize_afterward=None,
        pooling_settings=None,
        pooling_projection_mode='legacy_mean',
        pooling_aware_factorization_mode='none',
        pooling_aware_als_iteration_count=None,
        pooling_aware_als_ridge=1.0e-6,
        root_direct_mode=None,
        root_projection_scale=1.0,
        root_learning_mask=None,
        new_cortex_init_amplifier=1.0,
        polarity_split=None,
        polarity_split_component_count=None,
        polarity_split_duplicate_kernel_scales=None,
        polarity_split_duplicate_branch_weights=None,
        receiver_projection_balance_mode='none',
        receiver_projection_balance_epsilon=1.0e-12,
        contrast_branch_scale=1.0,
        dc_branch_scale=1.0,
        source_response_calibration_mode='none',
        source_response_calibration_ridge=1.0e-3,
        source_response_calibration_blend=1.0,
        source_response_calibration_epsilon=1.0e-12,
        source_response_calibration_group_count=None,
        factorization_component_gauge_receiver_group_weights=None,
        factorization_component_gauge_exponent=1.0,
        factorization_component_gauge_component_count=None,
        factorization_component_gauge_min_scale=0.5,
        factorization_component_gauge_max_scale=2.0,
        factorization_component_gauge_epsilon=1.0e-12,
        factorization_component_norm_gauge_mode='none',
        factorization_component_norm_gauge_component_count=None,
        factorization_component_norm_gauge_structure='independent',
        factorization_component_norm_gauge_min_scale=0.25,
        factorization_component_norm_gauge_max_scale=4.0,
        factorization_component_norm_gauge_epsilon=1.0e-12,
        factorization_component_selection_mode='none',
        factorization_component_selection_keep_count=None,
        factorization_component_selection_component_count=None,
        factorization_component_selection_structure='polarity_pairs',
        factorization_component_selection_receiver_group_count=None,
        factorization_component_selection_epsilon=1.0e-12,
        factorization_component_selection_recalibration_mode='none',
        factorization_component_selection_recalibration_ridge=1.0e-3,
        factorization_component_selection_recalibration_blend=1.0,
        factorization_component_selection_recalibration_epsilon=1.0e-12,
        factorization_component_selection_recalibration_group_count=None,
        source_response_residual_mode='none',
        source_response_residual_component_count=0,
        source_response_residual_scale=1.0,
        source_response_residual_group_indices=None,
        source_response_residual_group_count=None,
        source_response_residual_polarity_split=True,
        source_response_residual_kernel_scale=None,
        source_response_residual_basis_normalization='left_scaled',
    ):
        transform_name = 'patch_tucker_yield'
        local_kernel_size = int(local_kernel_size)
        patch_state = self._prepare_patch_yield_tensor(
            local_kernel_size,
            local_stride,
            transform_name,
            cover_edges=bool(cover_edges),
        )
        source_kernel = patch_state['source_kernel']
        center_mode = 'none' if center_mode is None else str(center_mode)
        base_kernel = self._patch_center_base_kernel(
            source_kernel,
            center_mode,
            transform_name,
        )
        if center_mode == 'none':
            tucker_tensor = patch_state['patch_tensor']
        else:
            tucker_tensor = self._patchify_kernel_with_state(
                source_kernel - base_kernel,
                patch_state,
                local_kernel_size,
            )

        factorization_tensor, factorization_objective_weights = (
            self._precondition_patch_tensor_receiver_groups(
                tucker_tensor,
                factorization_receiver_objective_group_weights,
                transform_name,
            )
        )

        pooling_aware_factorization_mode = (
            'none'
            if pooling_aware_factorization_mode is None
            else str(pooling_aware_factorization_mode).lower()
        )
        pooling_projection_mode_normalized = (
            'legacy_mean'
            if pooling_projection_mode is None
            else str(pooling_projection_mode).lower()
        )
        if (
            pooling_aware_factorization_mode not in {
                'none',
                'off',
                'false',
            }
            and pooling_projection_mode_normalized != 'legacy_mean'
        ):
            raise ValueError(
                f'{transform_name} pooling-aware factorization requires '
                'pooling_projection_mode=legacy_mean'
            )

        if pooling_aware_factorization_mode not in {'none', 'off', 'false'}:
            if factorization_receiver_group_count is not None:
                raise ValueError(
                    f'{transform_name} pooling-aware factorization does not '
                    'support factorization_receiver_group_count yet'
                )
            new_cortex_kernel, branch_projection_by_patch = (
                self._pooling_aware_factor_branch(
                    factorization_tensor,
                    patch_state,
                    pooling_settings,
                    rank_patch,
                    rank_local,
                    rank_receiver,
                    iteration_count,
                    random_state,
                    transform_name,
                    pooling_aware_factorization_mode,
                    full_rank_local_basis_mode=(
                        factorization_full_rank_local_basis_mode
                    ),
                    als_iteration_count=pooling_aware_als_iteration_count,
                    als_ridge=pooling_aware_als_ridge,
                )
            )
        elif factorization_receiver_group_count is None:
            core, factors = self._patch_tucker_decomposition(
                factorization_tensor,
                rank_patch,
                rank_local,
                rank_receiver,
                iteration_count,
                random_state,
                transform_name,
                full_rank_local_basis_mode=(
                    factorization_full_rank_local_basis_mode
                ),
            )
            new_cortex_kernel, branch_projection_by_patch = (
                self._patch_tucker_to_factor_branch(core, factors)
            )
        else:
            factorization_receiver_group_count = int(
                factorization_receiver_group_count
            )
            new_cortex_kernel, branch_projection_by_patch = (
                self._patch_grouped_tucker_to_factor_branch(
                    factorization_tensor,
                    rank_patch,
                    rank_local,
                    rank_receiver,
                    factorization_receiver_group_count,
                    iteration_count,
                    random_state,
                    transform_name,
                    full_rank_local_basis_mode=(
                        factorization_full_rank_local_basis_mode
                    ),
                )
            )
        if factorization_objective_weights is not None:
            branch_projection_by_patch = branch_projection_by_patch.copy()
            objective_group_slices = self._receiver_group_slices(
                int(branch_projection_by_patch.shape[2]),
                transform_name,
                group_count=int(factorization_objective_weights.size),
            )
            for objective_weight, (start, end) in zip(
                factorization_objective_weights,
                objective_group_slices,
            ):
                branch_projection_by_patch[:, :, start:end] /= float(
                    objective_weight
                )
        contrast_branch_scale = float(contrast_branch_scale)
        dc_branch_scale = float(dc_branch_scale)
        if contrast_branch_scale < 0.0:
            raise ValueError(
                f'{transform_name} contrast_branch_scale must be non-negative'
            )
        if dc_branch_scale < 0.0:
            raise ValueError(
                f'{transform_name} dc_branch_scale must be non-negative'
            )
        branch_projection_by_patch = (
            branch_projection_by_patch * contrast_branch_scale
        )

        source_response_target = tucker_tensor * contrast_branch_scale
        dc_component_count = int(dc_component_count)
        if center_mode != 'none' and dc_component_count > 0:
            base_patch_tensor = self._patchify_kernel_with_state(
                base_kernel,
                patch_state,
                local_kernel_size,
            )
            if pooling_aware_factorization_mode in {'none', 'off', 'false'}:
                base_kernel_matrix, base_branch_projection = (
                    self._patch_matrix_svd_factor_branch(
                        base_patch_tensor,
                        dc_component_count,
                        int(random_state) + 1000,
                        transform_name,
                    )
                )
            else:
                base_kernel_matrix, base_branch_projection = (
                    self._pooling_aware_matrix_svd_factor_branch(
                        base_patch_tensor,
                        patch_state,
                        pooling_settings,
                        dc_component_count,
                        int(random_state) + 1000,
                        transform_name,
                    )
                )
            base_branch_projection = base_branch_projection * dc_branch_scale
            new_cortex_kernel = np.concatenate(
                [new_cortex_kernel, base_kernel_matrix],
                axis=1,
            )
            branch_projection_by_patch = np.concatenate(
                [branch_projection_by_patch, base_branch_projection],
                axis=1,
            )
            source_response_target = (
                source_response_target
                + base_patch_tensor * dc_branch_scale
            )

        split_ratio = (
            self.new_cortex_split_ratio
            if new_cortex_split_ratio is None
            else float(new_cortex_split_ratio)
        )
        new_cortex_kernel_scale = float(new_cortex_kernel_scale)
        if new_cortex_kernel_scale <= 0.0:
            raise ValueError(
                f'{transform_name} new_cortex_kernel_scale must be positive'
            )
        new_cortex_kernel = (
            new_cortex_kernel
            * new_cortex_kernel_scale
            * split_ratio
        )
        branch_projection_by_patch = (
            branch_projection_by_patch / new_cortex_kernel_scale
        )
        duplicate_polarity_split = (
            isinstance(polarity_split, str)
            and polarity_split.lower() in {
                'duplicate',
                'duplicate_nonnegative',
            }
        )
        initial_polarity_split = (
            'posneg' if duplicate_polarity_split else polarity_split
        )
        new_cortex_kernel, branch_projection_by_patch = (
            self._apply_local_polarity_split(
                new_cortex_kernel,
                branch_projection_by_patch,
                initial_polarity_split,
                polarity_split_component_count,
                transform_name,
            )
        )
        branch_projection_by_patch = self._balance_branch_projection(
            branch_projection_by_patch,
            source_kernel,
            receiver_projection_balance_mode,
            transform_name,
            epsilon=receiver_projection_balance_epsilon,
        )
        branch_projection_by_patch = (
            self._calibrate_branch_projection_from_source_response(
                new_cortex_kernel,
                branch_projection_by_patch,
                source_response_target,
                source_response_calibration_mode,
                transform_name,
                ridge=source_response_calibration_ridge,
                blend=source_response_calibration_blend,
                epsilon=source_response_calibration_epsilon,
                group_count=source_response_calibration_group_count,
            )
        )
        selection_recalibration_mode = (
            'none'
            if factorization_component_selection_recalibration_mode is None
            else str(
                factorization_component_selection_recalibration_mode
            ).lower()
        )
        selection_recalibration_target = None
        if selection_recalibration_mode not in {'none', 'off', 'false'}:
            if (
                factorization_component_selection_mode is None
                or str(factorization_component_selection_mode).lower()
                in {'none', 'off', 'false'}
            ):
                raise ValueError(
                    f'{transform_name} component selection recalibration '
                    'requires an enabled component selection mode'
                )
            selection_recalibration_target = np.einsum(
                'lc,pcr->plr',
                new_cortex_kernel,
                branch_projection_by_patch,
                optimize=True,
            )
        new_cortex_kernel, branch_projection_by_patch = (
            self._select_factorization_components(
                new_cortex_kernel,
                branch_projection_by_patch,
                factorization_component_selection_mode,
                factorization_component_selection_keep_count,
                factorization_component_selection_component_count,
                transform_name,
                selection_structure=(
                    factorization_component_selection_structure
                ),
                receiver_group_count=(
                    factorization_component_selection_receiver_group_count
                ),
                epsilon=factorization_component_selection_epsilon,
            )
        )
        if selection_recalibration_target is not None:
            branch_projection_by_patch = (
                self._calibrate_branch_projection_from_source_response(
                    new_cortex_kernel,
                    branch_projection_by_patch,
                    selection_recalibration_target,
                    selection_recalibration_mode,
                    transform_name,
                    ridge=(
                        factorization_component_selection_recalibration_ridge
                    ),
                    blend=(
                        factorization_component_selection_recalibration_blend
                    ),
                    epsilon=(
                        factorization_component_selection_recalibration_epsilon
                    ),
                    group_count=(
                        factorization_component_selection_recalibration_group_count
                    ),
                )
            )
        new_cortex_kernel, branch_projection_by_patch = (
            self._apply_component_norm_gauge(
                new_cortex_kernel,
                branch_projection_by_patch,
                factorization_component_norm_gauge_mode,
                transform_name,
                component_count=(
                    factorization_component_norm_gauge_component_count
                ),
                structure=factorization_component_norm_gauge_structure,
                min_scale=factorization_component_norm_gauge_min_scale,
                max_scale=factorization_component_norm_gauge_max_scale,
                epsilon=factorization_component_norm_gauge_epsilon,
            )
        )
        new_cortex_kernel, branch_projection_by_patch = (
            self._apply_component_receiver_objective_gauge(
                new_cortex_kernel,
                branch_projection_by_patch,
                factorization_component_gauge_receiver_group_weights,
                factorization_component_gauge_exponent,
                transform_name,
                component_count=(
                    factorization_component_gauge_component_count
                ),
                min_scale=factorization_component_gauge_min_scale,
                max_scale=factorization_component_gauge_max_scale,
                epsilon=factorization_component_gauge_epsilon,
            )
        )
        if duplicate_polarity_split:
            split_count = (
                int(new_cortex_kernel.shape[1] // 2)
                if polarity_split_component_count is None
                else int(polarity_split_component_count)
            )
            inactive_start = split_count
            inactive_end = 2 * split_count
            inactive_kernel = new_cortex_kernel[
                :, inactive_start:inactive_end
            ]
            if (
                inactive_end > int(new_cortex_kernel.shape[1])
                or np.max(np.abs(inactive_kernel)) > 1.0e-12
            ):
                raise ValueError(
                    f'{transform_name} duplicate_nonnegative polarity split '
                    'requires the legacy negative-polarity block to be silent'
                )
            new_cortex_kernel = np.concatenate(
                [
                    new_cortex_kernel[:, :split_count],
                    new_cortex_kernel[:, inactive_end:],
                ],
                axis=1,
            )
            branch_projection_by_patch = np.concatenate(
                [
                    branch_projection_by_patch[:, :split_count, :],
                    branch_projection_by_patch[:, inactive_end:, :],
                ],
                axis=1,
            )
            new_cortex_kernel, branch_projection_by_patch = (
                self._apply_local_polarity_split(
                    new_cortex_kernel,
                    branch_projection_by_patch,
                    polarity_split,
                    polarity_split_component_count,
                    transform_name,
                    duplicate_kernel_scales=(
                        polarity_split_duplicate_kernel_scales
                    ),
                    duplicate_branch_weights=(
                        polarity_split_duplicate_branch_weights
                    ),
                )
            )
        residual_kernel_scale = (
            new_cortex_kernel_scale
            if source_response_residual_kernel_scale is None
            else source_response_residual_kernel_scale
        )
        new_cortex_kernel, branch_projection_by_patch = (
            self._append_source_response_residual_components(
                new_cortex_kernel,
                branch_projection_by_patch,
                source_response_target,
                source_response_residual_mode,
                transform_name,
                component_count=source_response_residual_component_count,
                residual_scale=source_response_residual_scale,
                group_indices=source_response_residual_group_indices,
                group_count=source_response_residual_group_count,
                polarity_split=source_response_residual_polarity_split,
                kernel_scale=residual_kernel_scale,
                basis_normalization=(
                    source_response_residual_basis_normalization
                ),
                random_state=int(random_state),
            )
        )

        return self._finalize_patch_factor_yield(
            cortex_constructor,
            local_kernel_size,
            int(local_stride),
            patch_state,
            new_cortex_kernel,
            branch_projection_by_patch,
            transform_name,
            regularize_afterward=regularize_afterward,
            pooling_settings=pooling_settings,
            pooling_projection_mode=pooling_projection_mode,
            root_direct_mode=root_direct_mode,
            root_projection_scale=root_projection_scale,
            root_learning_mask=root_learning_mask,
            new_cortex_init_amplifier=new_cortex_init_amplifier,
            cover_edges=bool(cover_edges),
        )

    def yield_patch_mean_pca_subcortex(
        self,
        cortex_constructor,
        local_kernel_size,
        local_stride,
        component_count,
        PCA_settings=None,
        new_cortex_split_ratio=None,
        regularize_afterward=None,
        pooling_settings=None,
        root_direct_mode=None,
        root_projection_scale=1.0,
        root_learning_mask=None,
        new_cortex_init_amplifier=1.0,
        polarity_split=None,
        polarity_split_component_count=None,
        receiver_projection_balance_mode='none',
        receiver_projection_balance_epsilon=1.0e-12,
    ):
        transform_name = 'patch_mean_pca_yield'
        patch_state = self._prepare_patch_yield_tensor(
            local_kernel_size,
            local_stride,
            transform_name,
        )
        pca_settings = copy.deepcopy(
            self.PCA_settings if PCA_settings is None else PCA_settings
        )
        pca_settings['remained_variance'] = int(component_count)
        split_ratio = (
            self.new_cortex_split_ratio
            if new_cortex_split_ratio is None
            else float(new_cortex_split_ratio)
        )

        patch_mean_matrix = patch_state['patch_tensor'].mean(axis=0)
        new_cortex_kernel, right_matrix, amplifier = self.PCA(
            patch_mean_matrix,
            **pca_settings
        )
        new_cortex_kernel *= split_ratio * float(amplifier)
        patch_num = patch_state['patch_num']
        branch_projection_by_patch = np.broadcast_to(
            right_matrix[None, :, :] / float(patch_num),
            (patch_num, right_matrix.shape[0], right_matrix.shape[1]),
        ).copy()
        new_cortex_kernel, branch_projection_by_patch = (
            self._apply_local_polarity_split(
                new_cortex_kernel,
                branch_projection_by_patch,
                polarity_split,
                polarity_split_component_count,
                transform_name,
            )
        )
        branch_projection_by_patch = self._balance_branch_projection(
            branch_projection_by_patch,
            patch_state['source_kernel'],
            receiver_projection_balance_mode,
            transform_name,
            epsilon=receiver_projection_balance_epsilon,
        )

        return self._finalize_patch_factor_yield(
            cortex_constructor,
            int(local_kernel_size),
            int(local_stride),
            patch_state,
            new_cortex_kernel,
            branch_projection_by_patch,
            transform_name,
            regularize_afterward=regularize_afterward,
            pooling_settings=pooling_settings,
            root_direct_mode=root_direct_mode,
            root_projection_scale=root_projection_scale,
            root_learning_mask=root_learning_mask,
            new_cortex_init_amplifier=new_cortex_init_amplifier,
        )

    def yield_patch_tensor_cp_subcortex(
        self,
        cortex_constructor,
        local_kernel_size,
        local_stride,
        component_count,
        PCA_settings=None,
        center_mode='none',
        dc_component_count=0,
        cover_edges=False,
        new_cortex_kernel_scale=1.0,
        random_state=722,
        rank1_iteration_count=1024,
        new_cortex_split_ratio=None,
        regularize_afterward=None,
        pooling_settings=None,
        pooling_projection_mode='legacy_mean',
        root_direct_mode=None,
        root_projection_scale=1.0,
        root_learning_mask=None,
        new_cortex_init_amplifier=1.0,
        polarity_split=None,
        polarity_split_component_count=None,
        receiver_projection_balance_mode='none',
        receiver_projection_balance_epsilon=1.0e-12,
        contrast_branch_scale=1.0,
        dc_branch_scale=1.0,
        source_response_calibration_mode='none',
        source_response_calibration_ridge=1.0e-3,
        source_response_calibration_blend=1.0,
        source_response_calibration_epsilon=1.0e-12,
        source_response_calibration_group_count=None,
        factorization_component_norm_gauge_mode='none',
        factorization_component_norm_gauge_component_count=None,
        factorization_component_norm_gauge_structure='independent',
        factorization_component_norm_gauge_min_scale=0.25,
        factorization_component_norm_gauge_max_scale=4.0,
        factorization_component_norm_gauge_epsilon=1.0e-12,
    ):
        transform_name = 'patch_tensor_cp_yield'
        local_kernel_size = int(local_kernel_size)
        patch_state = self._prepare_patch_yield_tensor(
            local_kernel_size,
            local_stride,
            transform_name,
            cover_edges=bool(cover_edges),
        )
        source_kernel = patch_state['source_kernel']
        center_mode = 'none' if center_mode is None else str(center_mode)
        base_kernel = self._patch_center_base_kernel(
            source_kernel,
            center_mode,
            transform_name,
        )
        if center_mode == 'none':
            cp_tensor = patch_state['patch_tensor']
        else:
            cp_tensor = self._patchify_kernel_with_state(
                source_kernel - base_kernel,
                patch_state,
                local_kernel_size,
            )
        pca_settings = copy.deepcopy(
            self.PCA_settings if PCA_settings is None else PCA_settings
        )
        pca_settings['remained_variance'] = int(component_count)

        patch_num, patch_dim, receiver_count = cp_tensor.shape
        mode_patch_feature_matrix = cp_tensor.transpose(1, 0, 2).reshape(
            patch_dim,
            patch_num * receiver_count,
        )
        new_cortex_kernel, _, amplifier = self.PCA(
            mode_patch_feature_matrix,
            **pca_settings
        )
        new_cortex_kernel *= float(amplifier)
        output_channel = int(new_cortex_kernel.shape[1])
        if output_channel <= 0:
            raise ValueError(f'{transform_name} produced no components')

        component_coefficients = np.einsum(
            'kd,pdr->pkr',
            np.linalg.pinv(new_cortex_kernel),
            cp_tensor,
        )
        (
            spatial_aggregation_by_component,
            receiver_projection,
            rank1_energy_fraction,
        ) = self._batched_rank1_svd(
            component_coefficients.transpose(1, 0, 2),
            rank1_iteration_count,
            int(random_state) + 500,
            transform_name,
        )
        spatial_aggregation = spatial_aggregation_by_component.T

        branch_projection_by_patch = (
            spatial_aggregation[:, :, None]
            * receiver_projection[None, :, :]
        )
        local_kernel_rms = self._rms_norm(new_cortex_kernel, axis=0)
        spatial_rms = self._rms_norm(spatial_aggregation, axis=0)
        receiver_rms = self._rms_norm(receiver_projection, axis=1)
        branch_rms = self._rms_norm(
            branch_projection_by_patch,
            axis=(0, 2),
        )
        factor_product_rms = local_kernel_rms * spatial_rms * receiver_rms
        factor_scalars = {}
        factor_scalars.update(self._prefixed_summary(
            'local_kernel_rms',
            local_kernel_rms,
        ))
        factor_scalars.update(self._prefixed_summary(
            'spatial_rms',
            spatial_rms,
        ))
        factor_scalars.update(self._prefixed_summary(
            'receiver_rms',
            receiver_rms,
        ))
        factor_scalars.update(self._prefixed_summary(
            'combined_branch_rms',
            branch_rms,
        ))
        factor_scalars.update(self._prefixed_summary(
            'factor_product_rms',
            factor_product_rms,
        ))
        factor_scalars.update(self._prefixed_summary(
            'rank1_energy_fraction',
            rank1_energy_fraction,
        ))
        self._append_post_load_transform_audit({
            'prefix': f'factorization/cp_factor_scale/{self.cortex_id}',
            'text': {
                'transform': transform_name,
                'allocation': 'spatial_factor_carries_rank1_singular_value',
                'rank1_iteration_count': int(rank1_iteration_count),
            },
            'scalars': factor_scalars,
        })
        contrast_branch_scale = float(contrast_branch_scale)
        dc_branch_scale = float(dc_branch_scale)
        if contrast_branch_scale < 0.0:
            raise ValueError(
                f'{transform_name} contrast_branch_scale must be non-negative'
            )
        if dc_branch_scale < 0.0:
            raise ValueError(
                f'{transform_name} dc_branch_scale must be non-negative'
            )
        branch_projection_by_patch *= contrast_branch_scale
        contrast_target = cp_tensor * contrast_branch_scale
        self._append_factorization_reconstruction_audit(
            contrast_target,
            new_cortex_kernel,
            branch_projection_by_patch,
            transform_name,
            'contrast_before_dc',
        )

        source_response_target = contrast_target
        dc_component_count = int(dc_component_count)
        if center_mode != 'none' and dc_component_count > 0:
            base_patch_tensor = self._patchify_kernel_with_state(
                base_kernel,
                patch_state,
                local_kernel_size,
            )
            base_kernel_matrix, base_branch_projection = (
                self._patch_matrix_svd_factor_branch(
                    base_patch_tensor,
                    dc_component_count,
                    int(random_state) + 1000,
                    transform_name,
                )
            )
            base_branch_projection *= dc_branch_scale
            new_cortex_kernel = np.concatenate(
                [new_cortex_kernel, base_kernel_matrix],
                axis=1,
            )
            branch_projection_by_patch = np.concatenate(
                [branch_projection_by_patch, base_branch_projection],
                axis=1,
            )
            source_response_target = (
                source_response_target + base_patch_tensor * dc_branch_scale
            )

        split_ratio = (
            self.new_cortex_split_ratio
            if new_cortex_split_ratio is None
            else float(new_cortex_split_ratio)
        )
        total_kernel_scale = float(split_ratio) * float(
            new_cortex_kernel_scale
        )
        if total_kernel_scale <= 0.0:
            raise ValueError(
                f'{transform_name} kernel scale must be positive'
            )
        new_cortex_kernel *= total_kernel_scale
        branch_projection_by_patch /= total_kernel_scale
        new_cortex_kernel, branch_projection_by_patch = (
            self._apply_local_polarity_split(
                new_cortex_kernel,
                branch_projection_by_patch,
                polarity_split,
                polarity_split_component_count,
                transform_name,
            )
        )
        branch_projection_by_patch = self._balance_branch_projection(
            branch_projection_by_patch,
            patch_state['source_kernel'],
            receiver_projection_balance_mode,
            transform_name,
            epsilon=receiver_projection_balance_epsilon,
        )
        branch_projection_by_patch = (
            self._calibrate_branch_projection_from_source_response(
                new_cortex_kernel,
                branch_projection_by_patch,
                source_response_target,
                source_response_calibration_mode,
                transform_name,
                ridge=source_response_calibration_ridge,
                blend=source_response_calibration_blend,
                epsilon=source_response_calibration_epsilon,
                group_count=source_response_calibration_group_count,
            )
        )
        new_cortex_kernel, branch_projection_by_patch = (
            self._apply_component_norm_gauge(
                new_cortex_kernel,
                branch_projection_by_patch,
                factorization_component_norm_gauge_mode,
                transform_name,
                component_count=(
                    factorization_component_norm_gauge_component_count
                ),
                structure=factorization_component_norm_gauge_structure,
                min_scale=factorization_component_norm_gauge_min_scale,
                max_scale=factorization_component_norm_gauge_max_scale,
                epsilon=factorization_component_norm_gauge_epsilon,
            )
        )
        self._append_factorization_reconstruction_audit(
            source_response_target,
            new_cortex_kernel,
            branch_projection_by_patch,
            transform_name,
            'full_after_gauge',
        )

        return self._finalize_patch_factor_yield(
            cortex_constructor,
            local_kernel_size,
            int(local_stride),
            patch_state,
            new_cortex_kernel,
            branch_projection_by_patch,
            transform_name,
            regularize_afterward=regularize_afterward,
            pooling_settings=pooling_settings,
            pooling_projection_mode=pooling_projection_mode,
            root_direct_mode=root_direct_mode,
            root_projection_scale=root_projection_scale,
            root_learning_mask=root_learning_mask,
            new_cortex_init_amplifier=new_cortex_init_amplifier,
            cover_edges=bool(cover_edges),
        )

    def yield_receiver_grouped_patch_pca_subcortex(
        self,
        cortex_constructor,
        local_kernel_size,
        local_stride,
        components_per_receiver_group,
        receiver_group_count=None,
        PCA_settings=None,
        new_cortex_split_ratio=None,
        regularize_afterward=None,
        pooling_settings=None,
        root_direct_mode=None,
        root_projection_scale=1.0,
        root_learning_mask=None,
    ):
        if self.subcortexs:
            raise ValueError(
                'receiver_grouped_patch_pca_yield currently supports only '
                'a target cortex without existing subcortexs'
            )

        h, w = self.shape
        input_channel = h // (self.kernel_size ** 2)
        if input_channel * (self.kernel_size ** 2) != h:
            raise ValueError(
                'receiver_grouped_patch_pca_yield cannot infer input_channel '
                'from target cortex shape'
            )

        local_kernel_size = int(local_kernel_size)
        local_stride = int(local_stride)
        components_per_receiver_group = int(components_per_receiver_group)
        if local_kernel_size <= 0:
            raise ValueError(
                'receiver_grouped_patch_pca_yield local_kernel_size must be positive'
            )
        if local_stride <= 0:
            raise ValueError(
                'receiver_grouped_patch_pca_yield local_stride must be positive'
            )
        if components_per_receiver_group <= 0:
            raise ValueError(
                'receiver_grouped_patch_pca_yield components_per_receiver_group '
                'must be positive'
            )
        if local_kernel_size > self.kernel_size:
            raise ValueError(
                'receiver_grouped_patch_pca_yield local_kernel_size cannot '
                'exceed target kernel_size'
            )

        if receiver_group_count is None:
            receiver_group_count = int(self.output_channel)
        else:
            receiver_group_count = int(receiver_group_count)
        if receiver_group_count <= 0:
            raise ValueError(
                'receiver_grouped_patch_pca_yield receiver_group_count must be positive'
            )
        if w % receiver_group_count != 0:
            raise ValueError(
                'receiver_grouped_patch_pca_yield receiver_group_count must '
                'divide target receiver count'
            )
        receiver_group_size = w // receiver_group_count

        patch_h = (self.kernel_size - local_kernel_size) // local_stride + 1
        patch_w = (self.kernel_size - local_kernel_size) // local_stride + 1
        patch_num = patch_h * patch_w
        if patch_num <= 0:
            raise ValueError(
                'receiver_grouped_patch_pca_yield produced no local patch positions'
            )

        pca_settings = copy.deepcopy(
            self.PCA_settings if PCA_settings is None else PCA_settings
        )
        pca_settings['remained_variance'] = components_per_receiver_group
        split_ratio = (
            self.new_cortex_split_ratio
            if new_cortex_split_ratio is None
            else float(new_cortex_split_ratio)
        )
        should_regularize = (
            self.regularize_afterward
            if regularize_afterward is None
            else bool(regularize_afterward)
        )
        root_direct_mode = self._get_root_direct_mode(
            root_direct_mode,
            should_regularize
        )
        root_projection_scale = self._get_root_projection_scale(
            root_projection_scale
        )
        if pooling_settings is None:
            pooling_settings = {'mode': 'max', 'kernel_size': 1}
        pooling = CortexPooling(pooling_settings)

        source_weight = self.kernel.weight[:h, :w].detach()
        source_device = source_weight.device
        source_dtype = source_weight.dtype
        source_kernel = source_weight.cpu().numpy().reshape(
            input_channel,
            self.kernel_size,
            self.kernel_size,
            w,
        )

        coverage = np.zeros(
            (input_channel, self.kernel_size, self.kernel_size, 1),
            dtype=source_kernel.dtype,
        )
        patch_slices = []
        for row in range(patch_h):
            top = row * local_stride
            bottom = top + local_kernel_size
            for col in range(patch_w):
                left = col * local_stride
                right = left + local_kernel_size
                patch_slices.append((top, bottom, left, right))
                coverage[:, top:bottom, left:right, :] += 1.0
        coverage = np.maximum(coverage, 1.0)

        patch_columns = []
        for top, bottom, left, right in patch_slices:
            weighted_patch = (
                source_kernel[:, top:bottom, left:right, :]
                / coverage[:, top:bottom, left:right, :]
            )
            patch_columns.append(
                weighted_patch.reshape(
                    input_channel * local_kernel_size * local_kernel_size,
                    w,
                )
            )

        group_results = []
        new_cortex_kernels = []
        for group_index in range(receiver_group_count):
            group_start = group_index * receiver_group_size
            group_end = group_start + receiver_group_size
            group_patch_matrix = np.concatenate(
                [
                    patch_column[:, group_start:group_end]
                    for patch_column in patch_columns
                ],
                axis=1,
            )
            group_kernel, group_right, group_amplifier = self.PCA(
                group_patch_matrix,
                **pca_settings
            )
            if group_kernel.shape[1] <= 0:
                raise ValueError(
                    'receiver_grouped_patch_pca_yield produced no PCA components'
                )
            group_kernel *= split_ratio * float(group_amplifier)
            group_results.append(
                (group_start, group_end, group_kernel, group_right)
            )
            new_cortex_kernels.append(group_kernel)

        new_cortex_kernel = np.concatenate(new_cortex_kernels, axis=1)
        output_channel = int(new_cortex_kernel.shape[1])
        if output_channel <= 0:
            raise ValueError(
                'receiver_grouped_patch_pca_yield produced no PCA components'
            )

        right_by_patch = np.zeros(
            (patch_num, output_channel, w),
            dtype=source_kernel.dtype,
        )
        component_offset = 0
        for group_start, group_end, group_kernel, group_right in group_results:
            group_component_count = int(group_kernel.shape[1])
            group_right_by_patch = group_right.reshape(
                group_component_count,
                patch_num,
                receiver_group_size,
            ).transpose(1, 0, 2)
            right_by_patch[
                :,
                component_offset:component_offset + group_component_count,
                group_start:group_end,
            ] = group_right_by_patch
            component_offset += group_component_count

        reconstructed = np.zeros_like(source_kernel)
        for patch_index, (top, bottom, left, right) in enumerate(patch_slices):
            patch_reconstruction = np.einsum(
                'ic,co->io',
                new_cortex_kernel,
                right_by_patch[patch_index],
            ).reshape(input_channel, local_kernel_size, local_kernel_size, w)
            reconstructed[:, top:bottom, left:right, :] += patch_reconstruction

        if root_direct_mode == 'source_passthrough':
            residual_kernel = source_kernel.reshape(h, w).copy()
        else:
            residual_kernel = (source_kernel - reconstructed).reshape(h, w)
        pooled_right_by_patch = self._pool_patch_root_projection(
            right_by_patch,
            patch_h,
            patch_w,
            pooling,
        )
        new_root_kernel = np.concatenate(
            [
                residual_kernel,
                pooled_right_by_patch.reshape(
                    pooled_right_by_patch.shape[0] * output_channel,
                    w,
                ) * root_projection_scale,
            ],
            axis=0
        )

        self.kernel.weight = torch.as_tensor(
            new_root_kernel,
            dtype=source_dtype,
            device=source_device
        )
        self.kernel.has_subcortexs = True
        self.kernel.output_len = len(self.hidden_nv)
        if should_regularize:
            self.kernel.kernel_regulation()
        self._apply_root_learning_mask(
            root_learning_mask,
            direct_input_len=h
        )

        subcortex_kernel = torch.as_tensor(
            new_cortex_kernel,
            dtype=source_dtype,
            device=source_device
        )
        self.subcortex_counter += 1
        new_cortex = cortex_constructor(
            cortex_id=f'{self.cortex_id}-{self.subcortex_counter}',
            kernel_size=local_kernel_size,
            stride=local_stride,
            input_channel=input_channel,
            output_channel=output_channel,
            kernel=subcortex_kernel,
            init_amplifier=1.0,
            pooling_settings=pooling_settings,
            competition_group_N=1,
        )
        new_cortex = move_to_device(new_cortex, source_device)
        self.subcortexs.append(new_cortex)
        self.kernel.has_subcortexs = len(self.subcortexs) > 0
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
        return new_cortex

    @staticmethod
    def _pool_patch_root_projection(
        right_by_patch,
        patch_h,
        patch_w,
        pooling,
        projection_mode='legacy_mean',
    ):
        if pooling.settings is None:
            return right_by_patch

        projection_mode = (
            'legacy_mean'
            if projection_mode is None
            else str(projection_mode).lower()
        )
        supported_modes = {'legacy_mean', 'spatial_lstsq'}
        if projection_mode not in supported_modes:
            raise ValueError(
                'pooling_projection_mode must be one of '
                f'{sorted(supported_modes)}, got {projection_mode!r}'
            )

        kernel, stride = pooling.get_kernel_stride(patch_h, patch_w)
        pooled_h, pooled_w, _ = pooling.get_pooled_grid(patch_h, patch_w)
        output_channel, output_len = right_by_patch.shape[1:]
        right_grid = right_by_patch.reshape(
            patch_h,
            patch_w,
            output_channel,
            output_len,
        )

        pooled_rows = []
        for row in range(pooled_h):
            top = row * stride[0]
            bottom = top + kernel[0]
            for col in range(pooled_w):
                left = col * stride[1]
                right = left + kernel[1]
                window = right_grid[top:bottom, left:right]
                pooled_rows.append(window.mean(axis=(0, 1)))
        legacy_projection = np.stack(pooled_rows, axis=0)
        if projection_mode == 'legacy_mean':
            return legacy_projection

        if pooling.settings['mode'] != 'mean':
            raise ValueError(
                'pooling_projection_mode=spatial_lstsq requires '
                'pooling_settings.mode=mean because max pooling is nonlinear'
            )

        patch_num = int(patch_h * patch_w)
        pooled_num = int(pooled_h * pooled_w)
        pool_operator = np.zeros(
            (pooled_num, patch_num),
            dtype=right_by_patch.dtype,
        )
        pooled_index = 0
        window_area = float(kernel[0] * kernel[1])
        for row in range(pooled_h):
            top = row * stride[0]
            for col in range(pooled_w):
                left = col * stride[1]
                for window_row in range(top, top + kernel[0]):
                    start = window_row * patch_w + left
                    pool_operator[
                        pooled_index,
                        start:start + kernel[1],
                    ] = 1.0 / window_area
                pooled_index += 1

        design = pool_operator.T
        projection_solver = np.linalg.pinv(design).astype(
            right_by_patch.dtype,
            copy=False,
        )
        flattened_projection = right_by_patch.reshape(
            patch_num,
            output_channel * output_len,
        )
        pooled_projection = projection_solver @ flattened_projection
        return pooled_projection.reshape(
            pooled_num,
            output_channel,
            output_len,
        )

    def _get_root_direct_mode(self, root_direct_mode, should_regularize):
        mode = (
            'reconstruction_residual'
            if root_direct_mode is None
            else str(root_direct_mode)
        )
        allowed_modes = {
            'factorized_only',
            'reconstruction_residual',
            'source_passthrough',
        }
        if mode not in allowed_modes:
            raise ValueError(
                'root_direct_mode must be one of '
                f'{sorted(allowed_modes)}, got {mode!r}'
            )
        if (
            mode == 'source_passthrough'
            and should_regularize
            and self.kernel.no_skip_links
        ):
            raise ValueError(
                'root_direct_mode=source_passthrough requires '
                'regularize_afterward=false when no_skip_links is true.'
            )
        return mode

    @staticmethod
    def _get_root_projection_scale(root_projection_scale):
        scale = float(root_projection_scale)
        if scale < 0.0:
            raise ValueError('root_projection_scale must be >= 0')
        return scale

    def _apply_root_learning_mask(self, root_learning_mask, direct_input_len):
        if root_learning_mask is None:
            self.set_kernel_learning_mask(None)
            return
        mode = str(root_learning_mask)
        if mode in {'all', 'none', 'false', 'False'}:
            self.set_kernel_learning_mask(None)
            return
        allowed_modes = {
            'branch_projection_only',
            'direct_input_only',
        }
        if mode not in allowed_modes:
            raise ValueError(
                'root_learning_mask must be one of: '
                'branch_projection_only, direct_input_only, all, none'
            )
        if (
            mode == 'branch_projection_only'
            and direct_input_len >= self.kernel.weight.shape[0]
        ):
            raise ValueError(
                'root_learning_mask=branch_projection_only requires '
                'at least one branch-projection input row'
            )
        if mode == 'direct_input_only' and direct_input_len <= 0:
            raise ValueError(
                'root_learning_mask=direct_input_only requires '
                'at least one direct-input row'
            )
        mask = torch.zeros_like(self.kernel.weight)
        if mode == 'branch_projection_only':
            mask[direct_input_len:, :] = 1.0
        else:
            mask[:direct_input_len, :] = 1.0
        self.set_kernel_learning_mask(mask)
