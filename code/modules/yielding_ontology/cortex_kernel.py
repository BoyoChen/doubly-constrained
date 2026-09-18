import torch
from modules.utils import remove_nan


class CortexKernel:
    def __init__(
        self,
        input_len,
        output_len,
        subcortexs,
        receptive_height,
        receptive_width,
        init_std,
        init_mean,
        no_skip_links,
        non_negative_weights,
        normalize_settings,
        minimal_weight,
        kernel=None
    ):
        self.sender_block_lengths = self._get_sender_block_lengths(
            input_len,
            subcortexs,
            receptive_height,
            receptive_width,
        )
        self.weight = self._init_weight(
            input_len,
            output_len,
            subcortexs,
            receptive_height,
            receptive_width,
            init_std,
            init_mean,
            kernel
        )
        self.input_len = input_len
        self.output_len = output_len
        self.has_subcortexs = len(subcortexs) > 0
        self.no_skip_links = no_skip_links
        self.non_negative_weights = non_negative_weights
        self.normalize_settings = normalize_settings
        self.minimal_weight = minimal_weight
        # A branch-only cortex must be branch-only from its very first forward,
        # not only after the first STDP regulation pass.
        if self.has_subcortexs and self.no_skip_links:
            self.weight[:self.input_len, :self.output_len] = 0
        self.normalize_count_up = 0
        self.normalization_operation_counts = {
            'post': 0,
            'pre': 0,
            'global_l1_sham': 0,
            'none': 0,
        }
        self.last_normalization_operations = ()
        self.last_global_l1_sham_target = None

    @staticmethod
    def _get_sender_block_lengths(
        input_len,
        subcortexs,
        receptive_height,
        receptive_width,
    ):
        """Return direct and structural-branch row counts in kernel order."""
        block_lengths = [int(input_len)]
        for subcortex in subcortexs:
            patch_h, patch_w, _ = subcortex._get_patch_grid(
                receptive_height,
                receptive_width,
            )
            _, _, pooled_patch_num = subcortex.pooling.get_pooled_grid(
                patch_h,
                patch_w,
            )
            block_lengths.append(
                int(pooled_patch_num * len(subcortex.output_nv))
            )
        return tuple(block_lengths)

    def _apply_global_l1_sham(self, use_shape_ratio):
        if int(self.normalize_settings['norm_p']) != 1:
            raise ValueError('global_l1_sham requires norm_p=1')

        row_l1 = self.weight.abs().sum(dim=1)
        active_row_count = int((row_l1 > 0).sum().item())
        per_row_target = self.shape_ratio if use_shape_ratio else 1.0
        target = float(active_row_count) * float(per_row_target)
        current = self.weight.abs().sum()
        if target == 0.0:
            self.weight = torch.zeros_like(self.weight)
        elif current > 0:
            self.weight = self.weight * (target / current)
        # A global scalar cannot create a direction from an all-zero matrix.
        # In that degenerate case the matrix remains zero and the audit target
        # records the unmet hypothetical pre-normalization mass.
        self.last_global_l1_sham_target = target

    def _normalize_per_sender_blockwise(self, norm_p, use_shape_ratio):
        """Normalize rows while conserving each structural source block."""
        if sum(self.sender_block_lengths) != int(self.weight.shape[0]):
            raise RuntimeError(
                'sender block lengths do not match kernel rows: '
                f'{self.sender_block_lengths} vs {tuple(self.weight.shape)}'
            )
        start = 0
        for block_length in self.sender_block_lengths:
            end = start + block_length
            block = torch.nn.functional.normalize(
                self.weight[start:end], p=norm_p, dim=1
            )
            if use_shape_ratio:
                block *= float(self.output_len) / float(block_length)
            self.weight[start:end] = block
            start = end

    def _normalize_per_receiver_blockwise(self, norm_p):
        """Normalize columns independently inside each structural source block."""
        if sum(self.sender_block_lengths) != int(self.weight.shape[0]):
            raise RuntimeError(
                'sender block lengths do not match kernel rows: '
                f'{self.sender_block_lengths} vs {tuple(self.weight.shape)}'
            )
        start = 0
        for block_length in self.sender_block_lengths:
            end = start + block_length
            self.weight[start:end] = torch.nn.functional.normalize(
                self.weight[start:end], p=norm_p, dim=0
            )
            start = end

    def _init_weight(
        self,
        input_len,
        output_len,
        subcortexs,
        receptive_height,
        receptive_width,
        init_std,
        init_mean,
        kernel
    ):
        if kernel is not None:
            return kernel

        kernel_shape = [input_len, output_len]
        for subcortex in subcortexs:
            patch_h, patch_w, _ = subcortex._get_patch_grid(
                receptive_height,
                receptive_width
            )
            _, _, pooled_patch_num = subcortex.pooling.get_pooled_grid(patch_h, patch_w)
            kernel_shape[0] += pooled_patch_num * len(subcortex.output_nv)
        return torch.randn(kernel_shape) * init_std + init_mean

    @property
    def shape_ratio(self):
        # shape of kernel, not input/output.
        kernel_input_len, kernel_output_len = self.weight.shape

        if self.has_subcortexs and self.no_skip_links:
            return float(kernel_output_len) / (kernel_input_len - self.input_len)
        return float(kernel_output_len) / kernel_input_len

    def normalize_kernel(self):
        # Source checkpoints can predate the operation audit counters.
        if not hasattr(self, 'normalization_operation_counts'):
            self.normalization_operation_counts = {
                'post': 0, 'pre': 0, 'global_l1_sham': 0, 'none': 0,
            }
        dim_0_freq = self.normalize_settings['dim_0_freq']
        dim_1_freq = self.normalize_settings['dim_1_freq']
        freq_diff = self.normalize_settings['freq_diff']
        out_weight_len = self.normalize_settings['out_weight_len']
        norm_p = self.normalize_settings['norm_p']
        use_shape_ratio = self.normalize_settings['use_shape_ratio']
        dim_0_mode = str(
            self.normalize_settings.get('dim_0_mode', 'per_receiver')
        )
        if dim_0_mode not in {
            'per_receiver', 'per_receiver_blockwise'
        }:
            raise ValueError(
                'normalize_settings.dim_0_mode must be one of '
                'per_receiver/per_receiver_blockwise'
            )
        dim_1_mode = str(
            self.normalize_settings.get('dim_1_mode', 'per_sender')
        )
        if dim_1_mode not in {
            'per_sender', 'per_sender_blockwise', 'global_l1_sham'
        }:
            raise ValueError(
                'normalize_settings.dim_1_mode must be one of '
                'per_sender/per_sender_blockwise/global_l1_sham'
            )
        self.normalize_count_up += 1
        self.normalize_count_up %= 10000
        normalized_flag = False
        operations = []

        # dim 0 norm = postsynaptic normalization = per neuron input is const
        if dim_0_freq > 0 and (self.normalize_count_up % dim_0_freq == 0):
            if dim_0_mode == 'per_receiver':
                self.weight = torch.nn.functional.normalize(
                    self.weight, p=norm_p, dim=0
                )
            else:
                self._normalize_per_receiver_blockwise(norm_p)
            normalized_flag = True
            operations.append('post')

        # dim 1 norm = presynaptic normalization = per neuron output is const
        if dim_1_freq > 0 and ((self.normalize_count_up + freq_diff) % dim_1_freq == 0):
            if dim_1_mode == 'per_sender':
                self.weight = torch.nn.functional.normalize(
                    self.weight, p=norm_p, dim=1
                )
                if use_shape_ratio:
                    self.weight *= self.shape_ratio
                operations.append('pre')
            elif dim_1_mode == 'per_sender_blockwise':
                self._normalize_per_sender_blockwise(
                    norm_p,
                    use_shape_ratio,
                )
                operations.append('pre')
            else:
                self._apply_global_l1_sham(use_shape_ratio)
                operations.append('global_l1_sham')
            normalized_flag = True

        if (out_weight_len != 1.0) and normalized_flag:
            self.weight *= out_weight_len

        if not operations:
            operations.append('none')
        for operation in operations:
            self.normalization_operation_counts[operation] += 1
        self.last_normalization_operations = tuple(operations)

    def kernel_regulation(self):
        # skip links needs to be removed before normalization
        if self.has_subcortexs and self.no_skip_links:
            self.weight[:self.input_len, :self.output_len] = 0

        if self.non_negative_weights:
            self.weight = torch.relu(self.weight)

        self.normalize_kernel()
        self.weight = torch.where(
            condition=torch.abs(self.weight) >= self.minimal_weight,
            input=self.weight,
            other=0.0
        )
        self.weight = remove_nan(self.weight, 1e-12)
