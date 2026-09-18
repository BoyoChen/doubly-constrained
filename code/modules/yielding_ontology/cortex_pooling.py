import torch
import torch.nn.functional as F


class CortexPooling:
    def __init__(self, pooling_settings=None):
        self.settings = self.normalize_settings(pooling_settings)

    @staticmethod
    def normalize_settings(pooling_settings):
        if not pooling_settings:
            return None

        mode = pooling_settings.get('mode', 'max')
        if mode not in ('max', 'mean'):
            raise ValueError('pooling_settings.mode must be "max" or "mean"')

        kernel_size = pooling_settings.get('kernel_size', None)
        if kernel_size is None:
            raise ValueError('pooling_settings.kernel_size is required')

        if isinstance(kernel_size, str):
            if kernel_size != 'global':
                raise ValueError('pooling_settings.kernel_size string must be "global"')
            stride = pooling_settings.get('stride', 1)
        else:
            stride = pooling_settings.get('stride', kernel_size)

        channel_phase_demultiplex = bool(
            pooling_settings.get('channel_phase_demultiplex', False)
        )
        channel_phase_period = pooling_settings.get(
            'channel_phase_period', None
        )
        if channel_phase_period is not None:
            channel_phase_period = int(channel_phase_period)
            if channel_phase_period <= 0:
                raise ValueError('channel_phase_period must be positive')
        if channel_phase_demultiplex and mode != 'max':
            raise ValueError(
                'channel_phase_demultiplex requires max pooling'
            )

        return {
            'mode': mode,
            'kernel_size': kernel_size,
            'stride': stride,
            'channel_phase_demultiplex': channel_phase_demultiplex,
            'channel_phase_period': channel_phase_period,
        }

    @staticmethod
    def as_hw_pair(value):
        if isinstance(value, (list, tuple)):
            if len(value) != 2:
                raise ValueError('pooling kernel_size/stride list must have length 2')
            return int(value[0]), int(value[1])
        return int(value), int(value)

    def get_kernel_stride(self, patch_h, patch_w):
        if self.settings is None:
            return None, None

        kernel_size = self.settings['kernel_size']
        if kernel_size == 'global':
            kernel = (patch_h, patch_w)
            stride = (1, 1)
        else:
            kernel = self.as_hw_pair(kernel_size)
            stride = self.as_hw_pair(self.settings['stride'])

        if kernel[0] <= 0 or kernel[1] <= 0 or stride[0] <= 0 or stride[1] <= 0:
            raise ValueError('pooling kernel_size and stride must be positive')
        if kernel[0] > patch_h or kernel[1] > patch_w:
            raise ValueError('pooling kernel_size cannot exceed cortex output grid')
        return kernel, stride

    def get_pooled_grid(self, patch_h, patch_w):
        if self.settings is None:
            return patch_h, patch_w, patch_h * patch_w

        kernel, stride = self.get_kernel_stride(patch_h, patch_w)
        pooled_h = (patch_h - kernel[0]) // stride[0] + 1
        pooled_w = (patch_w - kernel[1]) // stride[1] + 1
        return pooled_h, pooled_w, pooled_h * pooled_w

    def apply_output_pooling(self, output, patch_h, patch_w):
        if self.settings is None:
            return output

        *outer_dims, patch_num, channels = output.shape
        if patch_num != patch_h * patch_w:
            raise ValueError('output patch dimension does not match cached cortex grid')

        kernel, stride = self.get_kernel_stride(patch_h, patch_w)
        output_2d = output.reshape(-1, patch_h, patch_w, channels).permute(0, 3, 1, 2)
        if self.settings.get('channel_phase_demultiplex', False):
            if kernel != stride or kernel[0] * kernel[1] <= 1:
                raise ValueError(
                    'channel_phase_demultiplex requires non-overlapping '
                    'pooling windows larger than 1x1'
                )
            phase_count = kernel[0] * kernel[1]
            phase_period = self.settings.get('channel_phase_period', None)
            if phase_period is None:
                phase_period = channels
            if channels % phase_period != 0:
                raise ValueError(
                    'output channels must be divisible by '
                    'channel_phase_period'
                )
            channel_phase = (
                (torch.arange(channels, device=output.device) % phase_period)
                * phase_count
            ) // phase_period
            row_phase = torch.arange(patch_h, device=output.device) % kernel[0]
            col_phase = torch.arange(patch_w, device=output.device) % kernel[1]
            spatial_phase = (
                row_phase[:, None] * kernel[1] + col_phase[None, :]
            )
            phase_mask = channel_phase[:, None, None] == spatial_phase[None, :, :]
            output_2d = output_2d.masked_fill(~phase_mask[None, :, :, :], 0)
        if self.settings['mode'] == 'max':
            pooled = F.max_pool2d(output_2d, kernel_size=kernel, stride=stride)
        else:
            pooled = F.avg_pool2d(output_2d, kernel_size=kernel, stride=stride)

        pooled_h, pooled_w = pooled.shape[-2:]
        return pooled.permute(0, 2, 3, 1).reshape(
            *outer_dims, pooled_h * pooled_w, channels
        )
