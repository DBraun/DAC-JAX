# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import field
import math
import typing as tp
import warnings

from einops import rearrange
from jax import numpy as jnp
from flax import linen as nn

from dac_jax.nn.layers import make_initializer


CONV_NORMALIZATIONS = frozenset(
    ["none", "weight_norm", "spectral_norm", "time_group_norm"]
)


def apply_parametrization_norm(module: nn.Module, norm: str = "none"):
    assert norm in CONV_NORMALIZATIONS
    if norm == "weight_norm":
        # why we use scale_init: https://github.com/google/flax/issues/4138
        scale_init = nn.initializers.constant(1 / jnp.sqrt(3))
        return nn.WeightNorm(module, scale_init=scale_init)
    elif norm == "spectral_norm":
        return nn.SpectralNorm(module)
    else:
        # We already check was in CONV_NORMALIZATION, so any other choice
        # doesn't need reparametrization.
        return module


def get_norm_module(
    module: nn.Module, causal: bool = False, norm: str = "none", **norm_kwargs
):
    """Return the proper normalization module. If causal is True, this will ensure the returned
    module is causal, or return an error if the normalization doesn't support causal evaluation.
    """
    assert norm in CONV_NORMALIZATIONS
    if norm == "time_group_norm":
        if causal:
            raise ValueError("GroupNorm doesn't support causal evaluation.")
        assert isinstance(module, nn.Conv)
        return nn.GroupNorm(num_groups=1, **norm_kwargs)
    else:
        return lambda x: x


def get_extra_padding_for_conv1d(
    x: jnp.ndarray, kernel_size: int, stride: int, padding_total: int = 0
) -> int:
    """See `pad_for_conv1d`."""
    length = x.shape[-2]
    n_frames = (length - kernel_size + padding_total) / stride + 1
    ideal_length = (math.ceil(n_frames) - 1) * stride + (kernel_size - padding_total)
    return ideal_length - length


def pad_for_conv1d(
    x: jnp.ndarray, kernel_size: int, stride: int, padding_total: int = 0
):
    """Pad for a convolution to make sure that the last window is full.
    Extra padding is added at the end. This is required to ensure that we can rebuild
    an output of the same length, as otherwise, even with padding, some time steps
    might get removed.
    For instance, with total padding = 4, kernel size = 4, stride = 2:
        0 0 1 2 3 4 5 0 0   # (0s are padding)
        1   2   3           # (output frames of a convolution, last 0 is never used)
        0 0 1 2 3 4 5 0     # (output of tr. conv., but pos. 5 is going to get removed as padding)
            1 2 3 4         # once you removed padding, we are missing one time step !
    """
    extra_padding = get_extra_padding_for_conv1d(x, kernel_size, stride, padding_total)
    return jnp.pad(x, ((0, 0), (0, extra_padding), (0, 0)))


def pad1d(
    x: jnp.ndarray,
    paddings: tp.Tuple[int, int],
    mode: str = "constant",
    value: float = 0.0,
):
    """Tiny wrapper around F.pad, just to allow for reflect padding on small input.
    If this is the case, we insert extra 0 padding to the right before the reflection happen.
    """
    length = x.shape[-2]
    padding_left, padding_right = paddings
    assert padding_left >= 0 and padding_right >= 0, (padding_left, padding_right)
    if mode == "constant":
        pad_kwargs = {"constant_values": value}
    else:
        pad_kwargs = {}
    if mode == "reflect":
        max_pad = max(padding_left, padding_right)
        extra_pad = 0
        if length <= max_pad:
            extra_pad = max_pad - length + 1
            x = jnp.pad(x, ((0, 0), (0, extra_pad), (0, 0)))
        padded = jnp.pad(
            x, pad_width=((0, 0), paddings, (0, 0)), mode=mode, **pad_kwargs
        )
        end = padded.shape[-2] - extra_pad
        return padded[:, :end, :]
    else:
        return jnp.pad(x, pad_width=((0, 0), paddings, (0, 0)), mode=mode, **pad_kwargs)


def unpad1d(x: jnp.ndarray, paddings: tp.Tuple[int, int]):
    """Remove padding from x, handling properly zero padding. Only for 1d!"""
    padding_left, padding_right = paddings
    assert padding_left >= 0 and padding_right >= 0, (padding_left, padding_right)
    assert (padding_left + padding_right) <= x.shape[-2]
    end = x.shape[-2] - padding_right
    return x[:, padding_left:end, :]


class NormConv1d(nn.Conv):
    """Wrapper around Conv and normalization applied to this conv
    to provide a uniform interface across normalization approaches.
    """

    causal: bool = False
    norm: str = "none"
    norm_kwargs: tp.Dict[str, tp.Any] = field(default_factory=lambda: {})

    @nn.compact
    def __call__(self, x):

        # note: we just ignore whatever self.kernel_init is
        kernel_init = make_initializer(
            x.shape[-1],
            self.features,
            self.kernel_size,
            self.feature_group_count,
            mode="fan_in",
        )

        if self.use_bias:
            # note: we just ignore whatever self.bias_init is
            bias_init = make_initializer(
                x.shape[-1],
                self.features,
                self.kernel_size,
                self.feature_group_count,
                mode="fan_in",
            )
        else:
            bias_init = None

        conv = nn.Conv(
            features=self.features,
            kernel_size=(self.kernel_size,),
            strides=(self.strides,),
            padding="VALID",
            input_dilation=self.input_dilation,
            kernel_dilation=self.kernel_dilation,
            feature_group_count=self.feature_group_count,
            use_bias=self.use_bias,
            mask=self.mask,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
            kernel_init=kernel_init,
            bias_init=bias_init,
        )
        conv = apply_parametrization_norm(conv, self.norm)
        norm = get_norm_module(conv, self.causal, self.norm, **self.norm_kwargs)
        x = conv(x)
        x = norm(x)
        return x


class NormConv2d(nn.Conv):
    """Wrapper around Conv and normalization applied to this conv
    to provide a uniform interface across normalization approaches.
    """

    norm: str = "none"
    norm_kwargs: tp.Dict[str, tp.Any] = field(default_factory=lambda: {})

    @nn.compact
    def __call__(self, x):

        # note: we just ignore whatever self.kernel_init is
        kernel_init = make_initializer(
            x.shape[-1],
            self.features,
            self.kernel_size,
            self.feature_group_count,
            mode="fan_in",
        )

        if self.use_bias:
            # note: we just ignore whatever self.bias_init is
            bias_init = make_initializer(
                x.shape[-1],
                self.features,
                self.kernel_size,
                self.feature_group_count,
                mode="fan_in",
            )
        else:
            bias_init = None

        conv = nn.Conv(
            features=self.features,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding="VALID",
            input_dilation=self.input_dilation,
            kernel_dilation=self.kernel_dilation,
            feature_group_count=self.feature_group_count,
            use_bias=self.use_bias,
            mask=self.mask,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
            kernel_init=kernel_init,
            bias_init=bias_init,
        )
        conv = apply_parametrization_norm(conv, self.norm)
        norm = get_norm_module(conv, causal=False, norm=self.norm, **self.norm_kwargs)
        x = conv(x)
        x = norm(x)
        return x


class NormConvTranspose1d(nn.ConvTranspose):
    """Wrapper around ConvTranspose1d and normalization applied to this conv
    to provide a uniform interface across normalization approaches.
    """

    causal: bool = False
    norm: str = "none"
    norm_kwargs: tp.Dict[str, tp.Any] = field(default_factory=lambda: {})

    @nn.compact
    def __call__(self, x):
        groups = 1
        # note: we just ignore whatever self.kernel_init is
        kernel_init = make_initializer(
            x.shape[-1],
            self.features,
            self.kernel_size,
            groups,
            mode="fan_out",
        )

        if self.use_bias:
            # note: we just ignore whatever self.bias_init is
            bias_init = make_initializer(
                x.shape[-1],
                self.features,
                self.kernel_size,
                groups,
                mode="fan_out",
            )
        else:
            bias_init = None

        convtr = nn.ConvTranspose(
            features=self.features,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding="VALID",
            kernel_dilation=self.kernel_dilation,
            use_bias=self.use_bias,
            mask=self.mask,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
            kernel_init=kernel_init,
            bias_init=bias_init,
            transpose_kernel=True,  # note: this helps us load weights from PyTorch
        )
        convtr = apply_parametrization_norm(convtr, self.norm)
        norm = get_norm_module(convtr, self.causal, self.norm, **self.norm_kwargs)
        x = convtr(x)
        x = norm(x)
        return x


class NormConvTranspose2d(nn.ConvTranspose):
    """Wrapper around ConvTranspose2d and normalization applied to this conv
    to provide a uniform interface across normalization approaches.
    """

    norm: str = "none"
    norm_kwargs: tp.Dict[str, tp.Any] = field(default_factory=lambda: {})

    @nn.compact
    def __call__(self, x):
        groups = 1
        # note: we just ignore whatever self.kernel_init is
        kernel_init = make_initializer(
            x.shape[-1],
            self.features,
            self.kernel_size,
            groups,
            mode="fan_out",
        )

        if self.use_bias:
            # note: we just ignore whatever self.bias_init is
            bias_init = make_initializer(
                x.shape[-1],
                self.features,
                self.kernel_size,
                groups,
                mode="fan_out",
            )
        else:
            bias_init = None

        convtr = nn.ConvTranspose(
            features=self.features,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding="VALID",
            kernel_dilation=self.kernel_dilation,
            use_bias=self.use_bias,
            mask=self.mask,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
            kernel_init=kernel_init,
            bias_init=bias_init,
            transpose_kernel=True,  # note: this helps us load weights from PyTorch
        )
        convtr = apply_parametrization_norm(convtr, self.norm)
        norm = get_norm_module(convtr, causal=False, norm=self.norm, **self.norm_kwargs)
        x = convtr(x)
        x = norm(x)
        return x


class StreamableConv1d(nn.Module):
    """Conv1d with some builtin handling of asymmetric or causal padding
    and normalization.
    """

    out_channels: int
    kernel_size: int
    stride: int = 1
    dilation: int = 1
    groups: int = 1
    bias: bool = True
    causal: bool = False
    norm: str = "none"
    norm_kwargs: tp.Dict[str, tp.Any] = field(default_factory=lambda: {})
    pad_mode: str = "reflect"

    def __post_init__(self) -> None:
        # warn user on unusual setup between dilation and stride
        if self.stride > 1 and self.dilation > 1:
            warnings.warn(
                "StreamableConv1d has been initialized with stride > 1 and dilation > 1"
                f" (kernel_size={self.kernel_size} stride={self.stride}, dilation={self.dilation})."
            )
        super().__post_init__()

    @nn.compact
    def __call__(self, x):
        conv = NormConv1d(
            self.out_channels,
            kernel_size=self.kernel_size,
            strides=self.stride,
            kernel_dilation=self.dilation,
            feature_group_count=self.groups,
            use_bias=self.bias,
            causal=self.causal,
            norm=self.norm,
            norm_kwargs=self.norm_kwargs,
        )
        B, T, C = x.shape
        kernel_size = conv.kernel_size
        stride = conv.strides
        dilation = conv.kernel_dilation
        kernel_size = (
            kernel_size - 1
        ) * dilation + 1  # effective kernel size with dilations
        padding_total = kernel_size - stride
        extra_padding = get_extra_padding_for_conv1d(
            x, kernel_size, stride, padding_total
        )
        if self.causal:
            # Left padding for causal
            x = pad1d(x, (padding_total, extra_padding), mode=self.pad_mode)
        else:
            # Asymmetric padding required for odd strides
            padding_right = padding_total // 2
            padding_left = padding_total - padding_right
            x = pad1d(
                x, (padding_left, padding_right + extra_padding), mode=self.pad_mode
            )
        y = conv(x)
        return y


class StreamableConvTranspose1d(nn.Module):
    """ConvTranspose1d with some builtin handling of asymmetric or causal padding
    and normalization.
    """

    out_channels: int
    kernel_size: int
    stride: int = 1
    causal: bool = False
    norm: str = "none"
    trim_right_ratio: float = 1.0
    norm_kwargs: tp.Dict[str, tp.Any] = field(default_factory=lambda: {})

    def __post_init__(self):
        assert (
            self.causal or self.trim_right_ratio == 1.0
        ), "`trim_right_ratio` != 1.0 only makes sense for causal convolutions"
        assert self.trim_right_ratio >= 0.0 and self.trim_right_ratio <= 1.0
        super().__post_init__()

    @nn.compact
    def __call__(self, x):
        convtr = NormConvTranspose1d(
            self.out_channels,
            kernel_size=self.kernel_size,
            strides=self.stride,
            causal=self.causal,
            norm=self.norm,
            norm_kwargs=self.norm_kwargs,
        )
        kernel_size = convtr.kernel_size
        stride = convtr.strides
        padding_total = kernel_size - stride

        y = convtr(x)

        # We will only trim fixed padding. Extra padding from `pad_for_conv1d` would be
        # removed at the very end, when keeping only the right length for the output,
        # as removing it here would require also passing the length at the matching layer
        # in the encoder.
        if self.causal:
            # Trim the padding on the right according to the specified ratio
            # if trim_right_ratio = 1.0, trim everything from right
            padding_right = math.ceil(padding_total * self.trim_right_ratio)
            padding_left = padding_total - padding_right
            y = unpad1d(y, (padding_left, padding_right))
        else:
            # Asymmetric padding required for odd strides
            padding_right = padding_total // 2
            padding_left = padding_total - padding_right
            y = unpad1d(y, (padding_left, padding_right))
        return y


class StreamableLSTM(nn.Module):
    """LSTM without worrying about the hidden state, nor the layout of the data.
    Expects input as convolutional layout.
    """

    dimension: int
    num_layers: int = 2
    skip: int = 1  # bool

    @nn.compact
    def __call__(self, x):
        y = x
        for _ in range(self.num_layers):
            y = nn.RNN(nn.LSTMCell(self.dimension))(y)

        if self.skip:
            y = y + x

        return y
