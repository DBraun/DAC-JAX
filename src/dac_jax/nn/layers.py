import math
from typing import Optional, Sequence, Union

import flax.linen as nn
from flax.typing import (
  Array,
  Dtype,
  PrecisionLike,
  PaddingLike,
  Initializer,
)
import jax
import jax.numpy as jnp

from .weight_norm import MyWeightNorm


def default_stride(strides):
    if strides is None:
        return 1
    if isinstance(strides, int):
        return strides
    return strides[0]


def default_kernel_dilation(kernel_dilation):
    if kernel_dilation is None:
        return 1
    if isinstance(kernel_dilation, int):
        return kernel_dilation
    return kernel_dilation[0]


def default_kernel_size(kernel_size):
    if kernel_size is None:
        return 1
    if isinstance(kernel_size, int):
        return kernel_size
    return kernel_size[0]


def conv_to_delay(s, d, k, L):
    L = (L - 1) * s + d * (k - 1) + 1
    L = math.ceil(L)
    return L


def convtranspose_to_delay(s, d, k, L):
    L = ((L - d * (k - 1) - 1) / s) + 1
    L = math.ceil(L)
    return L


def conv_to_output_length(s, d, k, L):
    L = ((L - d * (k - 1) - 1) / s) + 1
    L = math.floor(L)
    return L


def convtranspose_to_output_length(s, d, k, L):
    L = (L - 1) * s + d * (k - 1) + 1
    L = math.floor(L)
    return L


class LeakyReLU(nn.Module):

    negative_slope: float = .01

    @nn.compact
    def __call__(self, x):
        return nn.leaky_relu(x, negative_slope=self.negative_slope)


class WNConv1d(nn.Module):

    # https://github.com/descriptinc/descript-audio-codec/blob/c7cfc5d2647e26471dc394f95846a0830e7bec34/dac/model/dac.py#L18-L21
    # kernel_init: nn.initializers.variance_scaling(.02, mode='fan_in', distribution='truncated_normal', dtype=jnp.float32)

    features: int
    kernel_size: Union[int, Sequence[int]]
    strides: Union[None, int, Sequence[int]] = 1
    padding: PaddingLike = 'SAME'
    input_dilation: Union[None, int, Sequence[int]] = 1
    kernel_dilation: Union[None, int, Sequence[int]] = 1
    feature_group_count: int = 1
    use_bias = True
    mask: Optional[Array] = None
    dtype: Optional[Dtype] = None
    param_dtype: Dtype = jnp.float32
    precision: PrecisionLike = None
    # https://github.com/descriptinc/descript-audio-codec/blob/c7cfc5d2647e26471dc394f95846a0830e7bec34/dac/model/dac.py#L18-L21
    # https://github.com/google/flax/issues/4091
    kernel_init: nn.initializers.Initializer = jax.nn.initializers.truncated_normal(.02, lower=-2/.02, upper=2/.02)
    bias_init: nn.initializers.Initializer = nn.initializers.zeros

    @nn.compact
    def __call__(self, x):
        conv = nn.Conv(
            features=self.features,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=self.padding,
            input_dilation=self.input_dilation,
            kernel_dilation=self.kernel_dilation,
            feature_group_count=self.feature_group_count,
            use_bias=self.use_bias,
            mask=self.mask,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init
        )
        # MyWeightNorm initializes itself as if the conv had been initialized the way PyTorch would have instead
        # of what we did (truncated normal).
        block = MyWeightNorm(conv)
        x = block(x)
        return x

    @staticmethod
    def delay(s, d, k, L):
        s = default_stride(s)
        d = default_kernel_dilation(d)
        k = default_kernel_size(k)
        return conv_to_delay(s, d, k, L)

    @staticmethod
    def output_length(s, d, k, L):
        s = default_stride(s)
        d = default_kernel_dilation(d)
        k = default_kernel_size(k)
        return conv_to_output_length(s, d, k, L)


class WNConvTranspose1d(nn.Module):

    # Note: set tranpose_kernel=True because PyTorch's kernels are transposed relative to JAX.
    # https://flax.readthedocs.io/en/latest/guides/converting_and_upgrading/convert_pytorch_to_flax.html#transposed-convolutions

    features: int
    kernel_size: Union[int, Sequence[int]]
    strides: Optional[Sequence[int]] = None
    padding: PaddingLike = 'SAME'
    kernel_dilation: Optional[Sequence[int]] = None
    use_bias = True
    mask: Optional[Array] = None
    dtype: Optional[Dtype] = None
    param_dtype: Dtype = jnp.float32
    precision: PrecisionLike = None
    kernel_init: Initializer = nn.initializers.kaiming_uniform()  # to match PyTorch
    bias_init: Initializer = nn.initializers.zeros_init()
    transpose_kernel = True  # note: non-standard

    @nn.compact
    def __call__(self, x):
        conv = nn.ConvTranspose(
            features=self.features,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=self.padding,
            kernel_dilation=self.kernel_dilation,
            use_bias=self.use_bias,
            mask=self.mask,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            transpose_kernel=self.transpose_kernel
        )
        block = nn.WeightNorm(conv)  # note: we use the epsilon default
        x = block(x)
        return x

    @staticmethod
    def delay(s, d, k, L):
        s = default_stride(s)
        d = default_kernel_dilation(d)
        k = default_kernel_size(k)
        return convtranspose_to_delay(s, d, k, L)

    @staticmethod
    def output_length(s, d, k, L):
        s = default_stride(s)
        d = default_kernel_dilation(d)
        k = default_kernel_size(k)
        return convtranspose_to_output_length(s, d, k, L)


class Snake1d(nn.Module):

    channels: int

    @nn.compact
    def __call__(self, x):
        alpha = self.param('alpha', nn.initializers.ones, (1, 1, self.channels))
        x = x + jnp.reciprocal(alpha + 1e-9) * jnp.square(jnp.sin(alpha * x))
        return x
