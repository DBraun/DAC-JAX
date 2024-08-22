import math
import flax.linen as nn

import jax
import jax.numpy as jnp


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


def make_initializer(in_channels, out_channels, kernel_size, groups, mode="fan_in"):
    # https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
    if mode == "fan_in":
        c = in_channels
    elif mode == "fan_out":
        c = out_channels
    else:
        raise ValueError(f"Unexpected mode: {mode}")
    k = groups / (c * jnp.prod(jnp.array(kernel_size)))
    scale = jnp.sqrt(k)
    return lambda key, shape, dtype: jax.random.uniform(key, shape, minval=-scale, maxval=scale, dtype=dtype)


class WNConv1d(nn.Conv):

    @nn.compact
    def __call__(self, x):
        # https://github.com/descriptinc/descript-audio-codec/blob/c7cfc5d2647e26471dc394f95846a0830e7bec34/dac/model/dac.py#L18-L21
        # https://github.com/google/flax/issues/4091
        # Note: we are just ignoring whatever self.kernel_init and self.bias_init are.
        kernel_init = jax.nn.initializers.truncated_normal(.02, lower=-2 / .02, upper=2 / .02)
        bias_init = nn.initializers.zeros

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
            kernel_init=kernel_init,
            bias_init=bias_init
        )
        scale_init = nn.initializers.constant(1/jnp.sqrt(3))
        block = nn.WeightNorm(conv, scale_init=scale_init)
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


class WNConvTranspose1d(nn.ConvTranspose):

    @nn.compact
    def __call__(self, x):

        groups = 1
        # note: we just ignore whatever self.kernel_init is
        kernel_init = make_initializer(
            x.shape[-1], self.features, self.kernel_size, groups, mode="fan_out",
        )

        if self.use_bias:
            # note: we just ignore whatever self.bias_init is
            bias_init = make_initializer(
                x.shape[-1], self.features, self.kernel_size, groups, mode="fan_out",
            )
        else:
            bias_init = None

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
            kernel_init=kernel_init,
            bias_init=bias_init,
            transpose_kernel=True  # note: this helps us load weights from PyTorch
        )
        scale_init = nn.initializers.constant(1 / jnp.sqrt(3))
        block = nn.WeightNorm(conv, scale_init=scale_init)
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
