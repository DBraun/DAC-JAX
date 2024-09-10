# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from dataclasses import field
import typing as tp

from einops import rearrange
from flax import linen as nn
from jax import numpy as jnp
import numpy as np

from dac_jax.nn.encodec_layers import (
    StreamableConv1d,
    StreamableConvTranspose1d,
    StreamableLSTM,
)


class SEANetResnetBlock(nn.Module):
    """Residual block from SEANet model.

    Args:
        dim (int): Dimension of the input/output.
        kernel_sizes (list): List of kernel sizes for the convolutions.
        dilations (list): List of dilations for the convolutions.
        activation (str): Activation function.
        activation_params (dict): Parameters to provide to the activation function.
        norm (str): Normalization method.
        norm_params (dict): Parameters to provide to the underlying normalization used along with the convolution.
        causal (bool): Whether to use fully causal convolution.
        pad_mode (str): Padding mode for the convolutions.
        compress (int): Reduced dimensionality in residual branches (from Demucs v3).
        true_skip (bool): Whether to use true skip connection or a simple
            (streamable) convolution as the skip connection.
    """

    dim: int
    kernel_sizes: tp.List[int] = field(default_factory=lambda: [3, 1])
    dilations: tp.List[int] = field(default_factory=lambda: [1, 1])
    activation: str = "elu"
    activation_params: dict = field(default_factory=lambda: {"alpha": 1.0})
    norm: str = "none"
    norm_params: tp.Dict[str, tp.Any] = field(default_factory=lambda: {})
    causal: int = 0  # bool
    pad_mode: str = "reflect"
    compress: int = 2
    true_skip: int = 1  # bool

    @nn.compact
    def __call__(self, x):
        assert len(self.kernel_sizes) == len(
            self.dilations
        ), "Number of kernel sizes should match number of dilations"
        act = lambda y: getattr(nn.activation, self.activation)(
            y, **self.activation_params
        )
        hidden = self.dim // self.compress
        block = []
        for i, (kernel_size, dilation) in enumerate(
            zip(self.kernel_sizes, self.dilations)
        ):
            out_chs = self.dim if i == len(self.kernel_sizes) - 1 else hidden
            block += [
                act,
                StreamableConv1d(
                    out_chs,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    norm=self.norm,
                    norm_kwargs=self.norm_params,
                    causal=self.causal,
                    pad_mode=self.pad_mode,
                ),
            ]
        block = nn.Sequential(block)
        if self.true_skip:
            return x + block(x)
        else:
            shortcut = StreamableConv1d(
                self.dim,
                kernel_size=1,
                norm=self.norm,
                norm_kwargs=self.norm_params,
                causal=self.causal,
                pad_mode=self.pad_mode,
            )

        return shortcut(x) + block(x)


class SEANetEncoder(nn.Module):
    """SEANet encoder.

    Args:
        channels (int): Audio channels.
        dimension (int): Intermediate representation dimension.
        n_filters (int): Base width for the model.
        n_residual_layers (int): nb of residual layers.
        ratios (Sequence[int]): kernel size and stride ratios. The encoder uses downsampling ratios instead of
            upsampling ratios, hence it will use the ratios in the reverse order to the ones specified here
            that must match the decoder order. We use the decoder order as some models may only employ the decoder.
        activation (str): Activation function.
        activation_params (dict): Parameters to provide to the activation function.
        norm (str): Normalization method.
        norm_params (dict): Parameters to provide to the underlying normalization used along with the convolution.
        kernel_size (int): Kernel size for the initial convolution.
        last_kernel_size (int): Kernel size for the initial convolution.
        residual_kernel_size (int): Kernel size for the residual layers.
        dilation_base (int): How much to increase the dilation with each layer.
        causal (bool): Whether to use fully causal convolution.
        pad_mode (str): Padding mode for the convolutions.
        true_skip (bool): Whether to use true skip connection or a simple
            (streamable) convolution as the skip connection in the residual network blocks.
        compress (int): Reduced dimensionality in residual branches (from Demucs v3).
        lstm (int): Number of LSTM layers at the end of the encoder.
        disable_norm_outer_blocks (int): Number of blocks for which we don't apply norm.
            For the encoder, it corresponds to the N first blocks.
    """

    channels: int = 1
    dimension: int = 128
    n_filters: int = 32
    n_residual_layers: int = 3
    ratios: tp.List[int] = field(default_factory=lambda: [8, 5, 4, 2])
    activation: str = "elu"
    activation_params: dict = field(default_factory=lambda: {"alpha": 1.0})
    norm: str = "none"
    norm_params: tp.Dict[str, tp.Any] = field(default_factory=lambda: {})
    kernel_size: int = 7
    last_kernel_size: int = 7
    residual_kernel_size: int = 3
    dilation_base: int = 2
    causal: bool = False
    pad_mode: str = "reflect"
    true_skip: bool = True
    compress: int = 2
    lstm: int = 0
    disable_norm_outer_blocks: int = 0

    def __post_init__(self) -> None:
        self.hop_length = np.prod(self.ratios)
        self.n_blocks = len(self.ratios) + 2  # first and last conv + residual blocks
        assert (
            self.disable_norm_outer_blocks >= 0
            and self.disable_norm_outer_blocks <= self.n_blocks
        ), (
            "Number of blocks for which to disable norm is invalid."
            "It should be lower or equal to the actual number of blocks in the network and greater or equal to 0."
        )
        super().__post_init__()

    @nn.compact
    def __call__(self, x):
        act = lambda y: getattr(nn.activation, self.activation)(
            y, **self.activation_params
        )
        mult = 1
        layers = [
            StreamableConv1d(
                mult * self.n_filters,
                kernel_size=self.kernel_size,
                norm="none" if self.disable_norm_outer_blocks >= 1 else self.norm,
                norm_kwargs=self.norm_params,
                causal=self.causal,
                pad_mode=self.pad_mode,
            )
        ]
        # Downsample to raw audio scale
        for i, ratio in enumerate(reversed(self.ratios)):
            block_norm = (
                "none" if self.disable_norm_outer_blocks >= i + 2 else self.norm
            )
            # Add residual layers
            for j in range(self.n_residual_layers):
                layers += [
                    SEANetResnetBlock(
                        mult * self.n_filters,
                        kernel_sizes=[self.residual_kernel_size, 1],
                        dilations=[self.dilation_base**j, 1],
                        norm=block_norm,
                        norm_params=self.norm_params,
                        activation=self.activation,
                        activation_params=self.activation_params,
                        causal=self.causal,
                        pad_mode=self.pad_mode,
                        compress=self.compress,
                        true_skip=self.true_skip,
                    )
                ]

            # Add downsampling layers
            layers += [
                act,
                StreamableConv1d(
                    mult * self.n_filters * 2,
                    kernel_size=ratio * 2,
                    stride=ratio,
                    norm=block_norm,
                    norm_kwargs=self.norm_params,
                    causal=self.causal,
                    pad_mode=self.pad_mode,
                ),
            ]
            mult *= 2

        if self.lstm:
            layers += [StreamableLSTM(mult * self.n_filters, num_layers=self.lstm)]

        layers += [
            act,
            StreamableConv1d(
                self.dimension,
                kernel_size=self.last_kernel_size,
                norm=(
                    "none"
                    if self.disable_norm_outer_blocks == self.n_blocks
                    else self.norm
                ),
                norm_kwargs=self.norm_params,
                causal=self.causal,
                pad_mode=self.pad_mode,
            ),
        ]

        model = nn.Sequential(layers)
        x = rearrange(x, "B C T -> B T C")
        return model(x)


class SEANetDecoder(nn.Module):
    """SEANet decoder.

    Args:
        channels (int): Audio channels.
        dimension (int): Intermediate representation dimension.
        n_filters (int): Base width for the model.
        n_residual_layers (int): nb of residual layers.
        ratios (Sequence[int]): kernel size and stride ratios.
        activation (str): Activation function.
        activation_params (dict): Parameters to provide to the activation function.
        final_activation (str): Final activation function after all convolutions.
        final_activation_params (dict): Parameters to provide to the activation function.
        norm (str): Normalization method.
        norm_params (dict): Parameters to provide to the underlying normalization used along with the convolution.
        kernel_size (int): Kernel size for the initial convolution.
        last_kernel_size (int): Kernel size for the initial convolution.
        residual_kernel_size (int): Kernel size for the residual layers.
        dilation_base (int): How much to increase the dilation with each layer.
        causal (bool): Whether to use fully causal convolution.
        pad_mode (str): Padding mode for the convolutions.
        true_skip (bool): Whether to use true skip connection or a simple.
            (streamable) convolution as the skip connection in the residual network blocks.
        compress (int): Reduced dimensionality in residual branches (from Demucs v3).
        lstm (int): Number of LSTM layers at the end of the encoder.
        disable_norm_outer_blocks (int): Number of blocks for which we don't apply norm.
            For the decoder, it corresponds to the N last blocks.
        trim_right_ratio (float): Ratio for trimming at the right of the transposed convolution under the causal setup.
            If equal to 1.0, it means that all the trimming is done at the right.
    """

    channels: int = 1
    dimension: int = 128
    n_filters: int = 32
    n_residual_layers: int = 3
    ratios: tp.List[int] = field(default_factory=lambda: [8, 5, 4, 2])
    activation: str = "elu"
    activation_params: dict = field(default_factory=lambda: {"alpha": 1.0})
    final_activation: tp.Optional[str] = None
    final_activation_params: tp.Optional[dict] = None
    norm: str = "none"
    norm_params: tp.Dict[str, tp.Any] = field(default_factory=lambda: {})
    kernel_size: int = 7
    last_kernel_size: int = 7
    residual_kernel_size: int = 3
    dilation_base: int = 2
    causal: bool = False
    pad_mode: str = "reflect"
    true_skip: bool = True
    compress: int = 2
    lstm: int = 0
    disable_norm_outer_blocks: int = 0
    trim_right_ratio: float = 1.0

    def __post_init__(self) -> None:
        self.hop_length = np.prod(self.ratios)
        self.n_blocks = len(self.ratios) + 2  # first and last conv + residual blocks
        assert (
            self.disable_norm_outer_blocks >= 0
            and self.disable_norm_outer_blocks <= self.n_blocks
        ), (
            "Number of blocks for which to disable norm is invalid."
            "It should be lower or equal to the actual number of blocks in the network and greater or equal to 0."
        )
        super().__post_init__()

    @nn.compact
    def __call__(self, z):
        z = z.transpose(0, 2, 1)
        act = lambda y: getattr(nn.activation, self.activation)(
            y, **self.activation_params
        )
        mult = int(2 ** len(self.ratios))
        layers = [
            StreamableConv1d(
                mult * self.n_filters,
                kernel_size=self.kernel_size,
                norm=(
                    "none"
                    if self.disable_norm_outer_blocks == self.n_blocks
                    else self.norm
                ),
                norm_kwargs=self.norm_params,
                causal=self.causal,
                pad_mode=self.pad_mode,
            )
        ]

        if self.lstm:
            layers += [StreamableLSTM(mult * self.n_filters, num_layers=self.lstm)]

        # Upsample to raw audio scale
        for i, ratio in enumerate(self.ratios):
            block_norm = (
                "none"
                if self.disable_norm_outer_blocks >= self.n_blocks - (i + 1)
                else self.norm
            )
            # Add upsampling layers
            layers += [
                act,
                StreamableConvTranspose1d(
                    mult * self.n_filters // 2,
                    kernel_size=ratio * 2,
                    stride=ratio,
                    norm=block_norm,
                    norm_kwargs=self.norm_params,
                    causal=self.causal,
                    trim_right_ratio=self.trim_right_ratio,
                ),
            ]
            # Add residual layers
            for j in range(self.n_residual_layers):
                layers += [
                    SEANetResnetBlock(
                        mult * self.n_filters // 2,
                        kernel_sizes=[self.residual_kernel_size, 1],
                        dilations=[self.dilation_base**j, 1],
                        activation=self.activation,
                        activation_params=self.activation_params,
                        norm=block_norm,
                        norm_params=self.norm_params,
                        causal=self.causal,
                        pad_mode=self.pad_mode,
                        compress=self.compress,
                        true_skip=self.true_skip,
                    )
                ]

            mult //= 2

        # Add final layers
        layers += [
            act,
            StreamableConv1d(
                self.channels,
                kernel_size=self.last_kernel_size,
                norm="none" if self.disable_norm_outer_blocks >= 1 else self.norm,
                norm_kwargs=self.norm_params,
                causal=self.causal,
                pad_mode=self.pad_mode,
            ),
        ]
        # Add optional final activation to decoder (eg. tanh)
        if self.final_activation is not None:
            final_act = getattr(nn, self.final_activation)
            final_activation_params = self.final_activation_params or {}
            layers += [final_act(**final_activation_params)]
        model = nn.Sequential(layers)
        y = model(z)
        y = rearrange(y, "B T C -> B C T")
        return y


class CompressionModel(ABC, nn.Module):
    """Base API for all compression models that aim at being used as audio tokenizers
    with a language model.
    """

    @abstractmethod
    def __call__(self, x: jnp.ndarray):  # todo: -> qt.QuantizedResult:
        ...

    @abstractmethod
    def encode(self, x: jnp.ndarray) -> tp.Tuple[jnp.ndarray, tp.Optional[jnp.ndarray]]:
        """See `EncodecModel.encode`."""
        ...

    @abstractmethod
    def decode(self, codes: jnp.ndarray, scale: tp.Optional[jnp.ndarray] = None):
        """See `EncodecModel.decode`."""
        ...

    @abstractmethod
    def decode_latent(self, codes: jnp.ndarray):
        """Decode from the discrete codes to continuous latent space."""
        ...

    @property
    @abstractmethod
    def channels(self) -> int: ...

    @property
    @abstractmethod
    def frame_rate(self) -> float: ...

    @property
    @abstractmethod
    def sample_rate(self) -> int: ...

    @property
    @abstractmethod
    def cardinality(self) -> int: ...

    @property
    @abstractmethod
    def num_codebooks(self) -> int: ...

    @property
    def n_codebooks(self):
        return self.num_codebooks

    @property
    def codebook_size(self):
        return self.cardinality

    @property
    @abstractmethod
    def total_codebooks(self) -> int: ...

    @abstractmethod
    def set_num_codebooks(self, n: int):
        """Set the active number of codebooks used by the quantizer."""
        ...


class EncodecModel(CompressionModel):
    """Encodec model operating on the raw waveform.

    Args:
        encoder (nn.Module): Encoder network.
        decoder (nn.Module): Decoder network.
        quantizer (qt.BaseQuantizer): Quantizer network.
        frame_rate (int): Frame rate for the latent representation.
        sample_rate (int): Audio sample rate.
        channels (int): Number of audio channels.
        causal (bool): Whether to use a causal version of the model.
        renormalize (bool): Whether to renormalize the audio before running the model.
    """

    encoder: nn.Module
    decoder: nn.Module
    quantizer: nn.Module  # todo: qt.BaseQuantizer,
    causal: int = 0  # bool
    renormalize: int = 0  # bool

    # todo: must declare these?
    frame_rate: float = 0  # todo: or int?
    sample_rate: int = 0
    channels: int = 0

    def __post_init__(self) -> None:
        if self.causal:
            # we force disabling here to avoid handling linear overlap of segments
            # as supported in original EnCodec codebase.
            assert not self.renormalize, "Causal model does not support renormalize"
        super().__post_init__()

    @property
    def total_codebooks(self):
        """Total number of quantizer codebooks available."""
        return self.quantizer.total_codebooks

    @property
    def num_codebooks(self):
        """Active number of codebooks used by the quantizer."""
        return self.quantizer.num_codebooks

    def set_num_codebooks(self, n: int):
        """Set the active number of codebooks used by the quantizer."""
        self.quantizer.set_num_codebooks(n)

    @property
    def cardinality(self):
        """Cardinality of each codebook."""
        return self.quantizer.bins

    def preprocess(
        self, x: jnp.ndarray
    ) -> tp.Tuple[jnp.ndarray, tp.Optional[jnp.ndarray]]:
        scale: tp.Optional[jnp.ndarray]
        if self.renormalize:
            mono = x.mean(axis=1, keepdims=True)
            volume = jnp.sqrt(jnp.square(mono).mean(axis=2, keepdims=True))
            scale = 1e-8 + volume
            x = x / scale
            scale = scale.reshape(-1, 1)
        else:
            scale = None
        return x, scale

    def postprocess(
        self, x: jnp.ndarray, scale: tp.Optional[jnp.ndarray] = None
    ) -> jnp.ndarray:
        if scale is not None:
            assert self.renormalize
            x = x * scale.reshape(-1, 1, 1)
        return x

    def __call__(self, x: jnp.ndarray, train=False):  # todo: -> qt.QuantizedResult:
        assert x.ndim == 3
        length = x.shape[-1]
        x, scale = self.preprocess(x)

        emb = self.encoder(x)
        q_res = self.quantizer(emb, self.frame_rate, train=train)
        out = self.decoder(q_res.x)

        # remove extra padding added by the encoder and decoder
        assert out.shape[-1] >= length, (out.shape[-1], length)
        out = out[..., :length]

        q_res.x = self.postprocess(out, scale)

        return q_res

    def encode(self, x: jnp.ndarray) -> tp.Tuple[jnp.ndarray, tp.Optional[jnp.ndarray]]:
        """Encode the given input tensor to quantized representation along with scale parameter.

        Args:
            x (jnp.ndarray): Float tensor of shape [B, C, T]

        Returns:
            codes, scale (tuple of jnp.ndarray, jnp.ndarray): Tuple composed of:
                codes: a float tensor of shape [B, K, T] with K the number of codebooks used and T the timestep.
                scale: a float tensor containing the scale for audio renormalization.
        """
        assert x.ndim == 3
        x, scale = self.preprocess(x)
        emb = self.encoder(x)
        emb = emb.transpose(0, 2, 1)
        codes = self.quantizer.encode(emb)
        return codes, scale

    def decode(self, codes: jnp.ndarray, scale: tp.Optional[jnp.ndarray] = None):
        """Decode the given codes to a reconstructed representation, using the scale to perform
        audio denormalization if needed.

        Args:
            codes (jnp.ndarray): Int tensor of shape [B, K, T]
            scale (jnp.ndarray, optional): Float tensor containing the scale value.

        Returns:
            out (jnp.ndarray): Float tensor of shape [B, C, T], the reconstructed audio.
        """
        emb = self.decode_latent(codes)
        out = self.decoder(emb)
        out = self.postprocess(out, scale)
        # out contains extra padding added by the encoder and decoder
        return out

    def decode_latent(self, codes: jnp.ndarray):
        """Decode from the discrete codes to continuous latent space."""
        return self.quantizer.decode(codes)
