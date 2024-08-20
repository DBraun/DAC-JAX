from dataclasses import field

from audiotree.resample import resample
from einops import rearrange
import flax.linen as nn
import jax
from jax import numpy as jnp

from dac_jax.audio_utils import stft
from dac_jax.nn.weight_norm import WeightNorm as MyWeightNorm


class LeakyReLU(nn.Module):

    negative_slope: float = .01

    @nn.compact
    def __call__(self, x):
        return nn.leaky_relu(x, negative_slope=self.negative_slope)


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


class WNConv(nn.Conv):

    act: bool = True

    @nn.compact
    def __call__(self, x):

        kernel_init = make_initializer(
            x.shape[-1], self.features, self.kernel_size, self.feature_group_count
        )

        if self.use_bias:
            # note: we just ignore whatever self.bias_init is
            bias_init = make_initializer(
                x.shape[-1], self.features, self.kernel_size, self.feature_group_count
            )
        else:
            bias_init = None

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
        x = MyWeightNorm("fan_in", conv)(x)

        if self.act:
            x = LeakyReLU(.1)(x)

        return x


class MPD(nn.Module):

    period: int

    def pad_to_period(self, x):
        t = x.shape[-1]
        x = jnp.pad(x, pad_width=((0, 0), (0, 0), (0, self.period - t % self.period)), mode='reflect')
        return x

    @nn.compact
    def __call__(self, x):
        convs = [
            WNConv(features=32, kernel_size=(5, 1), strides=(3, 1), padding=((2, 2), (0, 0))),
            WNConv(features=128, kernel_size=(5, 1), strides=(3, 1), padding=((2, 2), (0, 0))),
            WNConv(features=512, kernel_size=(5, 1), strides=(3, 1), padding=((2, 2), (0, 0))),
            WNConv(features=1024, kernel_size=(5, 1), strides=(3, 1), padding=((2, 2), (0, 0))),
            WNConv(features=1024, kernel_size=(5, 1), strides=(1, 1), padding=((2, 2), (0, 0))),
            WNConv(features=1, kernel_size=(3, 1), padding=((1, 1), (0, 0)), act=False)
        ]

        fmap = []

        x = self.pad_to_period(x)
        x = rearrange(x, "b c (l p) -> b l p c", p=self.period)

        for layer in convs:
            x = layer(x)
            fmap.append(x)

        return fmap


class MSD(nn.Module):

    rate: int = 1
    sample_rate: int = 44100

    @nn.compact
    def __call__(self, x):
        convs = [
            WNConv(features=16, kernel_size=15, strides=1, padding=7),
            WNConv(features=64, kernel_size=41, strides=4, feature_group_count=4, padding=20),
            WNConv(features=256, kernel_size=41, strides=4, feature_group_count=16, padding=20),
            WNConv(features=1024, kernel_size=41, strides=4, feature_group_count=64, padding=20),
            WNConv(features=1024, kernel_size=41, strides=4, feature_group_count=256, padding=20),
            WNConv(features=1024, kernel_size=5, strides=1, padding=2),
            WNConv(features=1, kernel_size=3, strides=1, padding=1, act=False)
        ]

        x = resample(x, old_sr=self.sample_rate, new_sr=self.sample_rate//self.rate)

        x = rearrange(x, "b c l -> b l c")

        fmap = []

        for layer in convs:
            x = layer(x)
            fmap.append(x)

        return fmap


BANDS = [(0.0, 0.1), (0.1, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)]


class MRD(nn.Module):

    window_length: int
    hop_factor: float = 0.25
    sample_rate: int = 44100
    bands: list = field(default_factory=lambda: BANDS)

    def __post_init__(self) -> None:
        n_fft = self.window_length // 2 + 1
        self.bands = [(int(low * n_fft), int(high * n_fft)) for low, high in self.bands]
        super().__post_init__()

    @nn.compact
    def __call__(self, x):

        """Complex multi-band spectrogram discriminator.
        Parameters
        ----------
        window_length : int
            Window length of STFT.
        hop_factor : float, optional
            Hop factor of the STFT, defaults to ``0.25 * window_length``.
        sample_rate : int, optional
            Sampling rate of audio in Hz, by default 44100
        bands : list, optional
            Bands to run discriminator over.
        """

        ch = 32
        convs = lambda: [
            WNConv(features=ch, kernel_size=(3, 9), strides=(1, 1), padding=((1, 1), (4, 4))),
            WNConv(features=ch, kernel_size=(3, 9),  strides=(1, 2), padding=((1, 1), (4, 4))),
            WNConv(features=ch, kernel_size=(3, 9),  strides=(1, 2), padding=((1, 1), (4, 4))),
            WNConv(features=ch, kernel_size=(3, 9),  strides=(1, 2), padding=((1, 1), (4, 4))),
            WNConv(features=ch, kernel_size=(3, 3),  strides=(1, 1), padding=((1, 1), (1, 1))),
        ]
        band_convs = [convs() for _ in range(len(self.bands))]
        conv_post = WNConv(features=1, kernel_size=(3, 3), strides=(1, 1), padding=((1, 1), (1, 1)), act=False)

        x_bands = self.get_bands(x)
        fmap = []

        x = []
        for band, stack in zip(x_bands, band_convs):
            band = rearrange(band, "b c t f -> b t f c")
            for layer in stack:
                band = layer(band)
                fmap.append(band)
            x.append(band)

        x = jnp.concatenate(x, axis=-2)  # concatenate along frequency axis
        x = conv_post(x)
        fmap.append(x)

        return fmap

    def get_bands(self, x):
        stft_data = stft(x, frame_length=self.window_length, hop_factor=self.hop_factor, match_stride=True)
        x = self.as_real(stft_data)
        x = rearrange(x, "b c f t ri -> (b c) ri t f", c=1, ri=2)  # ri is 2 for real and imaginary
        # Split into bands
        x_bands = [x[..., low:high] for low, high in self.bands]
        return x_bands

    @staticmethod
    def as_real(x: jnp.ndarray) -> jnp.ndarray:
        # https://github.com/google/jax/issues/9496#issuecomment-1033961377
        if not jnp.issubdtype(x.dtype, jnp.complexfloating):
            return x

        return jnp.stack([x.real, x.imag], axis=-1)


class Discriminator(nn.Module):

    rates: list = field(default_factory=lambda: [])
    periods: list = field(default_factory=lambda: [2, 3, 5, 7, 11])
    fft_sizes: list = field(default_factory=lambda: [2048, 1024, 512])
    sample_rate: int = 44100
    bands: list = field(default_factory=lambda: BANDS)

    @staticmethod
    def preprocess(y: jnp.ndarray):
        # Remove DC offset
        y = y - y.mean(axis=-1, keepdims=True)
        # Peak normalize the volume of input audio
        y = 0.8 * y / (jnp.abs(y).max(axis=-1, keepdims=True) + 1e-9)
        return y

    @nn.compact
    def __call__(self, x):
        """Discriminator that combines multiple discriminators.

        Parameters
        ----------
        rates : list, optional
            sampling rates (in Hz) to run MSD at, by default []
            If empty, MSD is not used.
        periods : list, optional
            periods (of samples) to run MPD at, by default [2, 3, 5, 7, 11]
        fft_sizes : list, optional
            Window sizes of the FFT to run MRD at, by default [2048, 1024, 512]
        sample_rate : int, optional
            Sampling rate of audio in Hz, by default 44100
        bands : list, optional
            Bands to run MRD at, by default `BANDS`
        """
        discriminators = []
        discriminators += [MPD(p) for p in self.periods]
        discriminators += [MSD(r, sample_rate=self.sample_rate) for r in self.rates]
        discriminators += [MRD(f, sample_rate=self.sample_rate, bands=self.bands) for f in self.fft_sizes]
        x = self.preprocess(x)
        fmaps = [d(x) for d in discriminators]
        return fmaps


if __name__ == "__main__":
    import numpy as np

    disc = Discriminator()
    x = jnp.zeros(shape=(1, 1, 44100))

    print(disc.tabulate(jax.random.key(1), x,
                        # compute_flops=True,
                        # compute_vjp_flops=True,
                        depth=3,
                        # column_kwargs={"width": 400},
                        console_kwargs={"width": 400},
                        ))

    results, variables = disc.init_with_output(jax.random.key(3), x)

    for i, result in enumerate(results):
        print(f"disc{i}")
        for i, _r in enumerate(result):
            r = np.array(_r)
            print(r.shape, f"{r.mean().item():,.5f}, {r.min().item():,.5f} {r.max().item():,.5f}")
    print('All Done!')
