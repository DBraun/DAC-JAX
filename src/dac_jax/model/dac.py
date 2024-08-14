from dataclasses import field, dataclass
from functools import lru_cache
import math
from pathlib import Path
import timeit
from typing import List, Union, Tuple

from audiotree.resample import resample
from einops import rearrange
from flax import linen as nn
import jax
from jax import numpy as jnp
from jax import random
import numpy as np
import tqdm

from dac_jax.nn.layers import Snake1d, WNConv1d, WNConvTranspose1d
from dac_jax.nn.quantize import ResidualVectorQuantize
from dac_jax.audio_utils import volume_norm

SUPPORTED_VERSIONS = ["1.0.0"]


@dataclass
class DACFile:
    codes: jnp.ndarray

    # Metadata
    chunk_length: int
    original_length: int
    input_db: Union[float, None]
    channels: int
    sample_rate: int
    dac_version: str

    def save(self, path):
        artifacts = {
            "codes": np.array(self.codes).astype(np.uint16),
            "metadata": {
                "input_db": np.array(self.input_db, dtype=jnp.float32) if self.input_db is not None else None,
                "original_length": self.original_length,
                "sample_rate": self.sample_rate,
                "chunk_length": self.chunk_length,
                "channels": self.channels,
                "dac_version": SUPPORTED_VERSIONS[-1],
            },
        }
        path = Path(path).with_suffix(".dac")
        jnp.save(path, artifacts)
        return path

    @classmethod
    def load(cls, path):
        # todo: use safetensors instead of allow_pickle
        artifacts = jnp.load(path, allow_pickle=True)[()]
        codes = jnp.array(artifacts["codes"].astype(int))
        if artifacts["metadata"].get("dac_version", None) not in SUPPORTED_VERSIONS:
            raise RuntimeError(
                f"Given file {path} can't be loaded with this version of descript-audio-codec."
            )
        return cls(codes=codes, **artifacts["metadata"])


class ResidualUnit(nn.Module):

    dim: int = 16
    dilation: int = 1
    padding: int = 1

    @staticmethod
    def delay(d, L):
        # remember to iterate in reverse
        L = WNConv1d.delay(1, 1, 1, L)
        L = WNConv1d.delay(1, d, 7, L)
        return L

    @staticmethod
    def output_length(d, L):
        # iterate forwards
        L = WNConv1d.output_length(1, d, 7, L)
        L = WNConv1d.output_length(1, 1, 1, L)
        return L

    @nn.compact
    def __call__(self, x):
        pad = ((7 - 1) * self.dilation) // 2 if self.padding else 0
        block = nn.Sequential([
            Snake1d(self.dim),
            WNConv1d(features=self.dim, kernel_size=(7,), kernel_dilation=(self.dilation,), padding=(pad,)),
            Snake1d(self.dim),
            WNConv1d(features=self.dim, kernel_size=(1,))
        ])
        y = block(x)
        pad = (x.shape[-2] - y.shape[-2]) // 2  # pad on time axis
        if pad > 0:
            x = x[..., pad:-pad, :]  # pad on time axis
        return x + y


class EncoderBlock(nn.Module):

    dim: int = 16
    stride: int = 1
    padding: int = 1

    @staticmethod
    def delay(s, L):
        # remember to iterate in reverse
        L = WNConv1d.delay(s, 1, 2*s, L)
        L = ResidualUnit.delay(9, L)
        L = ResidualUnit.delay(3, L)
        L = ResidualUnit.delay(1, L)
        return L

    @staticmethod
    def output_length(s, L):
        # iterate forwards
        L = ResidualUnit.output_length(1, L)
        L = ResidualUnit.output_length(3, L)
        L = ResidualUnit.output_length(9, L)
        L = WNConv1d.output_length(s, 1, 2*s, L)
        return L

    @nn.compact
    def __call__(self, x):
        block = nn.Sequential([
            ResidualUnit(self.dim // 2, dilation=1, padding=self.padding),
            ResidualUnit(self.dim // 2, dilation=3, padding=self.padding),
            ResidualUnit(self.dim // 2, dilation=9, padding=self.padding),
            Snake1d(self.dim // 2),
            WNConv1d(features=self.dim,
                kernel_size=(2 * self.stride,),
                strides=self.stride,
                padding=(math.ceil(self.stride / 2),) if self.padding else 0
            )
        ])
        x = block(x)
        return x


class Encoder(nn.Module):

    d_model: int = 64
    strides: list = field(default_factory=lambda: [2, 4, 8, 8])
    d_latent: int = 64
    padding: int = 1

    @staticmethod
    def delay(strides, L):
        # remember to iterate in reverse
        L = WNConv1d.delay(1, 1, 3, L)
        for stride in reversed(strides):
            L = EncoderBlock.delay(stride, L)
        L = WNConv1d.delay(1, 1, 7, L)
        return L

    @staticmethod
    def output_length(strides, L):
        # iterate forwards
        L = WNConv1d.output_length(1, 1, 7, L)
        for stride in strides:
            L = EncoderBlock.output_length(stride, L)
        L = WNConv1d.output_length(1, 1, 3, L)
        return L

    @nn.compact
    def __call__(self, x):
        d_model = self.d_model

        # Create first convolution
        block = [WNConv1d(features=d_model, kernel_size=(7,), padding='SAME' if self.padding else 0)]

        # Create EncoderBlocks that double channels as they downsample by `stride`
        for stride in self.strides:
            d_model *= 2
            block += [EncoderBlock(d_model, stride=stride, padding=self.padding)]

        # Create last convolution
        block += [
            Snake1d(d_model),
            WNConv1d(features=self.d_latent, kernel_size=(3,), padding='SAME' if self.padding else 0),
        ]

        # Wrap black into nn.Sequential
        block = nn.Sequential(block)
        x = rearrange(x, "b c l -> b l c")
        x = block(x)
        return x


class DecoderBlock(nn.Module):
    input_dim: int = 16
    output_dim: int = 8
    stride: int = 1
    padding: int = 1

    @staticmethod
    def delay(s, L):
        # remember to iterate in reverse
        L = ResidualUnit.delay(9, L)
        L = ResidualUnit.delay(3, L)
        L = ResidualUnit.delay(1, L)
        L = WNConvTranspose1d.delay(s, 1, 2*s, L)
        return L

    @staticmethod
    def output_length(s, L):
        # iterate forwards
        L = WNConvTranspose1d.output_length(s, 1, 2*s, L)
        L = ResidualUnit.output_length(1, L)
        L = ResidualUnit.output_length(3, L)
        L = ResidualUnit.output_length(9, L)
        return L

    @nn.compact
    def __call__(self, x):
        block = nn.Sequential([
            Snake1d(self.input_dim),
            WNConvTranspose1d(
                self.output_dim,
                (2 * self.stride,),
                strides=(self.stride,),
                padding='SAME' if self.padding else 'VALID'
            ),
            ResidualUnit(self.output_dim, dilation=1, padding=self.padding),
            ResidualUnit(self.output_dim, dilation=3, padding=self.padding),
            ResidualUnit(self.output_dim, dilation=9, padding=self.padding),
        ])
        x = block(x)
        return x


class Decoder(nn.Module):

    input_channel: int
    channels: int
    rates: List[int]
    d_out: int = 1
    padding: int = 1

    def __post_init__(self) -> None:
        assert self.rates is not None and len(self.rates)
        super().__post_init__()

    @staticmethod
    def delay(rates, L):
        # remember to iterate in reverse
        L = WNConv1d.delay(1, 1, 7, L)
        for stride in reversed(rates):
            L = DecoderBlock.delay(stride, L)
        L = WNConv1d.delay(1, 1, 7, L)
        return L

    @staticmethod
    def output_length(rates, L):
        # iterate forwards
        L = WNConv1d.output_length(1, 1, 7, L)
        for stride in rates:
            L = DecoderBlock.output_length(stride, L)
        L = WNConv1d.output_length(1, 1, 7, L)
        return L

    @nn.compact
    def __call__(self, x) -> jnp.ndarray:
        # Add first conv layer
        layers = [WNConv1d(features=self.channels, kernel_size=(7,), padding='SAME' if self.padding else 0)]

        output_dim = 1

        # Add upsampling + MRF blocks
        for i, stride in enumerate(self.rates):
            input_dim = self.channels // (2**i)
            output_dim = self.channels // (2 ** (i + 1))
            layers += [DecoderBlock(input_dim, output_dim, stride, self.padding)]

        # Add final conv layer
        layers += [
            Snake1d(output_dim),
            WNConv1d(features=self.d_out, kernel_size=(7,), padding='SAME' if self.padding else 0),
            nn.activation.tanh,
        ]

        block = nn.Sequential(layers)
        x = block(x)
        x = rearrange(x, 'b l c -> b c l')
        return x


class DAC(nn.Module):

    encoder_dim: int = 64
    encoder_rates: tuple = field(default_factory=lambda: [2, 4, 8, 8])
    latent_dim: int = None
    decoder_dim: int = 1536
    decoder_rates: tuple = field(default_factory=lambda: [8, 8, 4, 2])
    n_codebooks: int = 9
    codebook_size: int = 1024
    codebook_dim: Union[int, list] = 8
    quantizer_dropout: float = 0.0
    sample_rate: int = 44100
    padding: int = 1  # https://github.com/pseeth/argbind?tab=readme-ov-file#boolean-keyword-arguments

    def __post_init__(self) -> None:
        if self.latent_dim is None:
            self.latent_dim = self.encoder_dim * (2 ** len(self.encoder_rates))

        self.encoder_rates = tuple(self.encoder_rates)  # cast to tuple so that lru_cache works later
        self.decoder_rates = tuple(self.decoder_rates)

        self.hop_length = math.prod(self.encoder_rates)
        self.padding = bool(self.padding)  # https://github.com/pseeth/argbind?tab=readme-ov-file#boolean-keyword-arguments

        # Set the delay property
        # remember to iterate in reverse
        l_out = self.hop_length * 100  # Any number works here, delay is invariant to input length
        l_out = DAC.output_length(self.encoder_rates, self.decoder_rates, l_out)
        assert l_out > 0
        L = l_out
        L = Decoder.delay(self.decoder_rates, L)
        L = Encoder.delay(self.encoder_rates, L)
        l_in = L
        self.delay = (l_in - l_out) // 2

        super().__post_init__()

    def setup(self):

        self.encoder = Encoder(self.encoder_dim, self.encoder_rates, self.latent_dim, self.padding)

        self.quantizer = ResidualVectorQuantize(
            input_dim=self.latent_dim,
            n_codebooks=self.n_codebooks,
            codebook_size=self.codebook_size,
            codebook_dim=self.codebook_dim,
            quantizer_dropout=self.quantizer_dropout,
        )

        self.decoder = Decoder(input_channel=self.latent_dim, channels=self.decoder_dim, rates=self.decoder_rates,
                               d_out=1, padding=self.padding)

    @staticmethod
    @lru_cache(maxsize=16)
    def output_length(encoder_rates: tuple[int], decoder_rates: tuple[int], L: int):
        # iterate forwards
        L = Encoder.output_length(encoder_rates, L)
        L = Decoder.output_length(decoder_rates, L)
        return L

    def preprocess(self, audio_data, sample_rate):
        if sample_rate:
            assert sample_rate == self.sample_rate, f'Expected sample rate is {self.sample_rate}'

        length = audio_data.shape[-1]
        right_pad = math.ceil(length / self.hop_length) * self.hop_length - length
        audio_data = jnp.pad(audio_data, pad_width=((0, 0), (0, 0), (0, right_pad)))

        return audio_data

    def encode(
        self,
        audio_data: jnp.ndarray,
        n_quantizers: int = None,
        train=True,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Encode given audio data and return quantized latent codes

        Parameters
        ----------
        audio_data : jnp.ndarray[B x 1 x T]
            Audio data to encode
        n_quantizers : int, optional
            Number of quantizers to use, by default None
            If None, all quantizers are used.

        Returns
        -------
        dict
            A dictionary with the following keys:
            "z" : jnp.ndarray[B x T x D]
                Quantized continuous representation of input
            "codes" : jnp.ndarray[B x T x N]
                Codebook indices for each codebook
                (quantized discrete representation of input)
            "latents" : jnp.ndarray[B x T x N*D]
                Projected latents (continuous representation of input before quantization)
            "vq/commitment_loss" : jnp.ndarray[1]
                Commitment loss to train encoder to predict vectors closer to codebook
                entries
            "vq/codebook_loss" : jnp.ndarray[1]
                Codebook loss to update the codebook
            "length" : int
                Number of samples in input audio
        """
        z = self.encoder(audio_data)
        z, codes, latents, commitment_loss, codebook_loss = self.quantizer(
            z, n_quantizers, train=train
        )
        return z, codes, latents, commitment_loss, codebook_loss

    def compress_chunk(
        self,
        audio_data: jnp.ndarray,
        n_quantizers: int = None
    ) -> jnp.ndarray:
        """Encode given audio data and return quantized latent codes

        Parameters
        ----------
        audio_data : jnp.ndarray[B x 1 x T]
            Audio data to encode
        n_quantizers : int, optional
            Number of quantizers to use, by default None
            If None, all quantizers are used.

        Returns
        -------
        dict
            A dictionary with the following keys:
            "z" : jnp.ndarray[B x T x D]
                Quantized continuous representation of input
            "codes" : jnp.ndarray[B x T x N]
                Codebook indices for each codebook
                (quantized discrete representation of input)
            "latents" : jnp.ndarray[B x T x N*D]
                Projected latents (continuous representation of input before quantization)
            "vq/commitment_loss" : jnp.ndarray[1]
                Commitment loss to train encoder to predict vectors closer to codebook
                entries
            "vq/codebook_loss" : jnp.ndarray[1]
                Codebook loss to update the codebook
            "length" : int
                Number of samples in input audio
        """
        assert not self.padding, "Padding must be disabled in order to use a \"chunk\" method."
        audio_data = self.preprocess(audio_data, self.sample_rate)

        z = self.encoder(audio_data)
        z, codes, latents, commitment_loss, codebook_loss = self.quantizer(
            z, n_quantizers, train=False
        )
        return codes

    def decode(self, z: jnp.ndarray, length=None) -> jnp.ndarray:
        """Decode given latent codes and return audio data

        Parameters
        ----------
        z : jnp.ndarray[B x T x D]
            Quantized continuous representation of input
        length : int, optional
            Number of samples in output audio, by default None

        Returns
        -------
        "audio" : jnp.ndarray[B x 1 x length]
            Decoded audio data.
        """
        audio = self.decoder(z)
        if length is not None:
            audio = audio[..., :length]
        return audio

    def decompress_chunk(self, c):
        assert not self.padding, "Padding must be disabled in order to use a \"chunk\" method."
        z, _, _ = self.quantizer.from_codes(c)
        r = self.decode(z)
        return r

    def from_codes(self, c):
        z, _, _ = self.quantizer.from_codes(c)
        return z

    def __call__(
        self,
        audio_data: jnp.ndarray,
        sample_rate: int = None,
        n_quantizers: int = None,
        train=True
    ):
        """Model forward pass

        Parameters
        ----------
        audio_data : jnp.ndarray[B x 1 x T]
            Audio data to encode
        sample_rate : int, optional
            Sample rate of audio data in Hz, by default None
            If None, defaults to `self.sample_rate`
        n_quantizers : int, optional
            Number of quantizers to use, by default None.
            If None, all quantizers are used.

        Returns
        -------
        dict
            A dictionary with the following keys:
            "z" : jnp.ndarray[B x T x D]
                Quantized continuous representation of input
            "codes" : jnp.ndarray[B x T x N]
                Codebook indices for each codebook
                (quantized discrete representation of input)
            "latents" : jnp.ndarray[B x T x N*D]
                Projected latents (continuous representation of input before quantization)
            "vq/commitment_loss" : jnp.ndarray[1]
                Commitment loss to train encoder to predict vectors closer to codebook
                entries
            "vq/codebook_loss" : jnp.ndarray[1]
                Codebook loss to update the codebook
            "length" : int
                Number of samples in input audio
            "audio" : jnp.ndarray[B x 1 x length]
                Decoded audio data.
        """
        length = audio_data.shape[-1]
        audio_data = self.preprocess(audio_data, sample_rate)
        z, codes, latents, commitment_loss, codebook_loss = self.encode(
            audio_data, n_quantizers, train=train
        )

        audio = self.decode(z, length=length)
        return {
            "audio": audio,
            "z": z,
            "codes": codes,
            "latents": latents,
            "vq/commitment_loss": commitment_loss,
            "vq/codebook_loss": codebook_loss,
        }

    @staticmethod
    def ensure_max_of_audio(audio_data: jnp.ndarray, max: float = 1.0):
        """Ensures that ``abs(audio_data) <= max``.

        Parameters
        ----------
        audio_data : jnp.ndarray
            Audio data shaped [B, C, T]
        max : float, optional
            Max absolute value of signal, by default 1.0

        Returns
        -------
        audio_data
            audio data with values scaled between -max and max.
        """
        peak = jnp.abs(audio_data).max(axis=-1, keepdims=True)
        peak_gain = jnp.ones_like(peak)
        peak_gain = jnp.where(peak > max, max/peak, peak_gain)
        audio_data = audio_data * peak_gain
        return audio_data

    def compress(self, compress_chunk, audio_path_or_signal: Union[str, Path, jnp.ndarray], original_sr: int,
                 win_duration: float = 1, normalize_db: float = -16, n_quantizers: int = None, verbose=False,
                 benchmark=False) -> DACFile:

        audio_signal = audio_path_or_signal
        if isinstance(audio_signal, (str, Path)):
            import librosa
            audio_signal, original_sr = librosa.load(
                audio_path_or_signal,
                sr=None,
                mono=False,
            )
            audio_signal = jnp.array(audio_signal)
            while audio_signal.ndim < 3:
                audio_signal = jnp.expand_dims(audio_signal, axis=0)

        nb, nac, nt = audio_signal.shape
        original_length = nt

        audio_signal = rearrange(audio_signal, 'b c t -> (b c) 1 t')

        input_db = None

        # Use ffmpeg is audio duration is longer than 10 minutes
        # Here we compare the number of samples to 10 minutes * (60 sec/minute) * (original_sr samples/sec)
        use_ffmpeg = audio_signal.shape[-1] >= 10 * 60 * original_sr

        if use_ffmpeg:
            # resample with ffmpeg
            # then get input_db with ffmpeg
            # then normalize if normalize_db is not None
            raise RuntimeError('Not implemented yet')
        else:
            audio_signal = resample(audio_signal, original_sr, self.sample_rate)

            if normalize_db is not None:
                audio_signal, input_db = volume_norm(audio_signal, normalize_db, self.sample_rate)

        audio_signal = self.ensure_max_of_audio(audio_signal)

        # Chunked inference
        # Zero-pad signal on either side by the delay
        audio_signal = jnp.pad(audio_signal, pad_width=((0, 0), (0, 0), (self.delay, self.delay)))
        n_samples = int(win_duration * self.sample_rate)
        # Round n_samples to nearest hop length multiple
        n_samples = int(math.ceil(n_samples / self.hop_length) * self.hop_length)
        hop = self.output_length(self.encoder_rates, self.decoder_rates, n_samples)

        codes = []

        range_fn = range if not verbose else tqdm.trange

        for i in range_fn(0, nt, hop):
            x = audio_signal[..., i: i + n_samples]
            x = jnp.pad(x, pad_width=((0, 0), (0, 0), (0, max(0, n_samples - x.shape[-1]))))
            code = compress_chunk(x)
            chunk_length = code.shape[-2]
            codes.append(code)

        if benchmark:
            x = audio_signal[..., : n_samples]
            x = jnp.pad(x, pad_width=((0, 0), (0, 0), (0, max(0, n_samples - x.shape[-1]))))
            execution_times = timeit.repeat('compress_chunk(x).block_until_ready()',
                                            number=1, repeat=200,
                                            globals={'compress_chunk': compress_chunk, 'x': x})
            execution_times = np.array(execution_times) * 1000  # convert to ms
            mean_time = execution_times.mean()
            median_time = np.median(execution_times)
            min_time = execution_times.min()
            max_time = execution_times.max()
            std_time = execution_times.std()
            print('Requested win_duration:', win_duration)
            print('True window size:', n_samples)
            print('Hop size:', hop)
            print(f"Compress--Num executions:", execution_times.shape[0])
            print(f"Compress--Mean execution time: {mean_time:.2f} ms")
            print(f"Compress--Median execution time: {median_time:.2f} ms")
            print(f"Compress--Min execution time: {min_time:.2f} ms")
            print(f"Compress--Max execution time: {max_time:.2f} ms")
            print(f"Compress--Std execution time: {std_time:.2f} ms")

        codes = jnp.concatenate(codes, axis=-2)

        if n_quantizers is not None:
            codes = codes[:, :n_quantizers, :]

        dac_file = DACFile(
            codes=codes,
            chunk_length=chunk_length,
            original_length=original_length,
            input_db=input_db,
            channels=nac,
            sample_rate=original_sr,
            dac_version=SUPPORTED_VERSIONS[-1],
        )

        return dac_file

    def decompress(self, decompress_chunk, obj: Union[str, Path, DACFile], verbose=False, benchmark=False) \
            -> jnp.ndarray:

        if isinstance(obj, (str, Path)):
            obj = DACFile.load(obj)

        range_fn = range if not verbose else tqdm.trange
        codes = obj.codes
        chunk_length = obj.chunk_length
        recons = []

        for i in range_fn(0, codes.shape[-2], chunk_length):
            c = codes[..., i: i + chunk_length, :]
            r = decompress_chunk(c)
            recons.append(r)

        if benchmark:
            c = codes[..., 0:chunk_length, :]
            execution_times = timeit.repeat('decompress_chunk(c).block_until_ready()',
                                            number=1, repeat=200,
                                            globals={'decompress_chunk': decompress_chunk, 'c': c})
            execution_times = np.array(execution_times) * 1000  # convert to ms
            mean_time = execution_times.mean()
            median_time = np.median(execution_times)
            min_time = execution_times.min()
            max_time = execution_times.max()
            std_time = execution_times.std()

            print(f"Decompress--Mean execution time: {mean_time:.2f} ms")
            print(f"Decompress--Median execution time: {median_time:.2f} ms")
            print(f"Decompress--Min execution time: {min_time:.2f} ms")
            print(f"Decompress--Max execution time: {max_time:.2f} ms")
            print(f"Decompress--Std execution time: {std_time:.2f} ms")

        recons = jnp.concatenate(recons, axis=-1)

        use_ffmpeg = recons.shape[-1] >= 10 * 60 * self.sample_rate

        if use_ffmpeg:
            # todo: use ffmpeg if the audio is over 10 minutes long
            raise RuntimeError('Not implemented yet')
        else:
            # Normalize to original loudness
            if obj.input_db is not None:
                recons, _ = volume_norm(recons, obj.input_db, self.sample_rate)

            # Resample
            recons = resample(recons, old_sr=self.sample_rate, new_sr=obj.sample_rate)

        recons = recons[..., : obj.original_length]

        recons = recons.reshape(-1, obj.channels, obj.original_length)

        return recons


def receptive_field_test():

    SAMPLE_RATE = 44100

    model = DAC(sample_rate=SAMPLE_RATE)

    length = SAMPLE_RATE * 2 * 2
    key = random.key(0)
    key, subkey = random.split(key)
    x = random.uniform(subkey, shape=(1, 1, length), minval=-1, maxval=1)
    key, subkey = random.split(key)
    variables = model.init({'params': subkey, 'rng_stream': random.key(4)}, x, SAMPLE_RATE)
    params = variables['params']

    def fun(x):

        # Make a forward pass
        out = model.apply({'params': params}, x, rngs={'rng_stream': random.key(0)})["audio"]
        print("Input shape:", x.shape)
        print("Output shape:", out.shape)

        out = out.at[:, :, out.shape[-1]//2].set(1)  # todo: not sure about this
        out = out.sum()
        return out

    # print(model.tabulate({'params': random.key(0), 'rng_stream': random.key(1)}, x, SAMPLE_RATE,
    #                      compute_flops=True,
    #                      compute_vjp_flops=True
    #                      ))

    fun_grad = jax.grad(fun)
    # Check non-zero values
    gradmap = jnp.squeeze(fun_grad(x), axis=0)
    rf = jnp.where(gradmap != 0, jnp.ones_like(gradmap), jnp.zeros_like(gradmap)).sum()
    # todo: this doesn't report the same values as DAC in PyTorch
    print(f"Receptive field: {rf.item()}")


if __name__ == "__main__":

    receptive_field_test()

    print('All Done!')
