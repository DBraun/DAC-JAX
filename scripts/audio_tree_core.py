"""
This file borrows extensively from Descript AudioTools:
https://github.com/descriptinc/audiotools
specifically,
https://github.com/descriptinc/audiotools/blob/master/audiotools/core/audio_signal.py
"""

from dataclasses import field
from functools import partial
from pathlib import Path
from typing import Union

from einops import rearrange
from flax import struct
import jax
import jax.numpy as jnp
import jaxloudnorm as jln
import librosa
import numpy as np
import soundfile


@struct.dataclass
class SaliencyParams:
    enabled: bool = field(default=False)
    num_tries: int = 8
    loudness_cutoff: float = -40


@partial(jax.jit, static_argnames=('sample_rate', 'zeros'))
def jit_integrated_loudness(data, sample_rate, zeros: int):

    block_size = 0.4

    original_length = data.shape[-1]
    signal_duration = original_length / sample_rate

    if signal_duration < block_size:
        data = jnp.pad(data, pad_width=((0, 0), (0, 0), (0, int(block_size*sample_rate)-original_length)))

    data = rearrange(data, 'b c t -> b t c')
    meter = jln.Meter(sample_rate, block_size=block_size, use_fir=True, zeros=zeros)
    loudness = jax.vmap(meter.integrated_loudness)(data)
    return loudness


@struct.dataclass
class AudioTree:
    audio_data: jnp.ndarray
    sample_rate: int
    loudness: float = None
    metadata: dict = struct.field(pytree_node=True, default_factory=dict)

    def replace_loudness(self):
        loudness = jit_integrated_loudness(self.audio_data, self.sample_rate, zeros=512)
        return self.replace(loudness=loudness)

    @classmethod
    def from_file(cls, audio_path: str, sample_rate: int = None, offset: float = 0, duration: float = None, mono=False):
        data, sr = librosa.load(audio_path, sr=sample_rate, offset=offset, duration=duration, mono=mono)
        assert sr == sample_rate
        data = jnp.array(data, dtype=jnp.float32)
        if data.ndim == 1:
            data = data[None, :]  # Add channel dimension
        if data.ndim == 2:
            data = data[None, :, :]  # Add batch dimension

        if duration is not None and data.shape[-1] < round(duration * sample_rate):
            pad_right = round(duration * sample_rate) - data.shape[-1]
            data = jnp.pad(data, ((0, 0), (0, 0), (0, pad_right)))

        return cls(
            audio_data=data,
            sample_rate=sr,
            metadata={"offset": offset, "duration": duration}
        )

    @classmethod
    def from_array(cls, audio_array: np.ndarray, sample_rate: int):
        audio_data = jnp.array(audio_array, dtype=jnp.float32)
        if audio_data.ndim == 1:
            audio_data = audio_data[None, :]  # Add channel dimension
        if audio_data.ndim == 2:
            audio_data = audio_data[None, :, :]  # Add batch dimension
        return cls(audio_data=audio_data, sample_rate=sample_rate)

    @classmethod
    def excerpt(cls, audio_path: str, rng: np.random.RandomState, offset: float = None, duration: float = None,
                **kwargs):
        assert duration is not None and duration > 0
        info = soundfile.info(audio_path)
        total_duration = info.duration  # seconds

        lower_bound = 0 if offset is None else offset
        upper_bound = max(total_duration - duration, 0)
        offset = rng.uniform(lower_bound, upper_bound)

        audio_signal = cls.from_file(audio_path=audio_path, offset=offset, duration=duration, **kwargs)

        return audio_signal

    @classmethod
    def salient_excerpt(
        cls,
        audio_path: Union[str, Path],
        rng: np.random.RandomState,
        loudness_cutoff: float = None,
        num_tries: int = 8,
        **kwargs,
    ):
        assert 'offset' not in kwargs, "``salient_excerpt`` cannot be used with kwarg ``offset``."
        assert 'duration' in kwargs, "``salient_excerpt`` must be used with kwarg ``duration``."
        if loudness_cutoff is None:
            excerpt = cls.excerpt(audio_path, rng=rng, **kwargs)
        else:
            loudness = -np.inf
            num_try = 0
            while loudness <= loudness_cutoff:
                num_try += 1
                new_excerpt = cls.excerpt(audio_path, rng=rng, **kwargs).replace_loudness()
                if num_try == 1 or new_excerpt.loudness > loudness:
                    excerpt, loudness, = new_excerpt, new_excerpt.loudness
                if num_tries is not None and num_try >= num_tries:
                    break
        return excerpt

    def to_mono(self):
        audio_data = jnp.mean(self.audio_data, axis=1, keepdims=True)
        return self.replace(audio_data=audio_data)

    def resample(self, sample_rate: int):
        if sample_rate == self.sample_rate:
            return self
        raise NotImplementedError("Not implemented yet.")

    def write(self, audio_path: str):
        soundfile.write(audio_path, self.audio_data[0].T, self.sample_rate)
