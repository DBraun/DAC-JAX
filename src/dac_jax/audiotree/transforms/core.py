"""
This file is heavily influenced by Descript AudioTools:
https://github.com/descriptinc/audiotools/blob/master/audiotools/data/transforms.py

License: MIT
https://github.com/descriptinc/audiotools/blob/master/LICENSE
"""

from functools import partial
from typing import Optional, Tuple, Union

import chex
from einops import rearrange
import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np

from grain._src.core.transforms import TfRandomMapTransform
import grain.python as grain

from dac_jax.audiotree import AudioTree


def stft(x: jnp.ndarray, frame_length=2048, hop_factor=0.25, window='hann'):

    batch_size, num_channels, audio_length = x.shape

    frame_step = int(frame_length * hop_factor)

    x = rearrange(x, 'b c t -> (b c) t')

    # This probably uses less memory than the aux method, but it's definitely slower than aux.
    _, _, stft_data = jax.scipy.signal.stft(x,
                                            window=window,
                                            nperseg=frame_length,
                                            noverlap=(frame_length - frame_step),
                                            # padded=False,
                                            # boundary='even',
                                            )
    stft_data = rearrange(stft_data, '(b c) nf nt -> b c nf nt', b=batch_size)

    return stft_data


def istft(stft_matrix: chex.Array,
          noverlap: int,
          window: Optional[Union[str, float, Tuple[str, float]]] = 'hann',
          length: Optional[int] = None) -> chex.Array:
    _, reconstructed_signal = jax.scipy.signal.istft(stft_matrix,
                                                     noverlap=noverlap,
                                                     window=window,
                                                     )

    # Trim or pad the output signal to the desired length
    if length is not None:
        if length > reconstructed_signal.shape[-1]:
            # Pad the signal if it is shorter than the desired length
            pad_width = length - reconstructed_signal.shape[-1]
            reconstructed_signal = jnp.pad(reconstructed_signal, ((0, 0), (0, 0), (0, pad_width)),
                                           mode='constant')
        else:
            # Trim the signal if it is longer than the desired length
            reconstructed_signal = reconstructed_signal[..., :length]

    return reconstructed_signal


class Identity(grain.MapTransform):

    def map(self, element):
        return element


def _where_with_p(rng, modified: jnp.ndarray, original: jnp.ndarray, p: float):
    B = original.shape[0]

    shape = (B,) + (1,) * (original.ndim - 1)
    use_modified = random.uniform(rng, shape=shape, minval=0) < p

    return jnp.where(use_modified, modified, original)


def _db2linear(decibels):
    return jnp.pow(10.0, decibels / 20.0)


@partial(jax.jit, donate_argnums=0)
def _volume_norm_transform(audio_tree: AudioTree, key: jnp.ndarray, min_db: float, max_db: float, p: float) -> (
        AudioTree):

    audio_data = audio_tree.audio_data

    B = audio_data.shape[0]

    key, subkey = random.split(key)
    target_db = random.uniform(subkey, shape=(B,), minval=min_db, maxval=max_db)
    gain_db = target_db - audio_tree.loudness

    modified = audio_data * _db2linear(gain_db)[:, None, None]

    key, subkey = random.split(key)
    use_modified = random.uniform(subkey, shape=(B,)) < p
    use_modified = jnp.expand_dims(use_modified, axis=(1, 2))

    audio_data = jnp.where(use_modified, modified, audio_data)
    loudness = jnp.where(use_modified, target_db, audio_tree.loudness)

    audio_tree = audio_tree.replace(audio_data=audio_data, loudness=loudness)

    return audio_tree


@partial(jax.jit, donate_argnums=0)
def _volume_change_transform(audio_tree: AudioTree, key: jnp.ndarray, min_db, max_db, p) -> (
        tuple)[AudioTree, np.ndarray]:

    audio_data = audio_tree.audio_data

    B = audio_data.shape[0]

    key, subkey = random.split(key)
    gain_db = random.uniform(subkey, shape=(B,), minval=min_db, maxval=max_db)

    key, subkey = random.split(key)
    gain_db = _where_with_p(subkey, gain_db, jnp.zeros_like(gain_db), p)

    audio_data = audio_data * _db2linear(gain_db)[:, None, None]

    audio_tree = audio_tree.replace(audio_data=audio_data)

    return audio_tree, gain_db


class VolumeChange(grain.RandomMapTransform):

    def __init__(self, min_db: float = 0, max_db: float = 0, prob: float = 1):
        self.min_db = min_db
        self.max_db = max_db
        assert 0 <= prob <= 1
        self.prob = prob

    def random_map(self, audio_tree: AudioTree, rng: np.random.Generator) -> AudioTree:
        if self.prob == 0:
            return audio_tree

        subkey = random.key(rng.integers(2**63))
        audio_tree, gain_db = _volume_change_transform(audio_tree, subkey, self.min_db, self.max_db, self.prob)
        if audio_tree.loudness is not None:
            audio_tree = audio_tree.replace(loudness=(audio_tree.loudness+gain_db))

        return audio_tree


class VolumeNorm(grain.RandomMapTransform):

    def __init__(self, min_db: float = 0, max_db: float = 0, prob: float = 1):
        self.min_db = min_db
        self.max_db = max_db
        assert 0 <= prob <= 1
        self.prob = prob

    def random_map(self, audio_tree: AudioTree, rng: np.random.Generator) -> AudioTree:
        if self.prob == 0:
            return audio_tree

        if audio_tree.loudness is None:
            audio_tree = audio_tree.replace_loudness()

        subkey = random.key(rng.integers(2**63))
        
        return _volume_norm_transform(audio_tree, subkey, self.min_db, self.max_db, self.prob)


@partial(jax.jit, donate_argnums=0)
def _rescale_audio_transform(audio_tree: AudioTree, rng, p) -> AudioTree:
    """Rescales audio to the range [-1, 1] only if the original audio exceeds those bounds. Useful if transforms have
    caused the audio to clip. It won't change the relative balance of multichannel audio."""
    audio_data = audio_tree.audio_data
    maxes = jnp.max(jnp.absolute(audio_data), axis=[-2, -1], keepdims=True)
    maxes = jnp.maximum(maxes, jnp.ones_like(maxes))
    modified = audio_data / maxes

    audio_data = _where_with_p(rng, modified, audio_data, p)

    return audio_tree.replace(audio_data=audio_data)


class RescaleAudio(grain.RandomMapTransform):

    def __init__(self, prob: float = 1):
        assert 0 <= prob <= 1
        self.prob = prob

    def random_map(self, audio_tree: AudioTree, rng: np.random.Generator) -> AudioTree:
        subkey = random.key(rng.integers(2 ** 63))
        return _rescale_audio_transform(audio_tree, subkey, self.prob).replace(loudness=None)


@partial(jax.jit, donate_argnums=0)
def _invert_phase_audio_transform(audio_tree: AudioTree, rng: jnp.ndarray, p: float) -> AudioTree:
    audio_data = audio_tree.audio_data

    inverted = -audio_data

    audio_data = _where_with_p(rng, inverted, audio_data, p)

    return audio_tree.replace(audio_data=audio_data)


class InvertPhase(grain.RandomMapTransform):

    def __init__(self, prob: float = 1):
        """
        With a probability ``prob``, invert the phase of both channels of audio.

        :param prob: The probability between 0 and 1 of inverting the phase of both channels.
        """
        assert 0 <= prob <= 1
        self.prob = prob

    def random_map(self, audio_tree: AudioTree, rng: np.random.Generator) -> AudioTree:
        if self.prob == 0:
            return audio_tree

        subkey = random.key(rng.integers(2 ** 63))
        return _invert_phase_audio_transform(audio_tree, subkey, self.prob)


@partial(jax.jit, donate_argnames='audio_tree')
def _swap_stereo_audio_transform(audio_tree: AudioTree, rng: jnp.ndarray, p: float) -> AudioTree:
    audio_data = audio_tree.audio_data

    swapped = jnp.flip(audio_data, axis=1)

    audio_data = _where_with_p(rng, swapped, audio_data, p)

    return audio_tree.replace(audio_data=audio_data)


class SwapStereo(grain.RandomMapTransform):

    def __init__(self, prob: float = 1):
        """
        With a probability ``prob``, swap the channels of stereo audio.

        :param prob: The probability between 0 and 1 of swapping stereo channels.
        """
        assert 0 <= prob <= 1
        self.prob = prob

    def random_map(self, audio_tree: AudioTree, rng: np.random.Generator) -> AudioTree:
        if self.prob == 0:
            return audio_tree

        subkey = random.key(rng.integers(2 ** 63))
        return _swap_stereo_audio_transform(audio_tree, subkey, self.prob)


@partial(jax.jit, donate_argnums=0, static_argnums=(2, 3, 4, 5, 6))
def _corrupt_phase(
        audio_tree: AudioTree,
        rng: jnp.ndarray,
        p: float,
        amount: float,
        hop_factor: float = 0.5,
        frame_length: float = 2048,
        window: str = 'hann',
):
    audio_data = audio_tree.audio_data
    B, C, length = audio_data.shape

    frame_step = int(frame_length * hop_factor)
    noverlap = frame_length - frame_step

    stft_fun = partial(stft, frame_length=frame_length, hop_factor=hop_factor, window=window)
    istft_fun = partial(istft, noverlap=noverlap, window=window, length=length)

    stft_data = stft_fun(audio_data)

    amt = random.uniform(rng, shape=stft_data.shape[:-1], minval=-jnp.pi*amount, maxval=jnp.pi*amount)

    stft_data = stft_data * jnp.expand_dims(jnp.exp(1j * amt), axis=-1)
    shifted = istft_fun(stft_data)

    audio_data = _where_with_p(rng, shifted, audio_data, p)

    return audio_tree.replace(audio_data=audio_data)


@partial(jax.jit, donate_argnums=0, static_argnums=(2, 3, 4, 5, 6))
def _shift_phase(
        audio_tree: AudioTree,
        key: jnp.ndarray,
        p: float,
        amount: float,
        hop_factor: float = 0.5,
        frame_length: float = 2048,
        window: str = 'hann',
):
    audio_data = audio_tree.audio_data
    B, C, length = audio_data.shape

    frame_step = int(frame_length * hop_factor)
    noverlap = frame_length - frame_step

    stft_fun = partial(stft, frame_length=frame_length, hop_factor=hop_factor, window=window)
    istft_fun = partial(istft, noverlap=noverlap, window=window, length=length)

    stft_data = stft_fun(audio_data)

    key, subkey = random.split(key)
    amt = random.uniform(subkey, shape=stft_data.shape[:-2], minval=-jnp.pi*amount, maxval=jnp.pi*amount)

    stft_data = stft_data * jnp.expand_dims(jnp.exp(1j * amt), axis=(-2, -1))
    shifted = istft_fun(stft_data)

    key, subkey = random.split(key)
    audio_data = _where_with_p(subkey, shifted, audio_data, p)

    return audio_tree.replace(audio_data=audio_data)


class CorruptPhase(grain.RandomMapTransform):

    def __init__(self, amount: float = 1, prob: float = 1):
        """
        With a probability ``prob``, perform a phase corruption on the audio. The phase shift range is in the range
         [-pi * amount, pi * amount], and it's independently selected for each frequency in the STFT.

        :param p: The probability between 0 and 1 of swapping stereo channels.
        """
        assert 0 <= prob <= 1
        self.prob = prob
        self.amount = amount

    def random_map(self, audio_tree: AudioTree, rng: np.random.Generator) -> AudioTree:
        if self.prob == 0:
            return audio_tree

        subkey = random.key(rng.integers(2 ** 63))
        return _corrupt_phase(audio_tree, subkey, self.prob, self.amount)


class ShiftPhase(grain.RandomMapTransform):

    def __init__(self, amount: float = 1, prob: float = 1):
        """
        With a probability ``prob``, perform a phase shift on the audio. The phase shift range is in the range
         [-pi * amount, pi * amount].

        :param p: The probability between 0 and 1 of swapping stereo channels.
        """
        assert 0 <= prob <= 1
        self.prob = prob
        self.amount = amount

    def random_map(self, audio_tree: AudioTree, rng: np.random.Generator) -> AudioTree:
        if self.prob == 0:
            return audio_tree

        subkey = random.key(rng.integers(2 ** 63))
        return _shift_phase(audio_tree, subkey, self.prob, self.amount)


class Choose(grain.RandomMapTransform):

    """
    With probability ``prob``, choose a ``c`` transform(s) among ``transforms`` with optional probability weights
    ``weights``
    """

    def __init__(self, *transforms, c: int = 1, weights=None, prob: float = 1):

        if weights is not None:
            assert len(weights) == len(transforms)

        assert c <= len(transforms)

        self.c = c
        self.weights = weights
        assert 0 <= prob <= 1
        self.prob = prob

        self.transforms = transforms

    def random_map(self, element, rng: np.random.Generator):

        # Reference:
        # https://github.com/google/grain/blob/2a45a382a378a3737f0df76ba2c6ac7cc2dc43b6/grain/_src/python/lazy_dataset/
        # transformations/map.py#L95-L112

        if rng.random() >= self.prob:
            return element

        transforms = rng.choice(self.transforms, size=(self.c,), replace=False, p=self.weights)

        for transform in transforms:

            if isinstance(transform, grain.MapTransform):
                element = transform.map(element)
            elif isinstance(transform, grain.RandomMapTransform):
                element = transform.random_map(element, rng)
            elif isinstance(transform, TfRandomMapTransform):
                element = transform.np_random_map(element, rng)
            else:
                # If a `seed` is provided we treat the Callable as RandomMapTransform
                element = transform(element, rng)

        return element


class ReduceBatchTransform(grain.MapTransform):

    def map(self, audio_signal: AudioTree) -> AudioTree:

        def f(leaf):
            if isinstance(leaf, np.ndarray):
                if leaf.ndim > 1:
                    shape = leaf.shape
                    shape = (shape[0]*shape[1],) + shape[2:]
                    return jnp.reshape(leaf, shape=shape)
            return leaf

        audio_signal = jax.tree_util.tree_map(f, audio_signal)

        # We do this to make sample rate a scalar instead of an array.
        # We assume that all entries of audio_signal.sample_rate are the same.
        audio_signal = audio_signal.replace(sample_rate=audio_signal.sample_rate[0])

        return audio_signal
