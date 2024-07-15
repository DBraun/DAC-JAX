from functools import partial

import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np

import grain.python as grain

from dac_jax.audio_utils import stft, istft

from audio_tree_core import AudioTree


@partial(jax.jit, donate_argnums=0)
def _volume_transform(audio_tree: AudioTree, rng: jnp.ndarray, min_db, max_db) -> AudioTree:

    def db2linear(decibels):
        return jnp.pow(10.0, decibels / 20.0)

    audio_data = audio_tree.audio_data

    B = audio_data.shape[0]

    target_db = random.uniform(rng, shape=(B,), minval=min_db, maxval=max_db)
    gain_db = target_db - audio_tree.loudness

    audio_data = audio_data * db2linear(gain_db)[:, None, None]

    audio_tree = audio_tree.replace(audio_data=audio_data)

    return audio_tree


class VolumeTransform(grain.RandomMapTransform):

    def __init__(self, enabled: int = 1, min_db=0, max_db=0):
        self.enabled = bool(enabled)
        self.min_db = min_db
        self.max_db = max_db

    def random_map(self, audio_tree: AudioTree, rng: np.random.Generator) -> AudioTree:
        if not self.enabled:
            return audio_tree

        if audio_tree.loudness is None:
            audio_tree = audio_tree.replace_loudness()
        subkey = random.key(rng.integers(2**63))
        return _volume_transform(audio_tree, subkey, self.min_db, self.max_db).replace(loudness=None)


@partial(jax.jit, donate_argnums=0)
def _rescale_audio_transform(audio_tree: AudioTree) -> AudioTree:
    """Rescales audio to the range [-1, 1] only if the original audio exceeds those bounds. Useful if transforms have
    caused the audio to clip. It won't change the relative balance of multichannel audio."""
    audio_data = audio_tree.audio_data
    maxes = jnp.max(jnp.absolute(audio_data), axis=[-2, -1], keepdims=True)
    maxes = jnp.maximum(maxes, jnp.ones_like(maxes))
    audio_data = audio_data / maxes
    return audio_tree.replace(audio_data=audio_data)


class RescaleAudioTransform(grain.MapTransform):

    def map(self, audio_tree: AudioTree) -> AudioTree:
        return _rescale_audio_transform(audio_tree).replace(loudness=None)


@partial(jax.jit, donate_argnums=0)
def _invert_phase_audio_transform(audio_tree: AudioTree, rng: jnp.ndarray, p: float) -> AudioTree:
    audio_data = audio_tree.audio_data

    B = audio_data.shape[0]

    mult = 1-2*(random.uniform(rng, shape=(B,)) < p).astype(jnp.int32)
    mult = jnp.expand_dims(mult, axis=(1, 2))

    audio_data = audio_data * mult

    return audio_tree.replace(audio_data=audio_data)


class InvertPhaseAudioTransform(grain.RandomMapTransform):

    def __init__(self, p: float = 0):
        """
        With a probability ``p``, invert the phase of both channels of audio.

        :param p: The probability between 0 and 1 of inverting the phase of both channels.
        """
        assert 0 <= p <= 1
        self.p = p

    def random_map(self, audio_tree: AudioTree, rng: np.random.Generator) -> AudioTree:

        if self.p == 0:
            return audio_tree

        rng = random.key(rng.integers(2 ** 63))
        return _invert_phase_audio_transform(audio_tree, rng, self.p)


@partial(jax.jit, donate_argnames='audio_tree')
def _swap_stereo_audio_transform(audio_tree: AudioTree, rng: jnp.ndarray, p: float) -> AudioTree:
    audio_data = audio_tree.audio_data

    B = audio_data.shape[0]

    swapped = jnp.flip(audio_data, axis=1)

    do_swap = random.uniform(rng, shape=(B,), minval=0) < p
    do_swap = jnp.expand_dims(do_swap, axis=(1, 2))

    audio_data = jnp.where(do_swap, swapped, audio_data)

    return audio_tree.replace(audio_data=audio_data)


class SwapStereoAudioTransform(grain.RandomMapTransform):

    def __init__(self, p: float = 0):
        """
        With a probability ``p``, swap the channels of stereo audio.

        :param p: The probability between 0 and 1 of swapping stereo channels.
        """
        assert 0 <= p <= 1
        self.p = p

    def random_map(self, audio_tree: AudioTree, rng: np.random.Generator) -> AudioTree:
        if self.p == 0:
            return audio_tree

        rng = random.key(rng.integers(2 ** 63))
        return _swap_stereo_audio_transform(audio_tree, rng, self.p)


@partial(jax.jit, donate_argnums=0, static_argnums=(2, 3, 4))
def _phase_shift(
        audio_tree: AudioTree,
        rng: jnp.ndarray,
        p: float,
        hop_factor: float = 0.5,
        frame_length: float = 2048,
        window: str = 'hann',
):
    audio_data = audio_tree.audio_data
    B, C, length = audio_data.shape

    stft_fun = partial(stft, frame_length=frame_length, hop_factor=hop_factor, window=window, match_stride=False,
                       padding_type='reflect')
    istft_fun = partial(istft, window=window, length=length)

    stft_data = stft_fun(audio_data)

    amt = random.uniform(rng, shape=stft_data.shape[:-1], minval=-jnp.pi, maxval=jnp.pi)

    stft_data = stft_data * jnp.expand_dims(jnp.exp(1j * amt), axis=-1)
    shifted = istft_fun(stft_data)

    do_swap = random.uniform(rng, shape=(B,), minval=0) < p
    do_swap = jnp.expand_dims(do_swap, axis=(1, 2))

    audio_data = jnp.where(do_swap, shifted, audio_data)

    return audio_tree.replace(audio_data=audio_data)


class PhaseShiftAudioTransform(grain.RandomMapTransform):

    def __init__(self, p: float = 0):
        """
        With a probability ``p``, perform a phase shift on the audio

        :param p: The probability between 0 and 1 of swapping stereo channels.
        """
        assert 0 <= p <= 1
        self.p = p

    def random_map(self, audio_tree: AudioTree, rng: np.random.Generator) -> AudioTree:
        if self.p == 0:
            return audio_tree

        rng = random.key(rng.integers(2 ** 63))
        return _phase_shift(audio_tree, rng, self.p)
