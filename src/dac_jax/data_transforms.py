import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np

import grain.python as grain

from dac_jax.audio_utils import db2linear


class VolumeTransform(grain.RandomMapTransform):

    def __init__(self, min_db=0, max_db=0):
        self.min_db = min_db
        self.max_db = max_db

    def random_map(self, audio_data: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        rng = random.key(rng.integers(2**63))

        B = audio_data.shape[0]

        # note: apply target_db based on both channels
        target_db = random.uniform(rng, shape=(B,), minval=self.min_db, maxval=self.max_db)

        audio_data = audio_data * db2linear(target_db)[None, None, :]

        # audio_data, loudness = volume_norm(audio_data, target_db, SAMPLE_RATE, filter_class="K-weighting",
        #                                    block_size=.4)

        return audio_data


class RescaleAudioTransform(grain.RandomMapTransform):

    def random_map(self, audio_data: jnp.ndarray, rng: np.random.Generator):
        """Rescales audio to the range [-1, 1] only if the original audio exceeds those bounds. Useful if transforms have
        caused the audio to clip. It won't change the relative balance of multichannel audio."""
        maxes = jnp.max(jnp.absolute(audio_data), axis=[-2, -1], keepdims=True)
        maxes = jnp.maximum(maxes, jnp.ones_like(maxes))
        audio_data = audio_data / maxes
        return audio_data


class InvertPhaseAudioTransform(grain.RandomMapTransform):

    def __init__(self, p=0):
        """
        With a probability ``p``, invert the phase of both channels of audio.

        :param p: The probability between 0 and 1 of inverting the phase of both channels.
        """
        assert 0 <= p <= 1
        self.p = p

    def random_map(self, audio_data: jnp.ndarray, rng: np.random.Generator):
        if self.p == 0:
            return audio_data

        rng = random.key(rng.integers(2**63))

        B = audio_data.shape[0]

        p = self.p
        mult = random.choice(rng, a=np.array([1, -1]), shape=(B,), p=np.array([1-p, p]))  # results in either -1 or 1
        mult = jnp.expand_dims(mult, axis=(1, 2))

        audio_data = audio_data * mult

        return audio_data


class SwapStereoAudioTransform(grain.RandomMapTransform):

    def __init__(self, p=0):
        """
        With a probability ``p``, swap the channels of stereo audio.

        :param p: The probability between 0 and 1 of swapping stereo channels.
        """
        assert 0 <= p <= 1
        self.p = p

    def random_map(self, audio_data: jnp.ndarray, rng: np.random.Generator):
        if self.p == 0:
            return audio_data

        rng = random.key(rng.integers(2**63))

        B = audio_data.shape[0]

        swapped = jnp.flip(audio_data, axis=1)

        p = self.p
        do_swap = random.choice(rng, a=np.array([0, 1]), shape=(B,), p=np.array([1-p, p]))
        do_swap = jnp.expand_dims(do_swap, axis=(1, 2))

        audio_data = jnp.where(do_swap, swapped, audio_data)

        return audio_data


class ToJaxTensorTransform(grain.MapTransform):

    def __init__(self, dtype=jnp.float32):
        self._dtype = dtype

    def map(self, x) -> jnp.ndarray:
        return jnp.array(x, dtype=self._dtype)
