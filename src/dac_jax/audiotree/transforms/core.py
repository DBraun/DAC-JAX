"""
This file is heavily influenced by Descript AudioTools:
https://github.com/descriptinc/audiotools/blob/master/audiotools/data/transforms.py

License: MIT
https://github.com/descriptinc/audiotools/blob/master/LICENSE
"""

from typing import Dict, Any

import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np

from grain._src.core.transforms import TfRandomMapTransform
import grain.python as grain

from dac_jax.audiotree import AudioTree
from dac_jax.audiotree.transforms.helpers import _volume_norm_transform, _volume_change_transform, \
    _rescale_audio_transform, _invert_phase_audio_transform, _swap_stereo_audio_transform, _corrupt_phase, \
    _shift_phase, BaseRandomTransform, BaseMapTransform


class Identity(grain.MapTransform):

    def map(self, element):
        return element


class VolumeChange(BaseRandomTransform):
    @staticmethod
    def get_default_config() -> Dict[str, Any]:
        return {
            "min_db": 0,
            "max_db": 0,
        }

    @staticmethod
    def _apply_transform(audio_tree: AudioTree, rng: random.PRNGKey, min_db: float, max_db: float):
        audio_tree, gain_db = _volume_change_transform(audio_tree, rng, min_db, max_db)
        if audio_tree.loudness is not None:
            audio_tree = audio_tree.replace(loudness=(audio_tree.loudness+gain_db))
        return audio_tree


class VolumeNorm(BaseRandomTransform):

    @staticmethod
    def get_default_config() -> Dict[str, Any]:
        return {
            "min_db": 0,
            "max_db": 0,
        }

    @staticmethod
    def _apply_transform(audio_tree: AudioTree, rng: random.PRNGKey, min_db: float, max_db: float):
        return _volume_norm_transform(audio_tree, rng, min_db, max_db)


class RescaleAudio(BaseMapTransform):

    @staticmethod
    def get_default_config() -> Dict[str, Any]:
        return {}

    @staticmethod
    def _apply_transform(audio_tree: AudioTree):
        return _rescale_audio_transform(audio_tree).replace(loudness=None)


class InvertPhase(BaseMapTransform):
    """
    Invert the phase of both channels of audio.
    """

    @staticmethod
    def get_default_config() -> Dict[str, Any]:
        return {}

    @staticmethod
    def _apply_transform(audio_tree: AudioTree):
        return _invert_phase_audio_transform(audio_tree)


class SwapStereo(BaseMapTransform):
    """
    Swap the channels of stereo audio.
    """

    @staticmethod
    def get_default_config() -> Dict[str, Any]:
        return {}

    @staticmethod
    def _apply_transform(audio_tree: AudioTree):
        return _swap_stereo_audio_transform(audio_tree)


class CorruptPhase(BaseRandomTransform):
    """
    Perform a phase corruption on the audio. The phase shift range is in the range
     [-pi * amount, pi * amount], and it's independently selected for each frequency in the STFT.
    """

    @staticmethod
    def get_default_config() -> Dict[str, Any]:
        return {
            "amount": 1,
            "hop_factor": 0.5,
            "frame_length": 2048,
            "window": "hann",
        }

    @staticmethod
    def _apply_transform(audio_tree: AudioTree, rng: random.PRNGKey, amount: float, hop_factor: float,
                         frame_length: int, window: str):
        _corrupt_phase(audio_tree, rng, amount, hop_factor, frame_length, window)


class ShiftPhase(grain.RandomMapTransform):
    """
    Perform a phase shift on the audio. The phase shift range is in the range [-pi * amount, pi * amount].
    """

    @staticmethod
    def get_default_config() -> Dict[str, Any]:
        return {
            "amount": 1,
        }

    @staticmethod
    def _apply_transform(audio_tree: AudioTree, rng: random.PRNGKey, amount: float):
        return _shift_phase(audio_tree, rng, amount)


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
