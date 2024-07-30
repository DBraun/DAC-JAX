from functools import partial
from typing import Any, Dict, Optional, Tuple, Union

import chex
import jax
from einops import rearrange
from jax import numpy as jnp, random as random
from jax._src.tree_util import DictKey
from jax.tree_util import tree_map_with_path
from grain.python import MapTransform, RandomMapTransform


from dac_jax.audiotree import AudioTree

KeyLeafPairs = list[tuple[list[DictKey], Any]]


def _get_config_val(
        config: KeyLeafPairs,
        lookup_path: KeyLeafPairs,
        lookup_key: str,
        default: Any,
) -> Any:
    """
    Retrieve the configuration value for a given key and path.

    :param config: A list of key-leaf pairs from `tree_util.tree_flatten_with_path`.
    :param lookup_path: Path of the current element.
    :param lookup_key: Configuration key to look up
    :param default: Default value if key is not found
    :return: Configuration value
    """
    longest_len = 0
    matched_value = default
    for config_path, value in config:
        if config_path[-1].key == lookup_key:
            L = len(config_path)
            if config_path[:-1] == lookup_path[:L - 1] and L > longest_len:
                longest_len = L
                matched_value = value
    return matched_value


def _is_in_scope(
        scope: KeyLeafPairs,
        lookup_path: KeyLeafPairs,
) -> bool:
    """
    Retrieve the configuration value for a given key and path.

    :param scope: A list of key-leaf pairs from `tree_util.tree_flatten_with_path`.
    :param lookup_path: Path of the current element.
    :return: Boolean indicating if the path is in scope
    """
    if not scope:
        return True
    matched_value = False
    for config_path, value in scope:
        L = len(config_path)
        if config_path[:-1] == lookup_path[:L-1]:
            if not value:
                return False
            matched_value = True
    return matched_value


class BaseTransformMixIn:

    @staticmethod
    def get_default_config() -> Dict[str, Any]:
        """
        Get the default configuration for the transform.

        :return: Default configuration dictionary
        """
        raise NotImplementedError("Must be implemented in subclass")

    @staticmethod
    def _apply_transform(element, rng: random.PRNGKey, **kwargs):
        """
        Apply the transformation to the given element.

        :param element: Input element to transform
        :param rng: Random number generator key
        :param kwargs: Additional type-annotated keyword arguments for the transformation.
        """
        raise NotImplementedError("Must be implemented in subclass")


class BaseRandomTransform(BaseTransformMixIn, RandomMapTransform):

    def __init__(
            self,
            config: Dict[str, Any] = None,
            split_seed: bool = True,
            prob: float = 1.0,
            scope: Dict[str, Any] = None,
            output_key: str = None,
            ):
        """
        Initialize the base transform with a configuration, a flag for seed splitting, a probability, a scope, and an
        output key.

        :param config: Configuration dictionary for the transform
        :param split_seed: Whether to split the seed for each leaf
        :param prob: Probability of applying the transform
        :param scope: Dictionary indicating which modalities to apply the transform to
        :param output_key: Key under which to store the transformed value. By default, the values will be transformed
            in-place.
        """
        assert 0 <= prob <= 1
        self.default_config = self.get_default_config()
        self.config = jax.tree_util.tree_flatten_with_path(config or {})[0]
        self.split_seed = split_seed
        self.prob = prob
        self.scope = jax.tree_util.tree_flatten_with_path(scope or {})[0]
        self.output_key = output_key

    def random_map(self, element: Any, seed: int) -> Any:
        """
        Apply the random mapping to the given element using the provided seed.

        :param element: Input element to transform
        :param seed: Seed for the random number generator
        :return: Transformed element
        """
        key = random.PRNGKey(seed)

        # Determine if we should apply the transform
        apply_rng, key = random.split(key)
        apply_transform = random.uniform(apply_rng) < self.prob

        if not apply_transform:
            return element

        def map_func(path, leaf, rng: random.PRNGKey, *config):
            if _is_in_scope(self.scope, path):
                transformed_leaf = self._apply_transform(leaf, rng, **config[0])
                return {**leaf, self.output_key: transformed_leaf} if self.output_key else transformed_leaf
            return leaf

        def is_leaf(leaf):
            return isinstance(leaf, AudioTree) or isinstance(leaf, jnp.ndarray)  # todo: ask JAX experts about this

        treedef = jax.tree.flatten(element, is_leaf=is_leaf)[1]
        length = treedef.num_leaves
        subkeys = random.split(key, length) if self.split_seed else [key] * length
        subkeys = jax.tree.unflatten(treedef, subkeys)

        def map_use_default_config_val(path, leaf):
            return {
                key: _get_config_val(self.config, path, key, default)
                for key, default in self.default_config.items()
            }

        config = tree_map_with_path(map_use_default_config_val, element, is_leaf=is_leaf)

        return tree_map_with_path(map_func, element, subkeys, config, is_leaf=is_leaf)


class BaseMapTransform(BaseTransformMixIn, MapTransform):

    def __init__(
            self,
            config: Dict[str, Dict[str, Any]] = None,
            scope: Dict[str, Dict[str, Any]] = None,
            output_key: str = None,
            ):
        """
        Initialize the base transform with a configuration, a flag for seed splitting, a probability, a scope, and an
        output key.

        :param config: Configuration dictionary for the transform
        :param scope: Dictionary indicating which modalities to apply the transform to
        :param output_key: Key under which to store the transformed value. By default, the values will be transformed
            in-place.
        """
        self.default_config = self.get_default_config()
        self.config = jax.tree_util.tree_flatten_with_path(config or {})[0]
        self.scope = jax.tree_util.tree_flatten_with_path(scope or {})[0]
        self.output_key = output_key

    def map(self, element: Any) -> Any:
        """
        Apply the random mapping to the given element.

        :param element: Input element to transform
        :return: Transformed element
        """

        def map_func(path, leaf, *config):
            if _is_in_scope(self.scope, path):
                transformed_leaf = self._apply_transform(leaf, **config[0])
                return {**leaf, self.output_key: transformed_leaf} if self.output_key else transformed_leaf
            return leaf

        def map_use_default_config_val(path, leaf):
            return {
                key: _get_config_val(self.config, path, key, default)
                for key, default in self.default_config.items()
            }

        def is_leaf(leaf):
            return isinstance(leaf, AudioTree) or isinstance(leaf, jnp.ndarray)  # todo: ask JAX experts about this

        config = tree_map_with_path(map_use_default_config_val, element, is_leaf=is_leaf)
        return tree_map_with_path(map_func, element, config, is_leaf=is_leaf)


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


# def _where_with_p(rng, modified: jnp.ndarray, original: jnp.ndarray, p: float):
#     B = original.shape[0]
#
#     shape = (B,) + (1,) * (original.ndim - 1)
#     use_modified = random.uniform(rng, shape=shape, minval=0) < p
#
#     return jnp.where(use_modified, modified, original)


def _db2linear(decibels):
    return jnp.pow(10.0, decibels / 20.0)


@partial(jax.jit, donate_argnums=0)
def _volume_norm_transform(audio_tree: AudioTree, key: jnp.ndarray, min_db: float, max_db: float) -> (
        AudioTree):

    audio_data = audio_tree.audio_data

    B = audio_data.shape[0]

    key, subkey = random.split(key)
    target_db = random.uniform(subkey, shape=(B,), minval=min_db, maxval=max_db)
    gain_db = target_db - audio_tree.loudness

    audio_data = audio_data * _db2linear(gain_db)[:, None, None]

    audio_tree = audio_tree.replace(audio_data=audio_data, loudness=target_db)

    return audio_tree


@partial(jax.jit, donate_argnums=0)
def _volume_change_transform(audio_tree: AudioTree, key: jnp.ndarray, min_db: float, max_db: float) -> (
        tuple)[AudioTree, jnp.ndarray]:

    audio_data = audio_tree.audio_data

    B = audio_data.shape[0]

    key, subkey = random.split(key)
    gain_db = random.uniform(subkey, shape=(B,), minval=min_db, maxval=max_db)

    audio_data = audio_data * _db2linear(gain_db)[:, None, None]

    audio_tree = audio_tree.replace(audio_data=audio_data)

    return audio_tree, gain_db


@partial(jax.jit, donate_argnums=0)
def _rescale_audio_transform(audio_tree: AudioTree) -> AudioTree:
    """Rescales audio to the range [-1, 1] only if the original audio exceeds those bounds. Useful if transforms have
    caused the audio to clip. It won't change the relative balance of multichannel audio."""
    audio_data = audio_tree.audio_data
    maxes = jnp.max(jnp.absolute(audio_data), axis=[-2, -1], keepdims=True)
    maxes = jnp.maximum(maxes, jnp.ones_like(maxes))
    audio_data = audio_data / maxes

    return audio_tree.replace(audio_data=audio_data)


@partial(jax.jit, donate_argnums=0)
def _invert_phase_audio_transform(audio_tree: AudioTree) -> AudioTree:
    audio_data = audio_tree.audio_data
    audio_data = -audio_data
    return audio_tree.replace(audio_data=audio_data)


@partial(jax.jit, donate_argnums=0)
def _swap_stereo_audio_transform(audio_tree: AudioTree) -> AudioTree:
    audio_data = audio_tree.audio_data
    audio_data = jnp.flip(audio_data, axis=1)
    return audio_tree.replace(audio_data=audio_data)


# todo: potential slow-down with re-jitting due to changed static args
@partial(jax.jit, donate_argnums=0, static_argnums=(2, 3, 4, 5))
def _corrupt_phase(
        audio_tree: AudioTree,
        rng: jnp.ndarray,
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
    audio_data = istft_fun(stft_data)

    return audio_tree.replace(audio_data=audio_data)


# todo: potential slow-down with re-jitting due to changed static args
@partial(jax.jit, donate_argnums=0, static_argnums=(2, 3, 4, 5))
def _shift_phase(
        audio_tree: AudioTree,
        key: jnp.ndarray,
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
    audio_data = istft_fun(stft_data)

    return audio_tree.replace(audio_data=audio_data)
