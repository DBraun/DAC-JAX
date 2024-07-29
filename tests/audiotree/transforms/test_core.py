from typing import Dict, Any

import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np

import pytest

from dac_jax.audiotree import AudioTree
from dac_jax.audiotree.transforms.helpers import BaseRandomTransform, BaseMapTransform
from dac_jax.audiotree.transforms import VolumeChange


class ReturnConfigTransform(BaseRandomTransform):
    @staticmethod
    def get_default_config() -> Dict[str, Any]:
        return {
            "minval": 0,
            "maxval": 1,
        }

    @staticmethod
    def _apply_transform(element: jnp.ndarray, rng: random.PRNGKey, minval: float, maxval: float):
        return {
            'minval': minval,
            'maxval': maxval
        }


class AddSomethingTransform(BaseMapTransform):
    @staticmethod
    def get_default_config() -> Dict[str, Any]:
        return {
            "offset": 1,
        }

    @staticmethod
    def _apply_transform(element: jnp.ndarray, offset: int):
        return element + offset


@pytest.mark.parametrize("split_seed", [False, True])
def test_config_001(split_seed: bool):

    v = jnp.full((1,), fill_value=1)
    element = {"a": v, "b": v, "c": [v, v], "d": {"e": v, "f": v}}

    config = {
        "b": {"minval": -1, "maxval": 3},
        "c": {"maxval": 4},
        "d":
            # todo: it's ugly how a config parameter `minval` can potentially clash with the data `e` or `f`
            {"minval": -3,
             "e": {"minval": -2},
             "f": {"maxval": 2}
             }
    }

    transform = ReturnConfigTransform(config=config, split_seed=split_seed)
    seed = 42
    transformed_element = transform.random_map(element, seed)
    transformed_element = jax.tree.map(lambda x: np.array(x).tolist(), transformed_element)

    expected = {
        "a": {"minval": 0, "maxval": 1},
        "b": {"minval": -1, "maxval": 3},
        "c": [
            {"minval": 0, "maxval": 4},
            {"minval": 0, "maxval": 4}
        ],
        "d": {
            "e": {"minval": -2, "maxval": 1},
            "f": {"minval": -3, "maxval": 2},
        }
    }
    assert expected == transformed_element


def test_scope():

    v = jnp.zeros((1,))
    element = {
        "a": v,
        "b": v,
        "c": [v, v],
        "d": {
            "e": v,
            "f": v
        },
        "g": [v, v]
    }

    scope = {
        "b": {"scope": True},
        "d": {
            # todo: it's ugly how `scope` can potentially clash with `f`
            "scope": True,
            "f": {"scope": False}},
        "g": {"scope": True},
    }

    transform = AddSomethingTransform(scope=scope)
    transformed_element = transform.map(element)
    transformed_element = jax.tree.map(lambda x: np.array(x.reshape()).astype(int).tolist(), transformed_element)

    expected = {
        "a": 0,
        "b": 1,
        "c": [0, 0],
        "d": {
            "e": 1,
            "f": 0,
        },
        "g": [1, 1],
    }
    assert expected == transformed_element


def test_volume_change():

    audio_tree = AudioTree(audio_data=jnp.ones(shape=(1, 1, 44100)), sample_rate=44100)

    config = {
        "min_db": 20,
        "max_db": 20,
    }
    transform = VolumeChange(config=config)

    seed = 0
    transformed_element = transform.random_map(audio_tree, seed=seed)

    assert jnp.allclose(audio_tree.audio_data*10, transformed_element.audio_data)
