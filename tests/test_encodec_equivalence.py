import os

os.environ["XLA_FLAGS"] = (
    " --xla_gpu_deterministic_ops=true"  # todo: https://github.com/google/flax/discussions/3382
)
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from functools import partial

import pytest

# The torch reference path still needs audiocraft, which 57359e1 dropped as a
# dependency of the library itself (dac_jax loads the checkpoint straight from
# the Hub now). Skip rather than fail collection when it is not installed.
MusicGen = pytest.importorskip("audiocraft.models").MusicGen
import jax
from jax import numpy as jnp
from jax import random
import librosa
import numpy as np
import torch

from dac_jax import load_encodec_model, QuantizedResult


def run_jax_model1(np_data):

    x = jnp.array(np_data)

    encodec_model, variables = load_encodec_model("facebook/musicgen-small")

    result: QuantizedResult = encodec_model.apply(
        variables, x, train=False, rngs={"rng_stream": random.key(0)}
    )
    recons = result.recons
    codes = result.codes
    assert codes.shape[1] == encodec_model.num_codebooks

    return np.array(recons), np.array(codes)


def run_jax_model2(np_data):
    """jax.jit version of run_jax_model1"""

    model, variables = load_encodec_model()

    @jax.jit
    def encode_to_codes(x: jnp.ndarray):
        codes, scale = model.apply(
            variables,
            x,
            method="encode",
        )
        return codes, scale

    @partial(jax.jit, static_argnums=(1, 2))
    def decode_from_codes(codes: jnp.ndarray, scale, length: int = None):
        recons = model.apply(
            variables,
            codes,
            scale,
            length,
            method="decode",
        )

        return recons

    x = jnp.array(np_data)

    original_length = x.shape[-1]

    codes, scale = encode_to_codes(x)
    assert codes.shape[1] == model.num_codebooks

    recons = decode_from_codes(codes, scale, original_length)

    return np.array(recons), np.array(codes)


def run_torch_model(np_data):
    model = MusicGen.get_pretrained("facebook/musicgen-small")
    x = torch.from_numpy(np_data).cuda()
    result = model.compression_model(x)

    recons = result.x.detach().cpu().numpy()
    codes = result.codes.detach().cpu().numpy()
    assert codes.shape[1] == model.compression_model.num_codebooks

    return recons, codes


def test_encoded_equivalence():
    # A fixed offset into the excerpt: the residual quantizer picks codes with an
    # argmin, so a frame whose latent sits almost equidistant between two codebook
    # entries can round differently in JAX than in Torch and cascade through the
    # remaining codebooks. This segment has no such near-tie.
    np_data, sr = librosa.load(
        librosa.example("brahms"), sr=44100, mono=True, offset=35.0, duration=0.5
    )
    np_data = np.expand_dims(np.array(np_data), 0)
    np_data = np.expand_dims(np.array(np_data), 0)
    np_data = np.concatenate([np_data, np_data, np_data, np_data], axis=-1)

    np_data *= 0.5

    torch_recons, torch_codes = run_torch_model(np_data)
    jax_recons, jax_codes = run_jax_model1(np_data)

    assert np.allclose(torch_codes, jax_codes)
    assert np.allclose(torch_recons, jax_recons, atol=1e-4)  # todo: reduce atol to 1e-5

    jax_recons, jax_codes = run_jax_model2(np_data)

    assert np.allclose(torch_codes, jax_codes)
    assert np.allclose(torch_recons, jax_recons, atol=1e-4)  # todo: reduce atol to 1e-5


if __name__ == "__main__":
    test_encoded_equivalence()
