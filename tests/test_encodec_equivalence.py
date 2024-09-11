import os

os.environ["XLA_FLAGS"] = (
    " --xla_gpu_deterministic_ops=true"  # todo: https://github.com/google/flax/discussions/3382
)
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from pathlib import Path

from audiocraft.models import MusicGen
from jax import numpy as jnp
from jax import random
import librosa
import numpy as np
import torch

from dac_jax import load_encodec_model
from dac_jax.nn.encodec_quantize import QuantizedResult


def run_jax_model(np_data):

    x = jnp.array(np_data)

    encodec_model, variables = load_encodec_model("facebook/musicgen-small")

    result: QuantizedResult = encodec_model.apply(
        variables, x, rngs={"rng_stream": random.key(0)}
    )
    recons = result.recons
    codes = result.codes

    return np.array(recons), np.array(codes)


def run_torch_model(np_data):
    model = MusicGen.get_pretrained("facebook/musicgen-small")
    x = torch.from_numpy(np_data).cuda()
    result = model.compression_model(x)

    recons = result.x.detach().cpu().numpy()
    codes = result.codes.detach().cpu().numpy()

    return recons, codes


def test_encoded_equivalence():
    np_data, sr = librosa.load(
        Path(__file__).parent / "assets/60013__qubodup__whoosh.flac", sr=None, mono=True
    )
    np_data = np.expand_dims(np.array(np_data), 0)
    np_data = np.expand_dims(np.array(np_data), 0)
    np_data = np.concatenate([np_data, np_data, np_data, np_data], axis=-1)

    np_data *= 0.5

    torch_recons, torch_codes = run_torch_model(np_data)
    jax_recons, jax_codes = run_jax_model(np_data)

    assert np.allclose(torch_codes, jax_codes)
    assert np.allclose(torch_recons, jax_recons, atol=1e-4)  # todo: reduce atol to 1e-5


if __name__ == "__main__":
    test_encoded_equivalence()
