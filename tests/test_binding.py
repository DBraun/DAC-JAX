import os.path
from pathlib import Path
import tempfile

import jax.numpy as jnp
import librosa

import dac_jax


def test_binding():

    # Download a model and bind variables to it.
    model, variables = dac_jax.load_model(model_type="44khz")
    model = model.bind(variables)

    # Load audio file
    filepath = Path(__file__).parent / "assets" / "60013__qubodup__whoosh.flac"
    signal, sample_rate = librosa.load(filepath, sr=44100, mono=True, duration=0.5)

    signal = jnp.array(signal, dtype=jnp.float32)
    while signal.ndim < 3:
        signal = jnp.expand_dims(signal, axis=0)

    # Encode audio signal as one long file (may run out of GPU memory on long files)
    dac_file = model.encode_to_dac(signal, sample_rate)

    with tempfile.TemporaryDirectory() as tmpdirname:
        filepath = os.path.join(tmpdirname, "dac_file_001.dac")

        # Save to a file
        dac_file.save(filepath)

        # Load a file
        dac_file = dac_jax.DACFile.load(filepath)

    # Decode audio signal
    y = model.decode(dac_file)

    # reconstruction mean-square error
    mse = jnp.square(y - signal).mean()

    # Informal expected maximum MSE
    assert mse.item() < 0.005


if __name__ == "__main__":
    test_binding()
