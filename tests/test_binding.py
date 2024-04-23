import dac_jax
from dac_jax.audio_utils import volume_norm, db2linear

import jax.numpy as jnp
import librosa
from pathlib import Path


def test_binding():

    # Download a model and bind variables to it.
    model, variables = dac_jax.load_model(model_type="44khz")
    model = model.bind(variables)

    # Load audio file
    filepath = Path(__file__).parent / 'assets' / '60013__qubodup__whoosh.flac'
    signal, sample_rate = librosa.load(filepath, sr=44100, mono=True, duration=.5)

    signal = jnp.array(signal, dtype=jnp.float32)
    while signal.ndim < 3:
        signal = jnp.expand_dims(signal, axis=0)

    target_db = -16  # Normalize audio to -16 dB
    x, input_db = volume_norm(signal, target_db, sample_rate)

    # Encode audio signal as one long file (may run out of GPU memory on long files)
    x = model.preprocess(x, sample_rate)
    z, codes, latents, commitment_loss, codebook_loss = model.encode(x, train=False)

    # Decode audio signal
    y = model.decode(z, length=signal.shape[-1])

    # Undo previous loudness normalization
    y = y * db2linear(input_db - target_db)

    # reconstruction mean-square error
    mse = jnp.square(y-signal).mean()

    # Informal expected maximum MSE
    assert mse.item() < .005


if __name__ == '__main__':
    test_binding()
