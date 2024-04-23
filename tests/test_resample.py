import librosa
from scipy.io import wavfile
import numpy as np
import jax.numpy as jnp

from dac_jax.resample import resample


def test_resample(filepath='assets/60013__qubodup__whoosh.flac', new_sr=96_000):

    y, sr = librosa.load(filepath, sr=None, mono=False, duration=10)

    y = jnp.array(y)

    y = jnp.expand_dims(y, 0)
    y = y[:, :1, :]
    # print('y shape: ', y.shape)

    y = resample(y, old_sr=sr, new_sr=new_sr)
    # print('y shape: ', y.shape)
    y = y.squeeze(0).T
    y = np.array(y)

    # todo: use the torch version of julius and confirm the outputs match.
    # (DBraun did this manually once but didn't automate it.)

    wavfile.write("tmp_resampled_output.wav", new_sr, y)
