from functools import partial
from itertools import product

import jax.numpy as jnp
import jax.random as random
import pytest

from dac_jax.audiotree.transforms.helpers import stft, istft


@pytest.mark.parametrize("hop_factor,length", product(
    [0.25, 0.5],
    [44100, 44101],
))
def test_istft_invariance(hop_factor: float, length: int):

    # Show that it's possible to use STFT, then ISTFT and recover the input.

    audio_data = random.uniform(random.key(0), shape=(1, 1, length), minval=-1)

    frame_length = 2048
    window = 'hann'

    frame_step = int(frame_length * hop_factor)
    noverlap = frame_length - frame_step

    stft_fun = partial(stft, frame_length=frame_length, hop_factor=hop_factor, window=window)
    istft_fun = partial(istft, noverlap=noverlap, window=window, length=length)

    stft_data = stft_fun(audio_data)

    recons = istft_fun(stft_data)

    assert jnp.allclose(recons, audio_data, atol=1e-4)
