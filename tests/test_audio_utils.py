from dac_jax.audio_utils import stft, istft

from functools import partial
import jax.random as random
import pytest

import jax.numpy as jnp


@pytest.mark.parametrize("match_stride", [True, False])
def test_stft_equivalence(match_stride):

    # Test that use_scipy=True produces the same result as use_scipy=False, for both choices of match_stride

    audio_data = random.uniform(random.key(0), shape=(2, 1, 44100), minval=-1)

    hop_factor = 0.25
    frame_length = 2048
    window = 'hann'

    stft_fun = partial(stft, frame_length=frame_length, hop_factor=hop_factor, window=window, match_stride=match_stride,
                       padding_type='reflect')

    stft_fun_scipy = partial(stft_fun, use_scipy=True)

    stft_data = stft_fun(audio_data)
    stft_data2 = stft_fun_scipy(audio_data)

    assert jnp.allclose(stft_data, stft_data2, atol=1e-4)


def test_stft_equivalence2():

    # Test that use_scipy and dm_aux stft return same values for hop factor 0.5

    audio_data = random.uniform(random.key(0), shape=(2, 1, 44100), minval=-1)

    hop_factor = 0.5
    frame_length = 2048
    window = 'hann'

    stft_fun = partial(stft, frame_length=frame_length, hop_factor=hop_factor, window=window, match_stride=False,
                       padding_type='reflect')

    stft_fun_scipy = partial(stft_fun, use_scipy=True)

    stft_data = stft_fun(audio_data)
    stft_data2 = stft_fun_scipy(audio_data)

    assert jnp.allclose(stft_data, stft_data2, atol=1e-4)


@pytest.mark.parametrize("hop_factor", [0.25, 0.5])
def test_istft_invariance(hop_factor):

    audio_data = random.uniform(random.key(0), shape=(2, 1, 44100), minval=-1)

    B, C, length = audio_data.shape

    frame_length = 2048
    window = 'hann'

    stft_fun = partial(stft, frame_length=frame_length, hop_factor=hop_factor, window=window, match_stride=False,
                       padding_type='reflect')

    frame_step = int(frame_length * hop_factor)
    noverlap = frame_length - frame_step

    istft_fun = partial(istft, frame_length=frame_length, noverlap=noverlap, window=window, length=length)

    stft_data = stft_fun(audio_data)

    recons = istft_fun(stft_data)

    assert jnp.allclose(recons, audio_data, atol=1e-5)
