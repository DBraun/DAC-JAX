from functools import partial

from einops import rearrange
import jax.numpy as jnp
import jax.random as random
import numpy as np
import pytest

from dac_jax.audio_utils import stft, istft, mel_spectrogram
from dac_jax.nn.loss import mel_spectrogram_loss, multiscale_stft_loss

from dac.nn.loss import MelSpectrogramLoss, MultiScaleSTFTLoss
from audiotools import AudioSignal


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


def test_mel_same_as_audiotools():

    sample_rate = 44100

    B = 1
    x = np.random.uniform(low=-1, high=1, size=(B, 1, sample_rate))

    signal1 = AudioSignal(x, sample_rate=sample_rate)

    stft_kwargs = {
        'window_length': 2048,
        'hop_length': 1024,
        'window_type': 'hann',
        'match_stride': False,
        'padding_type': 'reflect',
    }

    n_mels = 80
    stft1 = signal1.stft(**stft_kwargs)

    mel1 = signal1.mel_spectrogram(n_mels=n_mels, **stft_kwargs)

    stft_data = stft(jnp.array(x), frame_length=stft_kwargs['window_length'],
                     hop_factor=stft_kwargs['hop_length']/stft_kwargs['window_length'],
                     window=stft_kwargs['window_type'],
                     match_stride=stft_kwargs['match_stride'], padding_type=stft_kwargs['padding_type'])

    # todo: due to the center=True in torch.stft in audiotools, the time lengths don't line up
    #  and we have shave off one time sample. This also forces us to have a large atol.
    stft_data = stft_data[..., :-1]

    assert np.allclose(np.abs(stft1).mean(), np.abs(stft_data).mean(), atol=1e-1)

    stft_data = rearrange(stft_data, 'b c nf nt -> (b c) nt nf')

    spectrogram = jnp.abs(stft_data)

    mel2 = mel_spectrogram(spectrogram, log_scale=False, sample_rate=sample_rate, num_features=n_mels,
                           frame_length=stft_kwargs['window_length'])

    mel2 = rearrange(mel2, '(b c) t bins -> b c bins t', b=B)

    assert np.allclose(np.array(mel2), mel1, atol=0.6)  # todo: better atol


def test_mel_loss_same_as_dac_torch():

    sample_rate = 44100

    x1 = np.random.uniform(low=-1, high=1, size=(1, 1, sample_rate))
    x2 = x1*0.9

    signal1 = AudioSignal(x1, sample_rate=sample_rate)
    signal2 = AudioSignal(x2, sample_rate=sample_rate)

    loss1 = mel_spectrogram_loss(jnp.array(x1), jnp.array(x2), sample_rate=sample_rate)
    loss2 = MelSpectrogramLoss()(signal1, signal2)

    assert np.isclose(np.array(loss1), loss2, atol=1e-3)  # todo: better atol


def test_multiscale_stft_loss():
    sample_rate = 44100

    x1 = np.random.uniform(low=-1, high=1, size=(1, 1, sample_rate))
    x2 = x1*0.9

    signal1 = AudioSignal(x1, sample_rate=sample_rate)
    signal2 = AudioSignal(x2, sample_rate=sample_rate)

    loss1 = multiscale_stft_loss(jnp.array(x1), jnp.array(x2))
    loss2 = MultiScaleSTFTLoss()(signal1, signal2)

    assert np.isclose(np.array(loss1), loss2, atol=1e-2)  # todo: better atol
