from itertools import product

from einops import rearrange
import jax.numpy as jnp
import numpy as np
import pytest

from dac_jax.audio_utils import stft, mel_spectrogram
from dac_jax.nn.loss import mel_spectrogram_loss, multiscale_stft_loss

from dac.nn.loss import MelSpectrogramLoss, MultiScaleSTFTLoss
from audiotools import AudioSignal


@pytest.mark.parametrize("match_stride,hop_factor,length,use_scipy", product(
    [False, True],
    [0.25, 0.5],
    [44100, 44101],
    [False, True],
))
def test_mel_same_as_audiotools(match_stride: bool, hop_factor: float, length: int, use_scipy: bool):

    if hop_factor == 0.5 and match_stride:
        return  # for some reason DAC torch disallows this

    sample_rate = 44100

    B = 1
    x = np.random.uniform(low=-1, high=1, size=(B, 1, length))

    signal1 = AudioSignal(x, sample_rate=sample_rate)

    window_length = 2048
    hop_length = int(window_length*hop_factor)

    stft_kwargs = {
        'window_length': window_length,
        'hop_length': hop_length,
        'window_type': 'hann',
        'match_stride': match_stride,
        'padding_type': 'reflect',
    }

    n_mels = 80

    mel1 = signal1.mel_spectrogram(n_mels=n_mels, **stft_kwargs)

    stft1 = signal1.stft_data

    stft_data = stft(jnp.array(x),
                     frame_length=stft_kwargs['window_length'],
                     hop_factor=hop_factor,
                     window=stft_kwargs['window_type'],
                     match_stride=stft_kwargs['match_stride'],
                     padding_type=stft_kwargs['padding_type'],
                     )

    assert np.allclose(np.abs(stft1), np.abs(stft_data), atol=1e-4)

    stft_data = rearrange(stft_data, 'b c nf nt -> (b c) nt nf')

    spectrogram = jnp.abs(stft_data)

    mel2 = mel_spectrogram(spectrogram, log_scale=False, sample_rate=sample_rate, num_features=n_mels,
                           frame_length=stft_kwargs['window_length'])

    mel2 = rearrange(mel2, '(b c) t bins -> b c bins t', b=B)

    assert np.allclose(mel1, np.array(mel2), atol=1e-4)


@pytest.mark.parametrize("length",
    (44100, 44101),
)
def test_mel_loss_same_as_dac_torch(length: int):

    sample_rate = 44100

    x1 = np.random.uniform(low=-1, high=1, size=(1, 1, length))
    x2 = x1*0.5

    signal1 = AudioSignal(x1, sample_rate=sample_rate)
    signal2 = AudioSignal(x2, sample_rate=sample_rate)

    loss1 = mel_spectrogram_loss(jnp.array(x1), jnp.array(x2), sample_rate=sample_rate)
    loss2 = MelSpectrogramLoss()(signal1, signal2)

    assert np.isclose(np.array(loss1), loss2)

@pytest.mark.parametrize("length",
    (44100, 44101),
)
def test_multiscale_stft_loss_same_as_dac_torch(length: int):
    sample_rate = 44100

    x1 = np.random.uniform(low=-1, high=1, size=(1, 1, length))
    x2 = x1*0.5

    signal1 = AudioSignal(x1, sample_rate=sample_rate)
    signal2 = AudioSignal(x2, sample_rate=sample_rate)

    loss1 = multiscale_stft_loss(jnp.array(x1), jnp.array(x2))
    loss2 = MultiScaleSTFTLoss()(signal1, signal2)

    assert np.isclose(np.array(loss1), loss2)


if __name__ == '__main__':
    # test_mel_same_as_audiotools()
    # test_mel_loss_same_as_dac_torch()
    # test_multiscale_stft_loss_same_as_dac_torch()
    # test_stft_equivalence(True)
    # test_stft_equivalence(False)
    # test_stft_equivalence2(0.5)
    test_mel_same_as_audiotools(False, 0.25, 44100)