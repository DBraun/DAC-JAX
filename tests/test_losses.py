"""Check that the losses and metrics match Descript's implementations in magnitude.

tests/test_audio_utils.py already covers the mel and multi-scale STFT losses; this
covers the rest of dac_jax.nn.loss against descript-audio-codec and audiotools.
"""

import numpy as np
import pytest

import jax.numpy as jnp
import torch
import torch.nn.functional as F
from audiotools import AudioSignal
from audiotools import metrics as at_metrics
import dac.nn.loss as dac_loss

from dac_jax.nn.loss import (
    l1_loss,
    sisdr_loss,
    generator_loss,
    discriminator_loss,
    stoi,
    pesq,
)

SR = 44100
RTOL = 1e-4


@pytest.fixture(scope="module")
def signals():
    rng = np.random.default_rng(0)
    a = (0.5 * rng.standard_normal((2, 1, SR))).astype(np.float32)
    b = (a + 0.05 * rng.standard_normal((2, 1, SR))).astype(np.float32)
    return a, b


@pytest.fixture(scope="module")
def disc_outputs():
    """Synthetic discriminator outputs, so the loss math is checked on its own.

    Eight discriminators (five periods plus three FFT sizes), matching the shape
    of conf/base.yml, each with a stack of feature maps whose last entry is the
    logit.
    """
    rng = np.random.default_rng(1)
    fake, real = [], []
    for n_maps in (6, 6, 6, 6, 6, 26, 26, 26):
        fake.append([rng.standard_normal((2, 4, 16)).astype(np.float32) for _ in range(n_maps)])
        real.append([rng.standard_normal((2, 4, 16)).astype(np.float32) for _ in range(n_maps)])
    return fake, real


def _gan_loss(fake, real):
    """descript's GANLoss with the discriminator short-circuited out."""
    gan = dac_loss.GANLoss(discriminator=None)
    tf_ = [[torch.from_numpy(x) for x in d] for d in fake]
    tr = [[torch.from_numpy(x) for x in d] for d in real]
    gan.forward = lambda a, b: (tf_, tr)
    return gan


def test_l1_loss_same_as_torch(signals):
    a, b = signals
    assert np.allclose(
        float(l1_loss(jnp.asarray(a), jnp.asarray(b))),
        float(F.l1_loss(torch.from_numpy(b), torch.from_numpy(a))),
        rtol=RTOL,
    )


def test_sisdr_loss_same_as_dac_torch(signals):
    a, b = signals
    jax_v = float(sisdr_loss(jnp.asarray(a), jnp.asarray(b)))
    torch_v = float(
        dac_loss.SISDRLoss()(AudioSignal(torch.from_numpy(b), SR),
                             AudioSignal(torch.from_numpy(a), SR))
    )
    assert np.allclose(jax_v, torch_v, rtol=RTOL)


def test_discriminator_loss_same_as_dac_torch(signals, disc_outputs):
    fake, real = disc_outputs
    a, b = signals
    jf = [[jnp.asarray(x) for x in d] for d in fake]
    jr = [[jnp.asarray(x) for x in d] for d in real]
    gan = _gan_loss(fake, real)
    sa, sb = AudioSignal(torch.from_numpy(a), SR), AudioSignal(torch.from_numpy(b), SR)
    assert np.allclose(
        float(discriminator_loss(jf, jr)),
        float(gan.discriminator_loss(sb, sa)),
        rtol=RTOL,
    )


def test_generator_loss_same_as_dac_torch(signals, disc_outputs):
    fake, real = disc_outputs
    a, b = signals
    jf = [[jnp.asarray(x) for x in d] for d in fake]
    jr = [[jnp.asarray(x) for x in d] for d in real]
    gan = _gan_loss(fake, real)
    sa, sb = AudioSignal(torch.from_numpy(a), SR), AudioSignal(torch.from_numpy(b), SR)

    jax_gen, jax_feat = generator_loss(jf, jr)
    torch_gen, torch_feat = gan.generator_loss(sb, sa)

    # The feature loss is a sum over every feature map of every discriminator and
    # is not normalized, exactly as in the PyTorch version -- conf/base.yml's
    # adv/feat_loss weight of 2.0 assumes that magnitude.
    assert np.allclose(float(jax_gen), float(torch_gen), rtol=RTOL)
    assert np.allclose(float(jax_feat), float(torch_feat), rtol=RTOL)


def _speech():
    import librosa

    clean, _ = librosa.load(
        librosa.example("brahms"), sr=SR, mono=True, offset=35.0, duration=2.0
    )
    clean = clean[None, None, :].astype(np.float32)
    noisy = (clean + 0.02 * np.random.default_rng(0).standard_normal(clean.shape)).astype(
        np.float32
    )
    return clean, noisy


def test_stoi_same_as_audiotools():
    clean, noisy = _speech()
    jax_v = float(np.ravel(np.asarray(stoi(jnp.asarray(noisy), jnp.asarray(clean), SR)))[0])
    torch_v = float(
        at_metrics.quality.stoi(
            AudioSignal(torch.from_numpy(noisy), SR), AudioSignal(torch.from_numpy(clean), SR)
        )[0]
    )
    assert np.allclose(jax_v, torch_v, rtol=RTOL)


def test_pesq_same_as_audiotools():
    clean, noisy = _speech()
    jax_v = float(
        np.ravel(np.asarray(pesq(jnp.asarray(noisy), SR, jnp.asarray(clean), SR)))[0]
    )
    torch_v = float(
        at_metrics.quality.pesq(
            AudioSignal(torch.from_numpy(noisy), SR), AudioSignal(torch.from_numpy(clean), SR)
        )[0]
    )
    assert np.allclose(jax_v, torch_v, rtol=RTOL)
