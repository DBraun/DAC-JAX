from typing import Callable, Optional
from functools import partial
import os

import numpy as np

import jax
import jax.numpy as jnp

from einops import rearrange

import dm_aux as aux

from dac_jax.audio_utils import stft, decibel_loudness


def l1_loss(y_true: jnp.ndarray,
            y_pred: jnp.ndarray,
            reduction='mean') -> jnp.ndarray:

    errors = jnp.abs(y_pred - y_true)
    if reduction == 'none':
        return errors
    elif reduction == 'mean':
        return jnp.mean(errors)
    elif reduction == 'sum':
        return jnp.sum(errors)
    else:
        raise ValueError(f"Invalid reduction method: {reduction}")


def sisdr_loss(y_true: jnp.ndarray,
               y_pred: jnp.ndarray,
               scaling: int = True,
               reduction: str = 'mean',
               zero_mean: int = True,
               clip_min: int = None):
    """
    Computes the Scale-Invariant Source-to-Distortion Ratio between a batch
    of estimated and reference audio signals or aligned features.

    Parameters
    ----------
    y_true : jnp.ndarray
        Estimate jnp.ndarray
    y_pred : jnp.ndarray
        Reference jnp.ndarray
    scaling : int, optional
        Whether to use scale-invariant (True) or
        signal-to-noise ratio (False), by default True
    reduction : str, optional
        How to reduce across the batch (either 'mean',
        'sum', or none).], by default ' mean'
    zero_mean : int, optional
        Zero mean the references and estimates before
        computing the loss, by default True
    clip_min : int, optional
        The minimum possible loss value. Helps network
        to not focus on making already good examples better, by default None

    Implementation copied from: https://github.com/descriptinc/lyrebird-audiotools/blob/961786aa1a9d628cca0c0486e5885a457fe70c1a/audiotools/metrics/distance.py
    """

    eps = 1e-8
    # nb, nc, nt
    references = y_true
    estimates = y_pred

    nb = references.shape[0]
    references = references.reshape(nb, 1, -1).transpose(0, 2, 1)
    estimates = estimates.reshape(nb, 1, -1).transpose(0, 2, 1)

    # samples now on axis 1
    if zero_mean:
        mean_reference = references.mean(axis=1, keepdims=True)
        mean_estimate = estimates.mean(axis=1, keepdims=True)
    else:
        mean_reference = 0
        mean_estimate = 0

    _references = references - mean_reference
    _estimates = estimates - mean_estimate

    references_projection = jnp.square(_references).sum(axis=-2) + eps
    references_on_estimates = (_estimates * _references).sum(axis=-2) + eps

    scale = (
        jnp.expand_dims(references_on_estimates / references_projection, 1)
        if scaling
        else 1
    )

    e_true = scale * _references
    e_res = _estimates - e_true

    signal = jnp.square(e_true).sum(axis=1)
    noise = jnp.square(e_res).sum(axis=1)
    sdr = -10 * jnp.log10(signal / noise + eps)

    if clip_min is not None:
        sdr = jnp.clip(sdr, a_min=clip_min)

    if reduction == "mean":
        sdr = sdr.mean()
    elif reduction == "sum":
        sdr = sdr.sum()
    return sdr


def discriminator_loss(fake, real):
    """
    Computes a discriminator loss, given the outputs of the discriminator
    used on a fake input and a real input.
    """
    d_fake, d_real = jax.lax.stop_gradient(fake), real

    loss_d = 0
    for x_fake, x_real in zip(d_fake, d_real):
        loss_d = loss_d + jnp.square(x_fake[-1]).mean()
        loss_d = loss_d + jnp.square(1 - x_real[-1]).mean()
    return loss_d


def generator_loss(fake, real):
    """
    Computes a generator loss, given the outputs of the discriminator
    used on a fake input and a real input.
    """
    d_fake, d_real = fake, jax.lax.stop_gradient(real)

    loss_g = 0
    for x_fake in d_fake:
        loss_g = loss_g + jnp.square(1 - x_fake[-1]).mean()

    loss_feature = 0

    for i in range(len(d_fake)):
        for j in range(len(d_fake[i]) - 1):
            loss_feature = loss_feature + l1_loss(d_fake[i][j], d_real[i][j])
    return loss_g, loss_feature


def multiscale_stft_loss(y_true: jnp.ndarray,
                         y_pred: jnp.ndarray,
                         window_lengths=None,
                         loss_fn: Callable = l1_loss,
                         clamp_eps: float = 1e-5,
                         mag_weight: float = 1.0,
                         log_weight: float = 1.0,
                         pow: float = 2.0,
                         match_stride: Optional[bool] = False,
                         window: str = 'hann'
                         ):
    """Computes the multiscale STFT loss from [1].

    Parameters
    ----------
    y_true : AudioSignal
        Estimate signal
    y_pred : AudioSignal
        Reference signal
    window_lengths : List[int], optional
        Length of each window of each STFT, by default [2048, 512]
    loss_fn : typing.Callable, optional
        How to compare each loss, by default l1_loss
    clamp_eps : float, optional
        Clamp on the log magnitude, below, by default 1e-5
    mag_weight : float, optional
        Weight of raw magnitude portion of loss, by default 1.0
    log_weight : float, optional
        Weight of log magnitude portion of loss, by default 1.0
    pow : float, optional
        Power to raise magnitude to before taking log, by default 2.0
    match_stride : bool, optional
        Whether to match the stride of convolutional layers, by default False
    window : str or tuple or array_like, optional
        Desired window to use. If `window` is a string or tuple, it is
        passed to `get_window` to generate the window values, which are
        DFT-even by default. See `get_window` for a list of windows and
        required parameters. If `window` is array_like it will be used
        directly as the window and its length must be nperseg. Defaults
        to a Hann window.

    Returns
    -------
    jnp.ndarray
        Multi-scale STFT loss.

    References
    ----------

    1.  Engel, Jesse, Chenjie Gu, and Adam Roberts.
        "DDSP: Differentiable Digital Signal Processing."
        International Conference on Learning Representations. 2019.

    Implementation copied from: https://github.com/descriptinc/lyrebird-audiotools/blob/961786aa1a9d628cca0c0486e5885a457fe70c1a/audiotools/metrics/spectral.py
    """

    x = y_pred
    y = y_true

    loss = jnp.zeros(())

    if window_lengths is None:
        window_lengths = [2048, 512]

    for frame_length in window_lengths:
        stft_fun = partial(stft, frame_length=frame_length, hop_factor=0.25, window=window,
                           match_stride=match_stride)
        x_stft = stft_fun(x)
        y_stft = stft_fun(y)

        loss = loss + log_weight * loss_fn(decibel_loudness(x_stft, clamp_eps=clamp_eps, pow=pow),
                                           decibel_loudness(y_stft, clamp_eps=clamp_eps, pow=pow))
        loss = loss + mag_weight * loss_fn(jnp.abs(x_stft), jnp.abs(y_stft))

    return loss


def mel_spectrogram_loss(y_true: jnp.ndarray,
                         y_pred: jnp.ndarray,
                         sample_rate: int,
                         n_mels=None,
                         window_lengths=None,
                         loss_fn: Callable = l1_loss,
                         clamp_eps: float = 1e-5,
                         mag_weight: float = 1.0,
                         log_weight: float = 1.0,
                         pow: float = 2.0,
                         match_stride: Optional[bool] = False,
                         lower_edge_hz=None,
                         upper_edge_hz=None,
                         window: str = 'hann',
                         ):
    """Compute distance between mel spectrograms. Can be used in a multiscale way.

    Parameters
    ----------
    y_true : jnp.ndarray
        Estimate signal
    y_pred : jnp.ndarray
        Reference signal
    sample_rate : int
        Sample rate
    n_mels : List[int]
        Number of mel bins per STFT, by default [150, 80],
    window_lengths : List[int], optional
        Length of each window of each STFT, by default [2048, 512]
    loss_fn : typing.Callable, optional
        How to compare each loss, by default L1Loss()
    clamp_eps : float, optional
        Clamp on the log magnitude, below, by default 1e-5
    mag_weight : float, optional
        Weight of raw magnitude portion of loss, by default 1.0
    log_weight : float, optional
        Weight of log magnitude portion of loss, by default 1.0
    pow : float, optional
        Power to raise magnitude to before taking log, by default 2.0
    match_stride : bool, optional
        Whether to match the stride of convolutional layers, by default False
    lower_edge_hz: List[float], optional
        Lowest frequency to consider to general mel filterbanks.
    upper_edge_hz: List[float], optional
        Highest frequency to consider to general mel filterbanks.
    window : str or tuple or array_like, optional
        Desired window to use. If `window` is a string or tuple, it is
        passed to `get_window` to generate the window values, which are
        DFT-even by default. See `get_window` for a list of windows and
        required parameters. If `window` is array_like it will be used
        directly as the window and its length must be nperseg. Defaults
        to a Hann window.

    Returns
    -------
    jnp.ndarray
        Mel loss.

    Implementation copied from: https://github.com/descriptinc/lyrebird-audiotools/blob/961786aa1a9d628cca0c0486e5885a457fe70c1a/audiotools/metrics/spectral.py
    """

    x = y_pred
    y = y_true

    if n_mels is None:
        n_mels = [150, 80]

    if window_lengths is None:
        window_lengths = [2048, 512]

    if lower_edge_hz is None:
        lower_edge_hz = [0., 0.]

    if upper_edge_hz is None:
        upper_edge_hz = [None, None]  # librosa converts None to sample_rate/2

    def decibel_fn(mels: jnp.ndarray) -> jnp.ndarray:
        return jnp.log10(jnp.pow(jnp.maximum(mels, clamp_eps), pow))

    loss = jnp.zeros(())
    for features, fmin, fmax, frame_length in zip(n_mels, lower_edge_hz, upper_edge_hz, window_lengths):

        def spectrogram_fn(signal):
            stft_data = stft(signal, frame_length=frame_length, hop_factor=0.25, window=window,
                             match_stride=match_stride)
            stft_data = rearrange(stft_data, 'b c nf nt -> (b c) nt nf')

            spectrogram = jnp.abs(stft_data)
            return spectrogram

        mel_fun = partial(aux.spectral.mel_spectrogram, log_scale=False, sample_rate=sample_rate,
                          frame_length=frame_length, num_features=features, lower_edge_hertz=fmin,
                          upper_edge_hertz=fmax)

        x_spectrogram = spectrogram_fn(x)
        y_spectrogram = spectrogram_fn(y)

        x_mels = mel_fun(x_spectrogram)
        y_mels = mel_fun(y_spectrogram)

        loss = loss + log_weight * loss_fn(decibel_fn(x_mels), decibel_fn(y_mels))
        loss = loss + mag_weight * loss_fn(x_mels, y_mels)

    return loss


def phase_loss(y_true: jnp.ndarray,
               y_pred: jnp.ndarray,
               window_length: int = 2048,
               hop_factor: float = 0.25,
               ):
    """Computes phase loss between an estimate and a reference signal.

    Parameters
    ----------
    y_true : AudioSignal
        Reference signal
    y_pred : AudioSignal
        Estimate signal
    window_length : int, optional
        Length of STFT window, by default 2048
    hop_factor : float, optional
        Hop factor between 0 and 1, which is multiplied by the length of STFT
        window length to determine the hop size.

    Returns
    -------
    jnp.ndarray
        Phase loss.

    Implementation adapted from https://github.com/descriptinc/audiotools/blob/7776c296c711db90176a63ff808c26e0ee087263/audiotools/metrics/spectral.py#L195
    """

    x = y_pred
    y = y_true

    stft_fun = partial(stft, frame_length=window_length, hop_factor=hop_factor, window='hann')

    x_stft = stft_fun(x)
    y_stft = stft_fun(y)

    def phase(spec):
        return jnp.angle(spec)

    # Take circular difference
    diff = phase(x_stft) - phase(y_stft)
    diff = diff.at[diff < -jnp.pi].set(diff[diff < -jnp.pi] + 2 * jnp.pi)
    diff = diff.at[diff > jnp.pi].set(diff[diff > jnp.pi - 2 * jnp.pi])

    # Scale true magnitude to weights in [0, 1]
    x_mag = jnp.abs(x_stft)
    x_min, x_max = x_mag.min(), x_mag.max()
    weights = (x_mag - x_min) / (x_max - x_min)

    # Take weighted mean of all phase errors
    loss = jnp.square(weights * diff).mean()
    return loss


def stoi(
    estimates: jnp.ndarray,
    references: jnp.ndarray,
    sample_rate: int,
    extended: int = False,
):
    """Short term objective intelligibility
    Computes the STOI (See [1][2]) of a de-noised signal compared to a clean
    signal, The output is expected to have a monotonic relation with the
    subjective speech-intelligibility, where a higher score denotes better
    speech intelligibility. Uses pystoi under the hood.

    Parameters
    ----------
    estimates : jnp.ndarray
        De-noised speech
    references : jnp.ndarray
        Clean original speech
    sample_rate: int
        Sample rate of the references
    extended : int, optional
        Boolean, whether to use the extended STOI described in [3], by default False

    Returns
    -------
    Tensor[float]
        Short time objective intelligibility measure between clean and
        de-noised speech

    References
    ----------
    1.  C.H.Taal, R.C.Hendriks, R.Heusdens, J.Jensen 'A Short-Time
        Objective Intelligibility Measure for Time-Frequency Weighted Noisy
        Speech', ICASSP 2010, Texas, Dallas.
    2.  C.H.Taal, R.C.Hendriks, R.Heusdens, J.Jensen 'An Algorithm for
        Intelligibility Prediction of Time-Frequency Weighted Noisy Speech',
        IEEE Transactions on Audio, Speech, and Language Processing, 2011.
    3.  Jesper Jensen and Cees H. Taal, 'An Algorithm for Predicting the
        Intelligibility of Speech Masked by Modulated Noise Maskers',
        IEEE Transactions on Audio, Speech and Language Processing, 2016.
    """
    import pystoi

    if estimates.ndim == 3:
        estimates = jnp.average(estimates, axis=-2)  # to mono
    if references.ndim == 3:
        references = jnp.average(references, axis=-2)  # to mono

    stois = []
    for reference, estimate in zip(references, estimates):
        _stoi = pystoi.stoi(
            np.array(reference),
            np.array(estimates),
            sample_rate,
            extended=extended,
        )
        stois.append(_stoi)
    return jnp.array(np.array(stois))


def pesq(
    estimates: jnp.ndarray,
    estimates_sample_rate: int,
    references: jnp.ndarray,
    references_sample_rate: int,
    mode: str = "wb",
    target_sr: int = 16000,
):
    """_summary_

    Parameters
    ----------
    estimates : jnp.ndarray
        Degraded audio signal
    estimates_sample_rate: int
        Sample rate of the estimates
    references : jnp.ndarray
        Reference audio signal
    references_sample_rate: int
        Sample rate of the references
    mode : str, optional
        'wb' (wide-band) or 'nb' (narrow-band), by default "wb"
    target_sr : int, optional
        Target sample rate, by default 16000

    Returns
    -------
    Tensor[float]
        PESQ score: P.862.2 Prediction (MOS-LQO)
    """
    from pesq import pesq as pesq_fn
    from ..resample import resample

    if estimates.ndim == 3:
        estimates = jnp.average(estimates, axis=-2, keepdims=True)  # to mono
    if references.ndim == 3:
        references = jnp.average(references, axis=-2, keepdims=True)  # to mono

    estimates = resample(estimates, old_sr=estimates_sample_rate, new_sr=target_sr)
    references = resample(references, old_sr=references_sample_rate, new_sr=target_sr)

    pesqs = []
    for reference, estimate in zip(references, estimates):
        _pesq = pesq_fn(
            estimates_sample_rate,
            np.array(reference[0]),
            np.array(estimate[0]),
            mode,
        )
        pesqs.append(_pesq)
    return jnp.array(np.array(pesqs))


def visqol(
    estimates: jnp.ndarray,
    estimates_sample_rate: int,
    references: jnp.ndarray,
    references_sample_rate: int,
    mode: str = "audio",
):  # pragma: no cover
    """ViSQOL score.

    Parameters
    ----------
    estimates : jnp.ndarray
        Degraded audio
    references : jnp.ndarray
        Reference audio
    mode : str, optional
        'audio' or 'speech', by default 'audio'

    Returns
    -------
    Tensor[float]
        ViSQOL score (MOS-LQO)
    """
    from visqol import visqol_lib_py
    from visqol.pb2 import visqol_config_pb2
    from visqol.pb2 import similarity_result_pb2
    from ..resample import resample

    config = visqol_config_pb2.VisqolConfig()
    if mode == "audio":
        target_sr = 48000
        config.options.use_speech_scoring = False
        svr_model_path = "libsvm_nu_svr_model.txt"
    elif mode == "speech":
        target_sr = 16000
        config.options.use_speech_scoring = True
        svr_model_path = "lattice_tcditugenmeetpackhref_ls2_nl60_lr12_bs2048_learn.005_ep2400_train1_7_raw.tflite"
    else:
        raise ValueError(f"Unrecognized mode: {mode}")
    config.audio.sample_rate = target_sr
    config.options.svr_model_path = os.path.join(
        os.path.dirname(visqol_lib_py.__file__), "model", svr_model_path
    )

    api = visqol_lib_py.VisqolApi()
    api.Create(config)

    if estimates.ndim == 3:
        estimates = jnp.average(estimates, axis=-2, keepdims=True)  # to mono
    if references.ndim == 3:
        references = jnp.average(references, axis=-2, keepdims=True)  # to mono

    estimates = resample(estimates, old_sr=estimates_sample_rate, new_sr=target_sr)
    references = resample(references, old_sr=references_sample_rate, new_sr=target_sr)

    visqols = []
    for reference, estimate in zip(references, estimates):
        _visqol = api.Measure(
            np.array(reference[0], dtype=np.float32),
            np.array(estimate[0], dtype=np.float32),
        )
        visqols.append(_visqol.moslqo)
    return jnp.array(np.array(visqols))
