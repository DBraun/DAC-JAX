import glob
import math
from pathlib import Path
from typing import List, Optional, Tuple, Union

import chex
import dm_aux as aux
from einops import rearrange
import jax.numpy as jnp
import jax.scipy.signal
import jaxloudnorm as jln
import librosa


def find_audio(folder: Union[str, Path], ext: List[str] = None) -> List[Path]:
    """Finds all audio files in a directory recursively.
    Returns a list.

    Parameters
    ----------
    folder : str
        Folder to look for audio files in, recursively.
    ext : List[str], optional
        Extensions to look for without the ., by default
        ``['.wav', '.flac', '.mp3', '.mp4']``.

    Copied from
    https://github.com/descriptinc/audiotools/blob/7776c296c711db90176a63ff808c26e0ee087263/audiotools/core/util.py#L225
    """
    if ext is None:
        ext = [".wav", ".flac", ".mp3", ".mp4"]

    folder = Path(folder)
    # Take care of case where user has passed in an audio file directly
    # into one of the calling functions.
    if str(folder).endswith(tuple(ext)):
        # if, however, there's a glob in the path, we need to
        # return the glob, not the file.
        if "*" in str(folder):
            return glob.glob(str(folder), recursive=("**" in str(folder)))
        else:
            return [folder]

    files = []
    for x in ext:
        files += folder.glob(f"**/*{x}")
    return files


def compute_stft_padding(
    length, window_length: int, hop_length: int, match_stride: bool
):
    """Compute how the STFT should be padded, based on match_stride.

    Parameters
    ----------
    length: int
    window_length : int
        Window length of STFT.
    hop_length : int
        Hop length of STFT.
    match_stride : bool
        Whether to match stride, making the STFT have the same alignment as convolutional layers.

    Returns
    -------
    tuple
        Amount to pad on either side of audio.
    """
    if match_stride:
        assert (
            hop_length == window_length // 4
        ), "For match_stride, hop must equal n_fft // 4"
        right_pad = math.ceil(length / hop_length) * hop_length - length
        pad = (window_length - hop_length) // 2
    else:
        right_pad = 0
        pad = 0

    return right_pad, pad


def stft(
    x: jnp.ndarray,
    frame_length=2048,
    hop_factor=0.25,
    window="hann",
    match_stride=False,
    padding_type: str = "reflect",
):
    """Reference:
    https://github.com/descriptinc/audiotools/blob/7776c296c711db90176a63ff808c26e0ee087263/audiotools/core/audio_signal.py#L1123
    """

    batch_size, num_channels, audio_length = x.shape

    frame_step = int(frame_length * hop_factor)

    right_pad, pad = compute_stft_padding(
        audio_length, frame_length, frame_step, match_stride
    )
    x = jnp.pad(
        x, pad_width=((0, 0), (0, 0), (pad, pad + right_pad)), mode=padding_type
    )

    x = rearrange(x, "b c t -> (b c) t")

    if window == "sqrt_hann":
        from scipy import signal as scipy_signal

        window = jnp.sqrt(scipy_signal.get_window("hann", frame_length))

    # todo: https://github.com/google-deepmind/dm_aux/issues/2
    stft_data = aux.spectral.stft(
        x,
        n_fft=frame_length,
        frame_step=frame_step,
        window_fn=window,
        pad_mode=padding_type,
        pad=aux.spectral.Pad.BOTH,
    )
    stft_data = rearrange(stft_data, "(b c) nt nf -> b c nf nt", b=batch_size)

    if match_stride:
        # Drop first two and last two frames, which are added
        # because of padding. Now num_frames * hop_length = num_samples.
        if hop_factor == 0.25:
            stft_data = stft_data[..., 2:-2]
        else:
            # I think this would be correct if DAC torch ever allowed match_stride==True and hop_factor==0.5
            stft_data = stft_data[..., 1:-1]

    return stft_data


def mel_spectrogram(
    spectrograms: chex.Array,
    log_scale: bool = True,
    sample_rate: int = 16000,
    frame_length: Optional[int] = 2048,
    num_features: int = 128,
    lower_edge_hertz: float = 0.0,
    upper_edge_hertz: Optional[float] = None,
) -> chex.Array:
    """Converts the spectrograms to Mel-scale.

    Adapted from dm_aux:
    https://github.com/google-deepmind/dm_aux/blob/77f5ed76df2928bac8550e1c5466c0dac2934be3/dm_aux/spectral.py#L312

    https://en.wikipedia.org/wiki/Mel_scale

    Args:
    spectrograms: Input spectrograms of shape [batch_size, time_steps,
      num_features].
    log_scale: Whether to return the mel_filterbanks in the log scale.
    sample_rate: The sample rate of the input audio.
    frame_length: The length of each spectrogram frame.
    num_features: The number of mel spectrogram features.
    lower_edge_hertz: Lowest frequency to consider to general mel filterbanks.
    upper_edge_hertz: Highest frequency to consider to general mel filterbanks.
      If None, use `sample_rate / 2.0`.

    Returns:
    Converted spectrograms in (log) Mel-scale.
    """
    # This setup mimics tf.signal.linear_to_mel_weight_matrix.
    linear_to_mel_weight_matrix = librosa.filters.mel(
        sr=sample_rate,
        n_fft=frame_length,
        n_mels=num_features,
        fmin=lower_edge_hertz,
        fmax=upper_edge_hertz,
    ).T
    spectrograms = jnp.matmul(spectrograms, linear_to_mel_weight_matrix)

    if log_scale:
        spectrograms = jnp.log(spectrograms + 1e-6)
    return spectrograms


def decibel_loudness(stft_data: jnp.ndarray, clamp_eps=1e-5, pow=2.0) -> jnp.ndarray:
    return jnp.log10(jnp.power(jnp.maximum(jnp.abs(stft_data), clamp_eps), pow))


def db2linear(decibels: jnp.ndarray):
    return jnp.pow(10.0, decibels / 20.0)


def volume_norm(
    audio_data: jnp.ndarray,
    target_db: jnp.ndarray,
    sample_rate: int,
    filter_class: str = "K-weighting",
    block_size: float = 0.400,
    min_loudness: float = -70,
    zeros: int = 2048,
):
    """Calculates loudness using an implementation of ITU-R BS.1770-4.
    Allows control over gating block size and frequency weighting filters for
    additional control. Measure the integrated gated loudness of a signal.

    API is derived from PyLoudnorm, but this implementation is ported to PyTorch
    and is tensorized across batches. When on GPU, an FIR approximation of the IIR
    filters is used to compute loudness for speed.

    Uses the weighting filters and block size defined by the meter
    the integrated loudness is measured based upon the gating algorithm
    defined in the ITU-R BS.1770-4 specification.

    Parameters
    ----------
    audio_data: jnp.ndarray
        audio signal [B, C, T]
    target_db: jnp.ndarray
        array of target decibel loudnesses [B]
    sample_rate: int
        sample rate of audio_data
    filter_class : str, optional
        Class of weighting filter used.
        K-weighting' (default), 'Fenton/Lee 1'
        'Fenton/Lee 2', 'Dash et al.'
        by default "K-weighting"
    block_size : float, optional
        Gating block size in seconds, by default 0.400
    min_loudness : float, optional
        Minimum loudness in decibels
    zeros : int, optional
        The length of the FIR filter. You should pick a power of 2 between 512 and 4096.

    Returns
    -------
    jnp.ndarray
        Audio normalized to `target_db` loudness
    jnp.ndarray
        Loudness of original audio data.

    Reference: https://github.com/descriptinc/audiotools/blob/master/audiotools/core/loudness.py
    """

    padded_audio = audio_data

    original_length = padded_audio.shape[-1]
    signal_duration = original_length / sample_rate

    if signal_duration < block_size:
        padded_audio = jnp.pad(
            padded_audio,
            pad_width=(
                (0, 0),
                (0, 0),
                (0, int(block_size * sample_rate) - original_length),
            ),
        )

    # create BS.1770 meter
    meter = jln.Meter(
        sample_rate,
        filter_class=filter_class,
        block_size=block_size,
        use_fir=True,
        zeros=zeros,
    )

    # measure loudness
    loudness = jax.vmap(meter.integrated_loudness)(
        rearrange(padded_audio, "b c t -> b t c")
    )

    loudness = jnp.maximum(loudness, jnp.full_like(loudness, min_loudness))

    audio_data = audio_data * db2linear(target_db - loudness)[:, None, None]

    return audio_data, loudness
