import math
from typing import Optional, Union, Tuple, List
from pathlib import Path
import glob

import chex

import jax.scipy.signal
import jax.numpy as jnp

import jaxloudnorm as jln

import dm_aux as aux

from einops import rearrange


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


def compute_stft_padding(length, window_length: int, hop_length: int, match_stride: bool):
    """Compute how the STFT should be padded, based on match\_stride.

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


def stft(x: jnp.ndarray, frame_length=2048, hop_factor=0.25, window='hann', match_stride=False,
         padding_type: str = 'reflect', use_scipy=False):

    """Reference:
    https://github.com/descriptinc/audiotools/blob/7776c296c711db90176a63ff808c26e0ee087263/audiotools/core/audio_signal.py#L1123
    """

    batch_size, num_channels, audio_length = x.shape

    frame_step = int(frame_length * hop_factor)

    right_pad, pad = compute_stft_padding(audio_length, frame_length, frame_step, match_stride)
    x = jnp.pad(x, pad_width=((0, 0), (0, 0), (pad, pad + right_pad)), mode=padding_type)
    x = rearrange(x, 'b c t -> (b c) t')

    if use_scipy:
        # This probably uses less memory than the aux method, but it's definitely slower than aux.
        _, _, stft_data = jax.scipy.signal.stft(x,
                                                window=window,
                                                nperseg=frame_length,
                                                noverlap=(frame_length - frame_step),
                                                nfft=frame_length,
                                                detrend=False,
                                                return_onesided=True,
                                                boundary='zeros',
                                                padded=True,
                                                )
        stft_data = rearrange(stft_data, '(b c) nf nt -> b c nf nt', b=batch_size)
    else:
        # todo: https://github.com/google-deepmind/dm_aux/issues/2
        stft_data = aux.spectral.stft(x, n_fft=frame_length, frame_step=frame_step, window_fn=window)
        stft_data = rearrange(stft_data, '(b c) nt nf -> b c nf nt', b=batch_size)

    if match_stride:
        # Drop first two and last two frames, which are added
        # because of padding. Now num_frames * hop_length = num_samples.
        stft_data = stft_data[..., 2:-2]

    return stft_data


def istft(stft_matrix: chex.Array,
          window: Optional[Union[str, float, Tuple[str, float]]] = 'hann',
          length: Optional[int] = None) -> chex.Array:
    """
    Computes the inverse Short-time Fourier Transform (iSTFT) of the signal using jax.scipy.signal.istft.

    Args:
        stft_matrix: input complex matrix of shape [batch_size, num_frames, n_fft // 2 + 1].
        frame_length: the size of each signal frame. If unspecified it defaults to be equal to `n_fft`.
        frame_step: the hop size of extracting signal frames. If unspecified it defaults to be equal to `int(frame_length // 2)`.
        window: applied to each frame to remove the discontinuities at the edge of the frame introduced by segmentation.
        pad: pad the signal at the end(s) by `int(n_fft // 2)`. Can either be `Pad.NONE`, `Pad.START`, `Pad.END`, `Pad.BOTH`, `Pad.ALIGNED`.
        length: the trim length of the time domain signal to output.
        precision: precision of the convolution. Either `None`, which means the default precision for the backend, or a `lax.Precision` enum value.

    Returns:
        The reconstructed time domain signal of shape `[batch_size, signal_length]`.

    Reference:
    https://github.com/descriptinc/audiotools/blob/7776c296c711db90176a63ff808c26e0ee087263/audiotools/core/audio_signal.py#L1214
    """
    # Compute iSTFT
    _, reconstructed_signal = jax.scipy.signal.istft(stft_matrix,
                                                     fs=1.0,
                                                     window=window,
                                                     input_onesided=True,
                                                     boundary=True,
                                                     # padded=True,
    )

    # Trim or pad the output signal to the desired length
    if length is not None:
        if length > reconstructed_signal.shape[-1]:
            # Pad the signal if it is shorter than the desired length
            pad_width = length - reconstructed_signal.shape[-1]
            reconstructed_signal = jnp.pad(reconstructed_signal, ((0, 0), (0, 0), (0, pad_width)), mode='constant')
        else:
            # Trim the signal if it is longer than the desired length
            reconstructed_signal = reconstructed_signal[..., :length]

    return reconstructed_signal


def decibel_loudness(stft_data: jnp.ndarray, clamp_eps=1e-5, pow=2.) -> jnp.ndarray:
    # todo: maybe do maximum right before log10
    return jnp.log10(jnp.power(jnp.maximum(jnp.abs(stft_data), clamp_eps), pow))


def db2linear(decibels):
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
        padded_audio = jnp.pad(padded_audio,
                               pad_width=((0, 0), (0, 0), (0, int(block_size*sample_rate)-original_length)))

    # create BS.1770 meter
    meter = jln.Meter(sample_rate, filter_class=filter_class, block_size=block_size, use_fir=True, zeros=zeros)

    # measure loudness
    loudness = jax.vmap(meter.integrated_loudness)(rearrange(padded_audio, 'b c t -> b t c'))

    loudness = jnp.maximum(loudness, jnp.full_like(loudness, min_loudness))

    audio_data = audio_data * db2linear(target_db-loudness)[:, None, None]

    return audio_data, loudness
