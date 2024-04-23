import warnings
from pathlib import Path

import argbind
import librosa
from tqdm import tqdm

import jax
import jax.numpy as jnp

from dac_jax import load_model
from dac_jax.audio_utils import find_audio

warnings.filterwarnings("ignore", category=UserWarning)  # ignore librosa warnings related to mel bins


@jax.jit
@argbind.bind(group="encode", positional=True, without_prefix=True)
def encode(
    input: str,
    output: str = "",
    weights_path: str = "",
    model_tag: str = "latest",
    model_bitrate: str = "8kbps",
    n_quantizers: int = None,
    model_type: str = "44khz",
    win_duration: float = 5.0,
    verbose: bool = False,
):
    """Encode audio files in input path to .dac format.

    Parameters
    ----------
    input : str
        Path to input audio file or directory
    output : str, optional
        Path to output directory, by default "". If `input` is a directory, the directory sub-tree relative to `input`
        is re-created in `output`.
    weights_path : str, optional
        Path to weights file, by default "". If not specified, the weights file will be downloaded from the internet
        using the model_tag and model_type.
    model_tag : str, optional
        Tag of the model to use, by default "latest". Ignored if `weights_path` is specified.
    model_bitrate: str
        Bitrate of the model. Must be one of "8kbps", or "16kbps". Defaults to "8kbps".
    n_quantizers : int, optional
        Number of quantizers to use, by default None. If not specified, all the quantizers will be used and the model
        will compress at maximum bitrate.
    model_type : str, optional
        The type of model to use. Must be one of "44khz", "24khz", or "16khz". Defaults to "44khz". Ignored if
        `weights_path` is specified.
    """
    model, variables = load_model(
        model_type=model_type,
        model_bitrate=model_bitrate,
        tag=model_tag,
        load_path=weights_path,
    )

    # Find all audio files in input path
    input = Path(input)
    audio_files = find_audio(input)

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)

    @jax.jit
    def compress_chunk(x):
        return model.apply(variables, x, method='compress_chunk')

    for audio_file in tqdm(audio_files, desc='Encoding files'):
        # Load file with original sample rate
        signal, sample_rate = librosa.load(audio_file, sr=None, mono=False)
        while signal.ndim < 3:
            signal = jnp.expand_dims(signal, axis=0)

        # Encode audio to .dac format
        dac_file = model.compress(compress_chunk, signal, sample_rate, win_duration=win_duration, verbose=verbose,
                                  n_quantizers=n_quantizers)

        # Compute output path
        relative_path = audio_file.relative_to(input)
        output_dir = output / relative_path.parent
        if not relative_path.name:
            output_dir = output
            relative_path = audio_file
        output_name = relative_path.with_suffix(".dac").name
        output_path = output_dir / output_name
        output_path.parent.mkdir(parents=True, exist_ok=True)

        dac_file.save(output_path)


if __name__ == "__main__":
    args = argbind.parse_args()
    with argbind.scope(args):
        encode()
