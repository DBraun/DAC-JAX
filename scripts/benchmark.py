from typing import List

import argbind
import jax
from jax import random

from dac_jax import load_model


@argbind.bind(without_prefix=True)
def benchmark_dac(model_type="44khz", model_bitrate='8kbps', win_durations: List[str] = None):

    if win_durations is None:
        win_durations = [0.37, 0.38, 0.42, 0.46, 0.5, 1, 5, 10, 20]
    else:
        win_durations = [float(x) for x in win_durations]

    # Set padding to False since we're using chunk functions.
    model, variables = load_model(model_type=model_type, model_bitrate=model_bitrate, padding=False)

    @jax.jit
    def compress_chunk(x):
        return model.apply(variables, x, method='compress_chunk')

    @jax.jit
    def decompress_chunk(c):
        return model.apply(variables, c, method='decompress_chunk')

    audio_sr = model.sample_rate  # not always a valid assumption, in case you copy-paste this elsewhere

    print(f'Benchmarking model: {model_type}, {model_bitrate}')

    for win_duration in win_durations:
        # Force chunk-encoding by making duration 1 more than win_duration:
        # (one day the compress function will default to unchunked if the audio length is <= win_duration)
        T = 1 + int(win_duration * model.sample_rate)
        x = random.normal(random.key(0), shape=(1, 1, T))
        try:
            dac_file = model.compress(compress_chunk, x, audio_sr, win_duration=win_duration, benchmark=True)
            recons = model.decompress(decompress_chunk, dac_file, benchmark=True)
        except Exception as e:
            print(f'Exception for win duration "{win_duration}": {e}')


if __name__ == "__main__":
    # example usage:
    # python3 benchmark.py --model_type=16khz --win_durations="0.5 1 5 10 20"
    print(f'devices: {jax.devices()}')

    args = argbind.parse_args()
    with argbind.scope(args):
        benchmark_dac()


# @argbind.bind(without_prefix=True)
# def benchmark_dac_encode(model_type="44khz", model_bitrate='8kbps', batch_size: int = 1, durations: List[str] = None):
#
#     if durations is None:
#         durations = [1, 2, 4, 8, 16, 32]
#     else:
#         durations = [float(x) for x in durations]
#
#     model, variables = load_model(model_type=model_type, model_bitrate=model_bitrate)
#
#     @jax.jit
#     def encode(audio):
#         audio = model.apply(variables, audio, model.sample_rate, method="preprocess")
#         _, codes, _, _, _ = model.apply(variables, audio, train=False, method="encode")
#         return codes
#
#     for duration in durations:
#         print(f'Benchmarking encode for model: {model_type}, {model_bitrate} with duration {duration} sec and batch size {batch_size}.')
#
#         T = int(duration * model.sample_rate)
#         x = random.normal(random.key(0), shape=(batch_size, 1, T))
#         import tqdm
#         for _ in tqdm.trange(100):
#             try:
#                 encode(x)
#             except Exception as e:
#                 print(f'Exception for duration "{duration}": {e}')


# if __name__ == "__main__":
#     # example usage:
#     # python3 benchmark.py --model_type=44khz --durations="5" --batch_size=8
#     print(f'devices: {jax.devices()}')
#
#     args = argbind.parse_args()
#     with argbind.scope(args):
#         benchmark_dac_encode()
