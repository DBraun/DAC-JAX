import logging
import random
import os
from typing import List
from pathlib import Path
import hashlib

import tensorflow as tf

import jax
import jax.numpy as jnp
import soundfile
import librosa


def find_files_with_extensions(directory: str, extensions: List[str], max_depth=None):
    """
    Searches for files with specified extensions up to a maximum depth in the directory,
    without modifying dirs while iterating.

    Parameters:
    - directory (str): The path to the directory to search.
    - extensions (list): A list of file extensions to search for.
    - max_depth (int): The maximum depth to search for files.

    Returns:
    - list: A list of paths to files that match the extensions within the maximum depth.
    """
    matching_files = []
    extensions = [ext.lower() for ext in extensions]  # Normalize extensions to lowercase for matching
    directory = os.path.abspath(directory)  # Ensure the directory path is absolute

    def recurse(current_dir, current_depth):
        if max_depth is not None and current_depth > max_depth:
            return
        for item in os.listdir(current_dir):
            item_path = os.path.join(current_dir, item)
            if os.path.isdir(item_path):
                recurse(item_path, current_depth + 1)
            elif os.path.isfile(item_path) and any(item.lower().endswith(ext) for ext in extensions):
                matching_files.append(item_path)

    recurse(directory, 0)
    return matching_files


class AudioDataset(tf.data.Dataset):

    @staticmethod
    def _generator(folders, duration: float, shuffle: bool, sample_rate: int, channels: int,
                   random_offset=False, repeat=False):

        # Caution: While this is a convenient approach it has limited portability and scalability.
        # It must run in the same python process that created the generator, and is still subject to the Python GIL.

        extensions = ['.wav', '.flac', '.mp3', '.mp4', '.ogg']

        file_paths = []
        for folder in folders:
            file_paths += find_files_with_extensions(folder.decode('utf-8'), extensions=extensions)

        key = jax.random.key(0)  # todo:
        loudness_cutoff = -40  # todo:

        cond = True
        while cond:
            if not repeat:
                cond = False

            if shuffle:
                random.shuffle(file_paths)

            for file_path in file_paths:
                if random_offset:  # todo: maybe just use AudioTools here
                    info = soundfile.info(file_path)
                    total_duration = info.duration  # seconds

                    lower_bound = 0
                    upper_bound = max(total_duration - duration, 0)

                    audio_data = jnp.zeros(shape=(channels, round(duration*sample_rate)))
                    best_loudness = -jnp.inf
                    attempts = 0
                    max_attempts = 100
                    while (best_loudness < loudness_cutoff) and attempts < max_attempts:
                        attempts += 1

                        key, subkey = jax.random.split(key)
                        offset = jax.random.uniform(key, shape=(1,), minval=lower_bound, maxval=upper_bound).item()
                        candidate_audio_data, _ = librosa.load(
                            file_path,
                            offset=offset,
                            duration=duration,
                            sr=sample_rate,
                            mono=False,
                        )

                        # Calculate the peak amplitude based on the raw audio file
                        peak_amplitude = jnp.max(jnp.abs(candidate_audio_data))

                        # Convert peak amplitude to dBFS
                        dBFS = 20 * jnp.log10(peak_amplitude)

                        if dBFS > best_loudness:
                            best_loudness = dBFS
                            audio_data = candidate_audio_data

                        if attempts == max_attempts:
                            logging.warning(f'Struggled to find salient audio section for file: {file_path}')
                            continue
                else:
                    audio_data, _ = librosa.load(file_path, sr=sample_rate, mono=False, duration=duration)

                if audio_data.ndim < 2:
                    audio_data = jnp.expand_dims(audio_data, axis=0)

                if audio_data.shape[-1] < round(duration*sample_rate):
                    pad_right = round(duration*sample_rate) - audio_data.shape[-1]
                    audio_data = jnp.pad(audio_data, ((0, 0), (0, int(pad_right))))

                assert audio_data.ndim == 2

                yield {'audio_data': audio_data}

    def __new__(cls, sources: dict, duration: float, batch_size=4, dtype=tf.float32, repeat=False, shuffle=False,
                channels=1, sample_rate=44100, random_offset=False):

        assert sources is not None, \
            "You must specify a sources as a dictionary mapping labels to lists of directories."

        dur_samples = round(duration * sample_rate)

        output_signature = tf.TensorSpec(shape=(channels, dur_samples), dtype=dtype)

        datasets = []
        for label, folders in sources.items():
            dataset = tf.data.Dataset.from_generator(
                cls._generator,
                output_signature={'audio_data': output_signature},
                args=(folders, duration, shuffle, sample_rate, channels, random_offset, repeat)
            )
            datasets.append(dataset)

        weights = None  # uniform distribution
        dataset = tf.data.Dataset.sample_from_datasets(datasets, weights=weights, rerandomize_each_iteration=shuffle)
        del datasets

        def convert_dtype(example):
            example = example.to(dtype)
            return example

        # dataset = dataset.map(convert_dtype)  # todo:
        # dataset = dataset.map(convert_dtype, num_parallel_calls=tf.data.experimental.AUTOTUNE)  # todo:

        dataset = dataset.batch(batch_size, drop_remainder=True)

        # hashed = hashlib.sha1(';'.join(sources).encode()).hexdigest()
        # cache_dir = str(Path("dataset_cache") / hashed)
        # os.makedirs(cache_dir, exist_ok=True)
        # cache_path = os.path.join(cache_dir, "cache")
        # dataset = dataset.cache(cache_path)  # todo:

        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset


def test_simple(folder):

    duration = 4  # seconds

    sources = {
        'foo': [folder, folder, folder]
    }

    dataset = AudioDataset(sources=sources, duration=duration, batch_size=4)

    i = 0
    for item in dataset:
        i += 1
        if i > 10:
            break


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str, required=True)

    args = parser.parse_args()

    tf.config.experimental.set_visible_devices([], 'GPU')

    import sys
    import traceback
    import pdb

    try:
        test_simple(args.folder)
    except:
        extype, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem()
