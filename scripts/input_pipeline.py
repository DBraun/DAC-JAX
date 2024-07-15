import os
from random import Random
from typing import List, SupportsIndex, Mapping

import grain.python as grain
import jax
import jax.numpy as jnp
import numpy as np

from audio_tree_core import AudioTree, SaliencyParams


class AudioDataSourceMixin:

    def load_audio(self, file_path, record_key: SupportsIndex):

        if self.saliency_params is not None and self.saliency_params.enabled:
            saliency_params = self.saliency_params
            return AudioTree.salient_excerpt(
                file_path,
                np.random.RandomState(int(record_key)),
                loudness_cutoff=saliency_params.loudness_cutoff,
                num_tries=saliency_params.num_tries,
                sample_rate=self.sample_rate,
                duration=self.duration,
                mono=self.mono,
            )

        return AudioTree.from_file(
            file_path,
            sample_rate=self.sample_rate,
            offset=0,
            duration=self.duration,
            mono=self.mono,
        )


class ReduceBatchTransform(grain.MapTransform):

    def map(self, audio_signal: AudioTree) -> AudioTree:

        def f(leaf):
            if isinstance(leaf, np.ndarray):
                if leaf.ndim > 1:
                    shape = leaf.shape
                    shape = (shape[0]*shape[1],) + shape[2:]
                    return jnp.reshape(leaf, shape=shape)
            return leaf

        audio_signal = jax.tree_util.tree_map(f, audio_signal)

        # We do this to make sample rate a scalar instead of an array.
        # We assume that all entries of audio_signal.sample_rate are the same.
        audio_signal = audio_signal.replace(sample_rate=audio_signal.sample_rate[0])

        return audio_signal


def find_files_with_extensions(directory: str, extensions: List[str], max_depth=None, follow_symlinks=False):
    """
    Searches for files with specified extensions up to a maximum depth in the directory,
    without modifying dirs while iterating.

    Parameters:
    - directory (str): The path to the directory to search.
    - extensions (list): A list of file extensions to search for. Each extension should include a period.
    - max_depth (int): The maximum depth to search for files.
    - follow_symlinks (bool): Whether to follow symbolic links during the search.

    Returns:
    - list: A list of paths to files that match the extensions within the maximum depth.
    """
    matching_files = []
    extensions_set = {ext.lower() for ext in extensions}  # Normalize extensions to lowercase for matching
    directory = os.path.abspath(directory)  # Ensure the directory path is absolute

    def recurse(current_dir, current_depth):
        if max_depth is not None and current_depth > max_depth:
            return
        with os.scandir(current_dir) as it:
            for entry in it:
                if entry.is_file(follow_symlinks=follow_symlinks) and any(entry.name.lower().endswith(ext) for ext in extensions_set):
                    matching_files.append(entry.path)
                elif entry.is_dir(follow_symlinks=follow_symlinks):
                    recurse(entry.path, current_depth + 1)

    recurse(directory, 0)
    return matching_files


class AudioDataSimpleSource(grain.RandomAccessDataSource, AudioDataSourceMixin):

    def __init__(
            self,
            sources: Mapping[str, List[str]],
            sample_rate: int = 44100,
            mono: int = 1,
            duration: float = 1.,
            extensions=None,
            num_steps=None,
            saliency_params: SaliencyParams = None,
    ):

        self.sample_rate = sample_rate
        self.mono = bool(mono)
        self.duration = duration
        if extensions is None:
            extensions = ['.wav', '.flac', '.mp3', '.mp4', '.ogg']
        self.saliency_params = saliency_params

        filepaths = []
        for group_name, folders in sources.items():
            for folder in folders:
                filepaths += find_files_with_extensions(folder, extensions=extensions)

        if num_steps is not None:
            filepaths = filepaths[:num_steps]

        self._file_paths = filepaths

        self._num_steps = len(filepaths)
        assert self._num_steps > 0

    def __len__(self) -> int:
        return self._num_steps

    def __getitem__(self, record_key: SupportsIndex):
        file_path = self._file_paths[record_key]
        return self.load_audio(file_path, record_key)


class AudioDataBalancedSource(grain.RandomAccessDataSource, AudioDataSourceMixin):

    # todo: make this algorithm work if the user specifies weights for the groups.
    #  Right now the groups are balanced uniformly.

    def __init__(
            self,
            sources: Mapping[str, List[str]],
            num_steps: int,
            sample_rate: int = 44100,
            mono: int = 1,
            duration: float = 1.,
            extensions=None,
            saliency_params: SaliencyParams = None,
    ):

        self.sample_rate = sample_rate
        self.mono = bool(mono)
        self.duration = duration
        if extensions is None:
            extensions = ['.wav', '.flac', '.mp3', '.mp4', '.ogg']
        self.saliency_params = saliency_params

        groups = []

        for group_name, folders in sources.items():
            filepaths = []
            for folder in folders:
                filepaths += find_files_with_extensions(folder, extensions=extensions)

            if filepaths:
                groups.append(filepaths)
            else:
                raise RuntimeError(f"Group '{group_name}' is empty. "
                                   f"The number of specified folders was {len(folders)}.")

        self._group_to_len = {i: len(group) for i, group in enumerate(groups)}
        self._groups = groups
        self._num_groups = len(groups)
        self._num_steps = num_steps
        assert self._num_steps > 0

    def __len__(self) -> int:
        return self._num_steps

    def __getitem__(self, record_key: SupportsIndex):
        record_key = int(record_key)

        group_idx = record_key % self._num_groups

        idx = record_key // self._num_groups

        x = idx % self._group_to_len[group_idx]
        y = idx // self._group_to_len[group_idx]

        group = self._groups[group_idx].copy()

        Random(y+4617*group_idx).shuffle(group)  # todo: 4617 is arbitrary and could probably be zero.

        file_path = group[x]

        return self.load_audio(file_path, record_key)


def create_audio_dataset(
        sources: Mapping[str, List[str]],
        duration: float = 1.,
        train: int = 1,
        batch_size: int = 4,
        sample_rate: int = 44100,
        mono: int = 1,
        seed: int = 0,
        num_steps=None,
        extensions=None,
        worker_count: int = 0,
        worker_buffer_size: int = 1,
        transforms=None,
        saliency_params: SaliencyParams = None,
):

    if train:
        assert num_steps is not None and num_steps > 0

        datasource = AudioDataBalancedSource(
            sources=sources,
            num_steps=num_steps*batch_size,
            sample_rate=sample_rate,
            mono=mono,
            duration=duration,
            extensions=extensions,
            saliency_params=saliency_params,
        )
    else:
        assert num_steps is None, "Train is False but num_steps is also None"
        datasource = AudioDataSimpleSource(
            sources=sources,
            sample_rate=sample_rate,
            mono=mono,
            duration=duration,
            extensions=extensions,
        )

    index_sampler = grain.IndexSampler(
      num_records=len(datasource),
      num_epochs=1,
      shard_options=grain.NoSharding(),
      shuffle=False,  # Keep shuffling off. AudioDataBalancedSource already does the shuffling deterministically,
                      #  and AudioDataSimpleSource doesn't need shuffling.
      seed=seed,
    )

    if transforms is None:
        transforms = []

    pygrain_ops = [
        grain.Batch(batch_size=batch_size, drop_remainder=True),
        ReduceBatchTransform(),
        *transforms,
    ]

    batched_dataloader = grain.DataLoader(
        data_source=datasource,
        sampler=index_sampler,
        operations=pygrain_ops,
        worker_count=worker_count,
        worker_buffer_size=worker_buffer_size,
        enable_profiling=False,
    )
    return batched_dataloader


if __name__ == '__main__':

    from tqdm import tqdm
    from data_transforms import VolumeNorm, RescaleAudio, SwapStereo, InvertPhase
    from absl import logging

    logging.set_verbosity(logging.INFO)

    folder1 = '/mnt/c/users/braun/Datasets/dx7/patches-DX7-AllTheWeb-Bridge-Music-Recording-Studio-Sysex-Set-4-Instruments-Bass-Bass3-bass-10-syx-01-SUPERBASS2-note69'
    folder2 = '/mnt/c/Users/braun/Datasets/dx7/patches-DX7-AllTheWeb-Bridge-Music-Recording-Studio-Sysex-Set-4-Instruments-Accordion-ACCORD01-SYX-06-AKKORDEON-note69'

    sources = {
        'a': [folder1],
        'b': [folder2],
    }

    num_steps = 1000

    ds = create_audio_dataset(
        sources=sources,
        duration=0.5,
        train=True,
        batch_size=32,
        sample_rate=44100,
        mono=True,
        seed=0,
        num_steps=num_steps,
        extensions=None,
        worker_count=0,
        worker_buffer_size=1,
        transforms=[
            VolumeNorm(),
            RescaleAudio(),
            SwapStereo(),
            InvertPhase(),
        ],
        saliency_params=SaliencyParams(False, 8, -70),
    )

    for x in tqdm(ds, total=num_steps, desc='Grain Dataset'):
        pass
