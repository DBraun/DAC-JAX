from typing import List, Mapping

import argbind
from audiotree import SaliencyParams
from audiotree.datasources import (
    AudioDataSimpleSource,
    AudioDataBalancedSource,
)
from audiotree.transforms import ReduceBatchTransform
from grain import python as grain

SaliencyParams = argbind.bind(SaliencyParams, "train", "val", "test", "sample")


@argbind.bind("train", "val", "test", "sample")
def create_dataset(
    batch_size: int,
    sample_rate: int,
    duration: float = 0.2,
    sources: Mapping[str, List[str]] = None,
    extensions: List[str] = None,
    mono: int = 1,  # bool
    train: int = 0,  # bool
    num_steps: int = None,
    seed: int = 0,
    worker_count: int = 0,
    worker_buffer_size: int = 2,
    enable_profiling: int = 0,  # bool
    num_epochs: int = 1,  # for train/val use 1, but for sample set it to None so that it loops forever.
):

    assert sources is not None

    if train:
        assert num_steps is not None and num_steps > 0
        datasource = AudioDataBalancedSource(
            sources=sources,
            num_steps=num_steps * batch_size,
            sample_rate=sample_rate,
            mono=mono,
            duration=duration,
            extensions=extensions,
            saliency_params=SaliencyParams(),  # rely on argbind,
        )
    else:
        datasource = AudioDataSimpleSource(
            sources=sources,
            num_steps=num_steps * batch_size if num_steps is not None else None,
            sample_rate=sample_rate,
            mono=mono,
            duration=duration,
            extensions=extensions,
        )

    shard_options = grain.NoSharding()  # todo:

    index_sampler = grain.IndexSampler(
        num_records=len(datasource),
        num_epochs=num_epochs,
        shard_options=shard_options,
        shuffle=bool(train),
        seed=seed,
    )

    pygrain_ops = [
        grain.Batch(batch_size=batch_size, drop_remainder=True),
        ReduceBatchTransform(sample_rate),
    ]

    dataloader = grain.DataLoader(
        data_source=datasource,
        sampler=index_sampler,
        operations=pygrain_ops,
        worker_count=worker_count,
        worker_buffer_size=worker_buffer_size,
        shard_options=shard_options,
        enable_profiling=bool(enable_profiling),
    )

    return dataloader


if __name__ == "__main__":

    from tqdm import tqdm
    from audiotree.transforms import VolumeNorm, RescaleAudio, SwapStereo, InvertPhase
    from absl import logging

    logging.set_verbosity(logging.INFO)

    folder1 = "/mnt/c/users/braun/Datasets/dx7/patches-DX7-AllTheWeb-Bridge-Music-Recording-Studio-Sysex-Set-4-Instruments-Bass-Bass3-bass-10-syx-01-SUPERBASS2-note69"
    folder2 = "/mnt/c/Users/braun/Datasets/dx7/patches-DX7-AllTheWeb-Bridge-Music-Recording-Studio-Sysex-Set-4-Instruments-Accordion-ACCORD01-SYX-06-AKKORDEON-note69"

    sources = {
        "a": [folder1],
        "b": [folder2],
    }

    num_steps = 1000

    ds = create_dataset(
        sources=sources,
        duration=0.5,
        train=True,
        batch_size=32,
        sample_rate=44_100,
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

    for x in tqdm(ds, total=num_steps, desc="Grain Dataset"):
        pass
