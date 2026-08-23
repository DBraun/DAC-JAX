from typing import List, Mapping

import argbind
from audiotree import AudioTree, ExcerptConfig
from audiotree.sources import (
    create_audio_dataset,
    create_balanced_audio_dataset,
)
import grain

ExcerptConfig = argbind.bind(ExcerptConfig, "train", "val", "test", "gen")


@argbind.bind("train", "val", "test", "gen")
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

    common_kwargs = dict(
        sample_rate=sample_rate,
        mono=bool(mono),
        duration=duration,
        extensions=extensions,
        excerpt=ExcerptConfig(),  # rely on argbind
        shuffle_seed=seed,
    )

    if train:
        assert num_steps is not None and num_steps > 0
        dataset = create_balanced_audio_dataset(
            sources=sources,
            shuffle=True,
            # Repeat without bound, so that the slice below is what fixes the
            # length. audiotree 1.0 dropped `num_records`, which the balanced
            # source used to take for exactly this, and which was likewise not
            # capped by how many files the corpus happens to hold: a training run
            # asks for `num_steps` batches whether that is a fraction of an epoch
            # or many epochs' worth.
            num_epochs=None,
            **common_kwargs,
        )
        dataset = dataset[: num_steps * batch_size]
    else:
        # `create_audio_dataset` takes a flat list of directories rather than a
        # mapping of groups: every file is weighted equally, which is what the
        # old AudioDataSimpleSource did with a mapping.
        folders = [folder for group in sources.values() for folder in group]
        dataset = create_audio_dataset(
            sources=folders,
            shuffle=False,
            num_epochs=num_epochs,
            **common_kwargs,
        )
        if num_steps is not None:
            dataset = dataset[: num_steps * batch_size]

    # Reads stay synchronous inside each worker; the workers are the parallelism.
    read_options = grain.ReadOptions(num_threads=0, prefetch_buffer_size=0)

    dataset = dataset.to_iter_dataset(read_options=read_options).batch(
        batch_size, drop_remainder=True, batch_fn=AudioTree.batch
    )

    if worker_count:
        dataset = dataset.mp_prefetch(
            grain.MultiprocessingOptions(
                num_workers=worker_count,
                per_worker_buffer_size=worker_buffer_size,
                enable_profiling=bool(enable_profiling),
            )
        )

    return dataset


if __name__ == "__main__":
    from tqdm import tqdm
    from absl import logging

    logging.set_verbosity(logging.INFO)

    folder1 = "/mnt/d/Datasets/dx7/patches-DX7-AllTheWeb-Bridge-Music-Recording-Studio-Sysex-Set-4-Instruments-Bass-Bass3-bass-10-syx-01-SUPERBASS2-note69"
    folder2 = "/mnt/d/Datasets/dx7/patches-DX7-AllTheWeb-Bridge-Music-Recording-Studio-Sysex-Set-4-Instruments-Accordion-ACCORD01-SYX-06-AKKORDEON-note69"

    sources = {
        "a": [folder1],
        "b": [folder2],
    }

    num_steps = 1000

    ds = create_dataset(
        batch_size=32,
        sample_rate=44_100,
        sources=sources,
        duration=0.5,
        train=True,
        mono=True,
        seed=0,
        num_steps=num_steps,
        extensions=None,
        worker_count=0,
        worker_buffer_size=1,
    )

    for x in tqdm(ds, total=num_steps, desc="Grain Dataset"):
        pass
