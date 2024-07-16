from typing import List, Mapping

import grain.python as grain

from dac_jax.audiotree.datasources import SaliencyParams, AudioDataSimpleSource, AudioDataBalancedSource
from dac_jax.audiotree.transforms import ReduceBatchTransform


def create_audio_dataset(
        sources: Mapping[str, List[str]],
        extensions=None,
        duration: float = 1.,
        train: int = 1,
        batch_size: int = 4,
        sample_rate: int = 44100,
        mono: int = 1,
        seed: int = 0,
        num_steps=None,
        worker_count: int = 0,
        worker_buffer_size: int = 1,
        transforms=None,
        saliency_params: SaliencyParams = None,
        enable_profiling=False,
):

    assert sources is not None

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

    pygrain_ops = [
        grain.Batch(batch_size=batch_size, drop_remainder=True),
        ReduceBatchTransform(),
    ]

    if transforms is not None:
        pygrain_ops += transforms

    batched_dataloader = grain.DataLoader(
        data_source=datasource,
        sampler=index_sampler,
        operations=pygrain_ops,
        worker_count=worker_count,
        worker_buffer_size=worker_buffer_size,
        enable_profiling=enable_profiling,
    )

    return batched_dataloader


if __name__ == '__main__':

    from tqdm import tqdm
    from dac_jax.audiotree.transforms import VolumeNorm, RescaleAudio, SwapStereo, InvertPhase
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
