import argbind

from typing import List, Mapping

from flax import jax_utils
import grain.python as grain
import jax

from dac_jax.audiotree import transforms as transforms_lib
from dac_jax.audiotree.datasources import SaliencyParams, AudioDataSimpleSource, AudioDataBalancedSource
from dac_jax.audiotree.transforms import ReduceBatchTransform

# Transforms
def filter_fn(fn):
    return (fn.__qualname__ != 'TfRandomMapTransform' and
            (hasattr(fn, 'random_map') or hasattr(fn, 'map') or hasattr(fn, 'np_random_map')))

# https://github.com/pseeth/argbind/tree/main/examples/bind_module
transforms_lib = argbind.bind_module(transforms_lib, 'train', 'val', filter_fn=filter_fn)


SaliencyParams = argbind.bind(SaliencyParams, 'train', 'val', 'test', 'sample')


@argbind.bind('train', 'val', 'test', 'sample')
def build_transforms(
        augment: list[str] = None,
):
    """
    :param augment: A list of str names of Transforms (from ``audiotree.transforms``) such as VolumeNorm
    :return: a list of instances of the Transforms.
    """
    def to_transform_instances(transform_classes):
        if transform_classes is None:
            return None
        instances = []
        for TransformClass in transform_classes:
            instance = getattr(transforms_lib, TransformClass)()
            instances.append(instance)
        return instances

    return to_transform_instances(augment)


def prepare_for_prefetch(xs):
    local_device_count = jax.local_device_count()

    def _prepare(x):
        return x.reshape((local_device_count, -1) + x.shape[1:])

    return jax.tree_util.tree_map(_prepare, xs)


@argbind.bind('train', 'val', 'test', 'sample')
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
        worker_count: int = 0,  # todo: https://github.com/google/grain/issues/495
        prefetch_size: int = 1,
        enable_profiling=False,
        num_epochs: int = 1,  # for train/val use 1, but for sample set it to None so that it loops forever.
) -> grain.DataLoader:

    assert sources is not None

    saliency_params = SaliencyParams()  # rely on argbind

    transforms = build_transforms()  # rely on argbind

    if train:
        assert num_steps is not None and num_steps > 0
        datasource = AudioDataBalancedSource(
            sources=sources,
            num_steps=num_steps * batch_size,
            sample_rate=sample_rate,
            mono=mono,
            duration=duration,
            extensions=extensions,
            saliency_params=saliency_params,
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

    index_sampler = grain.IndexSampler(
        num_records=len(datasource),
        num_epochs=num_epochs,
        shard_options=grain.NoSharding(),
        shuffle=not train,  # Keep shuffling off for training, which uses AudioDataBalancedSource.
                            # AudioDataBalancedSource already does the shuffling deterministically, and turning it on
                            # here would actually BREAK the locality of the balancing between batches.
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
        worker_buffer_size=1,  # set to 1 because we use jax_utils.prefetch_to_device for similar behavior.
        enable_profiling=enable_profiling,
    )

    # similar to flax.jax_utils.replicate
    batched_dataloader = map(prepare_for_prefetch, batched_dataloader)

    if prefetch_size > 1:
        # For prefetch to work, we must have already used prepare_for_prefetch
        batched_dataloader = jax_utils.prefetch_to_device(batched_dataloader, size=prefetch_size)

    batched_dataloader = iter(batched_dataloader)

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

    ds = create_dataset(
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
