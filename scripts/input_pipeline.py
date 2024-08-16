import collections
import itertools
from typing import List, Mapping

import argbind
from audiotree import transforms as transforms_lib
from audiotree.datasources import (
    SaliencyParams,
    AudioDataSimpleSource,
    AudioDataBalancedSource,
)
from audiotree.transforms import ReduceBatchTransform
from functools import partial
from grain import python as grain
import jax
from jax import random
from jax.sharding import NamedSharding, Mesh, PartitionSpec as P
from jax.experimental.shard_map import shard_map
from jax.experimental import mesh_utils


# Transforms
def filter_fn(fn):
    return hasattr(fn, 'random_map') or hasattr(fn, 'map') or hasattr(fn, 'np_random_map')


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
        instances = []
        for TransformClass in transform_classes:
            instance = getattr(transforms_lib, TransformClass)()
            instances.append(instance)
        return instances

    if augment is None:
        augment = []

    return to_transform_instances(augment)


def make_iterator(
    dataloader: grain.DataLoader,
    operations: List[grain.Transformation],
    seed: int,
    sample_rate: int,
):
    n_gpus = jax.device_count()
    devices = mesh_utils.create_device_mesh((n_gpus,))

    # replicate initial params on all devices, shard data batch over devices
    mesh = Mesh(devices, ("data",))
    named_sharding = NamedSharding(mesh, P("data"))

    @jax.jit
    @partial(
        shard_map,
        mesh=mesh,
        in_specs=(P(None), P("data")),
        out_specs=P("data"),
    )
    @partial(jax.vmap, in_axes=(0, None), out_axes=0)  # vmap over just `key`.  # todo: set `out_axes=None`
    def main_transform(key: jax.Array, batch):
        for transform in operations:
            if isinstance(transform, grain.RandomMapTransform):
                key, subkey = random.split(key)
                batch = transform.random_map(batch, subkey)
            elif isinstance(transform, grain.MapTransform):
                batch = transform.map(batch)
            elif hasattr(transform, "np_random_map"):  # TfRandomMapTransform
                key, subkey = random.split(key)
                batch = transform.np_random_map(batch, subkey)
            else:
                raise ValueError(f"Unknown operation type: {type(transform)}")
        return batch

    key = random.PRNGKey(seed)

    for batch in dataloader:

        # move numpy data to sharded
        batch_out = jax.device_put(batch, named_sharding)

        key, subkey = random.split(key)
        tmp_keys = random.split(subkey, jax.device_count())

        batch_out = main_transform(tmp_keys, batch_out)

        # todo: don't use this ReduceBatchTransform since we should be able to use `out_axes=None` above in the vmap.
        batch_out = ReduceBatchTransform(sample_rate=sample_rate).map(batch_out)

        yield batch_out


def prefetch_to_device(iterator, size):
    """Shard and prefetch batches on device.

    This is a variation of https://flax.readthedocs.io/en/latest/api_reference/flax.jax_utils.html#flax.jax_utils.prefetch_to_device

    This utility takes an iterator and returns a new iterator which fills an on
    device prefetch buffer. Eager prefetching can improve the performance of
    training loops significantly by overlapping compute and data transfer.

    This utility is mostly useful for GPUs, for TPUs and CPUs it should not be
    necessary -- the TPU & CPU memory allocators (normally) don't pick a memory
    location that isn't free yet so they don't block. Instead those allocators OOM.

    Args:
      iterator: an iterator that yields a pytree of ndarrays where the first
        dimension is sharded across devices.

      size: the size of the prefetch buffer.

        If you're training on GPUs, 2 is generally the best choice because this
        guarantees that you can overlap a training step on GPU with a data
        prefetch step on CPU.

    Yields:
      The original items from the iterator where each ndarray is now sharded to
      the specified devices.

    Reference:
    https://flax.readthedocs.io/en/latest/api_reference/flax.jax_utils.html#flax.jax_utils.prefetch_to_device
    """
    queue = collections.deque()

    def enqueue(n):  # Enqueues *up to* `n` elements from the iterator.
        for data in itertools.islice(iterator, n):
            queue.append(data)

    enqueue(size)  # Fill up the buffer.
    while queue:
        yield queue.popleft()
        enqueue(1)


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
    prefetch_size: int = 2,
    enable_profiling: bool = False,  # todo: haven't gotten it to work with True
    num_epochs: int = 1,  # for train/val use 1, but for sample set it to None so that it loops forever.
    post_transforms: List[grain.Transformation] = None,
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
            cpu=True,
        )
    else:
        datasource = AudioDataSimpleSource(
            sources=sources,
            num_steps=num_steps * batch_size if num_steps is not None else None,
            sample_rate=sample_rate,
            mono=mono,
            duration=duration,
            extensions=extensions,
            cpu=True,
        )

    if worker_count is None or worker_count > 1:
        true_num_steps = (
            len(datasource) // batch_size
        )  # floor since the batch mode is `drop_remainder`
        assert true_num_steps > 1, "This is an edge case."

    shard_options = grain.NoSharding()  # todo:

    index_sampler = grain.IndexSampler(
        num_records=len(datasource),
        num_epochs=num_epochs,
        shard_options=shard_options,
        # Keep shuffling off for training, which uses AudioDataBalancedSource.
        # AudioDataBalancedSource already does the shuffling deterministically, and turning it on
        # here would actually BREAK the locality of the balancing between batches.
        shuffle=not train,
        seed=seed,
    )

    pygrain_ops = [
        grain.Batch(batch_size=batch_size, drop_remainder=True),
        ReduceBatchTransform(sample_rate),
    ]

    tmp_dataloader = grain.DataLoader(
        data_source=datasource,
        sampler=index_sampler,
        operations=pygrain_ops,
        worker_count=worker_count,
        worker_buffer_size=worker_buffer_size,
        shard_options=shard_options,
        enable_profiling=enable_profiling,
    )

    operations = build_transforms()
    if post_transforms is not None:
        operations += post_transforms

    batched_dataloader = make_iterator(
        tmp_dataloader, operations, seed=seed + 1, sample_rate=sample_rate
    )

    if prefetch_size is not None and prefetch_size > 1:
        batched_dataloader = prefetch_to_device(batched_dataloader, size=prefetch_size)

    return batched_dataloader


if __name__ == '__main__':

    from tqdm import tqdm
    from audiotree.transforms import VolumeNorm, RescaleAudio, SwapStereo, InvertPhase
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
