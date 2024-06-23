# from __future__ import annotations  # we can't import this due to clu.metrics

import os
os.environ["XLA_FLAGS"] = (
    # '--xla_force_host_platform_device_count=1'
    ' --xla_gpu_deterministic_ops=true'  # todo: https://github.com/google/flax/discussions/3382
    # ' --xla_dump_to=tmp/xla_dump'
    )
# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Pre-allocate 90% of TPU memory to minimize memory fragmentation and allocation
# overhead
# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from typing import Tuple
import datetime
from functools import partial
import shutil
import warnings

import argbind
from absl import logging
import tqdm
import numpy as np
import jax
jax.config.update('jax_threefry_partitionable', True)
assert jax.config.jax_threefry_partitionable is True
assert jax.config.jax_default_prng_impl == 'threefry2x32'
from jax import random, numpy as jnp

from clu import metric_writers, periodic_actions
from clu.metrics import Average, Collection
from flax.training.train_state import TrainState
from flax.training.early_stopping import EarlyStopping
from flax import struct
from flax import jax_utils
import optax

import orbax.checkpoint as ocp

from einops import rearrange

import tensorflow as tf

from input_pipeline import AudioDataset

from dac_jax.nn.loss import l1_loss, multiscale_stft_loss, mel_spectrogram_loss, generator_loss, discriminator_loss
from dac_jax.model import DAC, Discriminator
from dac_jax import load_model
from dac_jax.audio_utils import volume_norm, phase_shift, rescale_audio

warnings.filterwarnings("ignore", category=UserWarning)  # ignore librosa warnings about mel filters

SAMPLE_RATE = 44100  # todo: get from config file

# Models
DAC = argbind.bind(DAC)
Discriminator = argbind.bind(Discriminator)

# Losses
multiscale_stft_loss = argbind.bind(multiscale_stft_loss)
mel_spectrogram_loss = argbind.bind(mel_spectrogram_loss)
mel_spectrogram_loss = partial(mel_spectrogram_loss, sample_rate=SAMPLE_RATE)


@argbind.bind()
def get_logger(level='DEBUG'):

    import logging as logging_py
    logger = logging_py.getLogger('train')

    # create console handler and set level to debug
    ch = logging_py.StreamHandler()
    ch.setLevel(level.upper())
    formatter = logging_py.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                                     datefmt='%m/%d/%Y %I:%M:%S %p')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger


logger = get_logger()


def is_process_main():
    return jax.process_index() == 0


def prepare_tf_data(xs):
    """Convert an input batch from tf Tensors to numpy arrays."""
    local_device_count = jax.local_device_count()
    def _prepare(x):
        # Use _numpy() for zero-copy conversion between TF and NumPy.
        x = x._numpy()  # pylint: disable=protected-access

        # reshape (host_batch_size, channels, duration samples) to
        # (local_devices, device_batch_size, channels, duration samples)
        return x.reshape((local_device_count, -1) + x.shape[1:])

    return jax.tree_util.tree_map(_prepare, xs)


@argbind.bind('train', 'val', 'test')
def create_dataset(batch_size: int, dtype=tf.float32, repeat=False, duration=0.2, sources: dict = None, shuffle=False,
                   channels=1, random_offset=False, sample_rate=SAMPLE_RATE) -> Tuple[tf.data.Dataset, float]:

    assert sources is not None

    ds = AudioDataset(sources=sources, duration=duration, batch_size=batch_size, dtype=dtype, repeat=repeat,
                      shuffle=shuffle, channels=channels, sample_rate=sample_rate, random_offset=bool(random_offset))
    ds = map(prepare_tf_data, ds)
    ds = jax_utils.prefetch_to_device(ds, size=2)  # todo: pick size
    return ds, duration


@struct.dataclass
class MyMetrics(Collection):
    loss: Average.from_output('loss')
    vq_commitment_loss: Average.from_output('vq/commitment_loss')
    codebook_loss: Average.from_output('vq/codebook_loss')
    disc_loss: Average.from_output('adv/disc_loss')
    stft_loss: Average.from_output('stft/loss')
    mel_loss: Average.from_output('mel/loss')
    l1_loss: Average.from_output('waveform/loss')
    gen_loss: Average.from_output('adv/gen_loss')
    feat_loss: Average.from_output('adv/feat_loss')


@struct.dataclass
class GenDiscState(struct.PyTreeNode):
    generator: TrainState
    discriminator: TrainState


@argbind.bind()
def create_generator(key,
                     shape,
                     learning_rate: float = 1e-4,
                     lr_gamma: float = 0.999996,
                     adam_b1: float = 0.8,
                     adam_b2: float = 0.99,
                     adam_weight_decay: float = 1e-4,
                     grad_clip: float = 1e3,
                     tabulate=False,
                     ) -> TrainState:

    # Exponential decay of the learning rate.
    scheduler = optax.exponential_decay(
        init_value=float(learning_rate),
        transition_steps=1,
        decay_rate=lr_gamma)

    # Combining gradient transforms using `optax.chain`.
    gradient_transform = optax.chain(
        optax.clip_by_global_norm(grad_clip),
        optax.scale_by_adam(b1=adam_b1, b2=adam_b2),
        optax.add_decayed_weights(float(adam_weight_decay)),  # this puts the W in AdamW
        optax.scale_by_schedule(scheduler),
        optax.scale(-1.0)  # gradient descent
    )

    x = jnp.ones(shape=shape)

    load_weights = False  # todo: if you're curious about how a pre-trained model performs
    if load_weights:
        model, variables = load_model(model_type="44khz", model_bitrate="8kbps")
        params = variables['params']
        del variables
    else:
        model = DAC()
        subkey1, subkey2, key = random.split(key, 3)
        params = model.init({'params': subkey1, 'rng_stream': subkey2}, x)['params']

    if tabulate:
        subkey1, subkey2, key = random.split(key, 3)
        print(model.tabulate({'params': subkey1, 'rng_stream': subkey2}, x,
                             depth=3,
                             compute_flops=True,
                             compute_vjp_flops=True,
                             # column_kwargs={'width': 200},
                             console_kwargs={'width': 400},
                             ))

    state = TrainState.create(apply_fn=model.apply, params=params, tx=gradient_transform)
    return state


@argbind.bind()
def create_discriminator(key,
                         shape,
                         learning_rate: float = 1e-4,
                         lr_gamma: float = 0.999996,
                         adam_b1: float = 0.8,
                         adam_b2: float = 0.9,
                         adam_weight_decay: float = 1e-4,
                         grad_clip: float = 10,
                         tabulate=False,
                         ) -> TrainState:

    # Exponential decay of the learning rate.
    scheduler = optax.exponential_decay(
        init_value=float(learning_rate),
        transition_steps=1,
        decay_rate=lr_gamma)

    gradient_transform = optax.chain(
        optax.clip_by_global_norm(grad_clip),
        optax.scale_by_adam(b1=adam_b1, b2=adam_b2),
        optax.add_decayed_weights(float(adam_weight_decay)),  # this puts the W in "AdamW"
        optax.scale_by_schedule(scheduler),
        optax.scale(-1.0)  # gradient descent
    )

    model = Discriminator()

    x = jnp.ones(shape=shape)
    params = model.init(key, x)['params']

    if tabulate:
        print(model.tabulate(random.key(0), x,
                             depth=3,
                             compute_flops=True,
                             compute_vjp_flops=True,
                             # column_kwargs={'width': 200},
                             console_kwargs={'width': 400},
                             ))

    state = TrainState.create(apply_fn=model.apply, params=params, tx=gradient_transform)
    return state


@partial(jax.pmap, axis_name='ensemble', static_broadcasted_argnums=(1,))
def create_train_state(key, shape) -> GenDiscState:

    key, subkey = random.split(key)

    generator_state = create_generator(subkey, shape)
    key, subkey = random.split(key)
    discriminator_state = create_discriminator(subkey, shape)

    state = GenDiscState(generator=generator_state, discriminator=discriminator_state)
    train_metrics = MyMetrics.empty()
    eval_metrics = MyMetrics.empty()

    return state, train_metrics, eval_metrics


def compute_train_metrics(*, train_metrics: MyMetrics, output) -> MyMetrics:
    return train_metrics.merge(jax_utils.replicate(train_metrics.single_from_model_output(**output)))


def compute_eval_metrics(*, eval_metrics: MyMetrics, output) -> MyMetrics:
    return eval_metrics.merge(jax_utils.replicate(eval_metrics.single_from_model_output(**output)))


@partial(jax.pmap, axis_name='ensemble')
@argbind.bind()
def eval_step(rng, state: GenDiscState, audio_data, lambdas=None):

    if lambdas is None:
        lambdas = {
            'mel/loss': 15.0,
            'adv/feat_loss': 2.0,
            'adv/gen_loss': 1.0,
            'vq/commitment_loss': 0.25,
            'vq/codebook_loss': 1.0,
        }

    audio_data = audio_data['audio_data']
    audio_data = jnp.squeeze(audio_data, axis=0)

    audio_data = rearrange(audio_data, 'b c t -> (b c) 1 t')

    rngs = {'rng_stream': rng}
    output = state.generator.apply_fn({'params': state.generator.params}, audio_data, SAMPLE_RATE, train=False,
                                      rngs=rngs)
    recons = output['audio']

    output['stft/loss'] = multiscale_stft_loss(audio_data, recons)
    output['mel/loss'] = mel_spectrogram_loss(audio_data, recons)
    output['waveform/loss'] = l1_loss(audio_data, recons)

    fake = state.discriminator.apply_fn({'params': state.discriminator.params}, recons)
    real = state.discriminator.apply_fn({'params': state.discriminator.params}, audio_data)

    output['adv/disc_loss'] = discriminator_loss(fake, real)
    (
        output['adv/gen_loss'],
        output['adv/feat_loss'],
    ) = generator_loss(fake, real)

    output['loss'] = sum([v * output[k] for k, v in lambdas.items() if k in output])

    return output


def train_step_discriminator(state: GenDiscState, audio_data, output) -> Tuple[GenDiscState, struct.PyTreeNode]:

    def loss_fn(params):
        # note: you could calculate with the generator again, since its weights were just updated,
        # but we prefer not to in order to run faster.
        # output = state.generator.apply_fn({'params': state.generator.params}, audio_data, SAMPLE_RATE, rngs=rngs,
        #                                   train=True  # todo: maybe pick Train=False even though DAC didn't
        #                                   )
        recons = output['audio']

        fake = state.discriminator.apply_fn({'params': params}, recons)
        real = state.discriminator.apply_fn({'params': params}, audio_data)

        loss = output['adv/disc_loss'] = discriminator_loss(fake, real)

        return loss

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.discriminator.params)
    loss = jax.lax.pmean(loss, axis_name='ensemble')
    grads = jax.lax.pmean(grads, axis_name='ensemble')

    state = state.replace(discriminator=state.discriminator.apply_gradients(grads=grads))
    return state, loss


@argbind.bind()
def train_step_generator(rng, state: GenDiscState, audio_data, lambdas=None) -> Tuple[GenDiscState, dict]:
    rngs = {'rng_stream': rng}

    if lambdas is None:
        lambdas = {
            'mel/loss': 15.0,
            'adv/feat_loss': 2.0,
            'adv/gen_loss': 1.0,
            'vq/commitment_loss': 0.25,
            'vq/codebook_loss': 1.0,
        }

    def loss_fn(params):

        output = state.generator.apply_fn({'params': params}, audio_data, SAMPLE_RATE, rngs=rngs)
        recons = output['audio']

        fake = state.discriminator.apply_fn({'params': state.discriminator.params}, recons)
        real = state.discriminator.apply_fn({'params': state.discriminator.params}, audio_data)

        output['stft/loss'] = multiscale_stft_loss(audio_data, recons)
        output['mel/loss'] = mel_spectrogram_loss(audio_data, recons)
        output['waveform/loss'] = l1_loss(audio_data, recons)
        (
            output['adv/gen_loss'],
            output['adv/feat_loss'],
        ) = generator_loss(fake, real)
        loss = output['loss'] = sum([v * output[k] for k, v in lambdas.items() if k in output])
        return loss, output

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, output), grads = grad_fn(state.generator.params)
    grads = jax.lax.pmean(grads, axis_name='ensemble')

    state = state.replace(generator=state.generator.apply_gradients(grads=grads))
    return state, output


@argbind.bind()
def augment_data(rng, audio_data, min_db=-16, max_db=-16):

    batch_size, channels = audio_data.shape[0:2]

    # note: apply target_db based on both channels
    target_db = random.uniform(rng, shape=(batch_size,), minval=min_db, maxval=max_db)

    audio_data, loudness = volume_norm(audio_data, target_db, SAMPLE_RATE, filter_class="K-weighting", block_size=0.400)

    # note: apply different phase to each channel
    phase_angles = random.uniform(rng, shape=(batch_size, channels), minval=-jnp.pi, maxval=jnp.pi)
    audio_data = phase_shift(audio_data, phase_angles)

    audio_data = rescale_audio(audio_data)

    return audio_data


@partial(jax.pmap, axis_name='ensemble', donate_argnums=(1,))
def train_step(rng, state: GenDiscState, audio_data) -> Tuple[GenDiscState, dict]:
    """Train for a single step."""

    audio_data = audio_data['audio_data']

    audio_data = jnp.squeeze(audio_data, axis=0)

    subkey1, subkey2 = random.split(rng, num=2)

    audio_data = augment_data(subkey1, audio_data)

    audio_data = rearrange(audio_data, 'b c t -> (b c) 1 t')

    state, output = train_step_generator(subkey2, state, audio_data)
    state, loss = train_step_discriminator(state, audio_data, output)

    output['adv/disc_loss'] = loss

    return state, output


@argbind.bind()
def save_checkpoint(checkpoint_manager: ocp.CheckpointManager, state: GenDiscState, step: int, metrics=None):

    if is_process_main():
        # get train state from the first replica
        main_state = jax.device_get(jax.tree_util.tree_map(lambda x: x[0], state))
        ckpt = {'generator': main_state.generator, 'discriminator': main_state.discriminator}
        save_args = ocp.args.StandardSave(ckpt)
        checkpoint_manager.save(step, ckpt, args=save_args, metrics=metrics)
        checkpoint_manager.wait_until_finished()


@partial(jax.pmap, axis_name='ensemble')
def save_samples(rng, state: GenDiscState, audio_data):
    """Save audio samples to tensorboard."""
    audio_data = audio_data['audio_data']
    audio_data = jnp.squeeze(audio_data, axis=0)
    batch_size = audio_data.shape[0]
    audio_data = rearrange(audio_data, 'b c t -> (b c) 1 t')

    rngs = {'rng_stream': rng}
    output = state.generator.apply_fn({'params': state.generator.params}, audio_data, SAMPLE_RATE, train=False,
                                      rngs=rngs)
    recons = output['audio']
    recons = rearrange(recons, '(b c) 1 t -> b c t', b=batch_size)
    return recons


@argbind.bind()
def log_training(train_metrics: MyMetrics, step: int, writer: metric_writers.MultiWriter, log_every_steps=1):

    if log_every_steps and (step % log_every_steps == 0):
        summary = {}
        summary.update({
            f'train/{k}': v.item()
            for k, v in train_metrics.unreplicate().compute().items()
        })
        writer.write_scalars(step, summary)
        writer.flush()
        # reset metrics for next logging
        train_metrics = jax_utils.replicate(train_metrics.empty())

    return train_metrics


def log_eval(eval_metrics: MyMetrics, step: int, writer: metric_writers.MultiWriter):

    summary = {}
    summary.update({
        f'eval/{k}': v.item()
        for k, v in eval_metrics.unreplicate().compute().items()
    })
    writer.write_scalars(step, summary)
    writer.flush()
    # reset metrics for next logging
    eval_metrics = jax_utils.replicate(eval_metrics.empty())

    return summary, eval_metrics


def split_device(key):
    return random.split(key, jax.device_count())


@argbind.bind()
def train(args,
          name=None,
          num_iterations=250_000,
          valid_freq=100,
          sample_freq=100,
          early_stop_patience=0,
          ckpt_max_keep=2,
          seed=0,
          batch_size=4,
          val_batch_size=1,
          half_precision=False,
          restore=False,
          best_key='eval/loss',
          best_mode='min',
          enable_async_checkpointing=False,
          log_level='info',
          save_path='runs',
          ):

    logging.set_verbosity(logging.ERROR)  # absl logging: # todo: set this to INFO but work well with tqdm
    logger.setLevel(log_level.upper())  # native python logging

    print(f'devices: {jax.devices()}')

    if batch_size % jax.device_count() > 0:
        raise ValueError('Batch size must be divisible by the number of devices')
    local_batch_size = batch_size // jax.process_count()
    local_val_batch_size = val_batch_size // jax.process_count()
    del batch_size

    platform = jax.local_devices()[0].platform

    if half_precision:
        if platform == 'tpu':
            input_dtype = tf.bfloat16
        else:
            input_dtype = tf.float16
    else:
        input_dtype = tf.float32

    if name is not None:
        workdir = os.path.join(save_path, name)
    else:
        workdir = os.path.join(save_path, datetime.datetime.now().strftime('%m-%d_%H-%M-%S'))
    del name

    # if os.path.isdir(workdir) and '..' not in workdir:
    #     logger.info(f"Deleting existing work directory: {workdir}")
    #     shutil.rmtree(workdir)

    writer = metric_writers.create_default_writer(logdir=workdir, just_logging=not is_process_main())

    ckpt_dir = '/tmp/flax_ckpt'

    if os.path.exists(ckpt_dir):
        shutil.rmtree(ckpt_dir)  # Remove any existing checkpoints from the last run.

    with argbind.scope(args, "train"):
        train_iter, duration = create_dataset(batch_size=local_batch_size, dtype=input_dtype, repeat=True, shuffle=True,
                                              random_offset=True)
    with argbind.scope(args, "val"):
        save_iter, _ = create_dataset(batch_size=local_val_batch_size, dtype=input_dtype, repeat=True)

    n_channels = 1
    shape = (local_batch_size, n_channels, round(SAMPLE_RATE * duration))
    del duration
    key = random.key(seed)

    key, subkey = random.split(key)
    state, train_metrics, eval_metrics = create_train_state(split_device(subkey), shape)

    del shape

    def best_fn(eval_metrics_summary) -> float:
        return eval_metrics_summary[best_key]

    checkpoint_manager = ocp.CheckpointManager(
        directory=ckpt_dir + '/orbax/managed',
        options=ocp.CheckpointManagerOptions(save_interval_steps=1, max_to_keep=ckpt_max_keep, best_fn=best_fn,
                                             best_mode=best_mode,
                                             enable_async_checkpointing=enable_async_checkpointing),
        item_handlers=ocp.StandardCheckpointHandler()
    )

    if restore:
        latest_ckpt = checkpoint_manager.latest()
        logger.info(f"Restoring latest checkpoint v{latest_ckpt}.")
        restored = checkpoint_manager.restore(latest_ckpt)
        state = restored.state
        state = jax_utils.replicate(state)
    else:
        save_checkpoint(checkpoint_manager, state, 0,
                        metrics={best_key: jnp.inf if best_mode == 'min' else -jnp.inf})

    hooks = []
    # todo: the report_progress won't show up in TensorBoard because the absl logging level is set to ERROR
    #  in order to not clash with tqdm printing.
    report_progress = periodic_actions.ReportProgress(num_train_steps=num_iterations,
                                                      every_secs=None,
                                                      every_steps=1,
                                                      writer=writer if is_process_main() else None)
    report_progress_eval = periodic_actions.ReportProgress(num_train_steps=num_iterations,
                                                           every_secs=None,
                                                           every_steps=valid_freq,
                                                           writer=writer if is_process_main() else None)
    if is_process_main():
        hooks += [report_progress]
        hooks += [report_progress_eval]
        # todo: the Profile hook seems slow, so we don't use it.
        # hooks += [periodic_actions.Profile(num_profile_steps=10, profile_duration_ms=0, logdir=workdir)]

    early_stop = EarlyStopping(min_delta=1e-3, patience=early_stop_patience)

    for step in tqdm.trange(1, num_iterations+1, desc='Train Steps'):

        with report_progress.timed("load_train_batch"):
            batch = next(train_iter)
            batch = jax_utils.replicate(batch)

        key, subkey = random.split(key)

        with report_progress.timed("train_step"):
            state, output = train_step(split_device(subkey), state, batch)

        with report_progress.timed("train_metrics"):
            train_metrics = compute_train_metrics(train_metrics=train_metrics, output=output)

        for hook in hooks:
            hook(step)

        train_metrics = log_training(train_metrics, step, writer)

        if step % sample_freq == 0:
            # Note: right now save_iter is just the validation dataset.
            # We perform reconstruction and save the input and output to WAV, which shows in tensorboard.
            # todo: we are just taking a single batch from the validation set. This could change.
            save_batch = next(save_iter)
            save_batch_replicated = jax_utils.replicate(save_batch)
            recons = save_samples(split_device(subkey), state, save_batch_replicated)

            signal = rearrange(save_batch['audio_data'], 'p b c t -> (p b) t c')
            recons = rearrange(recons, 'p b c t -> (p b) t c', c=signal.shape[-1])

            writer.write_audios(step=step,
                                audios={
                                    'signal': np.array(signal),
                                    'recons': np.array(recons),
                                },
                                sample_rate=SAMPLE_RATE)
            writer.flush()

        if step % valid_freq == 0:
            # Compute metrics on the validation set
            with argbind.scope(args, "val"):
                # todo: don't recreate the eval dataset multiple times
                eval_iter, _ = create_dataset(batch_size=local_val_batch_size, dtype=input_dtype)
            ran_once = False

            with report_progress_eval.timed("eval_step"):
                for test_batch in eval_iter:  # todo: use nested tqdm here
                    ran_once = True
                    test_batch = jax_utils.replicate(test_batch)
                    key, subkey = random.split(key)
                    output = eval_step(split_device(subkey), state, test_batch)
                    eval_metrics = compute_eval_metrics(eval_metrics=eval_metrics, output=output)
                assert ran_once

            summary, eval_metrics = log_eval(eval_metrics, step, writer)

            early_stop = early_stop.update(summary[best_key])
            if early_stop.should_stop:
                logger.info(f'Met early stopping criteria, breaking at step {step}')
                break

            with report_progress_eval.timed("checkpoint_step"):
                save_checkpoint(checkpoint_manager, state, step, summary)

            writer.flush()


if __name__ == '__main__':
    args = argbind.parse_args()
    with argbind.scope(args):
        train(args)
