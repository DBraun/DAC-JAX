# from __future__ import annotations  # we can't import this due to clu.metrics

import os

XLA_FLAGS = []

FORCE_DEVICE_COUNT = 0  # todo:
if FORCE_DEVICE_COUNT:
    XLA_FLAGS.append(f'--xla_force_host_platform_device_count={FORCE_DEVICE_COUNT}')

DETERMINISTIC = False  # todo:
if DETERMINISTIC:
    # Deterministic will be slower.
    # https://github.com/google/flax/discussions/3382
    XLA_FLAGS.append('--xla_gpu_deterministic_ops=true')
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

if XLA_FLAGS:
    os.environ["XLA_FLAGS"] = ' ' + ' '.join(XLA_FLAGS)

import datetime
from functools import partial
import shutil
from typing import List, Mapping, Tuple
import warnings

import jax
jax.config.update('jax_threefry_partitionable', True)
assert jax.config.jax_threefry_partitionable is True
assert jax.config.jax_default_prng_impl == 'threefry2x32'
from jax import random, numpy as jnp

from absl import logging
import argbind
from clu import metric_writers, periodic_actions
from clu.metrics import Average, Collection
from einops import rearrange
from flax.training.early_stopping import EarlyStopping
from flax.training.train_state import TrainState
from flax import jax_utils
from flax import struct
import numpy as np
import optax
import orbax.checkpoint as ocp

from audiotree import AudioTree
from dac_jax import load_model
from dac_jax.model import DAC, Discriminator
from dac_jax.nn.loss import l1_loss, multiscale_stft_loss, mel_spectrogram_loss, generator_loss, discriminator_loss

from input_pipeline import create_dataset as _create_dataset

warnings.filterwarnings("ignore", category=UserWarning)  # ignore librosa warnings about mel filters

# Models
DAC = argbind.bind(DAC)
Discriminator = argbind.bind(Discriminator)

# Losses
multiscale_stft_loss = argbind.bind(multiscale_stft_loss)
mel_spectrogram_loss = argbind.bind(mel_spectrogram_loss)

EarlyStopping = argbind.bind(EarlyStopping)


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


@struct.dataclass
class EvalMetrics(Collection):
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
class TrainMetrics(Collection):
    loss: Average.from_output('loss')
    vq_commitment_loss: Average.from_output('vq/commitment_loss')
    codebook_loss: Average.from_output('vq/codebook_loss')
    disc_loss: Average.from_output('adv/disc_loss')
    stft_loss: Average.from_output('stft/loss')
    mel_loss: Average.from_output('mel/loss')
    l1_loss: Average.from_output('waveform/loss')
    gen_loss: Average.from_output('adv/gen_loss')
    feat_loss: Average.from_output('adv/feat_loss')
    lr_generator: Average.from_output('lr/generator')
    lr_discriminator: Average.from_output('lr/discriminator')


@struct.dataclass
class GenDiscState(struct.PyTreeNode):
    generator: TrainState
    discriminator: TrainState


@argbind.bind()
def create_generator_schedule(
        learning_rate: float = 1e-4,
        lr_gamma: float = 0.999996,
):
    # Exponential decay of the learning rate.
    schedule = optax.exponential_decay(
        init_value=float(learning_rate),
        transition_steps=1,
        decay_rate=lr_gamma,
    )
    return schedule


@argbind.bind()
def create_generator_optimizer(
        schedule: optax.Schedule,
        adam_b1: float = 0.9,
        adam_b2: float = 0.999,
        adam_weight_decay: float = 0.0,
        grad_clip: float = 1,
):
    # Combining gradient transforms using `optax.chain`.
    gradient_transform = optax.chain(
        optax.clip_by_global_norm(float(grad_clip)),
        optax.scale_by_adam(b1=adam_b1, b2=adam_b2),
        optax.add_decayed_weights(float(adam_weight_decay)),  # this puts the W in AdamW
        optax.scale_by_schedule(schedule),
        optax.scale(-1.0)  # gradient descent
    )
    return gradient_transform


def create_generator(
        key,
        shape,
        gradient_transform,
        tabulate=False,
) -> TrainState:

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
def create_discriminator_schedule(
        learning_rate: float = 1e-4,
        lr_gamma: float = 0.999996,
):
    # Exponential decay of the learning rate.
    schedule = optax.exponential_decay(
        init_value=float(learning_rate),
        transition_steps=1,
        decay_rate=lr_gamma,
    )
    return schedule


@argbind.bind()
def create_discriminator_optimizer(
        schedule: optax.Schedule,
        adam_b1: float = 0.9,
        adam_b2: float = 0.999,
        adam_weight_decay: float = 0.0,
        grad_clip: float = 1,
):
    gradient_transform = optax.chain(
        optax.clip_by_global_norm(float(grad_clip)),
        optax.scale_by_adam(b1=adam_b1, b2=adam_b2),
        optax.add_decayed_weights(float(adam_weight_decay)),  # this puts the W in "AdamW"
        optax.scale_by_schedule(schedule),
        optax.scale(-1.0)  # gradient descent
    )
    return gradient_transform


def create_discriminator(
        key,
        shape,
        gradient_transform,
        tabulate=False,
) -> TrainState:

    model = Discriminator()

    x = jnp.ones(shape=shape)
    key, subkey = random.split(key)
    params = model.init(subkey, x)['params']

    if tabulate:
        key, subkey = random.split(key)
        print(model.tabulate(subkey, x,
                             depth=3,
                             compute_flops=True,
                             compute_vjp_flops=True,
                             # column_kwargs={'width': 200},
                             console_kwargs={'width': 400},
                             ))

    state = TrainState.create(apply_fn=model.apply, params=params, tx=gradient_transform)
    return state


@partial(jax.pmap, axis_name='ensemble', donate_argnums=3, static_broadcasted_argnums=(4,))
@argbind.bind(without_prefix=True)
def eval_step(rng, state: GenDiscState, audio_tree: AudioTree, eval_metrics, sample_rate, lambdas: Mapping[str, float] = None)\
        -> EvalMetrics:

    assert lambdas is not None

    audio_data = audio_tree.audio_data

    audio_data = rearrange(audio_data, 'b c t -> (b c) 1 t', c=1)

    output = state.generator.apply_fn({'params': state.generator.params}, audio_data, sample_rate, train=False,
                                      rngs={'rng_stream': rng})
    recons = output['audio']

    output['stft/loss'] = multiscale_stft_loss(audio_data, recons)
    output['mel/loss'] = mel_spectrogram_loss(audio_data, recons, sample_rate=sample_rate)
    output['waveform/loss'] = l1_loss(audio_data, recons)

    fake = state.discriminator.apply_fn({'params': state.discriminator.params}, recons)
    real = state.discriminator.apply_fn({'params': state.discriminator.params}, audio_data)

    output['adv/disc_loss'] = discriminator_loss(fake, real)
    (
        output['adv/gen_loss'],
        output['adv/feat_loss'],
    ) = generator_loss(fake, real)

    output['loss'] = sum([v * output[k] for k, v in lambdas.items()])

    eval_metrics = eval_metrics.merge(EvalMetrics.gather_from_model_output(axis_name='ensemble', **output))

    return eval_metrics


def train_step_discriminator(state: GenDiscState, audio_data, output) -> Tuple[GenDiscState, struct.PyTreeNode]:

    def loss_fn(params):
        # note: you could calculate with the ``state.generator`` again, since its weights were just updated,
        # but we prefer not to in order to run faster.
        # output = state.generator.apply_fn({'params': state.generator.params}, audio_data, sample_rate,
        #                                   rngs=rngs, train=True  # todo: maybe pick Train=False even though DAC didn't
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


# note that we use without_prefix=True and the same lambdas is used for eval_step
@argbind.bind(without_prefix=True)
def train_step_generator(rng, state: GenDiscState, audio_data, sample_rate, lambdas: Mapping[str, float] = None) ->\
        Tuple[GenDiscState, dict]:

    assert lambdas is not None

    def loss_fn(params):

        output = state.generator.apply_fn({'params': params}, audio_data, sample_rate, rngs={'rng_stream': rng})
        recons = output['audio']

        fake = state.discriminator.apply_fn({'params': state.discriminator.params}, recons)
        real = state.discriminator.apply_fn({'params': state.discriminator.params}, audio_data)

        output['stft/loss'] = multiscale_stft_loss(audio_data, recons)
        output['mel/loss'] = mel_spectrogram_loss(audio_data, recons, sample_rate=sample_rate)
        output['waveform/loss'] = l1_loss(audio_data, recons)
        (
            output['adv/gen_loss'],
            output['adv/feat_loss'],
        ) = generator_loss(fake, real)
        loss = output['loss'] = sum([v * output[k] for k, v in lambdas.items()])
        return loss, output

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, output), grads = grad_fn(state.generator.params)
    grads = jax.lax.pmean(grads, axis_name='ensemble')

    state = state.replace(generator=state.generator.apply_gradients(grads=grads))
    return state, output


@partial(jax.pmap, axis_name='ensemble', donate_argnums=(1, 2), static_broadcasted_argnums=(5, 6, 7),
         in_axes=(0, 0, 0, 0, None, None, None, None))
def train_step(rng, state: GenDiscState, train_metrics, audio_tree: AudioTree, step: int, sample_rate,
               generator_schedule: optax.Schedule, discriminator_schedule: optax.Schedule) \
        -> Tuple[GenDiscState, TrainMetrics]:
    """Train for a single step."""

    audio_data = audio_tree.audio_data

    subkey1, subkey2 = random.split(rng, num=2)

    audio_data = rearrange(audio_data, 'b c t -> (b c) 1 t', c=1)

    state, output = train_step_generator(subkey2, state, audio_data, sample_rate)
    state, loss = train_step_discriminator(state, audio_data, output)

    output['adv/disc_loss'] = loss
    output['lr/generator'] = generator_schedule(step)
    output['lr/discriminator'] = discriminator_schedule(step)

    train_metrics = train_metrics.merge(TrainMetrics.gather_from_model_output(axis_name='ensemble', **output))

    return state, train_metrics


def save_checkpoint(checkpoint_manager: ocp.CheckpointManager, state: GenDiscState, step: int, metrics=None):

    if is_process_main():
        # get train state from the first replica
        main_state = jax.device_get(jax.tree_util.tree_map(lambda x: x[0], state))
        ckpt = {'generator': main_state.generator, 'discriminator': main_state.discriminator}
        save_args = ocp.args.StandardSave(ckpt)
        checkpoint_manager.save(step, ckpt, args=save_args, metrics=metrics)
        checkpoint_manager.wait_until_finished()


@partial(jax.pmap, axis_name='ensemble', static_broadcasted_argnums=(3,))
def save_samples(rng, state: GenDiscState, audio_tree: AudioTree, sample_rate):
    """Save audio samples to tensorboard."""
    audio_data = audio_tree.audio_data
    batch_size = audio_data.shape[0]
    audio_data = rearrange(audio_data, 'b c t -> (b c) 1 t', c=1)

    output = state.generator.apply_fn({'params': state.generator.params}, audio_data, sample_rate, train=False,
                                      rngs={'rng_stream': rng})
    recons = output['audio']
    recons = rearrange(recons, '(b c) 1 t -> b t c', b=batch_size, c=1)
    return recons


@argbind.bind()
def log_training(train_metrics: TrainMetrics, step: int, writer: metric_writers.MultiWriter, log_every_steps=1):

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


def log_eval(eval_metrics: EvalMetrics, step: int, writer: metric_writers.MultiWriter):

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
def train(
        args,
        name: str = None,
        num_iterations: int = 250_000,
        valid_freq: int = 100,
        sample_freq: int = 100,
        ckpt_max_keep: int = 2,
        seed: int = 0,
        batch_size: int = 4,
        val_batch_size: int = 1,
        sample_batch_size: int = 1,
        restore: int = 0,  # bool
        best_key='eval/loss',
        best_mode='min',
        enable_async_checkpointing: int = 1,  # bool
        log_level='info',
        save_path='runs',
        tabulate: int = 0,  # bool
):

    logging.set_verbosity(log_level.upper())  # absl logging
    logger.setLevel(log_level.upper())  # native python logging

    print(f'devices: {jax.devices()}')

    local_batch_size = batch_size
    local_val_batch_size = val_batch_size
    # todo: or use jax.process_count()
    batch_size *= jax.device_count()
    val_batch_size *= jax.device_count()
    sample_batch_size *= jax.device_count()

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
        SAMPLE_RATE = args['DAC.sample_rate']

    create_dataset = partial(_create_dataset, sample_rate=SAMPLE_RATE)

    with argbind.scope(args, "train"):
        train_iter = create_dataset(batch_size=batch_size, train=True, num_steps=num_iterations)

    with argbind.scope(args, "sample"):
        # num_epochs=None so that it can iterate forever.
        sample_iter = create_dataset(batch_size=sample_batch_size, num_epochs=None)

    with argbind.scope(args, "train"):
        duration = args['train/create_dataset.duration']

    n_channels = 1
    shape = (local_batch_size, n_channels, round(SAMPLE_RATE * duration))
    del duration
    key = random.key(seed)

    generator_schedule = create_generator_schedule()
    discriminator_schedule = create_discriminator_schedule()

    tabulate = bool(tabulate)

    key, subkey = random.split(key)
    generator_state = create_generator(subkey, shape, create_generator_optimizer(generator_schedule), tabulate)

    key, subkey = random.split(key)
    discriminator_state = create_discriminator(subkey, shape, create_discriminator_optimizer(discriminator_schedule),
                                               tabulate)

    state = GenDiscState(generator=generator_state, discriminator=discriminator_state)

    state = jax_utils.replicate(state)
    train_metrics = jax_utils.replicate(TrainMetrics.empty())
    eval_metrics = jax_utils.replicate(EvalMetrics.empty())

    del shape

    def best_fn(eval_metrics_summary) -> float:
        return eval_metrics_summary[best_key]

    checkpoint_manager = ocp.CheckpointManager(
        directory=ckpt_dir + '/orbax/managed',
        options=ocp.CheckpointManagerOptions(save_interval_steps=1, max_to_keep=ckpt_max_keep, best_fn=best_fn,
                                             best_mode=best_mode,
                                             enable_async_checkpointing=bool(enable_async_checkpointing)),
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
    report_progress = periodic_actions.ReportProgress(num_train_steps=num_iterations,
                                                      every_secs=None,
                                                      every_steps=10,
                                                      writer=writer if is_process_main() else None)
    if is_process_main():
        hooks += [report_progress]
        # todo: the Profile hook seems slow, so we don't use it.
        # hooks += [periodic_actions.Profile(num_profile_steps=10, profile_duration_ms=0, logdir=workdir)]

    early_stop = EarlyStopping()

    with metric_writers.ensure_flushes(writer):
        for step in range(1, num_iterations+1):

            with report_progress.timed("load_train_batch"):
                batch = next(train_iter)

            with report_progress.timed("train_step"):
                key, subkey = random.split(key)
                state, train_metrics = train_step(split_device(subkey), state, train_metrics, batch, step, SAMPLE_RATE,
                                                  generator_schedule, discriminator_schedule)

            for hook in hooks:
                hook(step)

            train_metrics = log_training(train_metrics, step, writer)

            if step % sample_freq == 0:
                with report_progress.timed("sample_step"):
                    all_signal = []
                    all_recons = []
                    for _ in range(2):  # todo: argbind the 2 parameter
                        save_batch = next(sample_iter)
                        key, subkey = random.split(key)
                        recons = save_samples(split_device(subkey), state, save_batch, SAMPLE_RATE)
                        recons = jax_utils.unreplicate(recons)
                        save_batch = jax_utils.unreplicate(save_batch)
                        signal = rearrange(save_batch.audio_data, 'b c t -> b t c', c=1)
                        assert recons.ndim == 3
                        all_signal.append(np.array(signal))
                        all_recons.append(np.array(recons))

                    writer.write_audios(step=step,
                                        audios={
                                            'signal': np.concatenate(all_signal, axis=0),
                                            'recons': np.concatenate(all_recons, axis=0),
                                        },
                                        sample_rate=SAMPLE_RATE)
                    writer.flush()

            if step % valid_freq == 0:
                # Compute metrics on the validation set
                with report_progress.timed("eval_step"):
                    with argbind.scope(args, "val"):
                        eval_iter = create_dataset(batch_size=val_batch_size)
                    ran_once = False
                    for test_batch in eval_iter:
                        ran_once = True
                        key, subkey = random.split(key)
                        eval_metrics = eval_step(split_device(subkey), state, test_batch, eval_metrics, SAMPLE_RATE)
                    assert ran_once

                    summary, eval_metrics = log_eval(eval_metrics, step, writer)

                    early_stop = early_stop.update(summary[best_key])
                    if early_stop.should_stop:
                        logger.info(f'Met early stopping criteria, breaking at step {step}')
                        break

                with report_progress.timed("checkpoint_step"):
                    save_checkpoint(checkpoint_manager, state, step, summary)

    checkpoint_manager.wait_until_finished()


if __name__ == '__main__':
    args = argbind.parse_args()
    with argbind.scope(args):
        train(args)
