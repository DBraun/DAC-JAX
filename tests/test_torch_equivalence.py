import os
os.environ["XLA_FLAGS"] = (
    ' --xla_gpu_deterministic_ops=true'  # todo: https://github.com/google/flax/discussions/3382
)
os.environ["TF_CUDNN_DETERMINISTIC"] = '1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import torch
torch.use_deterministic_algorithms(True)

import jax
import jax.numpy as jnp

import dac as dac_torch
from audiotools import AudioSignal

import librosa
import numpy as np

import dac_jax


def _torch_padding(np_data) -> dict[np.array]:

    model_path = dac_torch.utils.download(model_type="44khz")
    model = dac_torch.DAC.load(model_path)

    x = torch.from_numpy(np_data)
    sample_rate = model.sample_rate  # note: not always true outside of this test
    x = model.preprocess(x, sample_rate)
    z, codes, latents, commitment_loss, codebook_loss = model.encode(x)

    # Decode audio signal
    audio = model.decode(z)

    d = {
        'audio': audio,
        'z': z,
        'codes': codes,
        'latents': latents,
        'vq/commitment_loss': commitment_loss,
        'vq/codebook_loss': codebook_loss
    }

    d = {k: v.detach().cpu().numpy() for k, v in d.items()}

    return d


def _torch_compress(np_data, win_duration: float):

    model = dac_torch.utils.load_model(model_type="44khz")

    sample_rate = model.sample_rate  # note: not always true outside of this test
    x = AudioSignal(np_data, sample_rate=sample_rate)

    dac_file = model.compress(x, win_duration=win_duration)
    # get an embedding z for just a single chunk, only for the sake of comparing to jax
    c = dac_file.codes[..., : dac_file.chunk_length]
    z = model.quantizer.from_codes(c)[0]
    z = z.detach().cpu().numpy()

    recons = model.decompress(dac_file).audio_data
    recons = recons.cpu().numpy()

    return dac_file.codes, z, recons


def _jax_padding(np_data) -> dict[np.array]:

    model, variables = dac_jax.load_model(model_type="44khz")

    y = model.apply(variables, jnp.array(np_data), model.sample_rate, method="preprocess")
    y = model.apply(variables, y, train=False)
    y['z'] = y['z'].transpose(0, 2, 1)
    y['codes'] = y['codes'].transpose(0, 2, 1)
    y['latents'] = y['latents'].transpose(0, 2, 1)

    # Multiply by model.n_codebooks since we normalize by n_codebooks and torch doesn't.
    y['vq/commitment_loss'] = y['vq/commitment_loss']*model.n_codebooks
    y['vq/codebook_loss'] = y['vq/codebook_loss']*model.n_codebooks

    y = jax.tree.map(lambda x: np.array(x), y)
    return y


def _jax_compress(np_data, win_duration: float):

    # set padding to False since we're using the chunk functions
    model, variables = dac_jax.load_model(model_type='44khz', padding=False)
    sample_rate = 44100

    @jax.jit
    def compress_chunk(x):
        return model.apply(variables, x, method='compress_chunk')

    @jax.jit
    def decompress_chunk(c):
        return model.apply(variables, c, method='decompress_chunk')

    @jax.jit
    def from_codes(c):
        return model.apply(variables, c, method='from_codes')

    key = jax.random.key(0)
    subkey1, subkey2, subkey3 = jax.random.split(key, 3)
    x = jax.random.normal(subkey1, shape=(1, 1, int(sample_rate*2)))

    _ = model.init({'params': subkey2, 'rng_stream': subkey3}, x, sample_rate)

    x = jnp.array(np_data)
    dac_file = model.compress(compress_chunk, x, sample_rate, win_duration=win_duration)

    codes = dac_file.codes

    c = codes[..., : dac_file.chunk_length, :]
    z = from_codes(c)
    z = np.array(z).transpose(0, 2, 1)

    recons = model.decompress(decompress_chunk, dac_file)
    recons = np.array(recons)

    codes = np.array(codes).transpose(0, 2, 1)

    return codes, z, recons


def test_equivalence_padding():

    np.random.seed(0)
    np_data = np.random.normal(loc=0, scale=1, size=(1, 1, 4096)).astype(np.float32)

    jax_result = _jax_padding(np_data)
    torch_result = _torch_padding(np_data)
    assert set(jax_result.keys()) == set(torch_result.keys())
    assert list(jax_result.keys())
    for key in jax_result.keys():
        if key == 'latents':
            # todo: why do we need to accept lower absolute tolerance for this key?
            atol = 1e-3
        elif key in ['vq/commitment_loss', 'vq/codebook_loss']:
            # todo: why do we need to accept lower absolute tolerance for these keys?
            atol = 1e-3
        elif key == 'codes':
            atol = 1e-8
        elif key == 'audio':
            atol = 1e-5
        elif key == 'z':
            atol = 1e-5
        else:
            raise ValueError(f"Unexpected key '{key}'.")
        assert np.allclose(jax_result[key], torch_result[key], atol=atol), \
            f'Failed to match outputs for key: {key} and atol: {atol}'


def test_equivalence_compress(verbose=False):

    def compress_helper(np_data, atol, win_duration=.38):

        jax_codes, jax_z, jax_recons = _jax_compress(np_data, win_duration)
        torch_codes, torch_z, torch_recons = _torch_compress(np_data, win_duration)
        assert np.allclose(jax_codes, torch_codes)
        np.testing.assert_almost_equal(torch_z, jax_z, decimal=5)  # todo: raise this to decimal=6
        if verbose:
            print('max diff: ', jnp.abs(jax_recons-torch_recons).max())
        assert np.allclose(jax_recons, torch_recons, atol=atol)

    np_data, sr = librosa.load('assets/60013__qubodup__whoosh.flac', sr=None, mono=True)
    np_data = np.expand_dims(np.array(np_data), 0)
    np_data = np.expand_dims(np.array(np_data), 0)
    np_data = np.concatenate([np_data, np_data, np_data, np_data], axis=-1)
    compress_helper(np_data, atol=1e-5)

    np.random.seed(0)
    num_samples = int(44100*10)
    np_data = 0.5*np.random.uniform(low=-1, high=1, size=(1, 1, num_samples)).astype(np.float32)
    # todo: for compressing/decompressing noise, why must we use a higher absolute tolerance?
    compress_helper(np_data, atol=.003)


if __name__ == '__main__':
    test_equivalence_padding()
    test_equivalence_compress()
    print('All Done!')
