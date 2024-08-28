from pathlib import Path

from audiocraft.models import MusicGen
from jax import random
from jax import numpy as jnp
import librosa
import numpy as np
import torch

from dac_jax import SEANetEncoder, SEANetDecoder, EncodecModel
from dac_jax.nn.encodec_quantize import ResidualVectorQuantizer
from dac_jax.utils.load_torch_weights_encodec import torch_to_linen


def run_jax_model(np_data):
    load_path = "encodec_torch_weights.npy"  # todo:
    allow_pickle = True  # todo:
    torch_params = np.load(load_path, allow_pickle=allow_pickle)
    torch_params = torch_params.item()

    kwargs = {
        "channels": 1,
        "dimension": 128,
        "n_filters": 64,
        "n_residual_layers": 1,
        "ratios": [8, 5, 4, 4],
        "activation": "elu",
        "activation_params": {"alpha": 1.0},
        "norm": "weight_norm",
        "norm_params": {},
        "kernel_size": 7,
        "last_kernel_size": 7,
        "residual_kernel_size": 3,
        "dilation_base": 2,
        "causal": False,
        "pad_mode": "reflect",
        "true_skip": True,
        "compress": 2,
        "lstm": 2,
        "disable_norm_outer_blocks": 0,
    }
    encoder_override_kwargs = {}
    decoder_override_kwargs = {
        "trim_right_ratio": 1.0,
        "final_activation": None,
        "final_activation_params": None,
    }
    encoder_kwargs = {**kwargs, **encoder_override_kwargs}
    decoder_kwargs = {**kwargs, **decoder_override_kwargs}

    encoder = SEANetEncoder(**encoder_kwargs)
    decoder = SEANetDecoder(**decoder_kwargs)
    quantizer = ResidualVectorQuantizer(
        dimension=encoder.dimension,
        n_q=4,
        q_dropout=False,
        bins=2048,
        decay=0.99,
        kmeans_init=True,
        kmeans_iters=50,
        threshold_ema_dead_code=0,  # todo: set to 2 if we needed to train
        orthogonal_reg_weight=0.0,
        orthogonal_reg_active_codes_only=False,
        orthogonal_reg_max_codes=None,
    )

    sample_rate = 32_000

    encodec_model = EncodecModel(
        encoder=encoder,
        decoder=decoder,
        quantizer=quantizer,
        causal=False,
        renormalize=False,
        frame_rate=sample_rate // encoder.hop_length,
        sample_rate=sample_rate,
        channels=1,
    )

    x = jnp.array(np_data)

    variables = encodec_model.init(
        {"params": random.key(0), "rng_stream": random.key(0)},
        x,
    )
    params = variables["params"]

    variables_from_torch = torch_to_linen(
        torch_params,
        encodec_model.encoder.ratios,
        encodec_model.decoder.ratios,
        encodec_model.num_codebooks,
    )
    params_from_torch = variables_from_torch["params"]

    result = encodec_model.apply(
        {"params": params_from_torch}, x, rngs={"rng_stream": random.key(0)}
    )
    recons = result.x
    codes = result.codes

    return np.array(recons), np.array(codes)


def run_torch_model(np_data):
    # Using small model, better results would be obtained with `medium` or `large`.
    model = MusicGen.get_pretrained("facebook/musicgen-small")
    x = torch.from_numpy(np_data).cuda()
    result = model.compression_model(x)

    recons = result.x.detach().cpu().numpy()
    codes = result.codes.detach().cpu().numpy()

    return recons, codes


def test_encoded_equivalence():
    np_data, sr = librosa.load(
        Path(__file__).parent / "assets/60013__qubodup__whoosh.flac", sr=None, mono=True
    )
    np_data = np.expand_dims(np.array(np_data), 0)
    np_data = np.expand_dims(np.array(np_data), 0)
    np_data = np.concatenate([np_data, np_data, np_data, np_data], axis=-1)

    np_data *= 0.5

    torch_recons, torch_codes = run_torch_model(np_data)
    jax_recons, jax_codes = run_jax_model(np_data)

    assert np.allclose(torch_codes, jax_codes)
    assert np.allclose(torch_recons, jax_recons, atol=1e-4)  # todo: reduce atol to 1e-5


if __name__ == "__main__":
    test_encoded_equivalence()
