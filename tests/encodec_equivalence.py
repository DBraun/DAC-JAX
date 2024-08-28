from pathlib import Path

import argbind
from einops import rearrange
from jax import random
from jax import numpy as jnp
import librosa
import numpy as np

from dac_jax import SEANetEncoder, SEANetDecoder, EncodecModel
from dac_jax.nn.encodec_quantize import ResidualVectorQuantizer

# SEANetEncoder = argbind.bind(SEANetEncoder)


def streamable(torch_params, prefix: str):
    return {
        "NormConv1d_0": {
            "WeightNorm_0": {
                "Conv_0/kernel/scale": torch_params[
                    f"{prefix}.conv.conv.weight_g"
                ].squeeze((1, 2)),
            },
            "Conv_0": {
                "bias": torch_params[f"{prefix}.conv.conv.bias"],
                "kernel": torch_params[f"{prefix}.conv.conv.weight_v"].T,
            },
        }
    }


def streamable_transpose(torch_params, prefix: str):
    return {
        "NormConvTranspose1d_0": {
            "WeightNorm_0": {
                "ConvTranspose_0/kernel/scale": torch_params[
                    f"{prefix}.convtr.convtr.weight_g"
                ].squeeze((1, 2)),
            },
            "ConvTranspose_0": {
                "bias": torch_params[f"{prefix}.convtr.convtr.bias"],
                "kernel": torch_params[f"{prefix}.convtr.convtr.weight_v"].T,
            },
        }
    }


def lstm(torch_params, prefix: str, i: int):
    weight_ih_l0 = torch_params[f"{prefix}.lstm.weight_ih_l{i}"]
    weight_hh_l0 = torch_params[f"{prefix}.lstm.weight_hh_l{i}"]
    bias_ih_l0 = torch_params[f"{prefix}.lstm.bias_ih_l{i}"]
    bias_hh_l0 = torch_params[f"{prefix}.lstm.bias_hh_l{i}"]

    weight_hh_l0 = weight_hh_l0.transpose(1, 0)
    weight_ih_l0 = weight_ih_l0.transpose(1, 0)

    # https://github.com/pytorch/pytorch/blob/40de63be097ce6d499aac15fc58ed27ca33e5227/aten/src/ATen/native/RNN.cpp#L1560-L1564
    kernel_hi, kernel_hf, kernel_hg, kernel_ho = jnp.split(weight_hh_l0, 4, axis=1)
    kernel_ii, kernel_if, kernel_ig, kernel_io = jnp.split(weight_ih_l0, 4, axis=1)

    bias = bias_ih_l0 + bias_hh_l0  # todo: correct?

    bias_i, bias_f, bias_g, bias_o = jnp.split(bias, 4)

    return {
        "hi": {
            "bias": bias_i,
            "kernel": kernel_hi,
        },
        "hf": {
            "bias": bias_f,
            "kernel": kernel_hf,
        },
        "hg": {
            "bias": bias_g,
            "kernel": kernel_hg,
        },
        "ho": {
            "bias": bias_o,
            "kernel": kernel_ho,
        },
        "ii": {
            "kernel": kernel_ii,
        },
        "if": {
            "kernel": kernel_if,
        },
        "ig": {
            "kernel": kernel_ig,
        },
        "io": {
            "kernel": kernel_io,
        },
    }


def torch_to_encoder(torch_params: dict, encoder_rates: tuple[int] = None):
    d = {}

    i = 0
    j = 0
    for _ in range(len(encoder_rates)):
        d[f"StreamableConv1d_{i}"] = streamable(torch_params, f"encoder.model.{j}")
        j += 1
        d[f"SEANetResnetBlock_{i}"] = {
            f"StreamableConv1d_0": streamable(
                torch_params, f"encoder.model.{j}.block.1"
            ),
            f"StreamableConv1d_1": streamable(
                torch_params, f"encoder.model.{j}.block.3"
            ),
        }
        i += 1
        j += 2

    d[f"StreamableConv1d_{i}"] = streamable(torch_params, f"encoder.model.{j}")

    j += 1
    assert j == 13  # todo: remove
    lstm_layers = 2  # todo:
    d[f"StreamableLSTM_0"] = {
        f"LSTMCell_{k}": lstm(torch_params, f"encoder.model.{j}", k)
        for k in range(lstm_layers)
    }
    j += lstm_layers

    i += 1
    assert i == 5  # todo: remove
    assert j == 15  # todo: remove
    d[f"StreamableConv1d_{i}"] = streamable(torch_params, f"encoder.model.{j}")

    return d


def torch_to_decoder(torch_params: dict, decoder_rates: tuple[int] = None):
    d = {}

    i = 0
    j = 0

    d[f"StreamableConv1d_{i}"] = streamable(torch_params, f"decoder.model.{j}")
    j += 1
    lstm_layers = 2  # todo:
    d[f"StreamableLSTM_0"] = {
        f"LSTMCell_{k}": lstm(torch_params, f"decoder.model.{j}", k)
        for k in range(lstm_layers)
    }
    j += lstm_layers
    assert j == 3  # todo: remove
    for k in range(len(decoder_rates)):
        d[f"StreamableConvTranspose1d_{i}"] = streamable_transpose(
            torch_params, f"decoder.model.{j}"
        )
        j += 1
        d[f"SEANetResnetBlock_{i}"] = {
            f"StreamableConv1d_0": streamable(
                torch_params, f"decoder.model.{j}.block.1"
            ),
            f"StreamableConv1d_1": streamable(
                torch_params, f"decoder.model.{j}.block.3"
            ),
        }
        i += 1
        j += 2

    assert j == 15  # todo: remove
    d[f"StreamableConv1d_1"] = streamable(torch_params, f"decoder.model.{j}")

    return d


def torch_to_quantizer(torch_params: dict, n_quantizers):
    d = {
        f"layers_{i}": {
            "_codebook": {
                "embed": torch_params[f"quantizer.vq.layers.{i}._codebook.embed"],
                "embed_avg": torch_params[
                    f"quantizer.vq.layers.{i}._codebook.embed_avg"
                ],
            }
        }
        for i in range(n_quantizers)
    }

    return {"vq": d}


def torch_to_linen(
    torch_params: dict,
    encoder_rates: tuple[int] = None,
    decoder_rates: tuple[int] = None,
    n_codebooks: int = 9,
) -> dict:
    """Convert PyTorch parameters to Linen nested dictionaries"""

    if encoder_rates is None:
        encoder_rates = [2, 4, 8, 8]
    if decoder_rates is None:
        decoder_rates = [8, 8, 4, 2]

    return {
        "params": {
            "encoder": torch_to_encoder(torch_params, encoder_rates=encoder_rates),
            "decoder": torch_to_decoder(torch_params, decoder_rates=decoder_rates),
            "quantizer": torch_to_quantizer(torch_params, n_codebooks),
        }
    }


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
        "pad_mode": "constant",
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

    encodec_model = EncodecModel(
        encoder=encoder,
        decoder=decoder,
        quantizer=quantizer,
        causal=False,
        renormalize=False,
        frame_rate=50,
        sample_rate=32_000,
        channels=1,
    )

    x = jnp.array(np_data)

    print(
        encodec_model.tabulate(
            {"params": random.key(0), "rng_stream": random.key(0)},
            x,
            console_kwargs={"width": 300},
            depth=3,
        )
    )

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

    pass
    result = encodec_model.apply(
        {"params": params_from_torch}, x, rngs={"rng_stream": random.key(0)}
    )
    recons = result.x
    codes = result.codes
    print("all done!")

    return np.array(recons), np.array(codes)


def run_torch_model(np_data):
    import torch
    from audiocraft.models import MusicGen

    # Using small model, better results would be obtained with `medium` or `large`.
    model = MusicGen.get_pretrained("facebook/musicgen-small")
    x = torch.from_numpy(np_data).cuda()
    result = model.compression_model(x)

    recons = result.x.detach().cpu().numpy()
    codes = result.codes.detach().cpu().numpy()

    return recons, codes


if __name__ == "__main__":

    np_data, sr = librosa.load(
        Path(__file__).parent / "assets/60013__qubodup__whoosh.flac", sr=None, mono=True
    )
    np_data = np.expand_dims(np.array(np_data), 0)
    np_data = np.expand_dims(np.array(np_data), 0)
    np_data = np.concatenate([np_data, np_data, np_data, np_data], axis=-1)

    np_data *= 0.5

    torch_recons, torch_codes = run_torch_model(np_data)
    jax_recons, jax_codes = run_jax_model(np_data)

    pass
