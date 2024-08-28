from jax import numpy as jnp


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

    bias = bias_ih_l0 + bias_hh_l0

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
    lstm_layers = 2  # todo:
    d[f"StreamableLSTM_0"] = {
        f"LSTMCell_{k}": lstm(torch_params, f"encoder.model.{j}", k)
        for k in range(lstm_layers)
    }
    j += lstm_layers

    i += 1
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
