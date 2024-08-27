import argbind
from dac_jax import SEANetEncoder, SEANetDecoder, EncodecModel
from dac_jax.nn.encodec_quantize import ResidualVectorQuantizer
import numpy as np
from jax import random
from jax import numpy as jnp

# SEANetEncoder = argbind.bind(SEANetEncoder)


def torch_to_encoder(torch_params: dict, encoder_rates: tuple[int] = None):
    d = {}
    def streamable(prefix):
        return {
            "NormConv1d_0": {
                "WeightNorm_0": {
                    "Conv_0/kernel/scale": torch_params[f"{prefix}.conv.conv.weight_g"],
                },
                "Conv_0": {
                    "bias": torch_params[f"{prefix}.conv.conv.bias"],
                    "kernel": torch_params[f"{prefix}.conv.conv.weight_v"].transpose(2, 1, 0),
                }
            }
        }

    i = 0
    for _ in range(4):  # todo:
        d[f'StreamableConv1d_{i}'] = streamable(f'encoder.model.{i}')
        i += 1
        d[f'SEANetResnetBlock_{i}'] = {
            f'StreamableConv1d_0': streamable(f'encoder.model.{i}.block.1'),
            f'StreamableConv1d_1': streamable(f'encoder.model.{i}.block.3'),
        }
        i += 2

    d[f'StreamableConv1d_{i}'] = streamable(f'encoder.model.{i}')

    d[f'StreamableLSTM_0'] = streamable(f'encoder.model.{i}')

    i += 1
    d[f'StreamableConv1d_{i}'] = streamable(f'encoder.model.{i}')

    pass

def torch_to_decoder(torch_params: dict, decoder_rates: tuple[int] = None):
    pass

def torch_to_quantizer(torch_params: dict):
    pass





def torch_to_linen(torch_params: dict,
                   encoder_rates: tuple[int] = None,
                   decoder_rates: tuple[int] = None,
                   n_codebooks: int = 9
                   ) -> dict:
    """Convert PyTorch parameters to Linen nested dictionaries"""

    if encoder_rates is None:
        encoder_rates = [2, 4, 8, 8]
    if decoder_rates is None:
        decoder_rates = [8, 8, 4, 2]

    return {
        'encoder': torch_to_encoder(torch_params, encoder_rates=encoder_rates),
        'decoder': torch_to_decoder(torch_params, decoder_rates=decoder_rates),
        'quantizer': torch_to_quantizer(torch_params),
    }

    def parse_wn_conv(flax_params, from_prefix, to_i: int):
        d = {}
        d[f'Conv_0'] = {
            'bias': torch_params[f'{from_prefix}.bias'],
            'kernel': torch_params[f'{from_prefix}.weight_v'].T
        }
        d[f'WeightNorm_0'] = {
            f'Conv_0/kernel/scale': torch_params[f'{from_prefix}.weight_g'].squeeze((1, 2))
        }
        flax_params[f'WNConv1d_{to_i}'] = d

    def parse_wn_convtranspose(flax_params, from_prefix, to_i: int):
        d = {}
        d[f'ConvTranspose_0'] = {
            'bias': torch_params[f'{from_prefix}.bias'],
            'kernel': torch_params[f'{from_prefix}.weight_v'].transpose()
        }
        d[f'WeightNorm_0'] = {
            f'ConvTranspose_0/kernel/scale': torch_params[f'{from_prefix}.weight_g'].squeeze((1, 2))
        }
        flax_params[f'WNConvTranspose1d_{to_i}'] = d

    def parse_residual_unit(flax_params, from_prefix, to_i):
        d = {}
        d['Snake1d_0'] = {
            'alpha': torch_params[f'{from_prefix}.block.0.alpha'].transpose(0, 2, 1)
        }
        parse_wn_conv(d, f'{from_prefix}.block.1', 0)
        d['Snake1d_1'] = {
            'alpha': torch_params[f'{from_prefix}.block.2.alpha'].transpose(0, 2, 1)
        }
        parse_wn_conv(d, f'{from_prefix}.block.3', 1)
        flax_params[f'ResidualUnit_{to_i}'] = d

    def parse_encoder_block(flax_params, from_prefix, to_i):
        d = {}
        for i in range(3):
            parse_residual_unit(d, f'{from_prefix}.block.{i}', i)

        d['Snake1d_0'] = {
            'alpha': torch_params[f'{from_prefix}.block.3.alpha'].transpose(0, 2, 1)
        }

        parse_wn_conv(d, f'{from_prefix}.block.4', 0)
        flax_params[f'EncoderBlock_{to_i}'] = d

    def parse_decoder_block(flax_params, from_prefix, to_i):
        d = {}
        d['Snake1d_0'] = {
            'alpha': torch_params[f'{from_prefix}.block.0.alpha'].transpose(0, 2, 1)
        }

        parse_wn_convtranspose(d, f'{from_prefix}.block.1', 0)

        for i in range(3):
            parse_residual_unit(d, f'{from_prefix}.block.{i+2}', i)

        flax_params[f'DecoderBlock_{to_i}'] = d

    flax_params = {'encoder': {}, 'decoder': {}, 'quantizer': {}}

    i = 0
    # add Encoder
    parse_wn_conv(flax_params['encoder'], f'encoder.block.{i}', 0)

    # add EncoderBlocks
    for _ in encoder_rates:
        parse_encoder_block(flax_params['encoder'], f'encoder.block.{i+1}', i)
        i += 1

    i += 1
    flax_params['encoder']['Snake1d_0'] = {
        'alpha': torch_params[f'encoder.block.{i}.alpha'].transpose(0, 2, 1)
    }

    i += 1
    parse_wn_conv(flax_params['encoder'], f'encoder.block.{i}', 1)

    # Add Quantizer
    for i in range(n_codebooks):
        quantizer = {}
        quantizer['in_proj'] = {
            'WeightNorm_0': {'Conv_0/kernel/scale': torch_params[f'quantizer.quantizers.{i}.in_proj.weight_g'].squeeze((1, 2))},
            'Conv_0': {
                'bias': torch_params[f'quantizer.quantizers.{i}.in_proj.bias'],
                'kernel': torch_params[f'quantizer.quantizers.{i}.in_proj.weight_v'].T,
            }
        }
        quantizer['codebook'] = {
            'embedding': torch_params[f'quantizer.quantizers.{i}.codebook.weight']
        }
        quantizer['out_proj'] = {
            'WeightNorm_0': {'Conv_0/kernel/scale': torch_params[f'quantizer.quantizers.{i}.out_proj.weight_g'].squeeze((1, 2))},
            'Conv_0': {
                'bias': torch_params[f'quantizer.quantizers.{i}.out_proj.bias'],
                'kernel': torch_params[f'quantizer.quantizers.{i}.out_proj.weight_v'].T,
            }
        }
        flax_params['quantizer'][f'quantizers_{i}'] = quantizer

    i = 0
    # Add Decoder
    parse_wn_conv(flax_params['decoder'], f'decoder.model.{i}', 0)

    # Add DecoderBlocks
    for _ in decoder_rates:
        parse_decoder_block(flax_params['decoder'], f'decoder.model.{i+1}', i)
        i += 1

    i += 1
    flax_params['decoder']['Snake1d_0'] = {
        'alpha': torch_params[f'decoder.model.{i}.alpha'].transpose(0, 2, 1)
    }

    i += 1
    parse_wn_conv(flax_params['decoder'], f'decoder.model.{i}', 1)

    return {'params': flax_params}


def main():
    load_path = "encodec_torch_weights.npy"  # todo:
    allow_pickle = True  # todo:
    torch_params = np.load(load_path, allow_pickle=allow_pickle)
    torch_params = torch_params.item()

    kwargs  = {
        "channels": 1,
        "dimension": 128,
        "n_filters": 64,  # todo: or 32?
        "n_residual_layers": 1,
        # "ratios": [8, 5, 4, 2],
        "ratios": [8, 5, 4, 4],  # todo:
        "activation": 'elu',
        "activation_params": {'alpha': 1.0},
        "norm": 'weight_norm',
        "norm_params": {},
        "kernel_size": 7,
        "last_kernel_size": 7,
        "residual_kernel_size": 3,
        "dilation_base": 2,
        "causal": False,
        "pad_mode": 'constant',
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
        # kmeans_init=False,  # todo: set to True if we needed to train
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
        frame_rate=640,
        sample_rate=32_000,
        channels=1,
    )

    x = jnp.zeros((2, 1, 44100))  # todo:
    print(encodec_model.tabulate(
        {"params": random.key(0), "rng_stream": random.key(0)},
        x,
        console_kwargs={"width": 300},
    ))

    variables = encodec_model.init(
        {"params": random.key(0), "rng_stream": random.key(0)},
        x,
    )
    params = variables["params"]

    # variables = torch_to_linen(torch_params, model.encoder_rates, model.decoder_rates, model.n_codebooks)

    pass


if __name__ == '__main__':
    main()
