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
            'layer_instance/kernel/scale': torch_params[f'quantizer.quantizers.{i}.in_proj.weight_g'].squeeze((1, 2)),
            'layer_instance': {
                'bias': torch_params[f'quantizer.quantizers.{i}.in_proj.bias'],
                'kernel': torch_params[f'quantizer.quantizers.{i}.in_proj.weight_v'].T,
            }
        }
        quantizer['codebook'] = {
            'embedding': torch_params[f'quantizer.quantizers.{i}.codebook.weight']
        }
        quantizer['out_proj'] = {
            'layer_instance/kernel/scale': torch_params[f'quantizer.quantizers.{i}.out_proj.weight_g'].squeeze((1, 2)),
            'layer_instance': {
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
