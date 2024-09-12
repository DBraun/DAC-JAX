# DAC-JAX and EnCodec-JAX

This repository holds **unofficial** JAX implementations of Descript's DAC and Meta's EnCodec.
We are not affiliated with Descript or Meta.

You can read the DAC-JAX paper [here](https://arxiv.org/abs/2405.11554).

## Background

In 2022, Meta published "[High Fidelity Neural Audio Compression](High Fidelity Neural Audio Compression)".
They eventually open-sourced the code inside [AudioCraft](https://github.com/facebookresearch/audiocraft/blob/main/docs/ENCODEC.md).

In 2023, Descript published a related work "[High-Fidelity Audio Compression with Improved RVQGAN](https://arxiv.org/abs/2306.06546)"
and released their code under the name [DAC](https://github.com/descriptinc/descript-audio-codec/) (Descript Audio Codec).

Both EnCodec and DAC are neural audio codecs which use residual vector quantization inside a fully convolutional
encoder-decoder architecture.

## Usage

### Installation

1. Upgrade `pip` and `setuptools`:
    ```bash
    pip install --upgrade pip setuptools
    ```

2. Install the **CPU** version of [PyTorch](https://pytorch.org/).
   We strongly suggest the CPU version because trying to install a GPU version can conflict with JAX's CUDA-related installation.
   PyTorch is required because it's used to load pretrained model weights.

3. Install [JAX](https://jax.readthedocs.io/en/latest/installation.html) (with GPU support).

4. Install DAC-JAX with one of the following:

    <!-- ```
    python -m pip install dac-jax
    ```
    OR -->
    
    ```
    pip install git+https://github.com/DBraun/DAC-JAX
    ```
    
    Or,
    
    ```bash
    python -m pip install .
    ```
    
    Or, if you intend to contribute, clone and do an [editable install](https://pip.pypa.io/en/stable/topics/local-project-installs/#editable-installs):
    ```bash
    python -m pip install -e ".[dev]"
    ```

### Weights
The original Descript repository releases model weights under the MIT license. These weights are for models that natively support 16 kHz, 24kHz, and 44.1kHz sampling rates. Our scripts download these PyTorch weights and load them into JAX.
Weights are automatically downloaded when you first run an `encode` or `decode` command. You can download them in advance with one of the following commands:
```bash
python -m dac_jax download_model # downloads the default 44kHz variant
python -m dac_jax download_model --model_type 44khz --model_bitrate 16kbps # downloads the 44kHz 16 kbps variant
python -m dac_jax download_model --model_type 44khz # downloads the 44kHz variant
python -m dac_jax download_model --model_type 24khz # downloads the 24kHz variant
python -m dac_jax download_model --model_type 16khz # downloads the 16kHz variant
```

EnCodec weights can be downloaded similarly. This will download the 32 kHz EnCodec used in [MusicGen](https://github.com/facebookresearch/audiocraft/blob/main/docs/MUSICGEN.md).
```bash
python -m dac_jax download_encodec
```

For both DAC and EnCodec, the default download location is `~/.cache/dac_jax`. You can change the location by setting an **absolute path** value for an environment variable `DAC_JAX_CACHE`. For example, on macOS/Linux:
```bash
export DAC_JAX_CACHE=/Users/admin/my-project/dac_jax_models
```

If you do this, remember to still have `DAC_JAX_CACHE` set before you use the `load_model` function.

### Compress audio
```
python -m dac_jax encode /path/to/input --output /path/to/output/codes
```

This command will create `.dac` files with the same name as the input files.
It will also preserve the directory structure relative to input root and
re-create it in the output directory. Please use `python -m dac_jax encode --help`
for more options.

### Reconstruct audio from compressed codes
```
python -m dac_jax decode /path/to/output/codes --output /path/to/reconstructed_input
```

This command will create `.wav` files with the same name as the input files.
It will also preserve the directory structure relative to input root and
re-create it in the output directory. Please use `python -m dac_jax decode --help`
for more options.

### Programmatic usage (DAC and EnCodec)

Here we use `jax.jit` for optimized encoding and decoding.
This does not do sample-rate conversion or volume normalization in the encoder or decoder.

```python
from functools import partial

import jax
from jax import numpy as jnp
import librosa

import dac_jax

model, variables = dac_jax.load_model(model_type="44khz")

# if you want to use pretrained 32 kHz EnCodec
# model, variables = dac_jax.load_encodec_model()


@jax.jit
def encode_to_codes(x: jnp.ndarray):
    codes, scale = model.apply(
        variables,
        x,
        method="encode",
    )
    return codes, scale

@partial(jax.jit, static_argnums=(1, 2))
def decode_from_codes(codes: jnp.ndarray, scale, length: int = None):
    recons = model.apply(
        variables,
        codes,
        scale,
        length,
        method="decode",
    )
    return recons

# Load a mono audio file
signal, sample_rate = librosa.load('input.wav', sr=44100, mono=True, duration=.5)

signal = jnp.array(signal, dtype=jnp.float32)
while signal.ndim < 3:
    signal = jnp.expand_dims(signal, axis=0)

original_length = signal.shape[-1]

codes, scale = encode_to_codes(signal)
assert codes.shape[1] == model.num_codebooks

recons = decode_from_codes(codes, scale, original_length)
```

### DAC with Binding

Here we use DAC-JAX as a "[bound](https://flax.readthedocs.io/en/latest/developer_notes/module_lifecycle.html#bind)" module, freeing us from repeatedly passing variables as an argument and using `.apply`. Note that bound modules are not meant to be used in fine-tuning.

```python
import dac_jax
from dac_jax import DACFile

from jax import numpy as jnp
import librosa

# Download a model and bind variables to it.
model, variables = dac_jax.load_model(model_type="44khz")
model = model.bind(variables)

# Load a mono audio file
signal, sample_rate = librosa.load('input.wav', sr=44100, mono=True, duration=.5)

signal = jnp.array(signal, dtype=jnp.float32)
while signal.ndim < 3:
    signal = jnp.expand_dims(signal, axis=0)

# Encode audio signal as one long file (may run out of GPU memory on long files).
# This performs resampling to the codec's sample rate and volume normalization.
dac_file = model.encode_to_dac(signal, sample_rate)

# Save to a file
dac_file.save("dac_file_001.dac")

# Load a file
dac_file = DACFile.load("dac_file_001.dac")

# Decode audio signal. Since we're passing a dac_file, this undoes the 
# previous sample rate conversion and volume normalization.
y = model.decode(dac_file)

# Calculate mean-square error of reconstruction in time-domain
mse = jnp.square(y-signal).mean()
```

### DAC compression with constant GPU memory regardless of input length:

```python
import dac_jax

import jax
import jax.numpy as jnp
import librosa

# Download a model and set padding to False because we will use the chunk functions.
model, variables = dac_jax.load_model(model_type="44khz", padding=False)

# Load a mono audio file
signal, sample_rate = librosa.load('input.wav', sr=44100, mono=True, duration=.5)

signal = jnp.array(signal, dtype=jnp.float32)
while signal.ndim < 3:
    # signal will eventually be shaped [B, C, T]
    signal = jnp.expand_dims(signal, axis=0)

# Jit-compile these functions because they're used inside a loop over chunks.
@jax.jit
def compress_chunk(x):
    return model.apply(variables, x, method='compress_chunk')

@jax.jit
def decompress_chunk(c):
    return model.apply(variables, c, method='decompress_chunk')

win_duration = 0.5  # Adjust based on your GPU's memory size
dac_file = model.compress(compress_chunk, signal, sample_rate, win_duration=win_duration)

# Save and load to and from disk
dac_file.save("compressed.dac")
dac_file = dac_jax.DACFile.load("compressed.dac")

# Decompress it back to audio
y = model.decompress(decompress_chunk, dac_file)
```

## DAC Training
The baseline model configuration can be trained using the following commands.

```bash
python scripts/train.py --args.load conf/final/44khz.yml --train.ckpt_dir="/tmp/dac_jax_runs"
```

In root directory, monitor with Tensorboard (`runs` will appear next to `scripts`):
```bash
tensorboard --logdir="/tmp/dac_jax_runs"
```

## Testing

```
python -m pytest tests
```

## Limitations

Pull requests—especially ones which address any of the limitations below—are welcome.

* We implement the "chunked" `compress`/`decompress` methods from the PyTorch repository, although this technique has some problems outlined [here](https://github.com/descriptinc/descript-audio-codec/issues/39).
* We have not run all evaluation scripts in the `scripts` directory. For some of them, it makes sense to just keep using PyTorch instead of JAX.
* The model architecture code (`model/dac.py`) has many static methods to help with finding DAC's `delay` and `output_length`. Please help us refactor this so that code is not so duplicated and at risk of typos.
* In `audio_utils.py` we use [DM_AUX's](https://github.com/google-deepmind/dm_aux) STFT function instead of `jax.scipy.signal.stft`. We believe this is faster but requires more memory.
* The source code of DAC-JAX has some `todo:` markings which indicate (mostly minor) improvements we'd like to have.
* We don't have a Docker image yet like the original [DAC repository](https://github.com/descriptinc/descript-audio-codec) does.
* Please check the limitations of [argbind](https://github.com/pseeth/argbind?tab=readme-ov-file#limitations-and-known-issues).
* We don't provide a training script for EnCodec.

## Citation

If you use this repository in your work, please cite  EnCodec:
```
@article{defossez2022high,
  title={High fidelity neural audio compression},
  author={D{\'e}fossez, Alexandre and Copet, Jade and Synnaeve, Gabriel and Adi, Yossi},
  journal={arXiv preprint arXiv:2210.13438},
  year={2022}
}
```

DAC:

```
@article{kumar2024high,
  title={High-fidelity audio compression with improved rvqgan},
  author={Kumar, Rithesh and Seetharaman, Prem and Luebs, Alejandro and Kumar, Ishaan and Kumar, Kundan},
  journal={Advances in Neural Information Processing Systems},
  volume={36},
  year={2024}
}
```



and DAC-JAX:

```
@misc{braun2024dacjax,
  title={{DAC-JAX}: A {JAX} Implementation of the Descript Audio Codec}, 
  author={David Braun},
  year={2024},
  eprint={2405.11554},
  archivePrefix={arXiv},
  primaryClass={cs.SD}
}
```
