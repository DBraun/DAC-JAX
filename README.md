# DAC-JAX and EnCodec-JAX

This repository holds **unofficial** JAX implementations of Descript's DAC and Meta's EnCodec.
We are not affiliated with Descript or Meta.

You can read the DAC-JAX paper [here](https://arxiv.org/abs/2405.11554).

## Background

In 2022, Meta published "[High Fidelity Neural Audio Compression](https://arxiv.org/abs/2210.13438)".
They eventually open-sourced the code inside [AudioCraft](https://github.com/facebookresearch/audiocraft/blob/main/docs/ENCODEC.md).

In 2023, Descript published a related work "[High-Fidelity Audio Compression with Improved RVQGAN](https://arxiv.org/abs/2306.06546)"
and released their code under the name [DAC](https://github.com/descriptinc/descript-audio-codec/) (Descript Audio Codec).

Both EnCodec and DAC are neural audio codecs which use residual vector quantization inside a fully convolutional
encoder-decoder architecture.

## Usage

### Installation

DAC-JAX requires **Python 3.12 or newer**.

The project is managed with [uv](https://docs.astral.sh/uv/). To work on it, clone
the repository and let `uv` build the environment from `uv.lock`:

```bash
git clone https://github.com/DBraun/DAC-JAX
cd DAC-JAX
uv sync --extra dev
```

That gives you a reproducible environment with everything: the library, the
training stack, the test suite and the PyTorch reference implementations it
compares against. Run anything inside it with `uv run`, e.g. `uv run pytest
tests`, and add or change dependencies with `uv add` / `uv remove` rather than by
editing `pyproject.toml` by hand.

There are two extras:

| Extra   | Adds                                                                  |
| ------- | --------------------------------------------------------------------- |
| `train` | What `scripts/train.py` needs -- grain, clu, optax, orbax, tensorboardX |
| `dev`   | `train`, plus pytest, ruff, and the PyTorch reference implementations  |

The library itself imports none of that, so plain `dac-jax` stays light. To
train without the test dependencies, use `uv sync --extra train`.

To use DAC-JAX as a dependency of your own project:

```bash
uv add "dac-jax @ git+https://github.com/DBraun/DAC-JAX"
```

<!-- ```bash
uv add dac-jax
``` -->

Or, with pip:

```bash
python -m pip install git+https://github.com/DBraun/DAC-JAX
```

Two notes on the dependencies:

* **PyTorch** is not declared as a dependency and is imported lazily. It is only
  needed the first time you download a pretrained Descript or Meta checkpoint, to
  read it and convert it to `.npy`; everything else works without it. Install it
  yourself, and prefer the **CPU** build -- a GPU build can conflict with JAX's
  CUDA installation. (The `dev` extra installs it anyway, since the equivalence
  tests compare against the PyTorch implementations.)
* For **GPU support**, install [JAX](https://jax.readthedocs.io/en/latest/installation.html)
  with the appropriate CUDA extra; the lockfile pins the CPU build.

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

# If you want to use pretrained 32 kHz EnCodec from Meta's MusicGen, use this:
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


# Load a mono audio file with the correct sample rate
signal, sample_rate = librosa.load(
    "input.wav", sr=model.sample_rate, mono=True, duration=0.5
)

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
signal, sample_rate = librosa.load("input.wav", sr=44100, mono=True, duration=0.5)

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
mse = jnp.square(y - signal).mean()
```

### DAC compression with constant GPU memory regardless of input length:

```python
import dac_jax

import jax
import jax.numpy as jnp
import librosa

# Download a model and set padding to False because we will use the chunk functions.
model, variables = dac_jax.load_model(model_type="44khz", padding=False)

# Load a mono audio file at any sample rate
signal, sample_rate = librosa.load("input.wav", sr=None, mono=True)

signal = jnp.array(signal, dtype=jnp.float32)
while signal.ndim < 3:
    # signal will eventually be shaped [B, C, T]
    signal = jnp.expand_dims(signal, axis=0)


# Jit-compile these functions because they're used inside a loop over chunks.
@jax.jit
def compress_chunk(x):
    return model.apply(variables, x, method="compress_chunk")


@jax.jit
def decompress_chunk(c):
    return model.apply(variables, c, method="decompress_chunk")


win_duration = 0.5  # Adjust based on your GPU's memory size
dac_file = model.compress(
    compress_chunk, signal, sample_rate, win_duration=win_duration
)

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

```bash
uv run pytest tests
```

Both PyTorch parity tests run by default and need no GPU. `test_dac_equivalence.py`
compares against [descript-audio-codec](https://github.com/descriptinc/descript-audio-codec),
and `test_encodec_equivalence.py` compares against `facebook/encodec_32khz` through
`transformers` -- that is MusicGen's compression model published on its own, so it
exercises the same weights without needing
[audiocraft](https://github.com/facebookresearch/audiocraft), which cannot be installed
on Python 3.12. Both are installed by the `dev` extra.

## Limitations

Pull requests—especially ones which address any of the limitations below—are welcome.

* We implement the "chunked" `compress`/`decompress` methods from the PyTorch repository, although this technique has some problems outlined [here](https://github.com/descriptinc/descript-audio-codec/issues/39).
* We have not run all evaluation scripts in the `scripts` directory. For some of them, it makes sense to just keep using PyTorch instead of JAX.
* The model architecture code (`model/dac.py`) has many static methods to help with finding DAC's `delay` and `output_length`. Please help us refactor this so that code is not so duplicated and at risk of typos.
* The source code of DAC-JAX has some `todo:` markings which indicate (mostly minor) improvements we'd like to have.
* We don't have a Docker image yet like the original [DAC repository](https://github.com/descriptinc/descript-audio-codec) does.
* Please check the limitations of [argbind-dbraun](https://github.com/DBraun/argbind?tab=readme-ov-file#limitations-and-known-issues), the fork of argbind that this project uses.
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
