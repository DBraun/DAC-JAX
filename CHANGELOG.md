# Change log

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
(categories: Added, Changed, Deprecated, Removed, Fixed, Security), and DAC-JAX
follows [Effort-based Versioning](https://jacobtomlinson.dev/effver/).

## [Unreleased]

Nothing yet.

## [2.0.0] - 2026-08-22

A dependency-modernization release. The floor moves to Python 3.12 and the current
JAX/Flax stack, `audiotree` goes from 0.2 to 1.0 (a near-total rewrite of its public
surface), and the build system moves to `uv` and a PEP 621 `pyproject.toml`.

**Existing `conf/*.yml` files will not load unchanged**, and training results are not
comparable with 1.1.0 -- audiotree 1.0 changed both its loudness meter and how excerpts
are chosen. The renames a config needs are listed under *Changed* below.

### Added

* **ruff as a pre-commit hook** (`.pre-commit-config.yaml`), pinned to the same version
  as the `ruff` in the dev extra. Its `types_or` is widened to include Markdown, because
  ruff 0.16 also formats Python code blocks inside Markdown and the README's examples are
  worth keeping in step with the code they mirror. `pyproject.toml` gains a `[tool.ruff]`
  block stating `line-length` and `target-version` explicitly -- both are ruff's current
  defaults, pinned so a later release formats the repository the same way. Run
  `uv run pre-commit install` once to activate it locally.
* `conf/mini.yml`, a deliberately tiny model used by the training smoke test, so that
  `tests/test_train.py` runs a couple of real iterations -- data pipeline, augmentation,
  generator and discriminator updates, evaluation, sampling and checkpointing -- in about
  a minute on CPU.
* `MetricWriter` and `ReportProgress` in `scripts/train.py`, small local replacements for
  the two pieces of `clu` that the script used. `clu.metric_writers` (and
  `clu.periodic_actions`, which imports it) pull in TensorFlow unconditionally, and
  TensorFlow's native runtime segfaults when loaded into the same process as grain's --
  grain hard-imports `array_record`, which links its own copy. The replacements write the
  same TensorBoard event files through `tensorboardX`. `clu.metrics` is still used.
* `tests/test_losses.py` checks the losses and metrics that `tests/test_audio_utils.py`
  did not cover -- `l1_loss`, `sisdr_loss`, the adversarial and feature-matching losses,
  `stoi` and `pesq` -- against descript-audio-codec and audiotools. All of them agree to
  within float32 rounding (worst relative difference ~2e-7), the `lambdas` in
  `conf/base.yml` match Descript's, and the VQ commitment/codebook losses and their
  residual accumulation are line-for-line equivalent.
* Tests take their audio fixture from `librosa.example("brahms")`, which downloads and
  caches on first use. The fixture they used to reference lived under `tests/assets/`,
  which is listed in `.gitignore` and was therefore never committed, so six tests failed
  on a fresh clone with a `LibsndfileError`. Nothing binary is tracked now. The
  equivalence tests read from a fixed offset, since the residual quantizer's `argmin`
  can break a near-tie differently in JAX than in Torch, and `test_equivalence_compress`
  now allows `atol=1e-3` on the reconstruction: codes still match exactly and the latents
  to ~2e-6, but the decoder turns that into ~4e-4 of waveform, and the old 1e-5 held only
  for the short noise burst that used to be the fixture.

### Changed

* **Requires Python 3.12+** (was 3.10). The current `jax`, `jaxlib`, `flax` and `librosa`
  releases all require 3.12.
* **Upgraded to `flax>=0.12.9` and `jax`/`jaxlib>=0.11.1`**, plus `numpy` 2.
* **Upgraded to `audiotree>=1.0.0`.** Configuration keys that must be renamed:
  * transform names are snake_case functions: `VolumeNorm` to `volume_norm`,
    `RescaleAudio` to `rescale_audio`, `ShiftPhase` to `shift_phase`, `VolumeChange` to
    `volume_change`; their parameters are flat, so `VolumeNorm.config: {min_db: ...}`
    becomes `volume_norm.min_db: ...`.
  * `SaliencyParams` is `ExcerptConfig`: `enabled: 1` plus `search_function` becomes
    `strategy: loudest` plus `search: uniform` (or `bias_early`), and `loudness_cutoff` is
    `lufs_cutoff`. Values are unchanged and still match Descript's `salient_excerpt`
    (`num_tries=8`, `-40` LUFS).
  * the validation, test and generation splits are now explicitly `strategy: start`,
    preserving the pre-1.0 behavior of reading from the beginning of each file.
* **The argbind scope `sample` is now `gen`**, so `sample/create_dataset.sources` becomes
  `gen/create_dataset.sources`, and so on.
* `scripts/input_pipeline.py` was rewritten for the grain dataset API: audiotree 1.0
  returns a `grain.MapDataset` rather than a data source, so `create_dataset` now builds
  `to_iter_dataset().batch(..., batch_fn=AudioTree.batch).mp_prefetch(...)` instead of a
  `DataLoader` with an `IndexSampler` and a `ReduceBatchTransform`.
* `AudioTree.audio_data` is `AudioTree.waveform` throughout `scripts/train.py`.
* **Switched from `argbind` to `argbind-dbraun>=0.6.0`.** The import name is unchanged.
  Note that 0.6.0 no longer silently drops keyword arguments a bound callable does not
  declare.
* **The build system moved to `uv` and a PEP 621 `[project]` table**; `setup.cfg` is gone
  and `uv.lock` is tracked. `jaxloudnorm` now comes from PyPI rather than a git URL, and
  with `dm_aux` gone (below) there are no git dependencies left.
* **`torch` is imported lazily.** It sat at module scope in `dac_jax.utils`, so importing
  `dac_jax` at all required it, even though it is only used to read a pretrained
  checkpoint and convert it. `import dac_jax` now works without PyTorch installed.
* **The training dependencies moved to a `train` extra**, and `dev` now includes it.
  `grain`, `clu`, `optax`, `orbax-checkpoint` and `tensorboardX` are only used by
  `scripts/train.py` and `scripts/input_pipeline.py`, never by `dac_jax` itself, so
  installing the library no longer drags in the training stack. `chex` and `numpy` are
  now declared, having always been imported directly.
* `jax.experimental.shard_map` is now `jax.shard_map`, whose `mesh` argument is
  keyword-only.
* README installation and testing instructions are written for `uv`.

### Removed

* **`jax-ai-stack` is no longer a dependency.** It pins `jax==0.8.0` and `flax==0.12.0`
  exactly, which made raising the JAX/Flax floor impossible. `jax`, `jaxlib` and `flax`
  are now direct dependencies, and `optax` and `orbax-checkpoint` moved to the `train`
  extra.
* **`dm_aux` is replaced by [librosax](https://github.com/DBraun/librosax).** It was the
  last git dependency and the last unmaintained one, pinning `numpy<1.27` and
  `librosa==0.10.1` (2022-era) which had to be overridden to allow numpy 2. Only
  `dm_aux.spectral.stft` was ever used, and librosax's `stft` matches it in shape and to
  ~5e-7 relative across every configuration this project uses, while also handling
  `sqrt_hann` and `(batch, channels, samples)` input natively -- so `audio_utils.stft`
  loses both of its `rearrange` calls and its special-casing. The only remaining uv
  override is `protobuf`, for descript-audiotools.
* `audiocraft` is gone from the development extra; it stopped being used in 57359e1.
* The duplicate `pandas` entry in the development extra.

### Fixed

* **Loudness normalization was measuring the wrong axis.** `jaxloudnorm` takes
  `(channels, samples)`, but `dac_jax.audio_utils.volume_norm` transposed each item to
  `(samples, channels)` first -- correct for the 0.2.0 fork, wrong for 0.3.2. The upgrade
  surfaced it as an error rather than a silent mis-normalization, because 0.3.2 validates
  its channel count. The transpose is removed.
* **`dac_jax encode` and `dac_jax decode` could never run.** Both were wrapped in
  `@jax.jit` despite doing file I/O and NumPy conversion, so any invocation died with a
  `TracerArrayConversionError`. The inner `compress_chunk`/`decompress_chunk` were already
  jitted, which is the right granularity.
* **`dac_jax encode`/`decode` loaded the model with padding enabled**, but `compress` and
  `decompress` are unconditionally chunked and the chunk methods assert that padding is
  off. They now load with `padding=False`, as `scripts/benchmark.py` and the README already
  did.
* **`dac_jax decode` never wrote its output**, calling `recons.write(...)` on a JAX array
  as though it were an `audiotools.AudioSignal`. It now writes with `soundfile`.
* `tests/test_cli.py` wrote its fixtures as `randn(1000, 44100)`, which soundfile reads as
  1000 frames of 44100 channels; libsndfile refused it and left zero-byte files. It also
  asserted the quantizer count against `codes.shape[2]`, the time axis, rather than
  `shape[1]`.
* `tests/test_train.py` wrote its generated config to `tmp_assets/` but loaded it from
  `assets/`, and set torch-DAC key names (`build_dataset.folders`,
  `AudioDataset.n_examples`) that this project does not use. It has been rewritten around
  `conf/mini.yml`.
* **`tests/test_encodec_equivalence.py` now runs.** It failed collection with a
  `ModuleNotFoundError` for `audiocraft`, which 57359e1 dropped as a dependency, and
  audiocraft cannot be reinstated -- it pins `torch==2.1.0` and `torchtext==0.16.0`, so it
  will not install on Python 3.12. The PyTorch reference is now `transformers`'
  `EncodecModel` loading `facebook/encodec_32khz`, which is MusicGen's compression model
  published on its own; it also drops the hard-coded `.cuda()`, so the test runs on CPU in
  about 15 seconds. Codes come out bit-identical and the reconstruction agrees to ~1.8e-7,
  so the `atol` is tightened from 1e-4 to 1e-5, closing the TODO that asked for it. The
  input is also read at the model's own 32 kHz rather than 44.1 kHz.
* **`stoi` passed the whole batch as the estimate.** Inside its per-item loop it handed
  `estimates` rather than `estimate` to `pystoi`, so it compared each reference against
  the entire batch array and raised on any input it could not broadcast.
* **`pesq` reported the wrong sample rate.** It resamples both signals to `target_sr` and
  then told `pesq` the rate they arrived at instead, so a 44.1 kHz input was scored as
  though the resampled audio were still 44.1 kHz.
* **`normalize` produced NaN gradients for a zero vector.** It guarded the division with
  `jnp.maximum(eps, norm)`, which fixes the forward pass but not the derivative:
  `jnp.linalg.norm` is `sqrt(sum(x**2))`, whose gradient at zero is infinite, and
  `jnp.maximum` contributes a local gradient of zero there, so the chain rule yields
  `0 * inf = NaN`. The clamp now happens inside the square root, which leaves the forward
  values unchanged (`sqrt(max(s, eps**2)) == max(sqrt(s), eps)`) and matches
  `torch.nn.functional.normalize` to ~1e-7 as before. This is live in training --
  `decode_latents` L2-normalizes both the encodings and the codebook, and an unused
  codebook entry can decay to exactly zero. `tests/test_gradients.py` covers it. See the
  [JAX FAQ](https://docs.jax.dev/en/latest/faq.html#gradients-contain-nan-where-using-where).
* **`phase_loss` could not run.** A misplaced bracket left it selecting
  `diff[diff > jnp.pi - 2 * jnp.pi]` -- that is, `diff > -pi` -- where it meant to
  subtract `2 * pi` from the entries above `pi`, so the two sides never matched in shape.
  It now wraps into `[-pi, pi]` properly, which deliberately departs from audiotools:
  that implementation reads `diff[diff > np.pi] -= -2 * np.pi` and adds where it means to
  subtract. `phase_loss` is not referenced anywhere, so nothing downstream changes.

### Notes

Work that landed after the 1.1.0 version bump but was never released under a version of
its own is included here: EnCodec support (`22068ac`), a fix to the discriminator loss,
initializers and parallelism (`ade6adf`), the audiotree 0.2.0 upgrade (`daebb60`), and the
removal of the audiocraft dependency (`57359e1`).

## [1.1.0] - 2024-08-08

* Replaced the data pipeline with [Grain](https://github.com/google/grain) and
  [AudioTree](https://github.com/DBraun/AudioTree) (`39e912b`).
* Resampling now goes through `audiotree.resample` (`490a878`).

## [1.0.0] - 2024-04-23

Initial JAX port of Descript's [descript-audio-codec](https://github.com/descriptinc/descript-audio-codec),
covering the DAC model, discriminator, losses, the `encode`/`decode` CLI, and a training
script driven by argbind configs (`cb66ef5`).
