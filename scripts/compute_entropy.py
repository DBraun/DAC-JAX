import argbind
import jax
from audiotools import AudioSignal
import numpy as np
import tqdm

from dac_jax import load_model
from dac_jax.audio_utils import find_audio


@argbind.bind(without_prefix=True, positional=True)
def main(
    folder: str,
    model_path: str,
    metadata_path: str,
    n_samples: int = 1024,
):
    files = find_audio(folder)[:n_samples]
    key = jax.random.key(0)
    key, subkey = jax.random.split(key)
    signals = [
        AudioSignal.salient_excerpt(f, subkey, loudness_cutoff=-20, duration=1.0)
        for f in files
    ]

    assert model_path is not None
    assert metadata_path is not None

    model, variables = load_model(load_path=model_path, metadata_path=metadata_path)
    model = model.bind(variables)

    codes = []
    for x in tqdm.tqdm(signals):
        x = jax.device_put(x, model.device)
        o = model.encode(x.audio_data, x.sample_rate)
        codes.append(np.array(o["codes"]))

    codes = np.concatenate(codes, axis=-1)
    entropy = []

    for i in range(codes.shape[1]):
        codes_ = codes[0, i, :]
        counts = np.bincount(codes_)
        counts = (counts / counts.sum())
        counts = np.maximum(counts, 1e-10)
        entropy.append(-(counts * np.log(counts)).sum().item() * np.log2(np.e))

    pct = sum(entropy) / (10 * len(entropy))
    print(f"Entropy for each codebook: {entropy}")
    print(f"Effective percentage: {pct * 100}%")


if __name__ == "__main__":
    args = argbind.parse_args()
    with argbind.scope(args):
        main()
