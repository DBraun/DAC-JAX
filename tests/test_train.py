"""
Smoke test for scripts/train.py.

Runs a couple of real training iterations of a deliberately tiny model
(conf/mini.yml) over a small synthetic dataset, so that the data pipeline,
augmentation, generator/discriminator updates, evaluation, sampling and
checkpointing all get exercised.
"""

import os
from pathlib import Path
import shlex
import shutil
import subprocess

import argbind
import numpy as np
import soundfile

SAMPLE_RATE = 16_000
GROUPS = ["speech", "music", "env"]
SPLITS = ["train", "val", "test"]


def make_fake_data(data_dir: Path, num_files: int = 4):
    data_dir.mkdir(exist_ok=True, parents=True)
    rng = np.random.default_rng(0)
    for i in range(num_files):
        signal = rng.uniform(-0.5, 0.5, size=SAMPLE_RATE)
        soundfile.write(data_dir / f"sample_{i}.wav", signal, samplerate=SAMPLE_RATE)
    return data_dir


def make_fake_data_tree(data_dir: Path):
    for split in SPLITS:
        for group in GROUPS:
            make_fake_data(data_dir / split / group)
    return {
        split: {group: [str(data_dir / split / group)] for group in GROUPS}
        for split in SPLITS
    }


def setup_module(module):
    tmp_dir = Path(__file__).parent / "tmp_assets"
    repo_root = Path(__file__).parent.parent

    input_datasets = make_fake_data_tree(tmp_dir)

    conf = argbind.load_args(repo_root / "conf" / "mini.yml")
    # `sample` draws from the validation split, the same way conf/base.yml does.
    for scope, split in [
        ("train", "train"),
        ("val", "val"),
        ("test", "test"),
        ("gen", "val"),
    ]:
        conf[f"{scope}/create_dataset.sources"] = input_datasets[split]

    argbind.dump_args(conf, tmp_dir / "conf.yml")


def teardown_module(module):
    repo_root = Path(__file__).parent.parent
    shutil.rmtree(repo_root / "tests" / "tmp_assets", ignore_errors=True)
    shutil.rmtree(repo_root / "tests" / "runs", ignore_errors=True)


def test_single_gpu_train():
    env = os.environ.copy()
    repo_root = Path(__file__).parent.parent
    conf_path = repo_root / "tests" / "tmp_assets" / "conf.yml"
    ckpt_dir = repo_root / "tests" / "runs"
    args = shlex.split(
        f"python {repo_root}/scripts/train.py"
        f" --args.load {conf_path}"
        f" --train.ckpt_dir {ckpt_dir}"
        f" --train.name baseline"
    )
    subprocess.run(args, env=env, check=True)

    assert (ckpt_dir / "baseline" / "orbax").is_dir()


def test_multi_gpu_train():
    pass  # todo:
