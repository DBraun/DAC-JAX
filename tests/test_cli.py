"""
Tests for CLI.
"""

import subprocess
from pathlib import Path

import argbind
import numpy as np
import pytest
import soundfile

from dac_jax.__main__ import run


def setup_module(module):
    data_dir = Path(__file__).parent / "tmp_assets"
    data_dir.mkdir(exist_ok=True, parents=True)
    input_dir = data_dir / "input"
    input_dir.mkdir(exist_ok=True, parents=True)

    for i in range(5):
        sample_rate = 44_100
        signal = np.random.randn(1000, sample_rate)
        soundfile.write(input_dir / f"sample_{i}.wav", signal, samplerate=sample_rate)
    return input_dir


def teardown_module(module):
    repo_root = Path(__file__).parent.parent
    subprocess.check_output(["rm", "-rf", f"{repo_root}/tests/tmp_assets"])


@pytest.mark.parametrize("model_type", ["44khz", "24khz", "16khz"])
def test_reconstruction(model_type):
    # Test encoding
    input_dir = Path(__file__).parent / "tmp_assets" / "input"
    output_dir = input_dir.parent / model_type / "encoded_output"
    args = {
        "input": str(input_dir),
        "output": str(output_dir),
        "model_type": model_type,
    }
    with argbind.scope(args):
        run("encode")

    # Test decoding
    input_dir = output_dir
    output_dir = input_dir.parent / model_type / "decoded_output"
    args = {
        "input": str(input_dir),
        "output": str(output_dir),
        "model_type": model_type,
    }
    with argbind.scope(args):
        run("decode")


def test_compression():
    # Test encoding
    input_dir = Path(__file__).parent / "tmp_assets" / "input"
    output_dir = input_dir.parent / "encoded_output_quantizers"
    args = {
        "input": str(input_dir),
        "output": str(output_dir),
        "n_quantizers": 3,
    }
    with argbind.scope(args):
        run("encode")

    # Open .dac file
    dac_file = output_dir / "sample_0.dac"
    allow_pickle = True  # todo:
    artifacts = np.load(dac_file, allow_pickle=allow_pickle)[()]
    codes = artifacts["codes"]

    # Ensure that the number of quantizers is correct
    assert codes.shape[2] == 3

    # Ensure that dtype of compression is uint16
    assert codes.dtype == np.uint16


# CUDA_VISIBLE_DEVICES=0 python -m pytest tests/test_cli.py -s
