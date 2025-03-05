import sys

import argbind

from dac_jax.utils import download_model
from dac_jax.utils import download_encodec
from dac_jax.utils.decode import decode
from dac_jax.utils.encode import encode

STAGES = ["encode", "decode", "download_model", "download_encodec"]


def run(stage: str):
    """Run stages.

    Parameters
    ----------
    stage : str
        Stage to run
    """
    if stage not in STAGES:
        raise ValueError(f"Unknown command: {stage}. Allowed commands are {STAGES}")
    stage_fn = globals()[stage]

    stage_fn()


if __name__ == "__main__":
    group = sys.argv.pop(1)
    args = argbind.parse_args(group=group)

    with argbind.scope(args):
        run(group)
