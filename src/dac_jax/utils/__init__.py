import os
from os import environ
from pathlib import Path
import json

import argbind
import numpy as np
from omegaconf import OmegaConf
import torch

from dac_jax.utils import load_torch_weights_encodec
from dac_jax.utils import load_torch_weights
from dac_jax.model import DAC
from dac_jax.model import EncodecModel, SEANetEncoder, SEANetDecoder
from dac_jax.nn.encodec_quantize import ResidualVectorQuantizer


__MODEL_LATEST_TAGS__ = {
    ("44khz", "8kbps"): "0.0.1",
    ("24khz", "8kbps"): "0.0.4",
    ("16khz", "8kbps"): "0.0.5",
    ("44khz", "16kbps"): "1.0.0",
}

__MODEL_URLS__ = {
    (
        "44khz",
        "0.0.1",
        "8kbps",
    ): "https://github.com/descriptinc/descript-audio-codec/releases/download/0.0.1/weights.pth",
    (
        "24khz",
        "0.0.4",
        "8kbps",
    ): "https://github.com/descriptinc/descript-audio-codec/releases/download/0.0.4/weights_24khz.pth",
    (
        "16khz",
        "0.0.5",
        "8kbps",
    ): "https://github.com/descriptinc/descript-audio-codec/releases/download/0.0.5/weights_16khz.pth",
    (
        "44khz",
        "1.0.0",
        "16kbps",
    ): "https://github.com/descriptinc/descript-audio-codec/releases/download/1.0.0/weights_44khz_16kbps.pth",
}


def convert_torch_weights_to_numpy(
    torch_weights_path: Path, write_path: Path, metadata_path: Path
):

    if write_path.exists() and metadata_path.exists():
        return

    if not write_path.exists():
        write_path.parent.mkdir(parents=True, exist_ok=True)

    weights = torch.load(str(torch_weights_path), map_location=torch.device("cpu"))

    kwargs = weights["metadata"]["kwargs"]
    with open(metadata_path, "w") as f:
        f.write(json.dumps(kwargs))

    weights = weights["state_dict"]
    weights = {key: value.numpy() for key, value in weights.items()}

    allow_pickle = (
        True  # todo: https://github.com/descriptinc/descript-audio-codec/issues/53
    )

    np.save(write_path, weights, allow_pickle=allow_pickle)


@argbind.bind(group="download_encodec", positional=True, without_prefix=True)
def download_encodec(
    name: str = "facebook/musicgen-small",
):
    if (
        "DAC_JAX_CACHE" in environ
        and environ["DAC_JAX_CACHE"].strip()
        and os.path.isabs(environ["DAC_JAX_CACHE"])
    ):
        cache_home = environ["DAC_JAX_CACHE"]
        cache_home = Path(cache_home)
    else:
        cache_home = Path.home() / ".cache" / "dac_jax"

    safename = name.replace("/", "_")

    metadata_path = cache_home / f"encodec_weights_{safename}.json"
    jax_write_path = cache_home / f"encodec_jax_weights_{safename}.npy"

    if jax_write_path.exists() and metadata_path.exists():
        return jax_write_path, metadata_path

    torch_model_path = cache_home / f"encodec_weights_{safename}.pth"

    if not torch_model_path.exists():
        torch_model_path.parent.mkdir(parents=True, exist_ok=True)

        from audiocraft.models.loaders import load_compression_model_ckpt

        file_or_url_or_id = name
        pkg = load_compression_model_ckpt(file_or_url_or_id, cache_dir=str(cache_home))
        cfg = OmegaConf.create(pkg["xp.cfg"])

        weights = pkg["best_state"]
        weights = {key: value.numpy() for key, value in weights.items()}

        jax_write_path.parent.mkdir(parents=True, exist_ok=True)

        allow_pickle = (
            True  # todo: https://github.com/descriptinc/descript-audio-codec/issues/53
        )

        np.save(jax_write_path, weights, allow_pickle=allow_pickle)

        OmegaConf.save(config=cfg, f=metadata_path)

    return jax_write_path, metadata_path


# todo: we don't call this function `download` because that would conflict with the PyTorch implementation's `download`.
# and we need to be able to run both in our tests.
# Reference issue: https://github.com/pseeth/argbind/?tab=readme-ov-file#bound-function-names-should-be-unique
@argbind.bind(group="download_model", positional=True, without_prefix=True)
def download_model(
    model_type: str = "44khz", model_bitrate: str = "8kbps", tag: str = "latest"
):
    """
    Function that downloads the weights file from URL if a local cache is not found.

    Parameters
    ----------
    model_type : str
        The type of model to download. Must be one of "44khz", "24khz", or "16khz". Defaults to "44khz".
    model_bitrate: str
        Bitrate of the model. Must be one of "8kbps", or "16kbps". Defaults to "8kbps".
        Only 44khz model supports 16kbps.
    tag : str
        The tag of the model to download. Defaults to "latest".

    Returns
    -------
    Path
        Directory path required to load model via audiotools.
    """
    model_type = model_type.lower()
    tag = tag.lower()

    if (
        "DAC_JAX_CACHE" in environ
        and environ["DAC_JAX_CACHE"].strip()
        and os.path.isabs(environ["DAC_JAX_CACHE"])
    ):
        cache_home = environ["DAC_JAX_CACHE"]
        cache_home = Path(cache_home)
    else:
        cache_home = Path.home() / ".cache" / "dac_jax"

    metadata_path = cache_home / f"weights_{model_type}_{model_bitrate}_{tag}.json"
    jax_write_path = cache_home / f"jax_weights_{model_type}_{model_bitrate}_{tag}.npy"

    if jax_write_path.exists() and metadata_path.exists():
        return jax_write_path, metadata_path

    assert model_type in [
        "44khz",
        "24khz",
        "16khz",
    ], "model_type must be one of '44khz', '24khz', or '16khz'"

    assert model_bitrate in [
        "8kbps",
        "16kbps",
    ], "model_bitrate must be one of '8kbps', or '16kbps'"

    if tag == "latest":
        tag = __MODEL_LATEST_TAGS__[(model_type, model_bitrate)]

    download_link = __MODEL_URLS__.get((model_type, tag, model_bitrate), None)

    if download_link is None:
        raise ValueError(
            f"Could not find model with tag {tag} and model type {model_type}"
        )

    torch_model_path = cache_home / f"weights_{model_type}_{model_bitrate}_{tag}.pth"

    if not torch_model_path.exists():
        torch_model_path.parent.mkdir(parents=True, exist_ok=True)

        # Download the model
        import requests

        response = requests.get(download_link)

        if response.status_code != 200:
            raise ValueError(
                f"Could not download model. Received response code {response.status_code}"
            )
        torch_model_path.write_bytes(response.content)

    convert_torch_weights_to_numpy(torch_model_path, jax_write_path, metadata_path)

    # remove torch model because it's not needed anymore.
    if torch_model_path.exists():
        os.remove(torch_model_path)

    return jax_write_path, metadata_path


def load_encodec_model(
    name: str = "facebook/musicgen-small",
    load_path: str = None,
    metadata_path: str = None,
):
    if not load_path or not metadata_path:
        load_path, metadata_path = download_encodec(name)

    kwargs = OmegaConf.load(metadata_path)

    seanet_kwargs = kwargs["seanet"]

    common_kwargs = {
        "channels": kwargs["channels"],
        "dimension": seanet_kwargs["dimension"],
        "n_filters": seanet_kwargs["n_filters"],
        "n_residual_layers": seanet_kwargs["n_residual_layers"],
        "ratios": seanet_kwargs["ratios"],
        "activation": seanet_kwargs["activation"].lower(),
        "activation_params": OmegaConf.to_object(seanet_kwargs["activation_params"]),
        "norm": seanet_kwargs["norm"],
        "norm_params": OmegaConf.to_object(seanet_kwargs["norm_params"]),
        "kernel_size": seanet_kwargs["kernel_size"],
        "last_kernel_size": seanet_kwargs["last_kernel_size"],
        "residual_kernel_size": seanet_kwargs["residual_kernel_size"],
        "dilation_base": seanet_kwargs["dilation_base"],
        "causal": kwargs["encodec"]["causal"],
        "pad_mode": seanet_kwargs["pad_mode"],
        "true_skip": seanet_kwargs["true_skip"],
        "compress": seanet_kwargs["compress"],
        "lstm": seanet_kwargs["compress"],
        "disable_norm_outer_blocks": seanet_kwargs["disable_norm_outer_blocks"],
    }
    encoder_override_kwargs = {}
    decoder_override_kwargs = {
        "trim_right_ratio": seanet_kwargs["decoder"]["trim_right_ratio"],
        "final_activation": seanet_kwargs["decoder"]["final_activation"],
        "final_activation_params": seanet_kwargs["decoder"]["final_activation_params"],
    }
    encoder_kwargs = {**common_kwargs, **encoder_override_kwargs}
    decoder_kwargs = {**common_kwargs, **decoder_override_kwargs}

    rvq_kwargs = kwargs["rvq"]
    quantizer_kwargs = {
        "dimension": seanet_kwargs["dimension"],
        "n_q": rvq_kwargs["n_q"],
        "q_dropout": rvq_kwargs["q_dropout"],
        "bins": rvq_kwargs["bins"],
        "decay": rvq_kwargs["decay"],
        "kmeans_init": rvq_kwargs["kmeans_init"],
        "kmeans_iters": rvq_kwargs["kmeans_iters"],
        "threshold_ema_dead_code": rvq_kwargs["threshold_ema_dead_code"],
        "orthogonal_reg_weight": rvq_kwargs["orthogonal_reg_weight"],
        "orthogonal_reg_active_codes_only": rvq_kwargs[
            "orthogonal_reg_active_codes_only"
        ],
        "orthogonal_reg_max_codes": None,  # todo:
    }

    encoder = SEANetEncoder(**encoder_kwargs)
    decoder = SEANetDecoder(**decoder_kwargs)
    quantizer = ResidualVectorQuantizer(**quantizer_kwargs)

    sample_rate = kwargs["sample_rate"]

    encodec_model = EncodecModel(
        encoder=encoder,
        decoder=decoder,
        quantizer=quantizer,
        causal=kwargs["encodec"]["causal"],
        renormalize=kwargs["encodec"]["renormalize"],
        frame_rate=sample_rate // encoder.hop_length,
        sample_rate=sample_rate,
        channels=kwargs["channels"],
    )

    allow_pickle = (
        True  # todo: https://github.com/descriptinc/descript-audio-codec/issues/53
    )

    torch_params = np.load(load_path, allow_pickle=allow_pickle)
    torch_params = torch_params.item()  # todo

    variables = load_torch_weights_encodec.torch_to_linen(
        torch_params,
        encodec_model.encoder.ratios,
        encodec_model.decoder.ratios,
        encodec_model.num_codebooks,
    )

    return encodec_model, variables


def load_model(
    model_type: str = "44khz",
    model_bitrate: str = "8kbps",
    tag: str = "latest",
    load_path: str = None,
    metadata_path: str = None,
    padding=True,
):
    # reference:
    # https://flax.readthedocs.io/en/latest/guides/training_techniques/transfer_learning.html#create-a-function-for-model-loading

    if not load_path or not metadata_path:
        load_path, metadata_path = download_model(
            model_type=model_type, model_bitrate=model_bitrate, tag=tag
        )

    with open(str(metadata_path), "r") as f:
        kwargs = json.loads(f.read())

    kwargs["padding"] = padding  # todo: seems like bad design

    model = DAC(**kwargs)

    allow_pickle = (
        True  # todo: https://github.com/descriptinc/descript-audio-codec/issues/53
    )

    torch_params = np.load(load_path, allow_pickle=allow_pickle)
    torch_params = torch_params.item()

    variables = load_torch_weights.torch_to_linen(
        torch_params, model.encoder_rates, model.decoder_rates, model.num_codebooks
    )

    return model, variables


if __name__ == "__main__":
    load_encodec_model()
