from abc import ABC, abstractmethod
from flax import linen as nn
from jax import numpy as jnp
import typing as tp

from dac_jax.nn.encodec_quantize import QuantizedResult


class CompressionModel(ABC, nn.Module):
    """Base API for all compression models that aim at being used as audio tokenizers
    with a language model.
    """

    @abstractmethod
    def __call__(self, x: jnp.ndarray) -> QuantizedResult: ...

    @abstractmethod
    def encode(self, x: jnp.ndarray) -> tp.Tuple[jnp.ndarray, tp.Optional[jnp.ndarray]]:
        """See `EncodecModel.encode`."""
        ...

    @abstractmethod
    def decode(
        self,
        codes: jnp.ndarray,
        scale: tp.Optional[jnp.ndarray] = None,
        length: int = None,
    ):
        """See `EncodecModel.decode`."""
        ...

    @abstractmethod
    def decode_latent(self, codes: jnp.ndarray):
        """Decode from the discrete codes to continuous latent space."""
        ...

    @property
    @abstractmethod
    def channels(self) -> int: ...

    @property
    @abstractmethod
    def frame_rate(self) -> float: ...

    @property
    @abstractmethod
    def sample_rate(self) -> int: ...

    @property
    @abstractmethod
    def cardinality(self) -> int: ...

    @property
    @abstractmethod
    def num_codebooks(self) -> int: ...

    @property
    @abstractmethod
    def total_codebooks(self) -> int: ...
