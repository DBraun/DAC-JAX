# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Base class for all quantizers.
"""

from dataclasses import dataclass, field
import typing as tp

from einops import einsum, rearrange
from flax import linen as nn
from flax.training.common_utils import onehot
import jax.lax
from jax import numpy as jnp
from jax import random


@dataclass
class QuantizedResult:
    x: jnp.ndarray
    codes: jnp.ndarray
    bandwidth: jnp.ndarray  # bandwidth in kb/s used, per batch item.
    penalty: tp.Optional[jnp.ndarray] = None
    metrics: dict = field(default_factory=dict)


class BaseQuantizer(nn.Module):
    """Base class for quantizers."""

    @nn.compact
    def __call__(self, x: jnp.ndarray, frame_rate: int) -> QuantizedResult:
        """
        Given input tensor x, returns first the quantized (or approximately quantized)
        representation along with quantized codes, bandwidth, and any penalty term for the loss.
        Finally, this returns a dict of metrics to update logging etc.
        Frame rate must be passed so that the bandwidth is properly computed.
        """
        raise NotImplementedError()

    def encode(self, x: jnp.ndarray) -> jnp.ndarray:
        """Encode a given input tensor with the specified sample rate at the given bandwidth."""
        raise NotImplementedError()

    def decode(self, codes: jnp.ndarray) -> jnp.ndarray:
        """Decode the given codes to the quantized representation."""
        raise NotImplementedError()

    @property
    def total_codebooks(self):
        """Total number of codebooks."""
        raise NotImplementedError()

    @property
    def num_codebooks(self):
        """Number of active codebooks."""
        raise NotImplementedError()

    def set_num_codebooks(self, n: int):
        """Set the number of active codebooks."""
        raise NotImplementedError()


class DummyQuantizer(BaseQuantizer):
    """Fake quantizer that actually does not perform any quantization."""

    @nn.compact
    def __call__(self, x: jnp.ndarray, frame_rate: int):
        q = x.unsqueeze(1)
        return QuantizedResult(
            x, q, jnp.array(q.numel() * 32 * frame_rate / 1000 / len(x))
        )

    def encode(self, x: jnp.ndarray) -> jnp.ndarray:
        """Encode a given input tensor with the specified sample rate at the given bandwidth.
        In the case of the DummyQuantizer, the codes are actually identical
        to the input and resulting quantized representation as no quantization is done.
        """
        return x.unsqueeze(1)

    def decode(self, codes: jnp.ndarray) -> jnp.ndarray:
        """Decode the given codes to the quantized representation.
        In the case of the DummyQuantizer, the codes are actually identical
        to the input and resulting quantized representation as no quantization is done.
        """
        return codes.squeeze(1)

    @property
    def total_codebooks(self):
        """Total number of codebooks."""
        return 1

    @property
    def num_codebooks(self):
        """Total number of codebooks."""
        return self.total_codebooks

    def set_num_codebooks(self, n: int):
        """Set the number of active codebooks."""
        raise AttributeError(
            "Cannot override the number of codebooks for the dummy quantizer"
        )


def exists(val: tp.Optional[tp.Any]) -> bool:
    return val is not None


def default(val: tp.Any, d: tp.Any) -> tp.Any:
    return val if exists(val) else d


def l2norm(t):
    return F.normalize(t, p=2, dim=-1)


def ema_inplace(moving_avg, new, decay: float):
    moving_avg.data.mul_(decay).add_(new, alpha=(1 - decay))


def laplace_smoothing(x, n_categories: int, epsilon: float = 1e-5):
    return (x + epsilon) / (x.sum() + n_categories * epsilon)


def uniform_init(*shape: int):
    return nn.initializers.kaiming_uniform()


def sample_vectors(samples, num: int):
    num_samples = samples.shape[0]

    if num_samples >= num:
        indices = torch.randperm(num_samples)[:num]
    else:
        indices = torch.randint(0, num_samples, (num,))

    return samples[indices]


def kmeans(samples, num_clusters: int, num_iters: int = 10):
    dim, dtype = samples.shape[-1], samples.dtype

    means = sample_vectors(samples, num_clusters)

    for _ in range(num_iters):
        diffs = rearrange(samples, "n d -> n () d") - rearrange(means, "c d -> () c d")
        dists = -(diffs**2).sum(dim=-1)

        buckets = dists.max(dim=-1).indices
        bins = torch.bincount(buckets, minlength=num_clusters)
        zero_mask = bins == 0
        bins_min_clamped = bins.masked_fill(zero_mask, 1)

        new_means = buckets.new_zeros(num_clusters, dim, dtype=dtype)
        new_means.scatter_add_(0, repeat(buckets, "n -> n d", d=dim), samples)
        new_means = new_means / bins_min_clamped[..., None]

        means = jnp.where(zero_mask[..., None], means, new_means)

    return means, bins


def orthogonal_loss_fn(t):
    # eq (2) from https://arxiv.org/abs/2112.00384
    n = t.shape[0]
    normed_codes = l2norm(t)
    identity = jnp.eye(n)
    cosine_sim = einsum("i d, j d -> i j", normed_codes, normed_codes)
    return ((cosine_sim - identity) ** 2).sum() / (n**2)


class EuclideanCodebook(nn.Module):
    """Codebook with Euclidean distance.

    Args:
        dim (int): Dimension.
        codebook_size (int): Codebook size.
        kmeans_init (bool): Whether to use k-means to initialize the codebooks.
            If set to true, run the k-means algorithm on the first training batch and use
            the learned centroids as initialization.
        kmeans_iters (int): Number of iterations used for k-means algorithm at initialization.
        decay (float): Decay for exponential moving average over the codebooks.
        epsilon (float): Epsilon value for numerical stability.
        threshold_ema_dead_code (int): Threshold for dead code expiration. Replace any codes
            that have an exponential moving average cluster size less than the specified threshold with
            randomly selected vector from the current batch.
    """

    dim: int
    codebook_size: int
    kmeans_init: int = 0  # bool
    kmeans_iters: int = 10
    decay: float = 0.8
    epsilon: float = 1e-5
    threshold_ema_dead_code: int = 2

    def setup(self):
        init_fn: tp.Union[tp.Callable[..., jnp.ndarray], tp.Any] = (
            uniform_init() if not self.kmeans_init else nn.initializers.zeros
        )
        self.embed = self.param("embed", init_fn, (self.codebook_size, self.dim))
        self.embed_avg = self.param(
            "embed_avg", init_fn, (self.codebook_size, self.dim)
        )
        self.inited = jnp.array([not self.kmeans_init])
        self.cluster_size = jnp.zeros(self.codebook_size)
        super().setup()

    # @torch.jit.ignore
    def init_embed_(self, data):
        if self.inited:
            return

        embed, cluster_size = kmeans(data, self.codebook_size, self.kmeans_iters)
        self.embed.data.copy_(embed)
        self.embed_avg.data.copy_(embed.clone())
        self.cluster_size.data.copy_(cluster_size)
        self.inited.data.copy_(jnp.array([True]))
        # Make sure all buffers across workers are in sync after initialization
        flashy.distrib.broadcast_tensors(self.buffers())

    def replace_(self, samples, mask):
        modified_codebook = jnp.where(
            mask[..., None], sample_vectors(samples, self.codebook_size), self.embed
        )
        self.embed.data.copy_(modified_codebook)

    def expire_codes_(self, batch_samples):
        if self.threshold_ema_dead_code == 0:
            return

        expired_codes = self.cluster_size < self.threshold_ema_dead_code
        if not jnp.any(expired_codes):
            return

        batch_samples = rearrange(batch_samples, "... d -> (...) d")
        self.replace_(batch_samples, mask=expired_codes)
        flashy.distrib.broadcast_tensors(self.buffers())

    def preprocess(self, x):
        x = rearrange(x, "... d -> (...) d")
        return x

    def quantize(self, x):
        embed = self.embed.T
        dist = -(
            jnp.square(x).sum(1, keepdims=True)
            - 2 * x @ embed
            + jnp.square(embed).sum(0, keepdims=True)
        )
        embed_ind = jnp.argmax(dist, axis=-1)
        return embed_ind

    def postprocess_emb(self, embed_ind, shape):
        return jnp.reshape(embed_ind, shape[:-1])

    def dequantize(self, embed_ind):
        quantize = self.embed[embed_ind]
        return quantize

    def encode(self, x):
        shape = x.shape
        # pre-process
        x = self.preprocess(x)
        # quantize
        embed_ind = self.quantize(x)
        # post-process
        embed_ind = self.postprocess_emb(embed_ind, shape)
        return embed_ind

    def decode(self, embed_ind):
        quantize = self.dequantize(embed_ind)
        return quantize

    # todo: set train=True by default
    def __call__(self, x, train=False):
        shape, dtype = x.shape, x.dtype
        x = self.preprocess(x)
        # self.init_embed_(x)  # todo:

        embed_ind = self.quantize(x)
        # embed_onehot = F.one_hot(embed_ind, self.codebook_size).type(dtype)
        embed_onehot = onehot(labels=embed_ind, num_classes=self.codebook_size).astype(
            dtype
        )
        embed_ind = self.postprocess_emb(embed_ind, shape)
        quantize = self.dequantize(embed_ind)

        if train:
            # We do the expiry of code at that point as buffers are in sync
            # and all the workers will take the same decision.
            self.expire_codes_(x)
            ema_inplace(self.cluster_size, embed_onehot.sum(0), self.decay)
            embed_sum = x.t() @ embed_onehot
            ema_inplace(self.embed_avg, embed_sum.t(), self.decay)
            cluster_size = (
                laplace_smoothing(self.cluster_size, self.codebook_size, self.epsilon)
                * self.cluster_size.sum()
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(1)
            self.embed.data.copy_(embed_normalized)

        return quantize, embed_ind


class VectorQuantization(nn.Module):
    """Vector quantization implementation.
    Currently supports only Euclidean distance.

    Args:
        dim (int): Dimension
        codebook_size (int): Codebook size
        codebook_dim (int): Codebook dimension. If not defined, uses the specified dimension in dim.
        decay (float): Decay for exponential moving average over the codebooks.
        epsilon (float): Epsilon value for numerical stability.
        kmeans_init (bool): Whether to use kmeans to initialize the codebooks.
        kmeans_iters (int): Number of iterations used for kmeans initialization.
        threshold_ema_dead_code (int):
        channels_last (bool): Channels are the last dimension in the input tensors.
        commitment_weight (float): Weight for commitment loss.
        orthogonal_reg_weight (float): Orthogonal regularization weights.
        orthogonal_reg_active_codes_only (bool): Apply orthogonal regularization only on active codes.
        orthogonal_reg_max_codes (optional int): Maximum number of codes to consider
            for orthogonal regularization.
        threshold_ema_dead_code (int): Threshold for dead code expiration. Replace any codes
            that have an exponential moving average cluster size less than the specified threshold with
            randomly selected vector from the current batch.
    """

    dim: int
    codebook_size: int
    codebook_dim: tp.Optional[int] = None
    decay: float = 0.8
    epsilon: float = 1e-5
    kmeans_init: int = 0  # bool
    kmeans_iters: int = 10
    threshold_ema_dead_code: int = 2
    channels_last: int = 0  # bool
    commitment_weight: float = 1.0
    orthogonal_reg_weight: float = 0.0
    orthogonal_reg_active_codes_only: int = 0  # bool
    orthogonal_reg_max_codes: tp.Optional[int] = None

    def setup(self):
        _codebook_dim: int = default(self.codebook_dim, self.dim)

        requires_projection = _codebook_dim != self.dim
        self.project_in = (
            nn.Dense(_codebook_dim) if requires_projection else lambda x: x
        )
        self.project_out = nn.Dense(self.dim) if requires_projection else lambda x: x

        self._codebook = EuclideanCodebook(
            dim=_codebook_dim,
            codebook_size=self.codebook_size,
            kmeans_init=self.kmeans_init,
            kmeans_iters=self.kmeans_iters,
            decay=self.decay,
            epsilon=self.epsilon,
            threshold_ema_dead_code=self.threshold_ema_dead_code,
        )

    @property
    def codebook(self):
        return self._codebook.embed

    @property
    def inited(self):
        return self._codebook.inited

    def _preprocess(self, x):
        if not self.channels_last:
            x = rearrange(x, "b d n -> b n d")
        return x

    def _postprocess(self, quantize):
        if not self.channels_last:
            quantize = rearrange(quantize, "b n d -> b d n")
        return quantize

    def encode(self, x):
        x = self._preprocess(x)
        x = self.project_in(x)
        embed_in = self._codebook.encode(x)
        return embed_in

    def decode(self, embed_ind):
        quantize = self._codebook.decode(embed_ind)
        quantize = self.project_out(quantize)
        quantize = self._postprocess(quantize)
        return quantize

    # todo: set train=True by default
    def __call__(self, x, train=False):
        x = self._preprocess(x)

        x = self.project_in(x)
        quantize, embed_ind = self._codebook(x, train=train)

        if train:
            quantize = x + (quantize - x).detach()

        loss = jnp.array([0.0])

        if train:
            if self.commitment_weight > 0:
                commit_loss = F.mse_loss(quantize.detach(), x)
                loss = loss + commit_loss * self.commitment_weight

            if self.orthogonal_reg_weight > 0:
                codebook = self.codebook

                if self.orthogonal_reg_active_codes_only:
                    # only calculate orthogonal loss for the activated codes for this batch
                    unique_code_ids = jnp.unique(embed_ind)
                    codebook = codebook[unique_code_ids]

                num_codes = codebook.shape[0]
                if (
                    exists(self.orthogonal_reg_max_codes)
                    and num_codes > self.orthogonal_reg_max_codes
                ):
                    rand_ids = random.choice(
                        self.make_rng("rng_stream"),
                        num_codes,
                        shape=(self.orthogonal_reg_max_codes,),
                        replace=False,
                    )
                    codebook = codebook[rand_ids]

                orthogonal_reg_loss = orthogonal_loss_fn(codebook)
                loss = loss + orthogonal_reg_loss * self.orthogonal_reg_weight

        quantize = self.project_out(quantize)
        quantize = self._postprocess(quantize)

        return quantize, embed_ind, loss


class ResidualVectorQuantization(nn.Module):
    """Residual vector quantization implementation.

    Follows Algorithm 1. in https://arxiv.org/pdf/2107.03312.pdf
    """

    num_quantizers: int
    vector_quantization: tp.Callable

    def setup(self):
        self.layers = [self.vector_quantization() for _ in range(self.num_quantizers)]
        super().__init__()

    # todo: set train=True by default
    @nn.compact
    def __call__(self, x, n_q: tp.Optional[int] = None, train=False):
        quantized_out = 0.0
        residual = x.transpose(0, 2, 1)

        all_losses = []
        all_indices = []

        n_q = n_q or len(self.layers)

        for i, layer in enumerate(self.layers[:n_q]):
            quantized, indices, loss = layer(residual, train=train)
            quantized = jax.lax.stop_gradient(quantized)
            residual = residual - quantized
            quantized_out = quantized_out + quantized
            all_indices.append(indices)
            all_losses.append(loss)

        if train:
            # Solving subtle bug with STE and RVQ: https://github.com/facebookresearch/encodec/issues/25
            # quantized_out = x + (quantized_out - x).detach()
            quantized_out = x + jax.lax.stop_gradient(quantized_out - x)

        out_losses, out_indices = map(jnp.stack, (all_losses, all_indices))
        return quantized_out, out_indices, out_losses

    def encode(self, x: jnp.ndarray, n_q: tp.Optional[int] = None) -> jnp.ndarray:
        residual = x
        all_indices = []
        n_q = n_q or len(self.layers)
        for layer in self.layers[:n_q]:
            indices = layer.encode(residual)
            quantized = layer.decode(indices)
            residual = residual - quantized
            all_indices.append(indices)
        out_indices = jnp.stack(all_indices)
        return out_indices

    def decode(self, q_indices: jnp.ndarray) -> jnp.ndarray:
        quantized_out = jnp.array(0.0)
        for i, indices in enumerate(q_indices):
            layer = self.layers[i]
            quantized = layer.decode(indices)
            quantized_out = quantized_out + quantized
        return quantized_out


class ResidualVectorQuantizer(BaseQuantizer):
    """Residual Vector Quantizer.

    Args:
        dimension (int): Dimension of the codebooks.
        n_q (int): Number of residual vector quantizers used.
        q_dropout (bool): Random quantizer drop out at train time.
        bins (int): Codebook size.
        decay (float): Decay for exponential moving average over the codebooks.
        kmeans_init (bool): Whether to use kmeans to initialize the codebooks.
        kmeans_iters (int): Number of iterations used for kmeans initialization.
        threshold_ema_dead_code (int): Threshold for dead code expiration. Replace any codes
            that have an exponential moving average cluster size less than the specified threshold with
            randomly selected vector from the current batch.
        orthogonal_reg_weight (float): Orthogonal regularization weights.
        orthogonal_reg_active_codes_only (bool): Apply orthogonal regularization only on active codes.
        orthogonal_reg_max_codes (optional int): Maximum number of codes to consider.
            for orthogonal regularization.
    """

    dimension: int = 256
    n_q: int = 8
    q_dropout: int = 0  # bool
    bins: int = 1024
    decay: float = 0.99
    kmeans_init: int = 1  # bool
    kmeans_iters: int = 10
    threshold_ema_dead_code: int = 2
    orthogonal_reg_weight: float = 0.0
    orthogonal_reg_active_codes_only: int = 0  # bool
    orthogonal_reg_max_codes: tp.Optional[int] = None

    def setup(self):
        vector_quantization = lambda: VectorQuantization(
            dim=self.dimension,
            codebook_size=self.bins,
            decay=self.decay,
            kmeans_init=self.kmeans_init,
            kmeans_iters=self.kmeans_iters,
            threshold_ema_dead_code=self.threshold_ema_dead_code,
            orthogonal_reg_weight=self.orthogonal_reg_weight,
            orthogonal_reg_active_codes_only=self.orthogonal_reg_active_codes_only,
            orthogonal_reg_max_codes=self.orthogonal_reg_max_codes,
            channels_last=False,
        )
        self.vq = ResidualVectorQuantization(
            num_quantizers=self.n_q, vector_quantization=vector_quantization
        )

    # todo: set train=True for default
    def __call__(self, x: jnp.ndarray, frame_rate: int, train=False):
        n_q = self.n_q
        if train and self.q_dropout:
            n_q = random.randint(
                self.make_rng("rng_stream"), shape=(1,), minval=1, maxval=self.n_q + 1
            )
        bw_per_q = jnp.log2(self.bins) * frame_rate / 1000
        quantized, codes, commit_loss = self.vq(x, n_q=n_q, train=train)
        codes = codes.transpose(1, 0, 2)
        # codes is [B, K, T], with T frames, K nb of codebooks.
        bw = jnp.array(n_q * bw_per_q)
        return QuantizedResult(quantized, codes, bw, penalty=jnp.mean(commit_loss))

    def encode(self, x: jnp.ndarray) -> jnp.ndarray:
        """Encode a given input tensor with the specified frame rate at the given bandwidth.
        The RVQ encode method sets the appropriate number of quantizer to use
        and returns indices for each quantizer.
        """
        n_q = self.n_q
        codes = self.vq.encode(x, n_q=n_q)
        codes = codes.transpose(0, 1)
        # codes is [B, K, T], with T frames, K nb of codebooks.
        return codes

    def decode(self, codes: jnp.ndarray) -> jnp.ndarray:
        """Decode the given codes to the quantized representation."""
        # codes is [B, K, T], with T frames, K nb of codebooks, vq.decode expects [K, B, T].
        codes = codes.transpose(0, 1)
        quantized = self.vq.decode(codes)
        return quantized

    @property
    def total_codebooks(self):
        return self.n_q

    @property
    def num_codebooks(self):
        return self.n_q

    def set_num_codebooks(self, n: int):
        assert n > 0 and n <= self.n_q
        self.n_q = n
