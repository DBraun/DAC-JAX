from typing import Union, Tuple

from einops import rearrange
import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.random

from dac_jax.nn.layers import WNConv1d


def mse_loss(predictions: jnp.ndarray, targets: jnp.ndarray, reduction='mean') -> jnp.ndarray:
    errors = (predictions - targets) ** 2
    if reduction == 'none':
        return errors
    elif reduction == 'mean':
        return jnp.mean(errors)
    elif reduction == 'sum':
        return jnp.sum(errors)
    else:
        raise ValueError(f"Invalid reduction method: {reduction}")


def normalize(x, ord=2, axis=1, eps=1e-12):
    """Normalizes an array along a specified dimension.

    Args:
    x: A JAX array to normalize.
    ord: The order of the norm (default is 2, corresponding to L2-norm).
    axis: The dimension along which to normalize.
    eps: A small constant to avoid division by zero.

    Returns:
    A JAX array with normalized vectors.

    Reference:
    https://pytorch.org/docs/stable/generated/torch.nn.functional.normalize.html
    """
    denom = jnp.linalg.norm(x, ord=ord, axis=axis, keepdims=True)
    denom = jnp.maximum(eps, denom)
    return x / denom


class VectorQuantize(nn.Module):
    """
    Implementation of VQ similar to Karpathy's repo:
    https://github.com/karpathy/deep-vector-quantization
    Additionally uses following tricks from Improved VQGAN
    (https://arxiv.org/pdf/2110.04627.pdf):
        1. Factorized codes: Perform nearest neighbor lookup in low-dimensional space
            for improved codebook usage
        2. l2-normalized codes: Converts Euclidean distance to cosine similarity which
            improves training stability
    """

    input_dim: int
    codebook_size: int
    codebook_dim: int

    def setup(self):
        self.in_proj = WNConv1d(features=self.codebook_dim, kernel_size=(1,))
        self.out_proj = WNConv1d(features=self.input_dim, kernel_size=(1,))
        # PyTorch uses a normal distribution for weight initialization of Embeddings.
        self.codebook = nn.Embed(num_embeddings=self.codebook_size, features=self.codebook_dim,
                                 embedding_init=nn.initializers.normal(stddev=1))

    def __call__(self, z) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Quantized the input tensor using a fixed codebook and returns the corresponding codebook vectors

        Parameters
        ----------
        z : Tensor[B x T x D]

        Returns
        -------
        Tensor[B x T x D]
            Quantized continuous representation of input
        Tensor[1]
            Commitment loss to train encoder to predict vectors closer to codebook
            entries
        Tensor[1]
            Codebook loss to update the codebook
        Tensor[B x T]
            Codebook indices (quantized discrete representation of input)
        Tensor[B x T x D]
            Projected latents (continuous representation of input before quantization)
        """

        # Factorized codes (ViT-VQGAN) Project input into low-dimensional space
        z_e = self.in_proj(z)  # z_e : (B x T x D)
        z_q, indices = self.decode_latents(z_e)

        commitment_loss = mse_loss(z_e, jax.lax.stop_gradient(z_q), reduction='none').mean([1, 2])
        codebook_loss = mse_loss(z_q, jax.lax.stop_gradient(z_e), reduction='none').mean([1, 2])

        z_q = (
            z_e + jax.lax.stop_gradient(z_q - z_e)
        )  # noop in forward pass, straight-through gradient estimator in backward pass

        z_q = self.out_proj(z_q)

        return z_q, commitment_loss, codebook_loss, indices, z_e

    def embed_code(self, embed_id):
        return self.codebook(embed_id)

    def decode_code(self, embed_id):
        return self.embed_code(embed_id)

    def decode_latents(self, latents: jnp.ndarray):
        encodings = rearrange(latents, "b t d -> (b t) d", d=self.codebook_dim)
        codebook = self.codebook.embedding  # codebook: (N x D)
        # L2 normalize encodings and codebook (ViT-VQGAN)
        encodings = normalize(encodings)
        codebook = normalize(codebook)

        # Compute Euclidean distance with codebook
        dist = (
            jnp.square(encodings).sum(1, keepdims=True)
            - 2 * encodings @ codebook.transpose()
            + jnp.square(codebook).sum(1, keepdims=True).transpose()
        )
        indices = rearrange(jnp.argmax(-dist, axis=1), "(b t) -> b t", b=latents.shape[0])
        z_q = self.decode_code(indices)
        return z_q, indices


class ResidualVectorQuantize(nn.Module):
    """
    Introduced in SoundStream: An End-to-End Neural Audio Codec
    https://arxiv.org/abs/2107.03312
    """

    input_dim: int = 512
    n_codebooks: int = 9
    codebook_size: int = 1024
    codebook_dim: Union[int, list] = 8
    quantizer_dropout: float = 0.0

    def __post_init__(self) -> None:
        if isinstance(self.codebook_dim, int):
            self.codebook_dim = [self.codebook_dim for _ in range(self.n_codebooks)]
        super().__post_init__()

    def setup(self) -> None:

        self.quantizers = [VectorQuantize(self.input_dim, self.codebook_size, self.codebook_dim[i])
                           for i in range(self.n_codebooks)]

    def __call__(self, z, n_quantizers: int = None, train=True) \
            -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Quantized the input tensor using a fixed set of `n` codebooks and returns
        the corresponding codebook vectors
        Parameters
        ----------
        z : Tensor[B x T x D]
        n_quantizers : int, optional
            No. of quantizers to use
            (n_quantizers < self.n_codebooks ex: for quantizer dropout)
            Note: if `self.quantizer_dropout` is True, this argument is ignored
                when in training mode, and a random number of quantizers is used.
        Returns
        -------
        "z" : Tensor[B x T x D]
            Quantized continuous representation of input
        "codes" : Tensor[B x T x N]
            Codebook indices for each codebook
            (quantized discrete representation of input)
        "latents" : Tensor[B x T x N*D]
            Projected latents (continuous representation of input before quantization)
        "vq/commitment_loss" : Tensor[1]
            Commitment loss to train encoder to predict vectors closer to codebook
            entries
        "vq/codebook_loss" : Tensor[1]
            Codebook loss to update the codebook
        """
        z_q = 0
        residual = z
        commitment_loss = jnp.zeros(())
        codebook_loss = jnp.zeros(())

        codebook_indices = []
        latents = []

        if n_quantizers is None:
            n_quantizers = self.n_codebooks
        if train:
            n_quantizers = jnp.ones((z.shape[0],)) * self.n_codebooks + 1
            dropout = jax.random.randint(self.make_rng('rng_stream'),
                                         shape=(z.shape[0],), minval=1, maxval=self.n_codebooks+1)
            n_dropout = int(z.shape[0] * self.quantizer_dropout)
            n_quantizers = n_quantizers.at[:n_dropout].set(dropout[:n_dropout])

        # todo: this loop would possibly compile faster if jax.lax.scan were used
        for i, quantizer in enumerate(self.quantizers):
            if not train and i >= n_quantizers:
                break

            z_q_i, commitment_loss_i, codebook_loss_i, indices_i, z_e_i = quantizer(
                residual
            )

            # Create mask to apply quantizer dropout
            mask = (
                jnp.full((z.shape[0],), fill_value=i) < n_quantizers
            )
            z_q = z_q + z_q_i * mask[:, None, None]
            residual = residual - z_q_i

            # Sum losses
            commitment_loss = commitment_loss + (commitment_loss_i * mask).mean()
            codebook_loss = codebook_loss + (codebook_loss_i * mask).mean()

            codebook_indices.append(indices_i)
            latents.append(z_e_i)

        codes = jnp.stack(codebook_indices, axis=2)
        latents = jnp.concatenate(latents, axis=2)

        # normalize based on number of codebooks
        # commitment_loss = commitment_loss / self.n_codebooks
        # codebook_loss = codebook_loss / self.n_codebooks

        return z_q, codes, latents, commitment_loss, codebook_loss

    def from_codes(self, codes: jnp.ndarray):
        """Given the quantized codes, reconstruct the continuous representation
        Parameters
        ----------
        codes : Tensor[B x T x N]
            Quantized discrete representation of input
        Returns
        -------
        Tensor[B x T x D]
            Quantized continuous representation of input
        """
        z_q = 0.0
        z_p = []
        n_codebooks = codes.shape[-1]

        # todo: use jax.lax.scan for this loop
        for i in range(n_codebooks):
            z_p_i = self.quantizers[i].decode_code(codes[:, :, i])
            z_p.append(z_p_i)

            z_q_i = self.quantizers[i].out_proj(z_p_i)
            z_q = z_q + z_q_i

        return z_q, jnp.concatenate(z_p, axis=2), codes

    def from_latents(self, latents: jnp.ndarray):
        # todo: this function hasn't been tested/used yet.

        """Given the unquantized latents, reconstruct the
        continuous representation after quantization.

        Parameters
        ----------
        latents : Tensor[B x T x N]
            Continuous representation of input after projection

        Returns  # todo: make this return info correct
        -------
        Tensor[B x T x D]
            Quantized representation of full-projected space
        Tensor[B x T x D]
            Quantized representation of latent space
        """
        z_q = 0
        z_p = []
        codes = []
        dims = jnp.cumsum([0] + [q.codebook_dim for q in self.quantizers])

        n_codebooks = jnp.where(dims <= latents.shape[2])[0].max(axis=0, keepdims=True)  # todo: check

        # todo: use jax.lax.scan for this loop
        for i in range(n_codebooks):
            j, k = dims[i], dims[i + 1]
            z_p_i, codes_i = self.quantizers[i].decode_latents(latents[:, :, j:k])
            z_p.append(z_p_i)
            codes.append(codes_i)

            z_q_i = self.quantizers[i].out_proj(z_p_i)
            z_q = z_q + z_q_i

        return z_q, jnp.concatenate(z_p, axis=2), jnp.stack(codes, axis=2)


if __name__ == "__main__":
    rvq = ResidualVectorQuantize(quantizer_dropout=True)
    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    x = jax.random.normal(key=subkey, shape=(16, 80, 512))

    key, subkey = jax.random.split(key)
    params = rvq.init({'params': subkey, 'rng_stream': jax.random.key(4)}, x)['params']
    z_q, codes, latents, commitment_loss, codebook_loss = rvq.apply({'params': params}, x,
                                                                    rngs={'rng_stream': jax.random.key(4)})
    print(latents.shape)
