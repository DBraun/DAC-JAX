"""Guards against NaN gradients from operations that are finite in the forward pass.

See https://docs.jax.dev/en/latest/faq.html#gradients-contain-nan-where-using-where
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dac_jax.nn.quantize import normalize


@pytest.mark.parametrize("shape", [(1, 3), (4, 8), (2, 16)])
def test_normalize_gradient_is_finite_at_zero(shape):
    """A codebook entry that goes unused can decay to exactly zero.

    `normalize` clamps the denominator, so the forward pass is finite either way;
    the gradient is what the clamp has to be placed correctly to protect.
    """

    def loss(x):
        return jnp.sum(normalize(x))

    grad = jax.grad(loss)(jnp.zeros(shape))
    assert not jnp.isnan(grad).any()
    assert jnp.isfinite(grad).all()


def test_normalize_matches_torch():
    """The gradient fix must not move the forward values."""
    torch = pytest.importorskip("torch")
    import torch.nn.functional as F

    rng = np.random.default_rng(0)
    for shape in [(4, 8), (2, 16, 32), (8, 1024)]:
        x = rng.standard_normal(shape).astype(np.float32)
        ours = np.asarray(normalize(jnp.asarray(x)))
        theirs = F.normalize(torch.from_numpy(x), p=2, dim=1, eps=1e-12).numpy()
        assert np.allclose(ours, theirs, atol=1e-6)
