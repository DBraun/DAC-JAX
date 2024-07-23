import dataclasses
from typing import Iterable, Optional, Tuple

from flax import linen as nn

import jax
from jax import random
from jax import numpy as jnp

from flax.linen import dtypes, module, transforms
from flax.typing import (
  Dtype,
  Axes,
)
from flax.linen.normalization import _l2_normalize, _canonicalize_axes

field = dataclasses.field
canonicalize_dtype = dtypes.canonicalize_dtype
compact = module.compact
Module = module.Module
merge_param = module.merge_param
map_variables = transforms.map_variables


class MyWeightNorm(nn.Module):

  layer_instance: nn.Module
  epsilon: float = 1e-12
  dtype: Optional[Dtype] = None
  param_dtype: Dtype = jnp.float32
  use_scale: bool = True
  feature_axes: Optional[Axes] = -1
  variable_filter: Optional[Iterable] = dataclasses.field(
    default_factory=lambda: {'kernel'}
  )

  @compact
  def __call__(self, *args, **kwargs):
    """Compute the l2-norm of the weights in ``self.layer_instance``
    and normalize the weights using this value before computing the
    ``__call__`` output.

    Args:
      *args: positional arguments to be passed into the call method of the
        underlying layer instance in ``self.layer_instance``.
      **kwargs: keyword arguments to be passed into the call method of the
        underlying layer instance in ``self.layer_instance``.

    Returns:
      Output of the layer using l2-normalized weights.
    """

    def layer_forward(layer_instance):
      return layer_instance(*args, **kwargs)

    return transforms.map_variables(
      layer_forward,
      trans_in_fn=lambda vs: jax.tree_util.tree_map_with_path(
        self._l2_normalize,
        vs,
      ),
      init=self.is_initializing(),
    )(self.layer_instance)

  def _l2_normalize(self, path, vs):
    """Compute the l2-norm and normalize the variables ``vs`` using this
    value. This is intended to be a helper function used in this Module's
    ``__call__`` method in conjunction with ``nn.transforms.map_variables``
    and ``jax.tree_util.tree_map_with_path``.

    Args:
      path: dict key path, used for naming the ``scale`` variable
      vs: variables to be l2-normalized
    """
    value = jnp.asarray(vs)
    str_path = (
      self.layer_instance.name
      + '/'
      + '/'.join((dict_key.key for dict_key in path[1:]))
    )
    if self.variable_filter:
      for variable_name in self.variable_filter:
        if variable_name in str_path:
          break
      else:
        return value

    if self.feature_axes is None:
      feature_axes = ()
      reduction_axes = tuple(i for i in range(value.ndim))
    else:
      feature_axes = _canonicalize_axes(value.ndim, self.feature_axes)
      reduction_axes = tuple(
        i for i in range(value.ndim) if i not in feature_axes
      )

    feature_shape = [1] * value.ndim
    reduced_feature_shape = []
    for ax in feature_axes:
      feature_shape[ax] = value.shape[ax]
      reduced_feature_shape.append(value.shape[ax])

    value_bar = _l2_normalize(value, axis=reduction_axes, eps=self.epsilon)

    def scale_init(key, _shape, dtype):
        # Initialize the weights of a Conv1d the way PyTorch would. Look at the "Variables" section and "weights".
        # https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
        kernel_size = value.shape[0]
        c_in = value.shape[1]
        out_features = value.shape[2]
        groups = 1
        conv_weights = random.uniform(key, (kernel_size, c_in, out_features), dtype, -1) * jnp.sqrt(groups/(c_in*kernel_size))
        scale = jnp.linalg.norm(conv_weights, axis=(0, 1))  # [out_features]
        return scale

    args = [vs]
    if self.use_scale:
      scale = self.param(
        str_path + '/scale',
        scale_init,
        reduced_feature_shape,
        self.param_dtype,
      ).reshape(feature_shape)
      value_bar *= scale
      args.append(scale)

    dtype = dtypes.canonicalize_dtype(*args, dtype=self.dtype)
    return jnp.asarray(value_bar, dtype)