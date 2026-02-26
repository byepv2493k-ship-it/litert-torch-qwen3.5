# Copyright 2026 The LiteRT Torch Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Cache for LFM2."""

from typing import List, Tuple
from litert_torch.generative.export_hf.core import cache as cache_lib
from litert_torch.generative.export_hf.core import cache_base as cache_base_lib
from litert_torch.generative.export_hf.core import exportable_module_config
import torch
import torch.utils._pytree as pytree


class LiteRTLFM2CacheLayer(cache_lib.LiteRTLMCacheLayer):
  """Optimized Cache layer class for LFM2 integration."""

  def __init__(
      self,
      conv_state: torch.Tensor,
      key_cache: cache_lib.KeyCache | None = None,
      value_cache: cache_lib.ValueCache | None = None,
      batch_size: int = 1,
      k_ts_idx: int = 2,
      v_ts_idx: int = 3,
      **kwargs,
  ):
    dummy_key_cache = torch.zeros((1, 1, 1, 1))
    dummy_value_cache = torch.zeros((1, 1, 1, 1))
    super().__init__(
        dummy_key_cache,
        dummy_value_cache,
        batch_size,
        k_ts_idx,
        v_ts_idx,
        **kwargs,
    )
    self.conv_state = conv_state

  @classmethod
  def create_from_config(
      cls,
      model_config,
      layer_index,
      export_config: exportable_module_config.ExportableModuleConfig,
      **kwargs,
  ) -> "LiteRTLFM2CacheLayer":
    """Creates a KV cache from the model config."""
    assert model_config.layer_types[layer_index] == "conv"
    c_state_shape = (
        export_config.batch_size,
        model_config.hidden_size,
        model_config.conv_L_cache - 1,
    )
    c_state = torch.zeros(c_state_shape, dtype=torch.float32)
    return cls(
        c_state,
        batch_size=export_config.batch_size,
        **kwargs,
    )


@cache_base_lib.register_cache_implementation
class LiteRTLFM2Cache(cache_lib.LiteRTLMCache):
  """Optimized Cache class for LFM2 integration."""

  @classmethod
  def create_from_config(
      cls,
      model_config,
      export_config: exportable_module_config.ExportableModuleConfig,
      **kwargs,
  ) -> "LiteRTLFM2Cache":
    """Creates a KV cache from the model config."""
    num_layers = model_config.num_hidden_layers
    layers = []
    for layer_index in range(num_layers):
      if model_config.layer_types[layer_index] == "conv":
        layers.append(
            LiteRTLFM2CacheLayer.create_from_config(
                model_config,
                layer_index,
                export_config,
            )
        )
      else:
        layers.append(
            cache_lib.LiteRTLMCacheLayer.create_from_config(
                model_config,
                layer_index,
                export_config,
            )
        )
    return cls(layers)


def _flatten_kvc_t(
    kvc: LiteRTLFM2Cache,
) -> Tuple[
    List[torch.Tensor], Tuple[List[str], Tuple[int, int, int, int, List[bool]]]
]:
  """Flattens the cache into a list of tensors."""
  flattened = []
  flat_names = []
  num_layers = len(kvc.layers)
  layer_0 = kvc.layers[0]
  is_conv = []
  assert isinstance(layer_0, cache_base_lib.LiteRTLMCacheLayerMixin)
  batch_size = layer_0.get_batch_size()
  k_ts_idx = layer_0.get_k_ts_idx()
  v_ts_idx = layer_0.get_v_ts_idx()
  for i, layer in enumerate(kvc.layers):
    if isinstance(layer, LiteRTLFM2CacheLayer):
      is_conv.append(True)
      flattened.append(layer.conv_state)
      flat_names.append(f"c_{i}")
    else:
      is_conv.append(False)
      flattened.append(layer.keys)
      flat_names.append(f"k_{i}")
      flattened.append(layer.values)
      flat_names.append(f"v_{i}")
  return flattened, (
      flat_names,
      (batch_size, num_layers, k_ts_idx, v_ts_idx, is_conv),
  )


def _unflatten_kvc_t(
    values: List[torch.Tensor],
    context: Tuple[List[str], Tuple[int, int, int, int, List[bool]]],
) -> LiteRTLFM2Cache:
  """Unflattens the cache from a list of tensors."""
  flat_names = context[0]
  batch_size, num_layers, k_ts_idx, v_ts_idx, is_conv = context[1]
  layers = []
  for i in range(num_layers):
    if is_conv[i]:
      c_cache_idx = flat_names.index(f"c_{i}")
      layers.append(
          LiteRTLFM2CacheLayer(
              conv_state=values[c_cache_idx],
              batch_size=batch_size,
          )
      )
    else:
      k_cache_idx = flat_names.index(f"k_{i}")
      v_cache_idx = flat_names.index(f"v_{i}")
      layers.append(
          cache_lib.LiteRTLMCacheLayer(
              key_cache=values[k_cache_idx],
              value_cache=values[v_cache_idx],
              batch_size=batch_size,
              k_ts_idx=k_ts_idx,
              v_ts_idx=v_ts_idx,
          )
      )
  obj = LiteRTLFM2Cache(layers)
  return obj


def _flatten_kvc_t_with_keys(
    kvc: LiteRTLFM2Cache,
):
  flattened, (flat_names, _) = _flatten_kvc_t(kvc)
  return [
      (pytree.MappingKey(k), v) for k, v in zip(flat_names, flattened)
  ], flat_names


pytree.register_pytree_node(
    LiteRTLFM2Cache,
    _flatten_kvc_t,
    _unflatten_kvc_t,
    flatten_with_keys_fn=_flatten_kvc_t_with_keys,
    serialized_type_name="",
)
