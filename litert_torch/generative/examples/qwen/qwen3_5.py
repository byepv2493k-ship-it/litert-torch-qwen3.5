"""Qwen 3.5 model definition."""
from typing import Callable, Dict, Optional
import litert_torch.generative.layers.model_config as cfg
from litert_torch.generative.layers import builder, gated_deltanet
from litert_torch.generative.layers import kv_cache as kv_utils
from litert_torch.generative.utilities import loader as loading_utils
import litert_torch.generative.layers.attention_utils as attn_utils
from litert_torch.generative.utilities import export_config as export_cfg
import torch
from torch import nn

LAYER_TYPES = (["linear_attention"] * 3 + ["full_attention"]) * 6

TENSOR_NAMES = loading_utils.ModelLoader.TensorNames(
    ff_up_proj="model.layers.{}.mlp.up_proj",
    ff_down_proj="model.layers.{}.mlp.down_proj",
    ff_gate_proj="model.layers.{}.mlp.gate_proj",
    attn_query_proj="model.layers.{}.self_attn.q_proj",
    attn_key_proj="model.layers.{}.self_attn.k_proj",
    attn_value_proj="model.layers.{}.self_attn.v_proj",
    attn_output_proj="model.layers.{}.self_attn.o_proj",
    pre_attn_norm="model.layers.{}.input_layernorm",
    post_attn_norm="model.layers.{}.post_attention_layernorm",
    embedding="model.embed_tokens",
    final_norm="model.norm",
)

def get_2b_model_config():
  norm = cfg.NormalizationConfig(type=cfg.NormalizationType.RMS_NORM, epsilon=1e-06)
  dn = cfg.GatedDeltaNetConfig(num_qk_heads=16, num_v_heads=16, qk_head_dim=128, v_head_dim=128, conv_kernel_dim=4)
  full_attn = cfg.AttentionConfig(num_heads=8, head_dim=256, num_query_groups=2, rotary_base=10000000, rotary_percentage=0.25, qkv_use_bias=False, qkv_transpose_before_split=True, attn_type=cfg.AttentionType.GLOBAL)
  lin_attn = cfg.AttentionConfig(num_heads=16, head_dim=128, num_query_groups=16, rotary_percentage=0.0, attn_type=cfg.AttentionType.LINEAR_ATTENTION)
  ff = cfg.FeedForwardConfig(type=cfg.FeedForwardType.GATED, activation=cfg.ActivationConfig(cfg.ActivationType.SILU), intermediate_size=6144)
  blocks = [cfg.TransformerBlockConfig(attn_config=lin_attn if t=="linear_attention" else full_attn, ff_config=ff, pre_attention_norm_config=norm, post_attention_norm_config=norm, gated_deltanet_config=dn if t=="linear_attention" else None) for t in LAYER_TYPES]
  return cfg.ModelConfig(vocab_size=248320, num_layers=24, max_seq_len=2048, embedding_dim=2048, block_configs=blocks, final_norm_config=norm, lm_head_share_weight_with_embedding=True)

class Qwen3_5(nn.Module):
  def __init__(self, config, mask_cache_size=0):
    super().__init__()
    self.tok_embedding = nn.Embedding(config.vocab_size, config.embedding_dim, padding_idx=0)
    self.lm_head = nn.Linear(config.embedding_dim, config.vocab_size, bias=False)
    if config.lm_head_share_weight_with_embedding:
      self.lm_head.weight.data = self.tok_embedding.weight.data
    self.transformer_blocks = nn.ModuleList([gated_deltanet.HybridTransformerBlock(config.block_config(i), config) for i in range(config.num_layers)])
    self.final_norm = builder.build_norm(config.embedding_dim, config.final_norm_config)
    self.config = config
    self.is_linear = [config.block_config(i).gated_deltanet_config is not None for i in range(config.num_layers)]
    self.mask_cache = attn_utils.build_causal_mask_cache(mask_cache_size) if mask_cache_size > 0 else None

  @torch.inference_mode
  def forward(self, tokens, input_pos, kv_cache, mask=None, export_config=None):
    x = self.tok_embedding(tokens)
    full_idx = next(i for i, lin in enumerate(self.is_linear) if not lin)
    ac = self.config.block_config(full_idx).attn_config
    rope = self.config.build_rope(input_pos, int(ac.rotary_percentage * ac.head_dim), ac.rotary_base)
    if mask is None and self.mask_cache is not None and kv_cache is not None:
      mask = self.mask_cache.index_select(2, input_pos)[:, :, :, :kv_cache.get_max_seq_len()]
    kv_idx, updated_kv = 0, []
    for i, block in enumerate(self.transformer_blocks):
      if self.is_linear[i]:
        x = block(x)
      else:
        kv_entry = kv_cache.caches[kv_idx] if kv_cache else None
        x, kv_entry = block(x, rope, mask, input_pos, kv_entry)
        if kv_entry: updated_kv.append(kv_entry)
        kv_idx += 1
    new_kv = kv_utils.KVCache(tuple(updated_kv))
    if export_config is not None:
      if torch.numel(input_pos) > 1 and not export_config.output_logits_on_prefill:
        return {"kv_cache": new_kv}
    return {"logits": self.lm_head(self.final_norm(x)), "kv_cache": new_kv}

def build_2b_model(checkpoint_path, custom_loader=None, mask_cache_size=0):
  config = get_2b_model_config()
  model = Qwen3_5(config, mask_cache_size)
  if checkpoint_path:
    loader = loading_utils.ModelLoader(checkpoint_path, TENSOR_NAMES, custom_loader)
    loader.load(model, strict=False)
  return model.eval()
