"""Qwen 3.5 model definition."""
from typing import Callable, Dict, Optional, Tuple, Union
import litert_torch.generative.layers.model_config as cfg
from litert_torch.generative.layers import builder, gated_deltanet
from litert_torch.generative.layers import kv_cache as kv_utils
from litert_torch.generative.layers import attention
from litert_torch.generative.layers import sdpa_with_kv_update
from litert_torch.generative.utilities import loader as loading_utils
import litert_torch.generative.layers.attention_utils as attn_utils
import litert_torch.generative.layers.rotary_position_embedding as rotary_pos_emb
from litert_torch.generative.utilities import export_config as export_cfg
import torch
from torch import nn
from safetensors import safe_open
import glob
import os

# Layer pattern: 3 linear_attention + 1 full_attention, repeated 6 times = 24 layers
LAYER_TYPES = (["linear_attention"] * 3 + ["full_attention"]) * 6


class GatedCausalSelfAttention(nn.Module):
  """Causal self-attention with output gating (Qwen3.5 Gated Attention).

  In HF Qwen3.5, q_proj outputs 2x size: [query, gate].
  The gate is applied as: output = attn_output * sigmoid(gate)
  """

  def __init__(self, dim, config, enable_hlfb=True):
    super().__init__()
    self.config = config
    self.dim = dim
    self.enable_hlfb = enable_hlfb
    self.head_dim = config.head_dim
    self.num_heads = config.num_heads
    self.num_query_groups = config.num_query_groups

    q_out_dim = config.num_heads * config.head_dim
    kv_out_dim = config.num_query_groups * config.head_dim

    # Separate projections to match HF checkpoint structure
    # q_proj outputs 2x: query + gate
    self.q_proj = nn.Linear(dim, q_out_dim * 2, bias=config.qkv_use_bias)
    self.k_proj = nn.Linear(dim, kv_out_dim, bias=config.qkv_use_bias)
    self.v_proj = nn.Linear(dim, kv_out_dim, bias=config.qkv_use_bias)
    self.output_projection = nn.Linear(
        q_out_dim, dim, bias=config.output_proj_use_bias
    )

    # QK norms
    self.q_norm = nn.RMSNorm(config.head_dim, eps=1e-6)
    self.k_norm = nn.RMSNorm(config.head_dim, eps=1e-6)

  def forward(self, x, rope=None, mask=None, input_pos=None, kv_cache=None):
    B, T, _ = x.size()

    # Query projection outputs [query, gate]
    q_and_gate = self.q_proj(x)
    q_and_gate = q_and_gate.view(B, T, self.num_heads, self.head_dim * 2)
    q, gate = q_and_gate.chunk(2, dim=-1)
    # gate: [B, T, num_heads, head_dim] -> reshape to [B, T, num_heads * head_dim]
    gate = gate.reshape(B, T, -1)

    k = self.k_proj(x)
    v = self.v_proj(x)

    q = q.view(B, T, self.num_heads, self.head_dim)
    k = k.view(B, T, self.num_query_groups, self.head_dim)
    v = v.view(B, T, self.num_query_groups, self.head_dim)

    # Apply QK norms
    q = self.q_norm(q)
    k = self.k_norm(k)

    # Apply RoPE
    if rope is not None:
      cos, sin = rope
      q, k = rotary_pos_emb.apply_rope_inline(q, k, cos, sin)

    # SDPA with KV cache update
    sdpa_out, kv_cache = sdpa_with_kv_update.sdpa_with_kv_update(
        q, k, v, kv_cache, input_pos, mask, self.config, self.enable_hlfb
    )
    # sdpa_out: [B, T, num_heads * head_dim]

    # Apply output gate: attn_output * sigmoid(gate)
    sdpa_out = sdpa_out * torch.sigmoid(gate)

    # Output projection
    y = self.output_projection(sdpa_out)
    return y if kv_cache is None else (y, kv_cache)


class HybridTransformerBlock(nn.Module):
  """Transformer block supporting both GatedDeltaNet and GatedCausalSelfAttention."""

  def __init__(self, config, model_config):
    super().__init__()
    self.pre_atten_norm = builder.build_norm(
        model_config.embedding_dim, config.pre_attention_norm_config
    )
    self.post_atten_norm = builder.build_norm(
        model_config.embedding_dim, config.post_attention_norm_config
    )
    self.ff = builder.build_ff(model_config.embedding_dim, config.ff_config)
    self.config = config
    self.is_linear_attention = config.gated_deltanet_config is not None

    if self.is_linear_attention:
      self.atten_func = gated_deltanet.GatedDeltaNetAttention(
          model_config.embedding_dim, config.gated_deltanet_config
      )
    else:
      self.atten_func = GatedCausalSelfAttention(
          model_config.embedding_dim, config.attn_config, model_config.enable_hlfb
      )

  def forward(self, x, rope=None, mask=None, input_pos=None, kv_cache=None):
    x_norm = self.pre_atten_norm(x)

    if self.is_linear_attention:
      attn_out = self.atten_func(x_norm)
      kv = None
    else:
      res = self.atten_func(x_norm, rope, mask, input_pos, kv_cache)
      if kv_cache is not None:
        attn_out, kv = res
      else:
        attn_out, kv = res, None

    x = x + attn_out
    output = x + self.ff(self.post_atten_norm(x))

    if kv is not None:
      return output, kv
    return output


def get_model_config(model_size="0.8b"):
  """Get model config for the specified Qwen3.5 variant."""
  norm = cfg.NormalizationConfig(
      type=cfg.NormalizationType.RMS_NORM, epsilon=1e-06
  )

  if model_size == "0.8b":
    hidden_size = 1024
    intermediate_size = 3584
    num_attention_heads = 8
    num_key_value_heads = 2
    head_dim = 256
    lin_num_heads = 16
    lin_head_dim = 128
  elif model_size == "2b":
    hidden_size = 2048
    intermediate_size = 6144
    num_attention_heads = 8
    num_key_value_heads = 2
    head_dim = 256
    lin_num_heads = 16
    lin_head_dim = 128
  else:
    raise ValueError(f"Unsupported model size: {model_size}")

  dn = cfg.GatedDeltaNetConfig(
      num_qk_heads=lin_num_heads, num_v_heads=lin_num_heads,
      qk_head_dim=lin_head_dim, v_head_dim=lin_head_dim,
      conv_kernel_dim=4, use_output_gate=True, use_bias=False,
  )
  full_attn = cfg.AttentionConfig(
      num_heads=num_attention_heads, head_dim=head_dim,
      num_query_groups=num_key_value_heads,
      rotary_base=10000000, rotary_percentage=0.25,
      qkv_use_bias=False, qkv_transpose_before_split=True,
      attn_type=cfg.AttentionType.GLOBAL,
  )
  lin_attn = cfg.AttentionConfig(
      num_heads=lin_num_heads, head_dim=lin_head_dim,
      num_query_groups=lin_num_heads,
      rotary_percentage=0.0, attn_type=cfg.AttentionType.LINEAR_ATTENTION,
  )
  ff = cfg.FeedForwardConfig(
      type=cfg.FeedForwardType.GATED,
      activation=cfg.ActivationConfig(cfg.ActivationType.SILU),
      intermediate_size=intermediate_size,
  )
  blocks = [
      cfg.TransformerBlockConfig(
          attn_config=lin_attn if t == "linear_attention" else full_attn,
          ff_config=ff,
          pre_attention_norm_config=norm,
          post_attention_norm_config=norm,
          gated_deltanet_config=dn if t == "linear_attention" else None,
      )
      for t in LAYER_TYPES
  ]
  return cfg.ModelConfig(
      vocab_size=248320, num_layers=24, max_seq_len=2048,
      embedding_dim=hidden_size, block_configs=blocks,
      final_norm_config=norm, lm_head_share_weight_with_embedding=True,
  )


# Keep backward compat alias
def get_2b_model_config():
  return get_model_config("2b")


class Qwen3_5(nn.Module):
  def __init__(self, config, mask_cache_size=0):
    super().__init__()
    self.tok_embedding = nn.Embedding(
        config.vocab_size, config.embedding_dim, padding_idx=0
    )
    self.lm_head = nn.Linear(config.embedding_dim, config.vocab_size, bias=False)
    if config.lm_head_share_weight_with_embedding:
      self.lm_head.weight = self.tok_embedding.weight
    self.transformer_blocks = nn.ModuleList([
        HybridTransformerBlock(config.block_config(i), config)
        for i in range(config.num_layers)
    ])
    self.final_norm = builder.build_norm(
        config.embedding_dim, config.final_norm_config
    )
    self.config = config
    self.is_linear = [
        config.block_config(i).gated_deltanet_config is not None
        for i in range(config.num_layers)
    ]
    self.mask_cache = (
        attn_utils.build_causal_mask_cache(mask_cache_size)
        if mask_cache_size > 0 else None
    )

  @torch.inference_mode
  def forward(self, tokens, input_pos, kv_cache, mask=None, export_config=None):
    x = self.tok_embedding(tokens)
    full_idx = next(i for i, lin in enumerate(self.is_linear) if not lin)
    ac = self.config.block_config(full_idx).attn_config
    rope = self.config.build_rope(
        input_pos, int(ac.rotary_percentage * ac.head_dim), ac.rotary_base
    )
    if mask is None and self.mask_cache is not None and kv_cache is not None:
      mask = self.mask_cache.index_select(2, input_pos)[
          :, :, :, :kv_cache.get_max_seq_len()
      ]

    # KVCache.from_model_config creates entries for ALL layers (including
    # linear attention layers). We must index by the overall layer index,
    # not by a separate counter for full-attention layers only.
    updated_kv = []
    for i, block in enumerate(self.transformer_blocks):
      if self.is_linear[i]:
        x = block(x)
        if kv_cache is not None:
          # Pass through the existing KV cache entry unchanged
          updated_kv.append(kv_cache.caches[i])
      else:
        kv_entry = kv_cache.caches[i] if kv_cache else None
        x, kv_entry = block(x, rope, mask, input_pos, kv_entry)
        if kv_entry:
          updated_kv.append(kv_entry)
        else:
          updated_kv.append(kv_cache.caches[i] if kv_cache else None)

    new_kv = kv_utils.KVCache(tuple(updated_kv))
    if export_config is not None:
      if (
          torch.numel(input_pos) > 1
          and not export_config.output_logits_on_prefill
      ):
        return {"kv_cache": new_kv}
    return {"logits": self.lm_head(self.final_norm(x)), "kv_cache": new_kv}


def _load_safetensors(full_path):
  """Load all safetensors from a directory."""
  pattern = (
      os.path.join(full_path, "*.safetensors")
      if os.path.isdir(full_path) else full_path
  )
  tensors = {}
  for file in glob.glob(pattern):
    with safe_open(file, framework="pt") as fp:
      for k in fp.keys():
        tensors[k] = fp.get_tensor(k)
  return tensors


def _load_checkpoint_custom(model, checkpoint_path):
  """Custom loader mapping HF Qwen3.5 checkpoint keys to our model.

  HF checkpoint key structure:
    model.language_model.embed_tokens.weight
    model.language_model.norm.weight
    model.language_model.layers.{i}.input_layernorm.weight
    model.language_model.layers.{i}.post_attention_layernorm.weight
    model.language_model.layers.{i}.mlp.{gate,up,down}_proj.weight

  Linear attention layers (linear_attn.*):
    in_proj_qkv.weight  -> fused [q_dim+k_dim+v_dim, hidden]
    in_proj_z.weight     -> output gate [v_dim, hidden]
    in_proj_a.weight     -> [num_v_heads, hidden]
    in_proj_b.weight     -> [num_v_heads, hidden]
    conv1d.weight        -> [q_dim+k_dim+v_dim, 1, kernel_size]
    conv1d.bias          -> [q_dim+k_dim+v_dim]
    A_log                -> [num_qk_heads]
    dt_bias              -> [num_qk_heads]
    norm.weight          -> [v_dim]
    out_proj.weight      -> [hidden, v_dim]

  Full attention layers (self_attn.*):
    q_proj.weight  -> [num_heads * head_dim * 2, hidden] (query + gate fused)
    k_proj.weight  -> [num_kv_heads * head_dim, hidden]
    v_proj.weight  -> [num_kv_heads * head_dim, hidden]
    o_proj.weight  -> [hidden, num_heads * head_dim]
    q_norm.weight  -> [head_dim]
    k_norm.weight  -> [head_dim]
  """
  state = _load_safetensors(checkpoint_path)
  converted = {}

  # Embedding (also used as lm_head via weight tying)
  converted["tok_embedding.weight"] = state.pop(
      "model.language_model.embed_tokens.weight"
  )
  # Final norm
  converted["final_norm.weight"] = state.pop(
      "model.language_model.norm.weight"
  )

  for i in range(model.config.num_layers):
    prefix = f"transformer_blocks.{i}"
    hf = f"model.language_model.layers.{i}"

    # Layer norms
    converted[f"{prefix}.pre_atten_norm.weight"] = state.pop(
        f"{hf}.input_layernorm.weight"
    )
    converted[f"{prefix}.post_atten_norm.weight"] = state.pop(
        f"{hf}.post_attention_layernorm.weight"
    )

    # Feed-forward (gated: w1=gate, w3=up, w2=down)
    converted[f"{prefix}.ff.w3.weight"] = state.pop(
        f"{hf}.mlp.up_proj.weight"
    )
    converted[f"{prefix}.ff.w2.weight"] = state.pop(
        f"{hf}.mlp.down_proj.weight"
    )
    converted[f"{prefix}.ff.w1.weight"] = state.pop(
        f"{hf}.mlp.gate_proj.weight"
    )

    if model.is_linear[i]:
      # === Linear attention (GatedDeltaNet) ===
      la = f"{hf}.linear_attn"
      dn_cfg = model.config.block_config(i).gated_deltanet_config
      q_dim = dn_cfg.num_qk_heads * dn_cfg.qk_head_dim
      k_dim = dn_cfg.num_qk_heads * dn_cfg.qk_head_dim
      v_dim = dn_cfg.num_v_heads * dn_cfg.v_head_dim

      # Split fused QKV
      fused_qkv = state.pop(f"{la}.in_proj_qkv.weight")
      q_w, k_w, v_w = torch.split(fused_qkv, [q_dim, k_dim, v_dim], dim=0)
      converted[f"{prefix}.atten_func.q_proj.weight"] = q_w
      converted[f"{prefix}.atten_func.k_proj.weight"] = k_w
      converted[f"{prefix}.atten_func.v_proj.weight"] = v_w

      # Output projection
      converted[f"{prefix}.atten_func.o_proj.weight"] = state.pop(
          f"{la}.out_proj.weight"
      )

      # Split conv1d for separate q/k/v convs (weight AND bias)
      conv_weight = state.pop(f"{la}.conv1d.weight")
      cq, ck, cv = torch.split(conv_weight, [q_dim, k_dim, v_dim], dim=0)
      converted[f"{prefix}.atten_func.q_conv.weight"] = cq
      converted[f"{prefix}.atten_func.k_conv.weight"] = ck
      converted[f"{prefix}.atten_func.v_conv.weight"] = cv

      # Conv1d bias
      conv_bias_key = f"{la}.conv1d.bias"
      if conv_bias_key in state:
        conv_bias = state.pop(conv_bias_key)
        bq, bk, bv = torch.split(conv_bias, [q_dim, k_dim, v_dim], dim=0)
        converted[f"{prefix}.atten_func.q_conv.bias"] = bq
        converted[f"{prefix}.atten_func.k_conv.bias"] = bk
        converted[f"{prefix}.atten_func.v_conv.bias"] = bv

      # Decay and beta
      converted[f"{prefix}.atten_func.A_log"] = state.pop(f"{la}.A_log")
      converted[f"{prefix}.atten_func.beta"] = state.pop(f"{la}.dt_bias")

      # Output gate
      if f"{la}.in_proj_z.weight" in state:
        converted[f"{prefix}.atten_func.output_gate.weight"] = state.pop(
            f"{la}.in_proj_z.weight"
        )

      # Pop auxiliary keys not directly used in our simplified model
      for key_suffix in ["norm.weight", "in_proj_a.weight", "in_proj_b.weight"]:
        state.pop(f"{la}.{key_suffix}", None)

    else:
      # === Full attention (GatedCausalSelfAttention) ===
      sa = f"{hf}.self_attn"

      # q_proj: contains [query, gate] fused → keep as-is
      converted[f"{prefix}.atten_func.q_proj.weight"] = state.pop(
          f"{sa}.q_proj.weight"
      )
      converted[f"{prefix}.atten_func.k_proj.weight"] = state.pop(
          f"{sa}.k_proj.weight"
      )
      converted[f"{prefix}.atten_func.v_proj.weight"] = state.pop(
          f"{sa}.v_proj.weight"
      )
      converted[f"{prefix}.atten_func.output_projection.weight"] = state.pop(
          f"{sa}.o_proj.weight"
      )

      # QK norms
      if f"{sa}.q_norm.weight" in state:
        converted[f"{prefix}.atten_func.q_norm.weight"] = state.pop(
            f"{sa}.q_norm.weight"
        )
      if f"{sa}.k_norm.weight" in state:
        converted[f"{prefix}.atten_func.k_norm.weight"] = state.pop(
            f"{sa}.k_norm.weight"
        )

  # Log remaining keys
  remaining_lm = [k for k in state if k.startswith("model.language_model.")]
  if remaining_lm:
    print(f"WARNING: {len(remaining_lm)} unmapped language_model keys:")
    for k in remaining_lm[:10]:
      print(f"  {k}: {state[k].shape}")

  other = [k for k in state if not k.startswith("model.language_model.")]
  if other:
    print(f"INFO: Skipping {len(other)} non-language-model keys (visual, mtp)")

  return model.load_state_dict(converted, strict=False)


def build_model(checkpoint_path, model_size="0.8b", custom_loader=None,
                mask_cache_size=0):
  """Build a Qwen3.5 model with the specified size."""
  config = get_model_config(model_size)
  model = Qwen3_5(config, mask_cache_size)
  if checkpoint_path:
    missing, unexpected = _load_checkpoint_custom(model, checkpoint_path)
    if missing:
      print(f"Missing keys ({len(missing)}):")
      for k in missing[:20]:
        print(f"  {k}")
    if unexpected:
      print(f"Unexpected keys ({len(unexpected)}):")
      for k in unexpected[:20]:
        print(f"  {k}")
  return model.eval()


# Backward-compatible aliases
def build_2b_model(checkpoint_path, custom_loader=None, mask_cache_size=0):
  return build_model(checkpoint_path, "2b", custom_loader, mask_cache_size)


def build_08b_model(checkpoint_path, custom_loader=None, mask_cache_size=0):
  return build_model(checkpoint_path, "0.8b", custom_loader, mask_cache_size)
