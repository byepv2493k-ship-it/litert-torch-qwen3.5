"""Gated DeltaNet (linear attention) layer for Qwen3.5."""
from typing import Optional, Tuple, Union
from litert_torch.generative.layers import builder
from litert_torch.generative.layers import kv_cache as kv_utils
import litert_torch.generative.layers.model_config as cfg
import torch
from torch import nn
import torch.nn.functional as F


class GatedDeltaNetAttention(nn.Module):
  def __init__(self, dim, config, enable_hlfb=True):
    super().__init__()
    self.config, self.dim = config, dim
    self.num_qk_heads = config.num_qk_heads
    self.num_v_heads = config.num_v_heads
    self.qk_head_dim = config.qk_head_dim
    self.v_head_dim = config.v_head_dim
    self.conv_kernel_dim = config.conv_kernel_dim
    q_dim = self.num_qk_heads * self.qk_head_dim
    k_dim = self.num_qk_heads * self.qk_head_dim
    v_dim = self.num_v_heads * self.v_head_dim
    self.q_proj = nn.Linear(dim, q_dim, bias=config.use_bias)
    self.k_proj = nn.Linear(dim, k_dim, bias=config.use_bias)
    self.v_proj = nn.Linear(dim, v_dim, bias=config.use_bias)
    self.o_proj = nn.Linear(v_dim, dim, bias=config.use_bias)
    self.q_conv = nn.Conv1d(q_dim, q_dim, kernel_size=config.conv_kernel_dim, padding=config.conv_kernel_dim-1, groups=q_dim)
    self.k_conv = nn.Conv1d(k_dim, k_dim, kernel_size=config.conv_kernel_dim, padding=config.conv_kernel_dim-1, groups=k_dim)
    self.v_conv = nn.Conv1d(v_dim, v_dim, kernel_size=config.conv_kernel_dim, padding=config.conv_kernel_dim-1, groups=v_dim)
    self.beta = nn.Parameter(torch.zeros(self.num_qk_heads))
    self.A_log = nn.Parameter(torch.zeros(self.num_qk_heads))
    self.output_gate = nn.Linear(dim, v_dim, bias=config.use_bias) if config.use_output_gate else None

  def _causal_conv(self, x, conv):
    x = x.transpose(1, 2)
    x = conv(x)[:, :, :-(self.conv_kernel_dim - 1)]
    return x.transpose(1, 2)

  def _recurrent(self, q, k, v, state=None):
    B, T, H, D = q.shape
    decay = torch.sigmoid(self.A_log).view(1, H, 1, 1)
    beta = torch.sigmoid(self.beta).view(1, H, 1, 1)
    k = F.normalize(k, p=2, dim=-1)
    if state is None:
      state = torch.zeros(B, H, self.v_head_dim, D, dtype=q.dtype, device=q.device)
    outputs = []
    for t in range(T):
      q_t, k_t, v_t = q[:, t], k[:, t], v[:, t]
      state = decay * state + beta * (v_t.unsqueeze(-1) * k_t.unsqueeze(-2))
      outputs.append(torch.einsum("bhvk,bhk->bhv", state, q_t))
    return torch.stack(outputs, dim=1).reshape(B, T, -1), state

  def forward(self, x, rope=None, mask=None, input_pos=None, kv_cache=None, recurrent_state=None):
    B, T, _ = x.size()
    q = F.silu(self._causal_conv(self.q_proj(x), self.q_conv))
    k = F.silu(self._causal_conv(self.k_proj(x), self.k_conv))
    v = self._causal_conv(self.v_proj(x), self.v_conv)
    q = q.view(B, T, self.num_qk_heads, self.qk_head_dim)
    k = k.view(B, T, self.num_qk_heads, self.qk_head_dim)
    v = v.view(B, T, self.num_v_heads, self.v_head_dim)
    output, new_state = self._recurrent(q, k, v, recurrent_state)
    if self.output_gate is not None:
      output = output * torch.sigmoid(self.output_gate(x))
    y = self.o_proj(output)
    return y if recurrent_state is None else (y, new_state)


class HybridTransformerBlock(nn.Module):
  def __init__(self, config, model_config):
    super().__init__()
    self.pre_atten_norm = builder.build_norm(model_config.embedding_dim, config.pre_attention_norm_config)
    self.post_atten_norm = builder.build_norm(model_config.embedding_dim, config.post_attention_norm_config)
    self.ff = builder.build_ff(model_config.embedding_dim, config.ff_config)
    self.config = config
    self.is_linear_attention = config.gated_deltanet_config is not None
    if self.is_linear_attention:
      self.atten_func = GatedDeltaNetAttention(model_config.embedding_dim, config.gated_deltanet_config)
    else:
      from litert_torch.generative.layers import attention
      self.atten_func = attention.CausalSelfAttention(model_config.embedding_dim, config.attn_config, model_config.enable_hlfb)

  def forward(self, x, rope=None, mask=None, input_pos=None, kv_cache=None, recurrent_state=None):
    x_norm = self.pre_atten_norm(x)
    if self.is_linear_attention:
      res = self.atten_func(x_norm, recurrent_state=recurrent_state)
      if recurrent_state is not None:
        attn_out, new_state = res
      else:
        attn_out, new_state = res, None
    else:
      res = self.atten_func(x_norm, rope, mask, input_pos, kv_cache)
      if kv_cache is not None:
        attn_out, kv = res
      else:
        attn_out, kv = res, None
      new_state = None
    x = x + attn_out
    output = x + self.ff(self.post_atten_norm(x))
    if self.is_linear_attention and new_state is not None:
      return output, new_state
    if not self.is_linear_attention and kv_cache is not None:
      return output, kv
    return output
