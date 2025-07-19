# student_only_attention.py
import math, torch, torch.nn as nn, torch.nn.functional as F
from einops import rearrange
from typing import Optional, Tuple
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3RotaryEmbedding, repeat_kv, apply_rotary_pos_emb
)
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2RotaryEmbedding,
    repeat_kv,
)
from fla.modules import RotaryEmbedding
from fla.ops.gla import chunk_gla, fused_recurrent_gla
from flash_attn import flash_attn_func, flash_attn_varlen_func
from fla.layers.utils import pad_input, unpad_input

try:
    from flash_attn import flash_attn_func
except ImportError:
    flash_attn_func = None


class LigerQwen3GatedLinearAttentionStudent(nn.Module):
    """
    Pure‑student GLA block.
    """
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config      = config
        self.layer_idx   = layer_idx
        self.hidden_size = config.hidden_size

        self.num_heads     = config.num_attention_heads          # H
        self.num_kv_heads  = config.num_key_value_heads          # H_kv
        self.num_kv_groups = self.num_heads // self.num_kv_heads # G = H / H_kv
        self.head_dim      = self.hidden_size // self.num_heads  # D
        self.attn_dropout  = config.attention_dropout
        self.rope_theta    = config.rope_theta
        self.window_size   = 64

        # projections 
        self.q_proj = nn.Linear(self.hidden_size,
                                  self.num_heads * self.head_dim,  bias=True)
        self.k_proj = nn.Linear(self.hidden_size,
                                  self.num_kv_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size,
                                  self.num_kv_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim,
                                  self.hidden_size, bias=False)

        # rotary + gate helpers
        self.rotary_emb = Qwen3RotaryEmbedding(config=config)
        self.rotary     = RotaryEmbedding(dim=self.head_dim,
                                          base=self.rope_theta)

        self.pool_g = nn.AdaptiveAvgPool1d(
            output_size=self.head_dim * self.num_kv_heads)  # pool *before* repeat

        self.gate_low_rank_dim = 16
        self.gk_proj = nn.Sequential(
            nn.Linear(self.hidden_size, self.gate_low_rank_dim, bias=False),
            nn.Linear(self.gate_low_rank_dim,
                      self.num_kv_heads * self.head_dim, bias=True),
        )

    def forward(
        self,
        hidden_states,                      # (B, L, D)
        attention_mask      = None,
        position_ids        = None,
        past_key_value      = None,
        output_attentions   = False,        # unused
        use_cache           = False,
        cache_position      = None,
        position_embeddings = None,
        **kwargs,
    ):

        cu_seqlens = None  # placeholder
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        gk = self.pool_g(k)

        gate_logit_normalizer = 16
        gk = F.logsigmoid(gk.float()) / gate_logit_normalizer

        batch_size, q_len, _ = hidden_states.size()
        seqlen_offset, max_seqlen = 0, q_len
        if past_key_value is not None:
            seqlen_offset = past_key_value.get_seq_length(self.layer_idx)
            max_seqlen = q.shape[1] + seqlen_offset

            if attention_mask is not None:
                # to delimit the offsets of padding tokens
                seqlen_offset = seqlen_offset + \
                    prepare_lens_from_mask(attention_mask) - attention_mask.shape[-1]
                max_seqlen = q.shape[1] + max(seqlen_offset)

        max_seqlen = max(max_seqlen, 4096)
        q  = rearrange(q,  'b n (h d) -> b n h d', h=self.num_heads)
        k  = rearrange(k,  'b n (h d) -> b n h d', h=self.num_kv_heads)
        v  = rearrange(v,  'b n (h d) -> b n h d', h=self.num_kv_heads)
        gk = rearrange(gk, 'b n (h m) -> b n h m', h=self.num_kv_heads)

        kv_groups = self.num_heads // self.num_kv_heads

        q = F.softmax(q.float(), dim=-1).to(v)
        k = F.softmax(k.float(), dim=-1).to(v)


        k_gla, v_gla, gk_gla = k, v, gk                          # (B,L,H_kv,…)
        k_rep = repeat_kv(k, kv_groups)                          # (B,L,H,D)
        v_rep = repeat_kv(v, kv_groups)

        # replicate along the *head* dimension with repeat_interleave
        k_gla_full  = k_gla.repeat_interleave(kv_groups, dim=2)      # (B, L, H, D)
        v_gla_full  = v_gla.repeat_interleave(kv_groups, dim=2)
        gk_gla_full = gk_gla.repeat_interleave(kv_groups, dim=2)     # (B, L, H, m)
    
        sq, sk, sv = q, k, v

        recurrent_state = None
        if past_key_value is not None and len(past_key_value) > self.layer_idx:
            k_cached, v_cached = past_key_value[self.layer_idx]
            if hasattr(past_key_value, "recurrent_state"):
                recurrent_state = past_key_value.recurrent_state.get(self.layer_idx, None)

        scale = 1
        sq, sk = self.rotary(sq, sk, seqlen_offset=seqlen_offset,
                             max_seqlen=max_seqlen, cu_seqlens=None)

        if past_key_value is not None:
            cache_has_content = past_key_value.get_seq_length(self.layer_idx) > 0
            new_k = sk.permute(0, 2, 1, 3).contiguous()   # (B, H, T, D)
            new_v = sv.permute(0, 2, 1, 3).contiguous()
            past_key_value.update(
                new_k, new_v, layer_idx=self.layer_idx,
                cache_kwargs=dict(window_size=self.window_size),
            )
            if cache_has_content:
                k_cached, v_cached = past_key_value[self.layer_idx]
                sk = k_cached.permute(0, 2, 1, 3).contiguous()
                sv = v_cached.permute(0, 2, 1, 3).contiguous()

        if self.training or q.shape[1] > 1:
            if attention_mask is not None:
                q, (k, v, gk), indices_q, cu_seqlens_rnns, max_seq_lens = unpad_input(
                    q, (k_gla, v_gla, gk_gla), attention_mask, q_len)
                o_, recurrent_state = chunk_gla(
                    q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), gk.unsqueeze(0),
                    scale=scale, initial_state=recurrent_state, output_final_state=True,
                    cu_seqlens=cu_seqlens_rnns[0]
                )
                o_ = pad_input(o_.squeeze(0), indices_q, batch_size, q_len)
            else:
                o_, recurrent_state = chunk_gla(
                    q, k_gla_full, v_gla_full, gk_gla_full, scale=scale,
                    initial_state=recurrent_state, output_final_state=True)
        else:
            o_, recurrent_state = fused_recurrent_gla(
                q, k_gla, v_gla, gk_gla, scale=scale,
                initial_state=recurrent_state, output_final_state=True)

        if past_key_value is not None:
            past_key_value.update(
                sk.permute(0, 2, 1, 3).contiguous(),
                sv.permute(0, 2, 1, 3).contiguous(),
                layer_idx=self.layer_idx,
                cache_kwargs=dict(window_size=self.window_size),
            )
            if not hasattr(past_key_value, "recurrent_state"):
                past_key_value.recurrent_state = {}
            past_key_value.recurrent_state[self.layer_idx] = recurrent_state

        # Flash‑Attention
        if attention_mask is not None:
            if self.window_size is not None and sq.shape[1] == 1:
                attention_mask = attention_mask[:, -self.window_size:]
            sq, (sk, sv), indices_q, cu_seqlens, max_seq_lens = unpad_input(
                sq, (sk, sv), attention_mask, q_len)
            cu_seqlens_q, cu_seqlens_k = cu_seqlens
            max_seqlen_q, max_seqlen_k = max_seq_lens

            k_rep = repeat_kv(sk, kv_groups)   # regen after unpad
            v_rep = repeat_kv(sv, kv_groups)

            y = flash_attn_varlen_func(
                sq, k_rep, v_rep,
                cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_q, max_seqlen_k=max_seqlen_k,
                causal=True,
                window_size=(-1, -1) if self.window_size is None else (self.window_size-1, 0)
            )
            y = pad_input(y, indices_q, batch_size, q_len)

        elif cu_seqlens is not None:
            y = flash_attn_varlen_func(
                sq.squeeze(0), repeat_kv(sk.squeeze(0), kv_groups),
                repeat_kv(sv.squeeze(0), kv_groups),
                cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seqlen, max_seqlen_k=max_seqlen,
                causal=True,
                window_size=(-1, -1) if self.window_size is None else (self.window_size-1, 0)
            ).unsqueeze(0)
        else:
            y = flash_attn_func(
                sq, k_rep, v_rep,
                causal=True,
                window_size=(-1, -1) if self.window_size is None else (self.window_size-1, 0)
            )

        # ---- harmonise head‑count BEFORE mixing (tile H_kv → H) --------
        if o_.shape[2] != y.shape[2]:                 # 8 → 32
            o_ = repeat_kv(o_, kv_groups)             # (B, L, H, D)

        o_ = 0.5 * y + 0.5 * o_
        o  = rearrange(o_, 'b n h d -> b n (h d)').to(hidden_states.dtype)
        o  = self.o_proj(o)

        return o, None   # (hidden_states, self_attn_weights)


class LigerQwen2GatedLinearAttentionStudent(nn.Module):
    """
    Pure-student *Gated Linear Attention* (GLA) block for Qwen-2.
    This version is architecturally identical to the original implementation,
    using full MHA projections.
    """
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        # ---- Dims & hyper-params ------------------------------------------------
        self.hidden_size: int = config.hidden_size
        self.num_heads: int = config.num_attention_heads
        self.head_dim: int = self.hidden_size // self.num_heads
        self.rope_theta: float = config.rope_theta
        self.window_size: int = 64

        # ---- Projections (MHA) ----------------------------------
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        # ---- Rotary embeddings --------------------------------------------------
        self.rotary = RotaryEmbedding(dim=self.head_dim, base=self.rope_theta)

        # ---- Gated key helper --------------------
        self.pool_g = nn.AdaptiveAvgPool1d(output_size=self.head_dim * self.num_heads)
        self.gate_logit_normalizer = 16

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional["Cache"] = None,
        **kwargs,
    ):
        # 1. Projections & gating
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        gk = self.pool_g(k)
        gk = F.logsigmoid(gk.float()) / self.gate_logit_normalizer

        # 2. Reshape to separate heads
        q = rearrange(q,  "b n (h d) -> b n h d", h=self.num_heads)
        k = rearrange(k,  "b n (h d) -> b n h d", h=self.num_heads)
        v = rearrange(v,  "b n (h d) -> b n h d", h=self.num_heads)
        gk = rearrange(gk, "b n (h m) -> b n h m", h=self.num_heads)

        # 3. Prepare inputs for each branch separately
        # Tensors for FlashAttention will get rotary embeddings
        q_for_flash, k_for_flash = q.clone(), k.clone()

        # Tensors for GLA will be softmax-normalized
        q_for_gla = F.softmax(q.float(), dim=-1).to(v.dtype)
        k_for_gla = F.softmax(k.float(), dim=-1).to(v.dtype)

        # 4. Apply rotary embeddings to the FlashAttention Q/K
        seqlen_offset = 0
        if past_key_value is not None:
            seqlen_offset = past_key_value.get_seq_length(self.layer_idx)
        max_seqlen = max(seqlen_offset + q.shape[1], 4096)
        q_for_flash, k_for_flash = self.rotary(q_for_flash, k_for_flash, seqlen_offset=seqlen_offset, max_seqlen=max_seqlen)

        # 5. Run GLA branch (using softmax-normalized Q/K)
        if self.training or q.shape[1] > 1:
            if attention_mask is not None:
                q_gla_unpad, (k_gla_unpad, v_gla_unpad, gk_unpad), idx, cu_seqlens_rnn, _ = unpad_input(q_for_gla, (k_for_gla, v, gk), attention_mask, q.shape[1])
                o_gla, _ = chunk_gla(
                    q_gla_unpad.unsqueeze(0), k_gla_unpad.unsqueeze(0), v_gla_unpad.unsqueeze(0), gk_unpad.unsqueeze(0),
                    scale=1.0, initial_state=None, output_final_state=True,
                    cu_seqlens=cu_seqlens_rnn[0]
                )
                o_gla = pad_input(o_gla.squeeze(0), idx, hidden_states.size(0), q.shape[1])
            else:
                o_gla, _ = chunk_gla(q_for_gla, k_for_gla, v, gk, scale=1.0, initial_state=None, output_final_state=True)
        else:
            o_gla, _ = fused_recurrent_gla(q_for_gla, k_for_gla, v, gk, scale=1.0, initial_state=None, output_final_state=True)

        # 6. Flash-Attention branch (using rotary-encoded Q/K)
        if flash_attn_func is None:
            raise RuntimeError("flash-attn is required. Please install via `pip install flash-attn`.")
        
        if attention_mask is not None:
            if self.window_size is not None and q.shape[1] == 1:
                attention_mask = attention_mask[:, -self.window_size:]
            q_flash_unpad, (k_flash_unpad, v_flash_unpad), idx, cu_seqlens, max_lens = unpad_input(q_for_flash, (k_for_flash, v), attention_mask, q.shape[1])
            y_flash = flash_attn_varlen_func(
                q_flash_unpad, k_flash_unpad, v_flash_unpad,
                cu_seqlens_q=cu_seqlens[0], cu_seqlens_k=cu_seqlens[1],
                max_seqlen_q=max_lens[0], max_seqlen_k=max_lens[1],
                causal=True,
                window_size=(-1, -1) if self.window_size is None else (self.window_size - 1, 0),
            )
            y_flash = pad_input(y_flash, idx, hidden_states.size(0), q.shape[1])
        else:
            y_flash = flash_attn_func(
                q_for_flash, k_for_flash, v,
                causal=True,
                window_size=(-1, -1) if self.window_size is None else (self.window_size - 1, 0),
            )

        # 7. Merge branches & project
        o_combined = 0.5 * y_flash + 0.5 * o_gla
        o_combined = rearrange(o_combined, "b n h d -> b n (h d)").to(hidden_states.dtype)
        output = self.o_proj(o_combined)

        return output, None