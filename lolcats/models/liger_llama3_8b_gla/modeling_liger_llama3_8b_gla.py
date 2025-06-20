import copy
import math
import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from einops import rearrange, repeat
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import (BaseModelOutputWithPast,
                                           CausalLMOutputWithPast)
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaForCausalLM,
    LlamaMLP,
    LlamaModel,
    LlamaPreTrainedModel,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
    apply_rotary_pos_emb,
    repeat_kv,
)
from transformers.utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    logging,
    replace_return_docstrings,
)

try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
except ImportError:
    warnings.warn(
        "Flash Attention is not installed. Please install it via `pip install flash-attn --no-build-isolation`",
        category=ImportWarning
    )
    flash_attn_func = None

from fla.layers.utils import pad_input, unpad_input
from fla.modules import RMSNorm, RotaryEmbedding
from fla.ops.utils.index import prepare_lens_from_mask
from fla.models.utils import Cache as FlaCache
from fla.ops.gla import chunk_gla, fused_chunk_gla, fused_recurrent_gla

from .configuration_liger_llama3_8b_gla import LigerLlama3GLAConfig

logger = logging.get_logger(__name__)

if is_flash_attn_2_available():
    from transformers.modeling_flash_attention_utils import _flash_attention_forward
else:
    print("flash_attn_2 is not available")


class LigerLlama3GatedLinearAttention(nn.Module):
    def __init__(
        self,
        config: LigerLlama3GLAConfig,
        layer_idx: Optional[int] = None,
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended and will "
                "to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        self.attention_dropout = config.attention_dropout

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        # Teacher model's weight. We should load them in Stage1 and freeze them.
        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=False)

        # Student model's weight. We should load them in Stage2 and train them.
        self.q_proj_s = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj_s = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj_s = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj_s = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.rotary_emb = LlamaRotaryEmbedding(config=config)

        self.rotary = RotaryEmbedding(
            dim=self.head_dim, base=config.rope_theta)

        self.pool_g = nn.AdaptiveAvgPool1d(
            output_size=self.head_dim * self.num_key_value_heads)

        self.window_size = 64

    def init_student_weights(self):
        self.q_proj_s.weight.data.copy_(self.q_proj.weight.data)
        self.k_proj_s.weight.data.copy_(self.k_proj.weight.data)
        self.v_proj_s.weight.data.copy_(self.v_proj.weight.data)
        self.o_proj_s.weight.data.copy_(self.o_proj.weight.data)
        if self.layer_idx == 0:
            print(
                "Student weight copied from teacher!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    def destroy_teacher_weights(self):
        del self.q_proj.weight
        del self.q_proj.bias
        del self.k_proj.weight
        del self.k_proj.bias
        del self.v_proj.weight
        del self.v_proj.bias
        del self.o_proj.weight
        self.q_proj = None
        self.k_proj = None
        self.v_proj = None
        self.o_proj = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[FlaCache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        with torch.no_grad():
            o_teacher, _, _ = self.forward_teacher(hidden_states, attention_mask, position_ids, past_key_value,
                                                   output_attentions, use_cache, cache_position, position_embeddings, **kwargs)
        o_student, _, _ = self.forward_student(hidden_states, attention_mask, position_ids, past_key_value,
                                               output_attentions, use_cache, cache_position, position_embeddings, **kwargs)
        return o_teacher, (o_teacher, o_student), None

    def forward_teacher(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[FlaCache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        query_states = query_states.view(
            bsz, q_len, -1, self.head_dim).transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, -1, self.head_dim).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, -1, self.head_dim).transpose(1, 2)

        q = query_states
        k = repeat_kv(key_states, self.num_key_value_groups)
        v = repeat_kv(value_states, self.num_key_value_groups)

        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            cos, sin = self.rotary_emb(v, position_ids)
        else:
            cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        o = _flash_attention_forward(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            attention_mask,
            q_len,
            position_ids=position_ids,
            dropout=0.0,
            is_causal=True,
            target_dtype=v.dtype,
        )
        o = rearrange(o, 'b n h d -> b n (h d)')
        o = self.o_proj(o)
        return o, None, past_key_value

    def forward_student(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[FlaCache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        last_state = None
        if past_key_value is not None and len(past_key_value) > self.layer_idx:
            last_state = past_key_value[self.layer_idx]

        cu_seqlens = None
        q = self.q_proj_s(hidden_states)
        k = self.k_proj_s(hidden_states)
        v = self.v_proj_s(hidden_states)
        g = self.pool_g(k)

        batch_size, q_len, _ = hidden_states.size()
        seqlen_offset, max_seqlen = 0, q_len
        if past_key_value is not None:
            seqlen_offset = past_key_value.get_seq_length(self.layer_idx)
            max_seqlen = q.shape[1] + seqlen_offset

            if attention_mask is not None:
                seqlen_offset = seqlen_offset + \
                    prepare_lens_from_mask(
                        attention_mask) - attention_mask.shape[-1]
                max_seqlen = q.shape[1] + max(seqlen_offset)

        max_seqlen = max(max_seqlen, 4096)
        q = rearrange(q, 'b n (h d) -> b n h d', h=self.num_heads)
        k = rearrange(k, 'b n (h d) -> b n h d', h=self.num_key_value_heads)
        v = rearrange(v, 'b n (h d) -> b n h d', h=self.num_key_value_heads)
        g = rearrange(g, 'b n (h m) -> b n h m', h=self.num_key_value_heads)

        k = repeat(k, 'b n h d -> b n (h g) d', g=self.num_key_value_groups)
        v = repeat(v, 'b n h d -> b n (h g) d', g=self.num_key_value_groups)
        g = repeat(g, 'b n h m -> b n (h g) m', g=self.num_key_value_groups)

        sq, sk, sv = q, k, v
        q = F.softmax(q.float(), dim=-1).to(v)
        k = F.softmax(k.float(), dim=-1).to(v)

        gate_logit_normalizer = 16
        g = F.logsigmoid(g.float()) / gate_logit_normalizer

        recurrent_state = last_state['recurrent_state'] if last_state is not None else None
        scale = 1
        sq, sk = self.rotary(sq, sk, seqlen_offset=seqlen_offset,
                             max_seqlen=max_seqlen, cu_seqlens=None)

        if past_key_value is not None:
            cache_has_content = past_key_value.get_seq_length(
                self.layer_idx) > 0
            k_cached, v_cached = past_key_value.update(
                attn_state=(sk.flatten(-2, -1), sv.flatten(-2, -1)),
                layer_idx=self.layer_idx,
                offset=q_len,
                cache_kwargs=dict(window_size=self.window_size)
            )['attn_state']
            if cache_has_content:
                sk, sv = k_cached, v_cached
                sk = rearrange(sk, '... (h d) -> ... h d', d=self.head_dim)
                sv = rearrange(sv, '... (h d) -> ... h d', d=self.head_dim)

        if self.training or q.shape[1] > 1:
            if attention_mask is not None:
                q, (k, v, g), indices_q, cu_seqlens_rnns, max_seq_lens = unpad_input(
                    q, (k, v, g), attention_mask, q_len)
                o_, recurrent_state = chunk_gla(
                    q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), g.unsqueeze(0), scale=scale, initial_state=recurrent_state, output_final_state=True,
                    cu_seqlens=cu_seqlens_rnns[0]
                )
                o_ = pad_input(o_.squeeze(0), indices_q, batch_size, q_len)
            else:
                o_, recurrent_state = chunk_gla(
                    q, k, v, g, scale=scale, initial_state=recurrent_state, output_final_state=True)
        else:
            o_, recurrent_state = fused_recurrent_gla(
                q, k, v, g, scale=scale, initial_state=recurrent_state, output_final_state=True)

        if past_key_value is not None:
            past_key_value.update(
                recurrent_state=recurrent_state,
                layer_idx=self.layer_idx,
                offset=0
            )

        q_len = hidden_states.size(1)

        if attention_mask is not None:
            if self.window_size is not None:
                if sq.shape[1] == 1 and self.window_size is not None:
                    attention_mask = attention_mask[:, -self.window_size:]
            sq, (sk, sv), indices_q, cu_seqlens, max_seq_lens = unpad_input(
                sq, (sk, sv), attention_mask, q_len)
            cu_seqlens_q, cu_seqlens_k = cu_seqlens
            max_seqlen_q, max_seqlen_k = max_seq_lens
            y = flash_attn_varlen_func(
                sq, sk, sv,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                causal=True,
                window_size=(-1, -1) if self.window_size is None else (self.window_size-1, 0)
            )
            y = pad_input(y, indices_q, batch_size, q_len)
        elif cu_seqlens is not None:
            y = flash_attn_varlen_func(
                sq.squeeze(0), sk.squeeze(0), sv.squeeze(0),
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seqlen,
                max_seqlen_k=max_seqlen,
                causal=True,
                window_size=(-1, -1) if self.window_size is None else (self.window_size-1, 0)
            ).unsqueeze(0)
        else:
            y = flash_attn_func(
                sq, sk, sv,
                causal=True,
                window_size=(-1, -1) if self.window_size is None else (self.window_size-1, 0)
            )
        o_ = 0.5 * y + 0.5 * o_
        o = rearrange(o_, 'b n h d -> b n (h d)').to(hidden_states.dtype)
        o = self.o_proj_s(o)
        return o, None, past_key_value


class LigerLlama3DecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: LigerLlama3GLAConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.hidden_size = config.hidden_size

        # if config.sliding_window and config._attn_implementation != "flash_attention_2":
        #     logger.warning_once(
        #         f"Sliding Window Attention is enabled but not implemented for `{config._attn_implementation}`; "
        #         "unexpected results may be encountered."
        #     )
        self.self_attn = LigerLlama3GatedLinearAttention(config, layer_idx)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps)


class LigerLlama3PreTrainedModel(LlamaPreTrainedModel):

    config_class = LigerLlama3GLAConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LlamaDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"


class LigerLlama3GLAModel(LlamaModel, LigerLlama3PreTrainedModel):

    def __init__(self, config: LigerLlama3GLAConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [LigerLlama3DecoderLayer(config, layer_idx)
             for layer_idx in range(config.num_hidden_layers)]
        )
        self._attn_implementation = config._attn_implementation
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)

        self.gradient_checkpointing = False
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values:  Optional[Union[Tuple,
                                         FlaCache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        return_legacy_cache = False
        if use_cache and not isinstance(past_key_values, FlaCache):
            past_key_values = FlaCache.from_legacy_cache(past_key_values)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length(
            ) if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = attention_mask

        hidden_states = inputs_embeds

        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        if output_attentions:
            all_softmax_hidden_states = ()

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
                if all_softmax_hidden_states is not None:
                    all_softmax_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                )

            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )
                hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class LigerLlama3GLAForCausalLM(LigerLlama3PreTrainedModel, LlamaForCausalLM, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}

    def __init__(self, config):
        super().__init__(config)
        self.model = LigerLlama3GLAModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(
            config.hidden_size, config.vocab_size, bias=False)

        self.post_init()

    def init_student_weights(self):
        for layer in self.model.layers:
            layer.self_attn.init_student_weights()
