import torch
import torch.nn as nn
from student_only_attention import LigerQwen2GatedLinearAttentionStudent # Assumes corrected student is in this file

class AttentionDistillationWrapper(nn.Module):
    def __init__(self, teacher_attn, student_cls, config, layer_idx):
        super().__init__()
        self.teacher_attn = teacher_attn.eval()
        for p in self.teacher_attn.parameters():
            p.requires_grad_(False)

        self.student_attn = student_cls(config, layer_idx)

class AttentionDistillationWrapper(nn.Module):
    def __init__(self, teacher_attn, student_cls, config, layer_idx):
        super().__init__()
        self.teacher_attn = teacher_attn.eval()
        for p in self.teacher_attn.parameters():
            p.requires_grad_(False)

        self.student_attn = student_cls(config, layer_idx)

        # Perform GQA -> MHA weight transplant
        with torch.no_grad():
            # 1. Copy Q and O projections (shapes are compatible)
            self.student_attn.q_proj.weight.data.copy_(self.teacher_attn.q_proj.weight.data)
            self.student_attn.o_proj.weight.data.copy_(self.teacher_attn.o_proj.weight.data)
            
            # Check if BOTH teacher and student have bias
            if self.teacher_attn.q_proj.bias is not None and self.student_attn.q_proj.bias is not None:
                self.student_attn.q_proj.bias.data.copy_(self.teacher_attn.q_proj.bias.data)

            # 2. Get dimensions for GQA->MHA expansion
            num_kv_heads = config.num_key_value_heads
            num_heads = config.num_attention_heads
            head_dim = config.hidden_size // num_heads
            num_kv_groups = num_heads // num_kv_heads

            # 3. Handle K projection weights and biases
            gqa_k_weights = self.teacher_attn.k_proj.weight.data
            reshaped_k = gqa_k_weights.view(num_kv_heads, head_dim, config.hidden_size)
            mha_k_weights = torch.repeat_interleave(reshaped_k, num_kv_groups, dim=0)
            self.student_attn.k_proj.weight.data.copy_(mha_k_weights.view_as(self.student_attn.k_proj.weight.data))

            # CORRECTED: Check if BOTH teacher and student have bias
            if self.teacher_attn.k_proj.bias is not None and self.student_attn.k_proj.bias is not None:
                gqa_k_bias = self.teacher_attn.k_proj.bias.data
                reshaped_k_bias = gqa_k_bias.view(num_kv_heads, head_dim)
                mha_k_bias = torch.repeat_interleave(reshaped_k_bias, num_kv_groups, dim=0)
                self.student_attn.k_proj.bias.data.copy_(mha_k_bias.view_as(self.student_attn.k_proj.bias.data))

            # 4. Handle V projection weights and biases
            gqa_v_weights = self.teacher_attn.v_proj.weight.data
            reshaped_v = gqa_v_weights.view(num_kv_heads, head_dim, config.hidden_size)
            mha_v_weights = torch.repeat_interleave(reshaped_v, num_kv_groups, dim=0)
            self.student_attn.v_proj.weight.data.copy_(mha_v_weights.view_as(self.student_attn.v_proj.weight.data))

            # Check if BOTH teacher and student have bias
            if self.teacher_attn.v_proj.bias is not None and self.student_attn.v_proj.bias is not None:
                gqa_v_bias = self.teacher_attn.v_proj.bias.data
                reshaped_v_bias = gqa_v_bias.view(num_kv_heads, head_dim)
                mha_v_bias = torch.repeat_interleave(reshaped_v_bias, num_kv_groups, dim=0)
                self.student_attn.v_proj.bias.data.copy_(mha_v_bias.view_as(self.student_attn.v_proj.bias.data))

        self.distill_loss = torch.tensor(0.0)

    def forward(self, *args, **kwargs):
        kwargs["output_attentions"] = False
        kwargs["use_cache"] = False # Disable cache for teacher pass during training

        # Teacher pass – no gradients
        with torch.no_grad():
            # CORRECTED: Unpack 3 values from the standard attention layer
            t_hidden, _ = self.teacher_attn(*args, **kwargs)

        # Student pass (this is what must flow back to the decoder)
        s_hidden, _ = self.student_attn(*args, **kwargs)

        # Stash distillation loss for the caller to consume
        self.distill_loss = torch.linalg.vector_norm(
            t_hidden - s_hidden, dim=-1
        ).mean() * (t_hidden.size(-1) ** -0.5)

        # ONLY return the teacher's hidden state to stabilize the rest of the network
        return t_hidden, None