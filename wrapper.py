# wrapper.py
import torch, torch.nn as nn

class AttentionDistillationWrapper(nn.Module):
    def __init__(self, teacher_attn, student_cls, config, layer_idx):
        super().__init__()
        self.teacher_attn = teacher_attn.eval()
        for p in self.teacher_attn.parameters():
            p.requires_grad_(False)

        self.student_attn = student_cls(config, layer_idx)

        # weight transplant
        with torch.no_grad():
            for name, p in self.teacher_attn.named_parameters():
                if name in self.student_attn.state_dict():
                    self.student_attn.state_dict()[name].copy_(p)

        self.distill_loss = torch.tensor(0.0)

    def forward(self, *args, **kwargs):
        kwargs["output_attentions"] = False
        # teacher pass – no gradients
        with torch.no_grad():
            t_hidden, _ = self.teacher_attn(*args, **kwargs)

        # student pass (this is what must flow back to the decoder)
        s_hidden, _ = self.student_attn(*args, **kwargs)

        # stash distillation loss for the caller to consume
        self.distill_loss = torch.linalg.vector_norm(
            t_hidden - s_hidden, dim=-1
        ).mean() * t_hidden.size(-1) ** -0.5

        # ONLY return (hidden_states, attn_weights/None)
        return t_hidden, None
