# hf_trainer.py
from __future__ import annotations
import torch
import torch.nn as nn
from transformers import Trainer
import torch.nn.functional as F

__all__ = ["DistillTrainer", "FinetuneTrainer"]

class _BaseTrainer(Trainer):
    """
    A thin wrapper around Huggingface Trainer that only overrides compute_loss().
    Everything else (DDP, ZeRO, gradient‐accumulation, mixed precision,
    checkpoint‑rotation, etc.) is delegated to Hugging Face / DeepSpeed / Accelerate.
    """
    def __init__(self, *args,
                 mse_factor: float = 1.0,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.mse_factor = mse_factor
        self.criterion_ce  = nn.CrossEntropyLoss(reduction="mean")
        self.criterion_mse = nn.MSELoss(reduction="mean")

class DistillTrainer(_BaseTrainer):
    """
    Stage‑1 trainer used by rapid‑distill / LoLCats‑AT etc.
    Assumes the model.forward(..) returns a tuple of attentions where
    `attn[layer][0]` is the *teacher* map and `attn[layer][1]` is the *student* map.
    """
    def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False):
        # Strip labels – we only need hidden states / attentions
        inputs = {k: v.to(model.device) for k, v in inputs.items() if k != "labels"}

        outputs = model(**inputs, output_attentions=True)
        attentions = outputs.attentions          # tuple[#layers][2, B, H, L, L]

        loss_mse = 0.0
        n_layers = 0
        for layer_attn in attentions:
            if layer_attn is not None:
                loss_mse += self.criterion_mse(layer_attn[0], layer_attn[1])
                n_layers += 1
        if n_layers:
            loss_mse = loss_mse / n_layers * self.mse_factor

        if return_outputs:
            extra = {"loss_mse": loss_mse.detach().cpu().item(),
                     "mse_factor": self.mse_factor}
            return (loss_mse, {**outputs, **extra})
        return loss_mse


class FinetuneTrainer(_BaseTrainer):
    """
    LM fine‑tuning stage – classic causal‑LM cross‑entropy.
    We compute loss manually so it stays identical to your current code
    (shift‑inputs‑by‑1 and ignore padding).
    """
    def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False):
        input_keys = {"input_ids", "attention_mask"}
        data = {k: v.to(model.device) for k, v in inputs.items() if k in input_keys}

        logits = model(**data, use_cache=False).logits          # (B, L, V)
        targets = inputs["labels"].to(logits.device)            # (B, L)

        # Shift
        logits  = logits[:, :-1, :].contiguous()
        targets = targets[:, 1:].contiguous()

        loss = self.criterion_ce(
            logits.view(-1, logits.size(-1)),
            targets.view(-1)
        )

        if return_outputs:
            return (loss,
                    {"ppl": torch.exp(loss).item(),
                     "seq_len": targets.size(1)+1,
                     "logits": logits})
        return loss



class KDTrainer(Trainer):
    def __init__(
        self,
        teacher_model,
        kl_weight=1.0,
        ce_weight=1.0,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        self.kl_weight = kl_weight
        self.ce_weight = ce_weight

        # teacher_model can be large: put in eval mode, possibly wrap in deepspeed
        self.teacher_model.eval()

    def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False):
        # teacher forward in no_grad mode
        with torch.no_grad():
            teacher_out = self.teacher_model(**inputs)
            teacher_logits = teacher_out.logits

        # student forward
        # If ce_weight > 0, pass labels for cross-entropy
        if self.ce_weight > 0:
            outputs_student = model(**inputs)
            cross_entropy_loss = outputs_student.loss
            student_logits = outputs_student.logits
        else:
            # no labels -> no cross-entropy
            new_inputs = {k: v for k, v in inputs.items() if k != "labels"}
            outputs_student = model(**new_inputs)
            cross_entropy_loss = 0.0
            student_logits = outputs_student.logits

        # KL divergence
        # F.kl_div takes log-probabilities as first argument
        kl_loss = F.kl_div(
            F.log_softmax(student_logits, dim=-1),
            F.softmax(teacher_logits, dim=-1),
            reduction='batchmean'
        )

        total_loss = self.kl_weight * kl_loss + self.ce_weight * cross_entropy_loss

        return (total_loss, outputs_student) if return_outputs else total_loss
