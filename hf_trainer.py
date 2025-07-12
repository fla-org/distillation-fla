# hf_trainer.py
from __future__ import annotations
import torch
import torch.nn as nn
from transformers import Trainer
import torch.nn.functional as F
from fla.modules.fused_kl_div import FusedKLDivLoss

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

        # 'attentions' is now a simple tuple of pre-computed loss tensors from each layer
        per_layer_losses = outputs.attentions

        # The total loss is just the mean of the per-layer losses.
        # Stack them into a single tensor and calculate the mean.
        if per_layer_losses:
            loss = torch.stack(per_layer_losses).mean() * self.mse_factor
        else:
            loss = torch.tensor(0.0, device=model.device, requires_grad=True)

        if return_outputs:
            extra = {"loss_mse": loss.detach().cpu().item(),
                     "mse_factor": self.mse_factor}
            return (loss, {**outputs, **extra})
        return loss


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
        # TODO: change to this https://github.com/fla-org/flash-linear-attention/blob/main/fla/modules/fused_kl_div.py\

        kl_loss = F.kl_div(
            F.log_softmax(student_logits, dim=-1),
            F.softmax(teacher_logits, dim=-1),
            reduction='mean'
        )

        total_loss = self.kl_weight * kl_loss + self.ce_weight * cross_entropy_loss

        return (total_loss, outputs_student) if return_outputs else total_loss



# class KDTrainer(Trainer):
#     def __init__(
#         self,
#         teacher_model,
#         kl_weight=1.0,
#         ce_weight=1.0,
#         *args, **kwargs
#     ):
#         super().__init__(*args, **kwargs)
#         self.teacher_model = teacher_model
#         self.kl_weight = kl_weight
#         self.ce_weight = ce_weight

#         # 1. Instantiate the FusedKLDivLoss
#         # The reduction='batchmean' matches the original F.kl_div usage.
#         self.kl_div_loss_fn = FusedKLDivLoss(reduction='batchmean')

#         # teacher_model can be large: put in eval mode, possibly wrap in deepspeed
#         self.teacher_model.eval()

#     def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False):
#         # --- 1. Teacher Forward ---
#         # Get teacher's last hidden states in no_grad mode.
#         with torch.no_grad():
#             # Request hidden states from the teacher model's output.
#             teacher_outputs = self.teacher_model(**inputs, output_hidden_states=True)
#             teacher_hidden_states = teacher_outputs.hidden_states[-1]

#         # --- 2. Student Forward ---
#         # The student forward pass also needs to output hidden states.
#         # We maintain the original logic for handling cross-entropy loss.
#         if self.ce_weight > 0:
#             # The model forward pass calculates cross-entropy loss if labels are provided.
#             student_outputs = model(**inputs, output_hidden_states=True)
#             cross_entropy_loss = student_outputs.loss
#             student_hidden_states = student_outputs.hidden_states[-1]
#         else:
#             # If ce_weight is 0, we don't compute the cross-entropy loss.
#             # Remove "labels" from inputs to avoid unused output calculation.
#             new_inputs = {k: v for k, v in inputs.items() if k != "labels"}
#             student_outputs = model(**new_inputs, output_hidden_states=True)
#             cross_entropy_loss = 0.0
#             student_hidden_states = student_outputs.hidden_states[-1]

#         # --- 3. KL Divergence with FusedKLDivLoss ---

#         # FusedKLDivLoss expects inputs of shape [*, hidden_size].
#         # Model outputs are typically [batch_size, seq_len, hidden_size].
#         # We flatten the batch and sequence dimensions.
#         student_hidden_states_flat = student_hidden_states.view(-1, student_hidden_states.size(-1))
#         teacher_hidden_states_flat = teacher_hidden_states.view(-1, teacher_hidden_states.size(-1))

#         # Get the language model head weights from both models.
#         # This assumes the models have a `get_output_embeddings` method,
#         # which is standard for Hugging Face CausalLM models.
#         student_lm_head_weight = model.get_output_embeddings().weight
#         teacher_lm_head_weight = self.teacher_model.get_output_embeddings().weight

#         # Calculate the KL divergence loss using the fused kernel.
#         # This replaces the old F.kl_div call.
#         kl_loss = self.kl_div_loss_fn(
#             x=student_hidden_states_flat,
#             target_x=teacher_hidden_states_flat,
#             weight=student_lm_head_weight,
#             target_weight=teacher_lm_head_weight,
#         )

#         # --- 4. Combine Losses ---
#         total_loss = self.kl_weight * kl_loss + self.ce_weight * cross_entropy_loss

#         # The return format must match the Trainer's expectation.
#         return (total_loss, student_outputs) if return_outputs else total_loss
