# hf_trainer.py
from __future__ import annotations
import torch
import torch.nn as nn
from transformers import Trainer

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
