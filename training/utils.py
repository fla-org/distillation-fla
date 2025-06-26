import numpy as np
import torch
import torch.optim
from transformers import get_scheduler
import math
from torch.optim.lr_scheduler import LambdaLR


def get_optimizer_and_scheduler(model, config, total_steps):
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.train.lr, betas=(0.9, 0.95), fused=True)
    scheduler = get_scheduler(
        config.train.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps,
    )
    return optimizer, scheduler

def count_model_params(model, requires_grad: bool = True):
    # code form lolcats
    """
    Return total # of trainable parameters
    """
    if requires_grad:
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    else:
        model_parameters = model.parameters()
    try:
        return sum([np.prod(p.size()) for p in model_parameters]).item()
    except:
        return sum([np.prod(p.size()) for p in model_parameters])