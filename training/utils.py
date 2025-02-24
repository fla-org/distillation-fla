import numpy as np
import torch
import torch.optim

def get_optimizer_and_scheduler(model, config):
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.train.lr, fused=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer, 
        mode='min',
        factor=0.1,
        patience=10,
        min_lr=0.00001
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