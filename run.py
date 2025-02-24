import argparse
import sys
import os
import random
import numpy as np
from omegaconf import OmegaConf
import torch

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from training.train import train


def set_random_seed(seed=0):
    os.environ['PYTHONHASHSEED'] = str(seed) 
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False # if benchmark=True, deterministic will be False
    torch.backends.cudnn.deterministic = True # choose a deterministic algorithm 

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="configs/liger.yaml")
    args = parser.parse_args()
    return args

def main():
    set_random_seed(seed=0)
    args = get_args()
    config = OmegaConf.load(args.cfg)
    output_dir = args.cfg.split('/')[-1].split('.')[0]
    config.train.output_dir = os.path.join(config.train.output_dir, output_dir) # 'checkpoints/${filename}'
    train(config)

if __name__ == "__main__":
    main()