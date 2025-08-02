import argparse, os, yaml, math, torch, importlib
import json
import deepspeed

from transformers import (AutoConfig, AutoTokenizer, AutoModelForCausalLM,
                          TrainingArguments)
from omegaconf import OmegaConf
from training.utils import count_model_params
from hf_trainer import DistillTrainer, FinetuneTrainer, KDTrainer
from accelerate import init_empty_weights
from wrapper import AttentionDistillationWrapper
from distill_model.config_distilled_student import StudentConfig
from distill_model.modeling_distilled_student import StudentForCausalLM, get_student_attention_class
from transformers import AutoConfig, AutoModelForCausalLM
AutoConfig.register('student', StudentConfig, exist_ok=True)
AutoModelForCausalLM.register(StudentConfig, StudentForCausalLM, exist_ok=True)
import subprocess

import sys
import logging
import os
import torch.distributed as dist

def get_logger(name: str = None) -> logging.Logger:
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S"
    )
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    if 'RANK' in os.environ and int(os.environ['RANK']) == 0:
        logger.setLevel(logging.INFO)
        logger.addHandler(handler)

    return logger
logger = get_logger(__name__)


# workaround for pytorch2.6
_original_torch_load = torch.load
def patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = patched_torch_load

def parse_config(path: str):
    with open(path) as f:
        return yaml.safe_load(f)

def json_serializer(obj):
    """
    A custom serializer for objects that are not serializable by default json code.
    Specifically, this handles torch.dtype and numpy float types.
    """
    if isinstance(obj, torch.dtype):
        return str(obj)
    if hasattr(obj, 'item'):
        return obj.item()
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


def _prepare_teacher_deepspeed(teacher_model, ds_config_path):
    """
    Wrap a large teacher model in a separate DeepSpeed engine so it can be
    sharded under ZeRO-3. Only needed if your teacher is huge and you want
    DS to manage it. Otherwise you can skip or adapt as needed.
    """
    with open(ds_config_path) as f:
        ds_cfg = json.load(f)

    # Teacher does not need grads
    for param in teacher_model.parameters():
        param.requires_grad = False

    # Force ZeRO-3
    ds_cfg["zero_optimization"]["stage"] = 3

    # Optionally tune bucket sizes
    hidden_size = getattr(teacher_model.config, "hidden_size", None)
    if hidden_size is None and getattr(teacher_model.config, "hidden_sizes", None):
        hidden_size = max(teacher_model.config.hidden_sizes)

    if hidden_size is not None and ds_cfg["zero_optimization"]["stage"] == 3:
        ds_cfg["zero_optimization"]["reduce_bucket_size"] = hidden_size * hidden_size
        ds_cfg["zero_optimization"]["stage3_param_persistence_threshold"] = 10 * hidden_size
        ds_cfg["zero_optimization"]["stage3_prefetch_bucket_size"] = int(0.9 * hidden_size * hidden_size)

    teacher_engine, _, _, _ = deepspeed.initialize(
        model=teacher_model,
        model_parameters=None,
        config=ds_cfg
    )
    teacher_engine.eval()
    return teacher_engine


def patch_model_for_stage1(model, base_model_cfg, cfg):
    """
    Replace `layer.attn` with a wrapper so the teacher’s
    hidden states still drive the rest of the frozen network.

    This version is MODIFIED to keep specified layers as full-attention.
    """
    # Get the correct student attention class dynamically
    student_attn_class = get_student_attention_class(cfg.student_model.name)
    logger.info(f"✅ Using student attention class: {student_attn_class.__name__}")

    # Get the list of layers to keep as full attention from the config.
    # Default to an empty list if not specified.
    keep_full_attention_layers = cfg.student_model.get('keep_full_attention_layers', [])
    if keep_full_attention_layers:
        logger.info(f"⚠️ Will keep the following layers as full-attention: {keep_full_attention_layers}")

    for idx, layer in enumerate(model.model.layers):
        # Conditionally skip patching if the layer index is in our keep list.
        if idx in keep_full_attention_layers:
            logger.info(f"  -> Skipping layer {idx}, keeping as full-attention.")
            # Ensure the kept layer is frozen, as it's not being trained in Stage 1.
            for param in layer.attn.parameters():
                param.requires_grad_(False)
            continue

        # The existing logic now only runs for layers NOT in the keep list.
        logger.info(f"  -> Patching layer {idx} with student attention wrapper.")
        teacher_attn = layer.attn
        wrapper = AttentionDistillationWrapper(
            teacher_attn,
            student_attn_class,
            base_model_cfg,
            idx
        )
        layer.attn = wrapper

def build_student_for_stage1(cfg):
    """
    Build and partially freeze the student for stage 1 (attention distillation).
    Typically we load from the base model and selectively unfreeze Q/K/V or
    additional trainable layers.
    """
    base_model_cfg = AutoConfig.from_pretrained(cfg.teacher_model.name)

    base_model_cfg.use_cache = False # important!

    # build the base model first
    model = AutoModelForCausalLM.from_pretrained(
        cfg.teacher_model.name,
        config=base_model_cfg,
        torch_dtype=torch.bfloat16,
    )

    # patch each layer with (teacher → wrapper → student)
    patch_model_for_stage1(model, base_model_cfg, cfg)

    # Enable gradient checkpointing if specified in config
    if cfg.train.get('gradient_checkpointing', False):
        logger.info("✅ Enabling gradient checkpointing for Stage 1 model")
        model.gradient_checkpointing_enable()

    # freeze everything that is NOT inside .student_attn.
    for name, p in model.named_parameters():
        p.requires_grad_( ".student_attn." in name )

    tr, tot = count_model_params(model, True), count_model_params(model, False)
    logger.info(f"Trainable = {tr/1e6:.1f}M | Total = {tot/1e6:.1f}M ({tr/tot:.2%})")
    return model

def build_student_for_stage2_and_3(cfg):
    """
    Build the stage 2 student by loading the checkpoint from stage 1,
    purifying it by removing the teacher wrapper, and preparing it for
    knowledge distillation.

    This version is to handle hybrid models with both student
    and full-attention layers.
    """
    student_config = AutoConfig.from_pretrained(cfg.train.student_init_ckpt)
    student_config.fuse_swiglu = False # to be compatible with DeepSpeed's Zero-3

    student_model = AutoModelForCausalLM.from_pretrained(
    cfg.train.student_init_ckpt,
    config=student_config,
    torch_dtype=torch.bfloat16
    )
    for name, p in student_model.named_parameters():
        p.requires_grad = True
    # Enable gradient checkpointing if specified in config
    if cfg.train.get('gradient_checkpointing', False):
        logger.info("✅ Enabling gradient checkpointing for Stage 2 model")
        student_model.gradient_checkpointing_enable()
    tr, tot = count_model_params(student_model, True), count_model_params(student_model, False)
    logger.info(f"[Stage 2] Purified Student: Trainable = {tr/1e6:.1f}M | Total = {tot/1e6:.1f}M ({tr/tot:.2%})")
    return student_model


def build_teacher_for_stage2(cfg):
    """
    Teacher is the base model with full attention. If you want to
    DeepSpeed-shard it, do so. Otherwise, just load it normally.
    """
    teacher_config = AutoConfig.from_pretrained(cfg.teacher_model.name)
    teacher_config.fuse_swiglu = False # do not fuse swiglu, to be compatible with DeepSpeed's Zero-3
    teacher_model = AutoModelForCausalLM.from_pretrained(
        cfg.teacher_model.name,
        config=teacher_config,
        torch_dtype=torch.bfloat16,
    )
    teacher_model.eval()

    # Enable gradient checkpointing for teacher if specified in config
    if cfg.train.get('gradient_checkpointing', False):
        logger.info("✅ Enabling gradient checkpointing for teacher model")
        teacher_model.gradient_checkpointing_enable()

    # If you need DS-sharding for the teacher:
    teacher_ds_cfg = "ds_config_2_teacher.json"
    teacher_model = _prepare_teacher_deepspeed(teacher_model, teacher_ds_cfg)
    return teacher_model



def main(cfg):
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.teacher_model.name,
        padding_side="left"
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # -------------------------------------------------------------------
    # 1. Determine stage
    #    (We assume user sets cfg.stage = 1, 2, 3)
    # -------------------------------------------------------------------
    stage = cfg.stage

    if stage == 1:
        logger.info("==== Stage 1 (Attention Transfer) ====")
        # Student: from base model
        model = build_student_for_stage1(cfg)
        trainer_class = DistillTrainer
        ds_config_path = os.path.join(os.getcwd(), "ds_config_1.json")

    elif stage == 2:
        import torch.nn.functional as F
        logger.info("==== Stage 2 (Logit Distillation) ====")
        # Student: from the checkpoint saved by stage 1
        model = build_student_for_stage2_and_3(cfg)
        # Teacher: base model with full attention
        teacher_model = build_teacher_for_stage2(cfg)
        trainer_class = KDTrainer
        ds_config_path = os.path.join(os.getcwd(), "ds_config_2.json")

    elif stage == 3:
        logger.info("==== Stage 3 (Long-Context Finetuning) ====")
        # Student is the checkpoint saved by stage 2
        model = build_student_for_stage2_and_3(cfg)
        # No teacher model in stage 3
        teacher_model = None
        # Use the standard fine-tuning trainer
        trainer_class = FinetuneTrainer
        ds_config_path = os.path.join(os.getcwd(), "ds_config_3.json")
    else:
        raise ValueError(f"Unknown stage: {stage}. Must be 1, 2, or 3.")


    if os.path.exists(ds_config_path):
        logger.info(f"Using DS config = {ds_config_path}")
    else:
        ds_config_path = None
    
    from data import get_dataloader
    train_loader = get_dataloader(cfg.data.cache_dir, batch_size=cfg.train.batch_size, shuffle=True, num_workers=8)

    def get_optimizer(model, config):
        attn_params = []
        other_params = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if "attn" in name:
                attn_params.append(param)
                logger.info(f"Attn params: {name}, lr: {config.train.lr_attn}")
            else:
                other_params.append(param)
                logger.info(f"Other params: {name}, lr: {config.train.lr}")
        optimizer_grouped_parameters = [
            {"params": attn_params, "lr": config.train.lr_attn},
            {"params": other_params, "lr": config.train.lr},
        ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, betas=(0.9, 0.95), fused=True)
        return optimizer
    
    max_steps = cfg.train.target_tokens // (cfg.train.micro_batch_size * cfg.train.train_seq_len)
    num_gpus  = torch.cuda.device_count()
    g_accum   = cfg.train.batch_size // (cfg.train.micro_batch_size * num_gpus)
    seq_len   = cfg.train.train_seq_len
    tgt_tok   = cfg.train.target_tokens
    max_steps = (tgt_tok // (cfg.train.batch_size * seq_len)) if tgt_tok else cfg.train.max_steps

    training_args = TrainingArguments(
        per_device_train_batch_size = cfg.train.micro_batch_size,
        gradient_accumulation_steps = g_accum,
        max_steps                   = max_steps,
        bf16                        = True,
        logging_steps               = 10,
        eval_strategy               = "no",
        eval_steps                  = 5000000,
        save_steps                  = 2000,  # good?
        save_total_limit            = 100,
        metric_for_best_model       = "loss",
        greater_is_better           = False,
        output_dir                  = cfg.train.output_dir,
        deepspeed                   = ds_config_path,
        report_to                   = "wandb",
        gradient_checkpointing      = cfg.train.get('gradient_checkpointing', False),
        learning_rate               = cfg.train.lr,
        lr_scheduler_type           = cfg.train.lr_scheduler_type,
    )

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_loader.dataset,
        "eval_dataset": None,
        "optimizers": (get_optimizer(model, cfg), None), # auto infer lr scheduler
        "tokenizer": tokenizer,
    }

    if stage == 1:
        trainer_kwargs["mse_factor"] = 1.0 # Or from cfg
        trainer = DistillTrainer(**trainer_kwargs)
    elif stage == 2:
        trainer_kwargs["teacher_model"] = teacher_model
        trainer_kwargs["kl_weight"] = 1
        trainer_kwargs["ce_weight"] = 0
        trainer = KDTrainer(**trainer_kwargs)
    elif stage == 3:
        # FinetuneTrainer takes no extra args from this list
        trainer = FinetuneTrainer(**trainer_kwargs)

    # 6. Initialize trainer
    # 7. Train
    if cfg.train.resume_from_checkpoint == "None":
        trainer.train(resume_from_checkpoint=None)
    else:
        trainer.train(resume_from_checkpoint=cfg.train.resume_from_checkpoint)
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True, help="Path to YAML config")
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()

    cfg_dict = parse_config(args.cfg)
    cfg = OmegaConf.create(cfg_dict)
    main(cfg)

    if dist.is_initialized():
        if dist.get_rank() == 0:
            keep_layers = OmegaConf.to_container(cfg.student_model.keep_full_attention_layers)
            convert_cmd = [
                "python", "convert_ckpt.py",
                "--cfg", args.cfg,
            ]
            subprocess.run(convert_cmd)


            

