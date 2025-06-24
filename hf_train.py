# train.py
import argparse, os, yaml, math, torch
from transformers import (AutoConfig, AutoTokenizer, AutoModelForCausalLM,
                          TrainingArguments)
from hf_trainer import DistillTrainer, FinetuneTrainer         # NEW
from training.dataloader import load_data
from training.utils import count_model_params, get_optimizer_and_scheduler
from omegaconf import OmegaConf
from hf_trainer import KDTrainer

def parse_config(path: str):
    with open(path) as f: return yaml.safe_load(f)

def build_model(cfg):
    base_cfg = AutoConfig.from_pretrained(cfg.model.pretrained_model_name_or_path)
    # -------- your liger / lolcats configs ----------
    if cfg.model.name == "rapid_distill_stage_1_qwen":
        from lolcats.models.rapid_distill_stage_1_qwen import LigerQwen2GLAConfig as LC
        lg_cfg = LC(); lg_cfg.__dict__.update(base_cfg.__dict__)
        base_cfg = lg_cfg
    # -----------------------------------------------
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.pretrained_model_name_or_path,
        config=base_cfg, torch_dtype=torch.bfloat16
    )
    print(base_cfg);                              # sanity
    model.init_student_weights()                  # still there

    # Freeze / un‑freeze
    for name, p in model.named_parameters():
        p.requires_grad = any(k in name for k in ("q_proj_s", "k_proj_s",
                                                  "v_proj_s", "o_proj_s"))

    tr, tot = count_model_params(model, True), count_model_params(model, False)
    print(f"trainable={tr/1e6:.1f} M  | total={tot/1e6:.1f} M ({tr/tot:.2%})")
    return model

def main(cfg):
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.pretrained_model_name_or_path,
                                              padding_side="left")
    tokenizer.pad_token_id = tokenizer.eos_token_id

    model = build_model(cfg)
    dataloaders = load_data(cfg)                  # unchanged
    train_loader, eval_loader = dataloaders["train"], dataloaders["validation"]

    num_gpus = torch.cuda.device_count()
    g_accum = cfg.data.batch_size // (cfg.data.micro_batch_size * num_gpus)
    seq_len  = cfg.model.max_length
    tgt_tok  = cfg.train.target_tokens
    max_steps = (tgt_tok // (cfg.data.batch_size * seq_len)) if tgt_tok else cfg.train.max_steps

    # ---------- DeepSpeed / Accelerate flags ----------
    deepspeed_cfg = os.path.join(os.getcwd(), "ds_config.json")  # see §3
    training_args = TrainingArguments(
        per_device_train_batch_size  = cfg.data.micro_batch_size,
        gradient_accumulation_steps = g_accum,
        num_train_epochs            = cfg.train.epochs or 1e6,
        learning_rate               = cfg.train.lr,
        bf16=True,
        logging_steps               = 10,
        evaluation_strategy         = "steps" if cfg.data.val_set_size > 0 else "no",
        eval_steps                  = 200,
        save_steps                  = 1000,
        save_total_limit            = 3,
        metric_for_best_model       = "loss",
        greater_is_better           = False,
        output_dir                  = cfg.train.output_dir,
        deepspeed                   = deepspeed_cfg,            # NEW
        max_steps                   = max_steps,
        report_to                   = "wandb",
    )

    optim, sched = get_optimizer_and_scheduler(model, cfg, max_steps)
    trainer_cls  = DistillTrainer if "distill" in cfg.model.name or "lolcats_at" in cfg.model.name else FinetuneTrainer

    trainer = trainer_cls(
        model            = model,
        args             = training_args,
        train_dataset    = train_loader.dataset,
        eval_dataset     = eval_loader.dataset if cfg.data.val_set_size > 0 else None,
        data_collator    = train_loader.collate_fn,
        optimizers       = (optim, sched),
        tokenizer        = tokenizer,
        # extra kwargs handled by _BaseTrainer
        mse_factor       = 1.0,
    )

    trainer.train(resume_from_checkpoint=None)
    trainer.save_model(os.path.join(training_args.output_dir, "best"))
    tokenizer.save_pretrained(os.path.join(training_args.output_dir, "best"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True, help="Path to YAML config")
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()
    cfg  = parse_config(args.cfg)
    cfg = OmegaConf.create(cfg)
    main(cfg)
