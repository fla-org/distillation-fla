
import argparse, os, yaml, math, torch
from transformers import (AutoConfig, AutoTokenizer, AutoModelForCausalLM,
                          TrainingArguments)
from training.dataloader import load_data
from training.utils import count_model_params, get_optimizer_and_scheduler
from omegaconf import OmegaConf

# For Stage 2
from hf_trainer import KDTrainer   # or wherever you've placed it

def parse_config(path: str):
    with open(path) as f:
        return yaml.safe_load(f)

def build_model(cfg):
    base_cfg = AutoConfig.from_pretrained(cfg.model.pretrained_model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.pretrained_model_name_or_path,
        config=base_cfg, torch_dtype=torch.bfloat16
    )
    # freeze or load partial adapters as needed
    for name, p in model.named_parameters():
        p.requires_grad = True  # or partial freeze, if desired
    return model

def main(cfg):
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.pretrained_model_name_or_path, padding_side="left")
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # 1. Build the student:
    model = build_model(cfg)

    # 2. Build or load teacher if in kd mode:
    teacher_model = None
    if "distillation" in cfg and cfg.distillation.teacher_model:
        teacher_config = AutoConfig.from_pretrained(cfg.distillation.teacher_model)
        teacher_model  = AutoModelForCausalLM.from_pretrained(
            cfg.distillation.teacher_model, 
            config=teacher_config,
            torch_dtype=torch.bfloat16
        )
        # put teacher in eval, optionally freeze
        teacher_model.eval()
        for param in teacher_model.parameters():
            param.requires_grad = False
    # TODO:huggingface Trainer can only warp the model passed in
    # teacher_model now does not have deepspeed support
    if teacher_model is not None and torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device = torch.device(f"cuda:{local_rank}")
        teacher_model = teacher_model.to(device)


    # 3. Prepare Dataloaders
    dataloaders = load_data(cfg)
    train_loader, eval_loader = dataloaders["train"], dataloaders["validation"]

    # 4. Figure out max_steps from tokens
    num_gpus = torch.cuda.device_count()
    g_accum  = cfg.data.batch_size // (cfg.data.micro_batch_size * num_gpus)
    seq_len  = cfg.model.max_length
    tgt_tok  = cfg.train.target_tokens
    max_steps = (tgt_tok // (cfg.data.batch_size * seq_len)) if tgt_tok else cfg.train.max_steps

    # 5. Training arguments
    deepspeed_cfg = os.path.join(os.getcwd(), "ds_config_2.json")
    training_args = TrainingArguments(
        per_device_train_batch_size  = cfg.data.micro_batch_size,
        gradient_accumulation_steps  = g_accum,
        num_train_epochs             = cfg.train.epochs or 1e6,
        learning_rate                = cfg.train.lr,
        bf16                         = True,
        logging_steps                = 10,
        evaluation_strategy          = "steps" if cfg.data.val_set_size > 0 else "no",
        eval_steps                   = 200,
        save_steps                   = 1000,
        save_total_limit             = 3,
        metric_for_best_model        = "loss",
        greater_is_better            = False,
        output_dir                   = cfg.train.output_dir,
        deepspeed                    = deepspeed_cfg,
        max_steps                    = max_steps,
        report_to                    = "wandb",
    )

    # 6. Build optimizer & scheduler
    optim, sched = get_optimizer_and_scheduler(model, cfg, max_steps)

    # 7. Select trainer type
    if "distill" in cfg.model.name:
        trainer_cls = KDTrainer
    else:
        # e.g., do your stage1 DistillTrainer or FinetuneTrainer here
        from hf_trainer import DistillTrainer
        trainer_cls = DistillTrainer

    # 8. Instantiate trainer
    trainer = trainer_cls(
        model           = model,
        teacher_model   = teacher_model,
        kl_weight       = cfg.distillation.kl_weight,
        ce_weight       = cfg.distillation.ce_weight,
        args            = training_args,
        train_dataset   = train_loader.dataset,
        eval_dataset    = eval_loader.dataset if cfg.data.val_set_size > 0 else None,
        data_collator   = train_loader.collate_fn,
        optimizers      = (optim, sched),
        tokenizer       = tokenizer,
    )

    # 9. Run training
    trainer.train(resume_from_checkpoint=None)

    # 10. Save final best
    trainer.save_model(os.path.join(training_args.output_dir, "best"))
    tokenizer.save_pretrained(os.path.join(training_args.output_dir, "best"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True, help="Path to YAML config")
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()
    cfg  = parse_config(args.cfg)
    cfg  = OmegaConf.create(cfg)
    main(cfg)