import argparse, os, yaml, math, torch
import json
import deepspeed
from transformers import (AutoConfig, AutoTokenizer, AutoModelForCausalLM,
                          TrainingArguments)
from omegaconf import OmegaConf
from training.dataloader import load_data
from training.utils import count_model_params, get_optimizer_and_scheduler
from hf_trainer import DistillTrainer, FinetuneTrainer, KDTrainer

from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl


def parse_config(path: str):
    with open(path) as f:
        return yaml.safe_load(f)


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


def build_student_for_stage1(cfg):
    """
    Build and partially freeze the student for stage 1 (attention distillation).
    Typically we load from the base model and selectively unfreeze Q/K/V or
    additional trainable layers.
    """
    base_cfg = AutoConfig.from_pretrained(cfg.model.pretrained_model_name_or_path)

    from lolcats.models.rapid_distill_stage_1_qwen import LigerQwen2GLAConfig as LC
    lg_cfg = LC()
    lg_cfg.__dict__.update(base_cfg.__dict__)
    base_cfg = lg_cfg

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.pretrained_model_name_or_path,
        config=base_cfg,
        torch_dtype=torch.bfloat16
    )

    model.init_student_weights()

    if cfg.stage == 2:
        model.destroy_teacher_weights()

    # Freeze or unfreeze as needed
    for name, p in model.named_parameters():
        # allow Q/K/V projection to be trainable
        p.requires_grad = any(k in name for k in ("q_proj_s", "k_proj_s", "v_proj_s", "o_proj_s"))

    tr, tot = count_model_params(model, True), count_model_params(model, False)
    print(f"Trainable = {tr/1e6:.1f}M | Total = {tot/1e6:.1f}M ({tr/tot:.2%})")
    return model


def build_student_for_stage2(cfg):
    """
    Build the stage 2 student by loading the checkpoint from stage 1,
    then destroying the redundant teacher weights to save memory.
    """
    # CRITICAL: Load the custom config so AutoModel knows which class to use.
    # We use the base model's config and update our custom one, just like in stage 1.
    base_cfg = AutoConfig.from_pretrained(cfg.model.pretrained_model_name_or_path)
    if cfg.model.name.startswith("rapid_distill_stage"): # Make this check more general
        # TODO: change name
        from lolcats.models.rapid_distill_stage_1_qwen import LigerQwen2GLAConfig as LC
        lg_cfg = LC()
        lg_cfg.__dict__.update(base_cfg.__dict__)
        model_config = lg_cfg
    else:
        model_config = base_cfg
        
    student_stage1_path = cfg.train.student_init_ckpt # Use the correct key from your YAML

    print(f"Loading Stage 1 student from: {student_stage1_path}")
    model = AutoModelForCausalLM.from_pretrained(
        student_stage1_path,
        config=model_config, # Use the custom config
        torch_dtype=torch.bfloat16
    )
    print(f"[DEBUG] Model loaded in training script is of type: {type(model)}")

    # After loading, purify the model by destroying the unneeded teacher weights!!!!
    if hasattr(model, "destroy_teacher_weights"):
        model.destroy_teacher_weights()

    # For Stage 2, all parameters of the student should be trainable.
    for name, p in model.named_parameters():
        p.requires_grad = True

    tr, tot = count_model_params(model, True), count_model_params(model, False)
    print(f"[Stage 2] Purified Student: Trainable = {tr/1e6:.1f}M | Total = {tot/1e6:.1f}M ({tr/tot:.2%})")
    return model


def build_teacher_for_stage2(cfg):
    """
    Teacher is the base model with full attention. If you want to
    DeepSpeed-shard it, do so. Otherwise, just load it normally.
    """
    teacher_config = AutoConfig.from_pretrained(cfg.distillation.teacher_model)
    teacher_model = AutoModelForCausalLM.from_pretrained(
        cfg.distillation.teacher_model,
        config=teacher_config,
        torch_dtype=torch.bfloat16
    )
    teacher_model.eval()

    # If you need DS-sharding for the teacher:
    if hasattr(cfg.distillation, "teacher_deepspeed_cfg"):
        teacher_ds_cfg = cfg.distillation.teacher_deepspeed_cfg
        teacher_model = _prepare_teacher_deepspeed(teacher_model, teacher_ds_cfg)

    return teacher_model


def build_model_for_stage3(cfg):
    """
    Build the student for Stage 3 by loading the final checkpoint from Stage 2.
    This model is already purified and ready for fine-tuning.
    """
    stage2_ckpt_path = cfg.train.student_init_ckpt
    print(f"Loading Stage 2 model from: {stage2_ckpt_path}")

    base_cfg = AutoConfig.from_pretrained(cfg.model.pretrained_model_name_or_path)

    from lolcats.models.rapid_distill_stage_1_qwen import LigerQwen2GLAConfig as LC
    lg_cfg = LC()
    lg_cfg.__dict__.update(base_cfg.__dict__)
    model_config = lg_cfg

    model = AutoModelForCausalLM.from_pretrained(
        stage2_ckpt_path,
        config=model_config,
        torch_dtype=torch.bfloat16
    )

    # After loading, purify the model by destroying the unneeded teacher weights!!!!
    if hasattr(model, "destroy_teacher_weights"):
        model.destroy_teacher_weights()

    # For Stage 3, all parameters should be trainable for fine-tuning.
    for name, p in model.named_parameters():
        p.requires_grad = True

    tr, tot = count_model_params(model, True), count_model_params(model, False)
    print(f"[Stage 3] Model Ready for Finetuning: Trainable = {tr/1e6:.1f}M | Total = {tot/1e6:.1f}M ({tr/tot:.2%})")
    
    # Optional: Verify that the weights are no longer unused.
    try:
        # This should now exist and not be None
        _ = model.model.layers[0].self_attn.q_proj_s
        print("✅ Verification successful: `q_proj_s` layer exists in the loaded model.")
    except AttributeError:
        print("❌ Verification FAILED: `q_proj_s` layer not found.")

    return model


def main(cfg):
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.pretrained_model_name_or_path,
        padding_side="left"
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # -------------------------------------------------------------------
    # 1. Determine stage
    #    (We assume user sets cfg.stage = 1, 2, 3)
    # -------------------------------------------------------------------
    stage = cfg.stage

    if stage == 1:
        print("==== Stage 1 (Attention Transfer) ====")
        # Student: from base model (with partial freeze or attention replacements)
        model = build_student_for_stage1(cfg)
        teacher_model = None       # No teacher in stage 1 or let's rely on DistillTrainer’s attention distill
        trainer_class = DistillTrainer if "distill" in cfg.model.name else FinetuneTrainer

    elif stage == 2:
        import torch.nn.functional as F
        print("==== Stage 2 (Logit Distillation) ====")
        # Student: from the checkpoint saved by stage 1
        model = build_student_for_stage2(cfg)
        # Teacher: base model with full attention
        teacher_model = build_teacher_for_stage2(cfg)
        trainer_class = KDTrainer

        ds_config_path = os.path.join(os.getcwd(), "ds_config_2.json")

    elif stage == 3:
        print("==== Stage 3 (Long-Context Finetuning) ====")
        # Student is the checkpoint saved by stage 2
        model = build_model_for_stage3(cfg)
        # No teacher model in stage 3
        teacher_model = None
        # Use the standard fine-tuning trainer
        trainer_class = FinetuneTrainer

        ds_config_path = os.path.join(os.getcwd(), "ds_config_3.json")

    else:
        raise ValueError(f"Unknown stage: {stage}. Must be 1, 2, or 3.")

    # 2. Build the dataloaders
    dataloaders = load_data(cfg)
    train_loader, eval_loader = dataloaders["train"], dataloaders["validation"]

    # 3. Figure out steps or epochs
    num_gpus  = torch.cuda.device_count()
    g_accum   = cfg.data.batch_size // (cfg.data.micro_batch_size * num_gpus)
    seq_len   = cfg.model.max_length
    tgt_tok   = cfg.train.target_tokens
    max_steps = (tgt_tok // (cfg.data.batch_size * seq_len)) if tgt_tok else cfg.train.max_steps

    # 4. Prepare the student's DS config / TrainingArguments
    
    if os.path.exists(ds_config_path):
        print(f"Using DS config = {ds_config_path}")
    else:
        ds_config_path = None

    training_args = TrainingArguments(
        per_device_train_batch_size = cfg.data.micro_batch_size,
        gradient_accumulation_steps = g_accum,
        num_train_epochs            = cfg.train.epochs or 1e6,
        learning_rate               = cfg.train.lr,
        bf16                        = True,
        logging_steps               = 10,
        evaluation_strategy         = "steps" if cfg.data.val_set_size > 0 else "no",
        eval_steps                  = 50,
        save_steps                  = 200,
        save_total_limit            = 10000,
        metric_for_best_model       = "loss",
        greater_is_better           = False,
        output_dir                  = cfg.train.output_dir,
        deepspeed                   = ds_config_path,
        max_steps                   = max_steps,
        report_to                   = "wandb",
    )

    # 5. Optimizer & scheduler
    optim, sched = get_optimizer_and_scheduler(model, cfg, max_steps)

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_loader.dataset,
        "eval_dataset": eval_loader.dataset if cfg.data.val_set_size > 0 else None,
        "data_collator": train_loader.collate_fn,
        "optimizers": (optim, sched),
        "tokenizer": tokenizer,
    }

    if stage == 1:
        trainer_kwargs["mse_factor"] = 1.0 # Or from cfg
        trainer = DistillTrainer(**trainer_kwargs)
    elif stage == 2:
        trainer_kwargs["teacher_model"] = teacher_model
        trainer_kwargs["kl_weight"] = cfg.distillation.kl_weight
        trainer_kwargs["ce_weight"] = cfg.distillation.ce_weight
        trainer = KDTrainer(**trainer_kwargs)
    elif stage == 3:
        # FinetuneTrainer takes no extra args from this list
        trainer = FinetuneTrainer(**trainer_kwargs)

    # 7. Train
    trainer.train(resume_from_checkpoint=None)

    # 8. Save final model
    best_dir = os.path.join(training_args.output_dir, "best")
    trainer.save_model(best_dir)
    tokenizer.save_pretrained(best_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True, help="Path to YAML config")
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()

    cfg_dict = parse_config(args.cfg)
    cfg = OmegaConf.create(cfg_dict)

    # Make sure your config has something like:
    # train:
    #   stage: 1  (1, 2 or 3)
    # or pass it another way if you prefer.

    main(cfg)
