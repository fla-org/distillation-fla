import argparse, os, yaml, math, torch
import json
import deepspeed
from transformers import (AutoConfig, AutoTokenizer, AutoModelForCausalLM,
                          TrainingArguments)
from omegaconf import OmegaConf

# If you have separate modules:
# from training.dataloader import load_data
# from training.utils import count_model_params, get_optimizer_and_scheduler
# from hf_trainer import DistillTrainer, FinetuneTrainer, KDTrainer
#
# Make sure these imports match your project’s layout.
from training.dataloader import load_data
from training.utils import count_model_params, get_optimizer_and_scheduler
from hf_trainer import DistillTrainer, FinetuneTrainer, KDTrainer


def measure_gpu_memory(model, label="Model"):
    """
    Measures the GPU memory footprint of a given model more accurately.
    """
    if not torch.cuda.is_available():
        print(f"{label}: CUDA not available, cannot measure GPU memory.")
        return

    # 1. Clear the cache BEFORE the measurement to get a clean slate.
    torch.cuda.empty_cache()

    # 2. Get a baseline memory reading.
    mem_before = torch.cuda.memory_allocated() / 1024**2  # in MB

    # 3. Move model to GPU
    model.to("cuda")

    # 4. Get the memory reading AFTER loading the model.
    mem_after = torch.cuda.memory_allocated() / 1024**2  # in MB

    # 5. The model's footprint is the difference.
    model_mem = mem_after - mem_before
    print(f"[{label}] GPU Memory Allocated: {model_mem:,.2f} MB (Total: {mem_after:,.2f} MB)")

    # 6. Move model back to CPU and clear cache for the next measurement.
    model.to("cpu")
    torch.cuda.empty_cache()

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


    # e.g. your custom method for “student init”
    # If you are extending huggingface, you might have a method:
    model.init_student_weights()

    if cfg.stage == 2:
        model.destroy_teacher_weights()

    # Freeze or unfreeze as needed
    for name, p in model.named_parameters():
        # Example: allow Q/K/V projection to be trainable if wanted
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

    # After loading, purify the model by destroying the unneeded teacher weights!!!!
    if hasattr(model, "destroy_teacher_weights"):
        model.destroy_teacher_weights()

    # For Stage 2, all parameters of the student should be trainable.
    for name, p in model.named_parameters():
        p.requires_grad = True

    tr, tot = count_model_params(model, True), count_model_params(model, False)
    print(f"[Stage 2] Purified Student: Trainable = {tr/1e6:.1f}M | Total = {tot/1e6:.1f}M ({tr/tot:.2%})")
    return model


# import gc # Make sure to import the garbage collector at the top of your file

# def build_student_for_stage2(cfg):
#     """
#     Build the stage 2 student by loading the checkpoint from stage 1,
#     then destroying the redundant teacher weights to save memory.
#     """
#     # ... (code to determine model_config) ...
#     base_cfg = AutoConfig.from_pretrained(cfg.model.pretrained_model_name_or_path)
#     if cfg.model.name.startswith("rapid_distill_stage"):
#         from lolcats.models.rapid_distill_stage_1_qwen import LigerQwen2GLAConfig as LC
#         lg_cfg = LC()
#         lg_cfg.__dict__.update(base_cfg.__dict__)
#         model_config = lg_cfg
#     else:
#         model_config = base_cfg
        
#     student_stage1_path = cfg.train.student_init_ckpt

#     print(f"Loading Stage 1 student from: {student_stage1_path}")
#     model = AutoModelForCausalLM.from_pretrained(
#         student_stage1_path,
#         config=model_config,
#         torch_dtype=torch.bfloat16
#     )

#     # ========================================================================
#     # COMPLETE VERIFICATION BLOCK
#     # ========================================================================
#     print("\n" + "="*60)
#     print("VERIFICATION: Checking model state BEFORE destroying teacher weights...")
    
#     # 1. Get initial parameter count
#     tot_before = count_model_params(model, False)
#     print(f"[Before] Total parameters: {tot_before/1e6:,.1f}M")

#     # 2. Check for the existence of a teacher weight attribute
#     q_proj_before = model.model.layers[0].self_attn.q_proj
#     print(f"[Before] `q_proj` attribute is a module: {isinstance(q_proj_before, torch.nn.Module)}")

#     # --- The Destruction Step ---
#     if hasattr(model, "destroy_teacher_weights"):
#         print("\nAttempting to destroy teacher weights...")
#         model.destroy_teacher_weights()
#         # Force garbage collection to reclaim memory
#         gc.collect()
#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()
#     # --------------------------

#     print("\nVERIFICATION: Checking model state AFTER destroying teacher weights...")

#     # 1. Get final parameter count and compare
#     tot_after = count_model_params(model, False)
#     print(f"[After]  Total parameters: {tot_after/1e6:,.1f}M")
#     print(f"--> Reduction of {(tot_before - tot_after) / 1e6:,.1f}M parameters.")

#     # 2. Assert the weight is now None
#     q_proj_after = model.model.layers[0].self_attn.q_proj
#     if q_proj_after is None:
#         print("✅ [After]  `q_proj` attribute is now None. Verification successful.")
#     else:
#         print("❌ [After]  `q_proj` attribute still exists. Verification FAILED.")

#     # 3. Check final GPU memory
#     if torch.cuda.is_available() and cfg.local_rank == 0:
#         torch.cuda.synchronize()
#         mem_after = torch.cuda.memory_allocated() / 1024**2
#         print(f"[After]  GPU Memory Allocated on Rank 0: {mem_after:,.2f} MB")
#         print(f"--> GPU memory freed: {mem_before - mem_after:,.2f} MB.")
    
#     print("="*60 + "\n")
#     # ========================================================================
#     # END OF VERIFICATION BLOCK
#     # ========================================================================

#     # For Stage 2, all parameters of the student should be trainable.
#     for name, p in model.named_parameters():
#         p.requires_grad = True

#     tr, tot = count_model_params(model, True), count_model_params(model, False)
#     print(f"[Stage 2] Purified Student: Trainable = {tr/1e6:.1f}M | Total = {tot/1e6:.1f}M ({tr/tot:.2%})")
#     return model

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


def main(cfg, measure_memory=False):
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

        if measure_memory:
            measure_gpu_memory(model, "Stage 1 Student")

    elif stage == 2:
        print("==== Stage 2 (Logit Distillation) ====")
        # Student: from the checkpoint saved by stage 1
        model = build_student_for_stage2(cfg)
        # Teacher: base model with full attention
        teacher_model = build_teacher_for_stage2(cfg)
        trainer_class = KDTrainer

        if measure_memory:
            measure_gpu_memory(model, "Stage 2 Student")
            # For DeepSpeed-sharded teacher, this will measure the shard on the current device
            measure_gpu_memory(teacher_model, "Teacher Model")
    else:
        raise ValueError(f"Unknown stage: {stage}. Must be 1 or 2.")

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
    ds_config_path = os.path.join(os.getcwd(), "ds_config_2.json")
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
        eval_steps                  = 200,
        save_steps                  = 500,
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

    # 6. Instantiate the trainer
    if stage == 1:
        # Stage 1 trainer
        trainer = trainer_class(
            model          = model,
            args           = training_args,
            train_dataset  = train_loader.dataset,
            eval_dataset   = eval_loader.dataset if cfg.data.val_set_size > 0 else None,
            data_collator  = train_loader.collate_fn,
            optimizers     = (optim, sched),
            tokenizer      = tokenizer,
            mse_factor     = 1.0,  # if DistillTrainer needs it
        )
    else:
        # Stage 2 trainer
        trainer = trainer_class(
            model          = model,
            teacher_model  = teacher_model,
            kl_weight      = cfg.distillation.kl_weight,
            ce_weight      = cfg.distillation.ce_weight,
            args           = training_args,
            train_dataset  = train_loader.dataset,
            eval_dataset   = eval_loader.dataset if cfg.data.val_set_size > 0 else None,
            data_collator  = train_loader.collate_fn,
            optimizers     = (optim, sched),
            tokenizer      = tokenizer,
        )

    # 7. Train
    trainer.train(resume_from_checkpoint=None)

    # 8. Save final model
    best_dir = os.path.join(training_args.output_dir, "best")
    trainer.save_model(best_dir)
    tokenizer.save_pretrained(best_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True, help="Path to YAML config")
    parser.add_argument("--measure_memory", action="store_true", help="Measure GPU memory of models")
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()

    cfg_dict = parse_config(args.cfg)
    cfg = OmegaConf.create(cfg_dict)

    # Make sure your config has something like:
    # train:
    #   stage: 1  (or 2)
    # or pass it another way if you prefer.

    main(cfg, args.measure_memory)