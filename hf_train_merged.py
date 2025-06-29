import argparse, os, yaml, math, torch
import json
import deepspeed
from transformers import (AutoConfig, AutoTokenizer, AutoModelForCausalLM,
                          TrainingArguments)
from omegaconf import OmegaConf
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
        config=model_config, # Use the CORRECT (custom) config
        torch_dtype=torch.bfloat16
    )

    # For Stage 3, all parameters should be trainable for fine-tuning.
    for name, p in model.named_parameters():
        p.requires_grad = True

    tr, tot = count_model_params(model, True), count_model_params(model, False)
    print(f"[Stage 3] Model Ready for Finetuning: Trainable = {tr/1e6:.1f}M | Total = {tot/1e6:.1f}M ({tr/tot:.2%})")
    
    # # Optional: Verify that the weights are no longer unused.
    # # The warning should disappear, but you can also manually check a parameter.
    # try:
    #     # This should now exist and not be None
    #     _ = model.model.layers[0].self_attn.q_proj_s
    #     print("✅ Verification successful: `q_proj_s` layer exists in the loaded model.")
    # except AttributeError:
    #     print("❌ Verification FAILED: `q_proj_s` layer not found.")

    return model


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
        import torch.nn.functional as F
        print("==== Stage 2 (Logit Distillation) ====")
        # Student: from the checkpoint saved by stage 1
        model = build_student_for_stage2(cfg)
        # Teacher: base model with full attention
        teacher_model = build_teacher_for_stage2(cfg)
        trainer_class = KDTrainer


        # ---> START DEBUG BLOCK <---
        print("\n[DEBUG] Pre‑flight check: per‑layer hidden‑state L2 loss")

        import torch.nn.functional as F
        model.eval()
        teacher_model.eval()

        # Move to GPU (or keep on CPU if that is what you use)
        model.to("cuda")
        teacher_model.to("cuda")

        # 1. Get one mini‑batch
        data_loader = load_data(cfg)["train"]
        single_batch = next(iter(data_loader))
        single_batch = {k: v.to("cuda") for k, v in single_batch.items()
                        if isinstance(v, torch.Tensor)}

        # 2. Forward passes with hidden states
        with torch.no_grad():
            # DeepSpeed engines wrap the real module in `.module`
            teacher_inference = teacher_model.module if hasattr(teacher_model, "module") else teacher_model

            student_out  = model(**single_batch,
                                output_hidden_states=True,
                                use_cache=False)
            teacher_out  = teacher_inference(**single_batch,
                                            output_hidden_states=True,
                                            use_cache=False)

        student_h = student_out.hidden_states      # Tuple [emb + L layers]
        teacher_h = teacher_out.hidden_states

        assert len(student_h) == len(teacher_h), "Mismatch in #layers of hidden states"

        per_layer_l2 = []

        # skip index‑0 (embedding) so we report only transformer layers
        for idx in range(1, len(student_h)):
            diff = (student_h[idx].float() - teacher_h[idx].float()).pow(2).mean().sqrt()
            l2   = diff.item()
            per_layer_l2.append(l2)
            print(f"  Layer {idx:02d}:  RMS‑L2 = {l2:.4f}")

        avg_l2 = sum(per_layer_l2) / len(per_layer_l2)
        print(f"\n[DEBUG] Mean RMS‑L2 across layers: {avg_l2:.4f}")

        # Optional: stop here so you can inspect the numbers
        import sys
        sys.exit("Hidden‑state L2 check complete. Exiting.")
        # ---> END DEBUG BLOCK <---


        # # ---> START DEBUG BLOCK <---
        # print("\n[DEBUG] Performing pre-flight check...")
        # print("[DEBUG] Setting model to EVAL mode to match evaluation script...")
        # model.eval()
        # model.to('cuda')
        # teacher_model.to('cuda')
        # # 1. Get a single batch of data
        # data_loader = load_data(cfg)["train"]
        # single_batch = next(iter(data_loader))
        # single_batch = {k: v.to('cuda') for k, v in single_batch.items() if isinstance(v, torch.Tensor)}

        # # Define an input dictionary that mimics lm-eval-harness (NO 'labels')
        # inference_inputs = {
        #     "input_ids": single_batch["input_ids"],
        #     "attention_mask": single_batch["attention_mask"]
        # }

        # # 2. Run all model variations
        # with torch.no_grad():
        #     print("[DEBUG] Getting teacher logits...")
        #     # Use the raw model if wrapped by DeepSpeed
        #     teacher_for_inference = teacher_model.module if hasattr(teacher_model, 'module') else teacher_model
        #     teacher_logits = teacher_for_inference(**single_batch).logits.float()

        #     print("[DEBUG] Getting student logits (with 'labels' passed)...")
        #     student_logits_with_labels = model(**single_batch).logits.float()

        #     print("[DEBUG] Getting student logits (NO 'labels' passed)...")
        #     student_logits_no_labels = model(**inference_inputs).logits.float()

        # # 3. Define a KL loss function for comparison
        # def calculate_kl(student_logits, teacher_logits):
        #     log_softmax_student = F.log_softmax(student_logits, dim=-1)
        #     softmax_teacher = F.softmax(teacher_logits, dim=-1)
        #     return F.kl_div(log_softmax_student, softmax_teacher, reduction='batchmean')

        # kl_loss_with_labels = calculate_kl(student_logits_with_labels, teacher_logits)
        # kl_loss_no_labels = calculate_kl(student_logits_no_labels, teacher_logits)

        # # 4. Inspect the outputs
        # print("\n--- DEBUG RESULTS ---")
        # print(f"KL Divergence (Student vs. Teacher) when 'labels' is PASSED:    {kl_loss_with_labels.item():.4f}")
        # print(f"KL Divergence (Student vs. Teacher) when 'labels' is NOT PASSED: {kl_loss_no_labels.item():.4f}")

        # # This checks if the two student forward passes produced the same result
        # logit_diff = torch.mean(torch.abs(student_logits_with_labels - student_logits_no_labels))
        # print(f"Mean absolute difference between student's own logits: {logit_diff.item():.6f}")
        # print("---------------------\n")

        # if logit_diff > 1e-4:
        #     print("[CONCLUSION] HYPOTHESIS CONFIRMED: The model's forward pass behaves differently based on the 'labels' argument.")
        #     print("The high KL loss is real, but your evaluation script was triggering a different, teacher-like code path.")
        # else:
        #     print("[CONCLUSION] HYPOTHESIS REJECTED: The 'labels' argument makes no difference. The mystery remains.")

        # # Exit after the check to avoid running a full training
        # import sys
        # sys.exit("Pre-flight check complete. Exiting.")
        # # ---> END DEBUG BLOCK <---

        if measure_memory:
            measure_gpu_memory(model, "Stage 2 Student")
            # For DeepSpeed-sharded teacher, this will measure the shard on the current device
            measure_gpu_memory(teacher_model, "Teacher Model")

    elif stage == 3:
        print("==== Stage 3 (Long-Context Finetuning) ====")
        # Student is the checkpoint saved by stage 2
        model = build_model_for_stage3(cfg)
        # No teacher model in stage 3
        teacher_model = None
        # Use the standard fine-tuning trainer
        trainer_class = FinetuneTrainer
        if measure_memory:
            measure_gpu_memory(model, "Stage 3 Model")

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
        eval_steps                  = 50,
        save_steps                  = 50,
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
