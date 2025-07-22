# hf_train_merged_copy.py
import argparse, os, yaml, math, torch, importlib
import json
import deepspeed
from transformers import (AutoConfig, AutoTokenizer, AutoModelForCausalLM,
                          TrainingArguments)
from omegaconf import OmegaConf
from training.dataloader import load_data
from training.utils import count_model_params, get_optimizer_and_scheduler
from hf_trainer_copy import DistillTrainer, FinetuneTrainer, KDTrainer

from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from student_only_attention import LigerQwen3GatedLinearAttentionStudent, LigerQwen2GatedLinearAttentionStudent
from wrapper import AttentionDistillationWrapper

def parse_config(path: str):
    with open(path) as f:
        return yaml.safe_load(f)

def get_student_attention_class(model_name: str):
    """
    Dynamically imports and returns the correct student attention class
    based on the model name from the config.
    """
    # Map model names to their student attention class import paths
    STUDENT_ATTENTION_MAP = {
        "qwen2_liger_gla": "student_only_attention.LigerQwen2GatedLinearAttentionStudent",
        "qwen3_liger_gla": "student_only_attention.LigerQwen3GatedLinearAttentionStudent",
        "qwen2_gla": "student_only_attention.Qwen2GatedLinearAttentionStudent",
        # Add other attention layers for future models here
        # "new_model_attention_type": "path.to.new.AttentionStudent"
    }

    if model_name not in STUDENT_ATTENTION_MAP:
        raise ValueError(f"Unknown student attention for model name: {model_name}. Please add it to STUDENT_ATTENTION_MAP.")

    # Dynamically import the module and get the class
    module_path, class_name = STUDENT_ATTENTION_MAP[model_name].rsplit('.', 1)
    module = importlib.import_module(module_path)
    attention_class = getattr(module, class_name)
    
    return attention_class


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
    Replace `layer.self_attn` with a wrapper so the teacher’s
    hidden states still drive the rest of the frozen network.

    This version is MODIFIED to keep specified layers as full-attention.
    """
    # Get the correct student attention class dynamically
    student_attn_class = get_student_attention_class(cfg.model.name)
    print(f"✅ Using student attention class: {student_attn_class.__name__}")

    # Get the list of layers to keep as full attention from the config.
    # Default to an empty list if not specified.
    keep_full_attention_layers = cfg.model.get('keep_full_attention_layers', [])
    if keep_full_attention_layers:
        print(f"⚠️ Will keep the following layers as full-attention: {keep_full_attention_layers}")

    for idx, layer in enumerate(model.model.layers):
        # Conditionally skip patching if the layer index is in our keep list.
        if idx in keep_full_attention_layers:
            print(f"  -> Skipping layer {idx}, keeping as full-attention.")
            # Ensure the kept layer is frozen, as it's not being trained in Stage 1.
            for param in layer.self_attn.parameters():
                param.requires_grad_(False)
            continue

        # The existing logic now only runs for layers NOT in the keep list.
        print(f"  -> Patching layer {idx} with student attention wrapper.")
        teacher_attn = layer.self_attn
        wrapper = AttentionDistillationWrapper(
            teacher_attn,
            student_attn_class,
            base_model_cfg,
            idx
        )
        layer.self_attn = wrapper

def build_student_for_stage1(cfg):
    """
    Build and partially freeze the student for stage 1 (attention distillation).
    Typically we load from the base model and selectively unfreeze Q/K/V or
    additional trainable layers.
    """
    base_model_cfg = AutoConfig.from_pretrained(cfg.model.pretrained_model_name_or_path)

    # build the base model first
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.pretrained_model_name_or_path,
        config=base_model_cfg,
        torch_dtype=torch.bfloat16,
    )

    # patch each layer with (teacher → wrapper → student)
    patch_model_for_stage1(model, base_model_cfg, cfg)

    # freeze everything that is NOT inside .student_attn.
    for name, p in model.named_parameters():
        p.requires_grad_( ".student_attn." in name )

    tr, tot = count_model_params(model, True), count_model_params(model, False)
    print(f"Trainable = {tr/1e6:.1f}M | Total = {tot/1e6:.1f}M ({tr/tot:.2%})")
    return model

def build_student_for_stage2(cfg):
    """
    Build the stage 2 student by loading the checkpoint from stage 1,
    purifying it by removing the teacher wrapper, and preparing it for
    knowledge distillation.

    This version is to handle hybrid models with both student
    and full-attention layers.
    """
    stage1_ckpt_path = cfg.train.student_init_ckpt # Path to Stage 1 output
    print(f"Purifying Stage 1 checkpoint from: {stage1_ckpt_path}")

    # 1. Load the base model configuration.
    config = AutoConfig.from_pretrained(stage1_ckpt_path, trust_remote_code=True)
    
    # Get the specific student attention class needed
    student_attn_class = get_student_attention_class(cfg.model.name)
    print(f"✅ Building clean student model with attention class: {student_attn_class.__name__}")

    # 2. Build the clean HYBRID student model structure on a "meta" device.
    try:
        from accelerate import init_empty_weights
        from safetensors.torch import load_file
    except ImportError:
        raise ImportError("Please install accelerate & safetensors (`pip install accelerate safetensors`) to use this script.")
    
    # Get the list of layers that were kept as full attention.
    keep_full_attention_layers = cfg.model.get('keep_full_attention_layers', [])
    if keep_full_attention_layers:
        print(f"⚠️ Reconstructing hybrid model, keeping layers {keep_full_attention_layers} as full-attention.")

    with init_empty_weights():
        # First, create a model with the standard architecture (all full-attention).
        student_model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
        
        # --- MODIFICATION START ---
        # Now, iterate and replace layers that are NOT in the keep list.
        for idx, layer in enumerate(student_model.model.layers):
            if idx in keep_full_attention_layers:
                print(f"  -> Layer {idx} remains full-attention.")
                continue
            else:
                # This layer should be the student attention type.
                print(f"  -> Layer {idx} becomes {student_attn_class.__name__}.")
                layer.self_attn = student_attn_class(config, idx)
        # --- MODIFICATION END ---

    student_model.to_empty(device='cpu')
    student_model = student_model.to(torch.bfloat16)

    # 3. Load the raw state dictionary from the Stage 1 checkpoint (NO CHANGE NEEDED HERE).
    stage1_state_dict = {}
    index_path = os.path.join(stage1_ckpt_path, 'model.safetensors.index.json')
    safetensors_path = os.path.join(stage1_ckpt_path, 'model.safetensors')
    pytorch_bin_path = os.path.join(stage1_ckpt_path, 'pytorch_model.bin')

    if os.path.exists(index_path):
        print("Detected sharded safetensors checkpoint.")
        with open(index_path, 'r') as f:
            index = json.load(f)
        shard_files = set(index['weight_map'].values())
        for shard_file in shard_files:
            shard_path = os.path.join(stage1_ckpt_path, shard_file)
            stage1_state_dict.update(load_file(shard_path, device="cpu"))
    elif os.path.exists(safetensors_path):
        stage1_state_dict = load_file(safetensors_path, device="cpu")
    elif os.path.exists(pytorch_bin_path):
        stage1_state_dict = torch.load(pytorch_bin_path, map_location="cpu")
    else:
        raise FileNotFoundError(f"Could not find model weights in {stage1_ckpt_path}")

    # 4. Remap weights to the clean student structure (NO CHANGE NEEDED HERE).
    # This logic is robust. It correctly handles both remapped student weights
    # and passthrough full-attention weights.
    purified_state_dict = {}
    for key, value in stage1_state_dict.items():
        if ".student_attn." in key:
            # This handles the layers that were converted.
            new_key = key.replace(".student_attn", "")
            purified_state_dict[new_key] = value
        elif ".teacher_attn" not in key:
            # This handles MLPs, norms, embeddings, AND the full-attention layers.
            purified_state_dict[key] = value

    # 5. Load the remapped weights into the clean student model (NO CHANGE NEEDED HERE).
    missing_keys, unexpected_keys = student_model.load_state_dict(purified_state_dict, strict=False)
    if unexpected_keys:
        print(f"⚠️ [Warning] Found unexpected keys which were ignored: {unexpected_keys}")
    if missing_keys:
        raise RuntimeError(f"❌ [ERROR] The student model is missing keys: {missing_keys}")

    print("✅ Stage 1 hybrid model successfully purified for Stage 2 training.")

    # 6. For Stage 2, all parameters of the student should be trainable (NO CHANGE NEEDED HERE).
    for name, p in student_model.named_parameters():
        p.requires_grad = True

    tr, tot = count_model_params(student_model, True), count_model_params(student_model, False)
    print(f"[Stage 2] Purified Student: Trainable = {tr/1e6:.1f}M | Total = {tot/1e6:.1f}M ({tr/tot:.2%})")

    return student_model

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
    stage2_ckpt_path = cfg.train.student_init_ckpt # Path to Stage 2 output
    print(f"Loading Stage 2 model from: {stage2_ckpt_path}")

    # Since the Stage 2 model is clean, we can load it directly.
    # The config saved with the model already knows the correct architecture.
    model = AutoModelForCausalLM.from_pretrained(
        stage2_ckpt_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True # Good practice for custom models
    )

    # For Stage 3, all parameters should be trainable for fine-tuning.
    for name, p in model.named_parameters():
        p.requires_grad = True

    tr, tot = count_model_params(model, True), count_model_params(model, False)
    print(f"[Stage 3] Model Ready for Finetuning: Trainable = {tr/1e6:.1f}M | Total = {tot/tot:.2%})")
    
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
        # Student: from base model
        model = build_student_for_stage1(cfg)
        trainer_class = DistillTrainer
        ds_config_path = os.path.join(os.getcwd(), "ds_config_1.json")

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
        eval_strategy               = "steps" if cfg.data.val_set_size > 0 else "no",
        eval_steps                  = 50,
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
    if cfg.train.resume_from_checkpoint == "None":
        trainer.train(resume_from_checkpoint=None)
    else:
        trainer.train(resume_from_checkpoint=cfg.train.resume_from_checkpoint)

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
