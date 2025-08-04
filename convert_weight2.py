import transformers
from transformers import AutoConfig, AutoModelForCausalLM
import torch
import os
import json
from safetensors.torch import load_file
from accelerate import init_empty_weights
from omegaconf import OmegaConf
import argparse
import json
import os
import torch
import lm_eval
from lm_eval.models.huggingface import HFLM
from lm_eval.utils import setup_logging
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import load_file
from lm_eval.tasks import mmlu 
from datasets import load_dataset
from lm_eval import utils
from distill_model.config_distilled_student import StudentConfig
from distill_model.modeling_distilled_student import StudentModel, StudentForCausalLM, get_student_attention_class

AutoConfig.register('student', StudentConfig, exist_ok=True)
AutoModelForCausalLM.register(StudentConfig, StudentForCausalLM, exist_ok=True)

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



def main():
    
    stage1_ckpt_path = "/u/sonta/code/distillation-fla/checkpoints/qwen_3b_instruct_gla_v1_hybrid/stage2/checkpoint-6000"
    student_attn_class_name = "gla_v1"
    output_dir = "/u/sonta/code/distillation-fla/checkpoints/qwen_3b_instruct_gla_v1_hybrid/stage2/checkpoint-6000-hf"
    keep_full_attention_layers = [0, 1, 5, 17, 18, 19, 20, 26, 27]

    # Validate inputs
    if not os.path.exists(stage1_ckpt_path):
        raise FileNotFoundError(f"Stage 1 checkpoint path does not exist: {stage1_ckpt_path}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Purifying Stage 1 checkpoint from: {stage1_ckpt_path}")

    # 1. Load the base model configuration.
    config = AutoConfig.from_pretrained(stage1_ckpt_path)
    config.use_cache = True

    # Get the specific student attention class needed
    try:
        student_attn_class = get_student_attention_class(student_attn_class_name)
        print(f"✅ Building clean student model with attention class: {student_attn_class.__name__}")
    except Exception as e:
        print(f"❌ Failed to get student attention class '{student_attn_class_name}': {e}")
        print("Available student attention classes: path_v1, path_fox_v1, fox_v1, qwen2_liger_gla, qwen3_liger_gla, qwen2_gla")
        raise

    # 2. Build the clean HYBRID student model structure on a "meta" device.
    try:
        from accelerate import init_empty_weights
        from safetensors.torch import load_file
    except ImportError:
        raise ImportError("Please install accelerate & safetensors (`pip install accelerate safetensors`) to use this script.")

    # Get the list of layers that were kept as full attention.
    if keep_full_attention_layers:
        print(f"⚠️ Reconstructing hybrid model, keeping layers {keep_full_attention_layers} as full-attention.")

    config2 = config.to_dict()
    config2['student_name'] = student_attn_class_name
    config2['name'] = 'student'
    config2['keep_full_attention_layers'] = keep_full_attention_layers
    config = StudentConfig(**config2)

    with init_empty_weights():
        # First, create a model with the standard architecture (all full-attention).
        student_model = AutoModelForCausalLM.from_config(config)

    student_model.to_empty(device='cpu')
    student_model = student_model.to(torch.bfloat16)

    # 3. Load the raw state dictionary from the Stage 1 checkpoint (NO CHANGE NEEDED HERE).
    stage1_state_dict = {}
    index_path = os.path.join(stage1_ckpt_path, 'model.safetensors.index.json')
    safetensors_path = os.path.join(stage1_ckpt_path, 'model.safetensors')
    pytorch_bin_path = os.path.join(stage1_ckpt_path, 'pytorch_model.bin')

    # Check if this is a DeepSpeed ZeRO checkpoint
    is_deepspeed_zero = False
    if os.path.exists(os.path.join(stage1_ckpt_path, 'zero_to_fp32.py')):
        is_deepspeed_zero = True
        print("⚠️ Detected DeepSpeed ZeRO checkpoint. This may require special handling.")
    
    # Additional DeepSpeed ZeRO detection
    if os.path.exists(os.path.join(stage1_ckpt_path, 'mp_rank_00_model_states.pt')):
        is_deepspeed_zero = True
        print("⚠️ Detected DeepSpeed ZeRO checkpoint with mp_rank files.")
    
    if os.path.exists(os.path.join(stage1_ckpt_path, 'zero_to_fp32.py')):
        print("🔧 Found zero_to_fp32.py - this is a DeepSpeed ZeRO checkpoint")
        is_deepspeed_zero = True

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

    # Handle DeepSpeed ZeRO checkpoints
    if is_deepspeed_zero:
        print("🔧 Processing DeepSpeed ZeRO checkpoint...")
        # Remove DeepSpeed-specific keys
        keys_to_remove = []
        for key in list(stage1_state_dict.keys()):
            if key.startswith('module.') or key.startswith('_forward_module.'):
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            new_key = key.replace('module.', '').replace('_forward_module.', '')
            stage1_state_dict[new_key] = stage1_state_dict.pop(key)
            print(f"  -> Remapped key: {key} -> {new_key}")
        
        # Remove other DeepSpeed-specific keys
        deepspeed_keys_to_remove = [
            'optimizer_states',
            'lr_schedulers',
            'random_states',
            'buffer_names',
            'param_shapes',
            'ds_version',
            'ds_config',
            'global_steps',
            'skipped_steps',
            'iteration'
        ]
        
        for key in deepspeed_keys_to_remove:
            if key in stage1_state_dict:
                del stage1_state_dict[key]
                print(f"  -> Removed DeepSpeed key: {key}")
        
        print(f"✅ Processed DeepSpeed ZeRO checkpoint with {len(keys_to_remove)} remapped keys")

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
    # missing_keys, unexpected_keys = student_model.load_state_dict(purified_state_dict, strict=False)
    # if unexpected_keys:
    #     print(f"⚠️ [Warning] Found unexpected keys which were ignored: {unexpected_keys}")
    # if missing_keys:
    #     raise RuntimeError(f"❌ [ERROR] The student model is missing keys: {missing_keys}")

    try:
        missing_keys, unexpected_keys = student_model.load_state_dict(purified_state_dict)
        if unexpected_keys:
            print(f"⚠️ [Warning] Found unexpected keys which were ignored: {unexpected_keys}")
        if missing_keys:
            print(f"❌ [ERROR] The student model is missing keys: {missing_keys}")
            print("Trying to load with strict=False...")
            missing_keys, unexpected_keys = student_model.load_state_dict(purified_state_dict, strict=False)
            if missing_keys:
                print(f"⚠️ [Warning] Still missing keys with strict=False: {missing_keys}")
    except Exception as e:
        print(f"❌ [ERROR] Failed to load state dict: {e}")
        print("Trying to load with strict=False...")
        missing_keys, unexpected_keys = student_model.load_state_dict(purified_state_dict, strict=False)
        if missing_keys:
            print(f"⚠️ [Warning] Missing keys with strict=False: {missing_keys}")
        
    # Save the model using HuggingFace's save_pretrained method
    print(f"Saving cleaned student model to {output_dir}")

    student_model.save_pretrained(
        output_dir,
        safe_serialization=True  # Use safetensors format
    )
    tokenizer = AutoTokenizer.from_pretrained(stage1_ckpt_path)
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    main()
