import argparse
import os
import json
import random
import re
import copy
import logging
from typing import Callable

import torch
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from english_words import get_english_words_set

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --------------------------------------------------------------------------
# ## Step 1: Dataset Generation
# All dataset creation functions follow a similar pattern but produce
# different prompts and answers based on the task.
# --------------------------------------------------------------------------

def create_retrieval_dataset(num_samples: int, num_pairs: int, output_dir: str, **kwargs) -> list:
    """
    Generates a synthetic key-value retrieval dataset. `kwargs` is accepted for compatibility
    with the generic experiment runner but is not used.
    """
    dataset_path = os.path.join(output_dir, 'dataset', f'retrieval_task_{num_pairs}.json')
    if os.path.exists(dataset_path):
        logging.info(f"Dataset already exists at {dataset_path}. Loading it.")
        with open(dataset_path, 'r') as f:
            return json.load(f)

    word_source = list(get_english_words_set(['web2']))
    logging.info(f"Creating a new retrieval dataset with {num_samples} samples, each with {num_pairs} pairs.")
    dataset = []
    for _ in range(num_samples):
        keys = random.sample(word_source, num_pairs)
        values = [random.randint(0, 100) for _ in range(num_pairs)]
        data_dict = dict(zip(keys, values))
        query_key = random.choice(keys)
        correct_value = data_dict[query_key]
        dict_str = "\n".join([f"{k}:{v}" for k, v in data_dict.items()])
        prompt = f"Memorize the following dictionary:\n{dict_str}\nThe value of the key '{query_key}' is"
        dataset.append({"prompt": prompt, "answer": str(correct_value)})

    os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
    with open(dataset_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2)
    logging.info(f"Dataset saved to {dataset_path}")
    return dataset

def create_ar_dataset(num_samples: int, num_pairs: int, output_dir: str, tokenizer=None, **kwargs) -> list:
    """
    Generates a synthetic associative recall (AR) mathematical reasoning dataset.
    Requires a tokenizer to generate keys from its vocabulary.
    """
    if tokenizer is None:
        raise ValueError("The 'associative_recall' task requires a tokenizer to be provided for dataset creation.")
    
    safe_model_name = tokenizer.name_or_path.replace("/", "_")
    dataset_path = os.path.join(output_dir, 'dataset', f'ar_task_{safe_model_name}_{num_pairs}.json')
    if os.path.exists(dataset_path):
        logging.info(f"AR Dataset already exists at {dataset_path}. Loading it.")
        with open(dataset_path, 'r') as f:
            return json.load(f)

    logging.info("Generating a word source from the tokenizer's vocabulary...")
    vocab = tokenizer.get_vocab()
    word_source = [tokenizer.decode([token_id]).strip() for token_id in vocab.values() if re.fullmatch(r'[a-zA-Z]{3,}', tokenizer.decode([token_id]).strip())]
    logging.info(f"Found {len(word_source)} suitable words in vocabulary.")
    if len(word_source) < num_pairs:
        raise ValueError(f"Not enough suitable words in tokenizer vocab ({len(word_source)}) to create pairs ({num_pairs}).")

    logging.info(f"Creating a new AR dataset with {num_samples} samples, each with {num_pairs} pairs.")
    dataset = []
    for _ in tqdm(range(num_samples), desc="Generating AR Dataset"):
        keys = random.sample(word_source, num_pairs)
        values = [random.randint(0, 9) for _ in range(num_pairs)]
        data_dict = dict(zip(keys, values))
        query_keys = random.sample(keys, 3)
        correct_value = sum(data_dict[k] for k in query_keys)
        dict_str = "\n".join([f"{k}: {v}" for k, v in data_dict.items()])
        prompt = (f"Given the following key-value pairs:\n{dict_str}\n"
                  f"What is the value of {query_keys[0]} + {query_keys[1]} + {query_keys[2]}?")
        dataset.append({"prompt": prompt, "answer": str(correct_value)})

    os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
    with open(dataset_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2)
    logging.info(f"AR Dataset saved to {dataset_path}")
    return dataset

# --------------------------------------------------------------------------
# ## Step 2: Evaluation and Ablation Logic (Shared Helpers)
# --------------------------------------------------------------------------
def parse_generated_text(text: str) -> str:
    """Extracts the first integer from the model's generated text using regex."""
    match = re.search(r'\d+', text)
    return match.group(0) if match else ""

def evaluate(model, tokenizer, dataset: list, task_name: str, device: torch.device, batch_size: int = 8) -> float:
    """Evaluates the model's performance on a given task using Exact Match accuracy."""
    model.eval()
    correct = 0
    with torch.no_grad():
        for i in tqdm(range(0, len(dataset), batch_size), desc="Evaluating Model in Batches"):
            batch = dataset[i:i + batch_size]
            prompts = [item['prompt'] for item in batch]
            expected_answers = [item['answer'] for item in batch]
            inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)
            outputs = model.generate(**inputs, max_new_tokens=100, pad_token_id=tokenizer.eos_token_id)
            generated_ids = outputs[:, inputs['input_ids'].shape[1]:]
            generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            for j, text in enumerate(generated_texts):
                if task_name == "synthetic_retrieval":
                    if parse_generated_text(text) == expected_answers[j]:
                        correct += 1
                elif task_name == "associative_recall":
                    if expected_answers[j] in text:
                        correct += 1
    return correct / len(dataset)

def zero_out_attention_hook(module, input, output):
    """A forward hook that zeroes out the attention output tensor."""
    output[0].zero_()
    return output

# --------------------------------------------------------------------------
# ## Main Experiment Runner (Refactored)
# --------------------------------------------------------------------------
def run_experiment(args: argparse.Namespace, task_config: dict):
    """
    Runs a full, generic experiment including dataset creation, model loading,
    baseline evaluation, layer ablation, and result visualization.

    Args:
        args (argparse.Namespace): The command-line arguments.
        task_config (dict): A dictionary containing task-specific configurations.
    """
    task_name = args.task
    task_display_name = task_name.replace('_', ' ').title()
    logging.info(f"--- Starting Task: {task_display_name} ---")

    # --- Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)
    for sub_dir in ['dataset', 'results', 'plots']:
        os.makedirs(os.path.join(args.output_dir, sub_dir), exist_ok=True)

    safe_model_name = args.model_name.replace("/", "_")
    experiment_suffix = f'{safe_model_name}_{task_name}_pairs_{args.num_pairs}'

    # --- Load Tokenizer and Model ---
    logging.info(f"Loading tokenizer and model for '{args.model_name}'")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, padding_side='left')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16).to(device)
    
    # --- Create Dataset using the task-specific function ---
    dataset_creator = task_config['creator']
    dataset = dataset_creator(args.num_samples, args.num_pairs, args.output_dir, tokenizer=tokenizer)

    # --- Baseline Evaluation ---
    logging.info(f"Evaluating baseline performance for {task_display_name} task...")
    baseline_accuracy = evaluate(model, tokenizer, dataset, task_name, device, args.batch_size)
    logging.info(f"✅ {task_display_name} Baseline Accuracy: {baseline_accuracy:.4f}")
    baseline_path = os.path.join(args.output_dir, 'results', f'baseline_{experiment_suffix}.json')
    with open(baseline_path, 'w') as f:
        json.dump({"model_name": args.model_name, "task": task_name, "baseline_accuracy": baseline_accuracy, "num_pairs": args.num_pairs}, f)

    # --- Layer Ablation ---
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        layers = model.transformer.h
        get_attn = lambda m, i: m.transformer.h[i].attn
    elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
        layers = model.model.layers
        get_attn = lambda m, i: m.model.layers[i].self_attn
    else:
        logging.error(f"Cannot determine layer path for model '{args.model_name}'.")
        return

    num_layers = len(layers)
    logging.info(f"Found {num_layers} layers. Starting layer-wise ablation...")
    layer_performance = []
    for i in range(num_layers):
        logging.info(f"--- Ablating Layer {i}/{num_layers-1} ---")
        ablated_model = copy.deepcopy(model).to(device)
        attn_layer = get_attn(ablated_model, i)
        hook = attn_layer.register_forward_hook(zero_out_attention_hook)
        accuracy = evaluate(ablated_model, tokenizer, dataset, task_name, device, args.batch_size)
        performance_drop = baseline_accuracy - accuracy
        layer_performance.append({"layer_index": i, "accuracy": accuracy, "performance_drop": performance_drop})
        logging.info(f"Layer {i} Ablated Accuracy: {accuracy:.4f}, Drop: {performance_drop:.4f}")
        hook.remove()
        del ablated_model
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # --- Save Results and Visualize ---
    results_df = pd.DataFrame(layer_performance)
    results_path = os.path.join(args.output_dir, 'results', f'ablation_{experiment_suffix}.csv')
    results_df.to_csv(results_path, index=False)
    logging.info(f"📊 Layer-wise stats saved to {results_path}")

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))
    bars = ax.bar(results_df['layer_index'], results_df['performance_drop'], color=task_config['plot_color'], edgecolor='k')
    ax.set_xlabel("Transformer Layer Index", fontsize=12, weight='bold')
    ax.set_ylabel("Performance Drop (Baseline - Ablated Accuracy)", fontsize=12, weight='bold')
    ax.set_title(f"Impact of Zeroing Out Attention Output per Layer\nModel: {args.model_name} | Task: {task_display_name} | Pairs: {args.num_pairs}", fontsize=14, weight='bold')
    ax.set_xticks(results_df['layer_index'])
    ax.axhline(0, color='grey', linewidth=0.8)
    ax.bar_label(bars, fmt='%.3f', padding=3, fontsize=9)
    fig.tight_layout()

    plot_path = os.path.join(args.output_dir, 'plots', f'ablation_{experiment_suffix}.png')
    plt.savefig(plot_path, dpi=300)
    logging.info(f"🖼️  Visualization saved to {plot_path}. Mission complete!")


def main():
    """Parses command-line arguments and dispatches to the correct task function."""
    
    # A dictionary to hold the unique configurations for each task.
    # To add a new task, just add a new entry here.
    TASK_CONFIGS = {
        'synthetic_retrieval': {
            'creator': create_retrieval_dataset,
            'plot_color': 'c',
        },
        'associative_recall': {
            'creator': create_ar_dataset,
            'plot_color': 'm',
        }
    }

    parser = argparse.ArgumentParser(description="Run layer-wise analysis experiments for different tasks.")
    parser.add_argument("--task", type=str, required=True, choices=TASK_CONFIGS.keys(), help="The specific task to run.")
    parser.add_argument("--model_name", type=str, default="distilgpt2", help="Name of the Hugging Face model to evaluate.")
    parser.add_argument("--output_dir", type=str, default="experiment_results", help="Directory to save all artifacts.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for evaluation.")
    parser.add_argument("--num_samples", type=int, default=500, help="Number of samples to generate for the task.")
    parser.add_argument("--num_pairs", type=int, default=50, help="Number of key-value pairs in each prompt.")

    args = parser.parse_args()

    # --- Get the config for the selected task and run the experiment ---
    selected_task_config = TASK_CONFIGS[args.task]
    run_experiment(args, selected_task_config)

if __name__ == "__main__":
    main()