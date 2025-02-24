import os
import shutil
import random
from tqdm import tqdm
from functools import partial
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import Dataset as HFDataset
from datasets import load_dataset, load_from_disk
import evaluate
from huggingface_hub import hf_hub_download

from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from transformers import DataCollatorForSeq2Seq  


PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:\n"
    ),
}

def encode_response(response: str, tokenizer) -> list[int]:
    tokens = tokenizer.encode(response.strip(), add_special_tokens=False)
    # For Llama 3 Instruct: tokens.append(tokenizer.get_added_vocab()["<|eot_id|>"])
    tokens.append(tokenizer.eos_token_id)  
    try:  # Llama 3 Instruct
        tokens.append(tokenizer.get_added_vocab()["<|end_of_text|>"])
    except KeyError:
        pass
    return tokens

def load_data(config):
    cache_dir =  "/root/.cache" 
    input_len = config.model.max_length
    concat_data = True

    tokenizer_path = config.model.pretrained_model_name_or_path
    tokenizer_name = tokenizer_path.split('/')[-1]

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f'Setting tokenizer.pad_token to {tokenizer.pad_token}')

    tokenizer.padding_side = 'left'  # for decoder-only generation
        # Get initial data
    ignore_kwargs = ['concat_data', 'chunk_size', 'pose_kwargs']
    dataset_config = {
        "name": "default", 
        "path": "yahma/alpaca-cleaned",
        "chunk_size": input_len,
        "concat_data": concat_data,
        "cache_dir": cache_dir,
    }
    dataset = load_dataset(
        **{k: v for k, v in dataset_config.items() if k not in ignore_kwargs}
    )
    dataset = dataset['train']
    train_set = convert_to_hf_dataset([dataset[ix] for ix in range(200, len(dataset))], cache_dir)
    val_set   = convert_to_hf_dataset([dataset[ix] for ix in range(200)], cache_dir)
    test_set  = convert_to_hf_dataset([dataset[ix] for ix in range(200)], cache_dir)

    # Convert to dicts of {input_ids, attention_mask, labels}
    train_set = train_set.map(
        partial(template_and_tokenize, tokenizer=tokenizer, include_label=True), 
        remove_columns=list(dataset.features),) 
    val_set = val_set.map(
        partial(template_and_tokenize, tokenizer=tokenizer, include_label=True),
        remove_columns=list(dataset.features),) 
    test_set  = test_set.map(
        partial(template_and_tokenize, tokenizer=tokenizer, include_label=False),
        remove_columns=list(dataset.features),) 

    # Chunk together train and val sets
    if concat_data:
        train_set = ConcatDataset(train_set, chunk_size=input_len)
        val_set = ConcatDataset(val_set, chunk_size=input_len)

    loader_kwargs = {
        "batch_size": config.data.micro_batch_size,
        "num_workers": 0,
        "drop_last": False,
        "pin_memory": True,
    }

    # Get dataloaders
    dataloaders = {
        'train': get_lm_loader(train_set, tokenizer, 'train', input_len, **loader_kwargs),
        'validation': get_lm_loader(val_set, tokenizer, 'validation', input_len, **loader_kwargs),
        'test': get_seq2seq_loader(test_set, tokenizer, 'test', **loader_kwargs),
    }
    # Evaluation metric
    try:
        # metric = load_metric(download_metric(), 'gov_report')  # hack but we want rouge
        metric = evaluate.load(download_metric(), 'gov_report') 
    except Exception as e:
        print(f'Error loading metric: {e}')
        metric = None

    # Finishing touches
    for k, v in dataloaders.items():  # Make tokenizer accessible
        dataloaders[k].dataset.tokenizer = tokenizer
        dataloaders[k].dataset.metric = metric
    return dataloaders


def convert_to_hf_dataset(dataset, cache_dir: str):
    """
    Convert iterable dataset to HuggingFace HFDataset object
    """
    def gen():
        for _, sample in enumerate(dataset):
            yield sample  # dataset[idx]
    return HFDataset.from_generator(gen, cache_dir=cache_dir)

def template_and_tokenize(sample, tokenizer, include_label: bool = True):
    """
    Format dataset context and answers into single-sequence prompts
    """
    if sample.get('input', '') == '':
        prompt = PROMPT_DICT["prompt_no_input"].format_map(sample)
    else:
        prompt = PROMPT_DICT["prompt_input"].format_map(sample)

    prompt = tokenizer.encode(prompt, add_special_tokens=True)
    if include_label:
        answer = tokenizer.encode(f'{sample["output"]}{tokenizer.eos_token}', 
                                  add_special_tokens=False)
        target = None
    else:
        answer = []
        target = tokenizer.encode(f'{sample["output"]}{tokenizer.eos_token}', 
                                  add_special_tokens=False)
    input_ids = prompt + answer
    attn_mask = [1] * len(input_ids)

    sample =  {
        "input_ids": input_ids,
        "attention_mask" : attn_mask,
        "labels": [-100] * len(prompt) + answer if include_label else target,
    }
    return sample

def get_lm_loader(dataset: Dataset, tokenizer: AutoTokenizer,
                  split: str, max_length: int = None, **loader_kwargs: any):
    """
    Get dataloader for language modeling (training)
    -> Currently this ends up being the same as get_seq2seq_loader
    """
    # collate_fn = DefaultDataCollator(return_tensors='pt')
    # collate_fn = DataCollatorWithPadding(tokenizer=tokenizer, padding=True,
    #                                      max_length=max_length, return_tensors='pt')
    collate_fn = DataCollatorForSeq2Seq(
        tokenizer, label_pad_token_id=-100, return_tensors='pt')
    return DataLoader(
        dataset, shuffle='train' in split, collate_fn=collate_fn, **loader_kwargs)

def get_seq2seq_loader(dataset: Dataset, tokenizer: AutoTokenizer,
                       split: str, **loader_kwargs: any):
    """
    Get dataloader for seq2seq tasks (evaluation)
    """
    tokenizer.padding_side = 'right'
    collate_fn = DataCollatorForSeq2Seq(
        tokenizer, label_pad_token_id=-100, return_tensors='pt')
    return DataLoader(
        dataset, shuffle='train' in split, collate_fn=collate_fn, **loader_kwargs)

def download_metric():
    """
    Download ROUGE, F1, and other accuracy metrics included in the SCROLLS dataset
    """
    scrolls_metric_path = hf_hub_download(
        repo_id="tau/scrolls", filename="metrics/scrolls.py", repo_type="dataset"
    )
    updated_scrolls_metric_path = (
        os.path.dirname(scrolls_metric_path) + 
        os.path.basename(scrolls_metric_path).replace(".", "_") + ".py"
    )
    shutil.copy(scrolls_metric_path, updated_scrolls_metric_path)
    return updated_scrolls_metric_path

class ConcatDataset(Dataset):
    """
    Concatenates or packs samples of a dataset into chunks of size `chunk_size`
    """
    def __init__(self, dataset, chunk_size: int = 1024, seed: int = 42,) -> None:
        self.dataset = dataset
        self.chunk_size = chunk_size
        self.samples = []
        buffer = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
            }
        random.seed(seed)
        for sample in tqdm(self.dataset, desc="Preprocessing dataset", dynamic_ncols=True):
            buffer = {k: v + sample[k] for k,v in buffer.items()}
            
            while len(next(iter(buffer.values()))) > self.chunk_size:
                self.samples.append({k: v[:self.chunk_size] for k,v in buffer.items()})
                buffer = {k: v[self.chunk_size:] for k,v in buffer.items()}
        # Slow hack, but filter out any samples without valid labels (all -100)
        self.filtered_samples = []
        for s in self.samples:
            if sum(s['labels']) != chunk_size * -100:
                self.filtered_samples.append(s)
        if len(self.filtered_samples) < len(self.samples):
            print(f'OG dataset: {len(self.samples)} samples -> Filtered dataset: {len(self.filtered_samples)}')
            print(f'-> Filtered out {len(self.samples) - len(self.filtered_samples)} samples')
                
    def __getitem__(self, idx):
        return self.filtered_samples[idx]
    
    def __len__(self):
        return len(self.filtered_samples)
