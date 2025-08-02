import argparse
import numpy as np
from datasets import load_from_disk, Dataset
from itertools import chain
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Chunk tokenized dataset")
    parser.add_argument(
        "--tokenized_dataset_path",
        type=str,
        required=True,
        help="Path to the saved tokenized dataset (from save_to_disk)"
    )
    parser.add_argument(
        "--context_length",
        type=int,
        required=True,
        help="Context length for each chunk"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for chunked dataset"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="uint16",
        choices=["uint16", "uint32", "int32"],
        help="Storage dtype for concatenated tokens"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading tokenized dataset...")
    dataset = load_from_disk(args.tokenized_dataset_path)

    print("Concatenating all input_ids...")
    all_tokens = list(chain.from_iterable(dataset["input_ids"]))
    all_tokens = np.array(all_tokens, dtype=args.dtype)

    total_len = (len(all_tokens) // args.context_length) * args.context_length
    all_tokens = all_tokens[:total_len]
    chunks = all_tokens.reshape(-1, args.context_length)

    print(f"Total tokens: {len(all_tokens)}, Num chunks: {len(chunks)}")

    # Save as HuggingFace Dataset in Arrow format
    new_dataset = Dataset.from_dict({"input_ids": chunks.tolist()})
    arrow_path = os.path.join(args.output_dir, f"chunked_context{args.context_length}")
    new_dataset.save_to_disk(arrow_path)

    print(f"Saved chunked dataset to {arrow_path}")

if __name__ == "__main__":
    main()
