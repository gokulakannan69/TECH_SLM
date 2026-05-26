import json
import random
import os
import argparse
from typing import List, Dict, Any

def validate_entry(entry: Dict[str, Any], index: int) -> bool:
    """Validates the schema of a conversational dataset entry."""
    if "messages" not in entry:
        print(f"Error at row {index}: Missing 'messages' key.")
        return False
    
    messages = entry["messages"]
    if not isinstance(messages, list):
        print(f"Error at row {index}: 'messages' must be a list.")
        return False
        
    for msg_idx, msg in enumerate(messages):
        if not isinstance(msg, dict):
            print(f"Error at row {index}, message {msg_idx}: Message is not a dictionary.")
            return False
        if "role" not in msg or "content" not in msg:
            print(f"Error at row {index}, message {msg_idx}: Missing 'role' or 'content' keys.")
            return False
        if msg["role"] not in ["system", "user", "assistant"]:
            print(f"Error at row {index}, message {msg_idx}: Invalid role '{msg['role']}'.")
            return False
            
    return True

def prepare_data(input_file: str, output_dir: str, train_ratio: float = 0.8, seed: int = 42):
    """Reads, shuffles, validates, and splits conversational dataset into train/validation sets."""
    print(f"Reading dataset from: {input_file}")
    
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found at {input_file}")
        
    os.makedirs(output_dir, exist_ok=True)
    
    valid_entries = []
    with open(input_file, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                if validate_entry(entry, idx):
                    valid_entries.append(entry)
            except json.JSONDecodeError as e:
                print(f"Error: Row {idx} is not valid JSON. {e}")
                
    total_count = len(valid_entries)
    print(f"Successfully validated {total_count} dataset records.")
    
    if total_count == 0:
        print("Warning: No valid entries found to split!")
        return

    # Shuffle for random splitting (reproducible seed)
    random.seed(seed)
    random.shuffle(valid_entries)
    
    # Split calculation
    split_index = int(total_count * train_ratio)
    train_data = valid_entries[:split_index]
    val_data = valid_entries[split_index:]
    
    # Define outputs
    train_path = os.path.join(output_dir, "train_dataset.jsonl")
    val_path = os.path.join(output_dir, "val_dataset.jsonl")
    
    # Write train
    with open(train_path, "w", encoding="utf-8") as f:
        for entry in train_data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            
    # Write val
    with open(val_path, "w", encoding="utf-8") as f:
        for entry in val_data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            
    print("\n--- Data Preparation Complete ---")
    print(f"Training split:   {len(train_data)} items -> {train_path}")
    print(f"Validation split: {len(val_data)} items -> {val_path}")
    print("---------------------------------")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare and split conversational dataset.")
    parser.add_argument("--input", type=str, default="data/sample_dataset.jsonl", help="Path to raw JSONL dataset")
    parser.add_argument("--output_dir", type=str, default="data", help="Directory to save split datasets")
    parser.add_argument("--ratio", type=float, default=0.8, help="Ratio of data for training (default: 0.8)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for replication")
    
    args = parser.parse_args()
    prepare_data(args.input, args.output_dir, args.ratio, args.seed)
