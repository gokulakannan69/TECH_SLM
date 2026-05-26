import json
import random
import os
import argparse
from typing import List, Dict, Any

def validate_entry(entry: Dict[str, Any], index: int) -> bool:
    if "messages" not in entry:
        print(f"Error at row {index}: Missing 'messages' key.")
        return False
    messages = entry["messages"]
    if not isinstance(messages, list):
        print(f"Error at row {index}: 'messages' must be a list.")
        return False
    for msg_idx, msg in enumerate(messages):
        if not isinstance(msg, dict):
            print(f"Error at row {index}, message {msg_idx}: Message is not a dict.")
            return False
        if "role" not in msg or "content" not in msg:
            print(f"Error at row {index}, message {msg_idx}: Missing 'role' or 'content' keys.")
            return False
        if msg["role"] not in ["system", "user", "assistant"]:
            print(f"Error at row {index}, message {msg_idx}: Invalid role '{msg['role']}'.")
            return False
    return True

def prepare_data(input_file: str, output_dir: str, train_ratio: float = 0.8, seed: int = 42):
    print(f"Reading dataset from: {input_file}")
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found at {input_file}")
    os.makedirs(output_dir, exist_ok=True)
    valid_entries = []
    with open(input_file, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line: continue
            try:
                entry = json.loads(line)
                if validate_entry(entry, idx):
                    valid_entries.append(entry)
            except json.JSONDecodeError as e:
                print(f"Error: Row {idx} is not valid JSON. {e}")
    total_count = len(valid_entries)
    print(f"Successfully validated {total_count} records.")
    if total_count == 0: return
    random.seed(seed)
    random.shuffle(valid_entries)
    split_index = int(total_count * train_ratio)
    train_data = valid_entries[:split_index]
    val_data = valid_entries[split_index:]
    train_path = os.path.join(output_dir, "train_dataset.jsonl")
    val_path = os.path.join(output_dir, "val_dataset.jsonl")
    with open(train_path, "w", encoding="utf-8") as f:
        for entry in train_data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    with open(val_path, "w", encoding="utf-8") as f:
        for entry in val_data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print("\n--- Data Splits Saved Successfully! ---")
    print(f"Train file: {train_path} ({len(train_data)} elements)")
    print(f"Val file:   {val_path} ({len(val_data)} elements)")

if __name__ == "__main__":
    prepare_data("data/sample_dataset.jsonl", "data", 0.8, 42)
