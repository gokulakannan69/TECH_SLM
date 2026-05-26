import os
import torch
import argparse

try:
    from datasets import load_dataset
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        TrainingArguments,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTTrainer
except ImportError as e:
    print("\n" + "="*70)
    print("[!] [Dependency Error] Missing required Machine Learning Libraries!")
    print(f"Details: {e}")
    print("="*70)
    print("This script is designed for GPU-accelerated environments.")
    print("To install all dependencies locally on your machine, run:")
    print("  pip install -r requirements.txt")
    print("\nOr open the project notebook in Google Colab to run it on a free T4 GPU!")
    print("="*70 + "\n")
    import sys
    sys.exit(1)

def print_gpu_utilization():
    """Prints current GPU utilization if CUDA is available."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 3)
        reserved = torch.cuda.memory_reserved() / (1024 ** 3)
        print(f"GPU Memory: Allocated = {allocated:.2f} GB | Reserved = {reserved:.2f} GB")
    else:
        print("Device: CPU (CUDA not available)")

def train_slm(args):
    print("=== Step 1: Loading Dataset ===")
    # Load dataset from split files
    dataset_files = {
        "train": args.train_path,
        "validation": args.val_path
    }
    
    # Using datasets to load local JSONL files
    raw_datasets = load_dataset("json", data_files=dataset_files)
    print(f"Dataset summary:\n{raw_datasets}")
    
    print("\n=== Step 2: Configuring Quantization & Device ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Targeting device: {device.upper()}")
    
    # 4-bit Quantization Config (Only applicable if CUDA is available)
    if device == "cuda":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        print("QLoRA 4-bit quantization enabled.")
    else:
        bnb_config = None
        print("Running on CPU. 4-bit quantization disabled (bitsandbytes is CUDA-only).")

    print("\n=== Step 3: Loading Base Model & Tokenizer ===")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    
    # Set padding token if missing (extremely common in models like Llama or TinyLlama)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Ensure correct padding side for causal LM training
    tokenizer.padding_side = "right"

    # Load model
    model_kwargs = {}
    if bnb_config:
        model_kwargs["quantization_config"] = bnb_config
        model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["device_map"] = "cpu"
        
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.float32 if device == "cpu" else torch.float16,
        **model_kwargs
    )
    
    # Print device mappings and size
    print(f"Base model loaded successfully from {args.model_id}.")
    print_gpu_utilization()

    print("\n=== Step 4: Formatting Dataset using Chat Template ===")
    # Standard format function to apply the conversational chat template
    def apply_chat_formatting(batch):
        formatted_texts = []
        for conversation in batch["messages"]:
            # Format using Hugging Face's tokenizer apply_chat_template
            formatted_text = tokenizer.apply_chat_template(
                conversation, 
                tokenize=False, 
                add_generation_prompt=False
            )
            formatted_texts.append(formatted_text)
        return {"text": formatted_texts}
        
    # Map raw data to processed strings
    formatted_datasets = raw_datasets.map(
        apply_chat_formatting,
        batched=True,
        remove_columns=["messages"]
    )
    print("First formatted sample:")
    print("-" * 50)
    print(formatted_datasets["train"][0]["text"])
    print("-" * 50)

    print("\n=== Step 5: Preparing PEFT / LoRA Adapter Config ===")
    if device == "cuda":
        # Prepares model for kbit (4-bit/8-bit) training
        model = prepare_model_for_kbit_training(model)
        
    # Configure LoRA
    lora_config = LoraConfig(
        r=args.r,
        lora_alpha=args.alpha,
        # Target standard linear layers for LLMs (Q, K, V, O, etc.)
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # Wrap model with PEFT adapters
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print("\n=== Step 6: Setting Training Arguments ===")
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=1,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        fp16=(device == "cuda"), # Use FP16 only if training on GPU
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_dir=os.path.join(args.output_dir, "logs"),
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        save_total_limit=2,
        # Optimize performance on supported architectures
        gradient_checkpointing=True if device == "cuda" else False,
    )

    print("\n=== Step 7: Initializing SFTTrainer and Training ===")
    trainer = SFTTrainer(
        model=model,
        train_dataset=formatted_datasets["train"],
        eval_dataset=formatted_datasets["validation"],
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        tokenizer=tokenizer,
        args=training_args,
    )

    # Trigger model training!
    print("Starting training...")
    trainer.train()
    print("Training finished successfully!")

    # Save the fine-tuned adapter weights
    adapter_output_dir = os.path.join(args.output_dir, "final_adapter")
    trainer.model.save_pretrained(adapter_output_dir)
    tokenizer.save_pretrained(adapter_output_dir)
    print(f"Fine-tuned LoRA adapters and tokenizer successfully saved to: {adapter_output_dir}")
    
    # Save the base model configurations as well
    print("Training process finished.")
    print_gpu_utilization()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QLoRA Fine-Tuning for Small Language Models (SLMs)")
    
    # Model and Data Paths
    parser.add_argument("--model_id", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0", help="Hugging Face base model ID")
    parser.add_argument("--train_path", type=str, default="data/train_dataset.jsonl", help="Path to formatted training dataset")
    parser.add_argument("--val_path", type=str, default="data/val_dataset.jsonl", help="Path to formatted validation dataset")
    parser.add_argument("--output_dir", type=str, default="checkpoints/tech_tutor_slm", help="Directory to save model checkpoints")
    
    # Hyperparameters
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size per device")
    parser.add_argument("--grad_accum", type=int, default=2, help="Number of gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--max_seq_length", type=int, default=512, help="Maximum token sequence length")
    
    # LoRA / PEFT Parameters
    parser.add_argument("--r", type=int, default=16, help="LoRA rank parameter")
    parser.add_argument("--alpha", type=int, default=32, help="LoRA alpha scaling factor")
    
    args = parser.parse_args()
    train_slm(args)
