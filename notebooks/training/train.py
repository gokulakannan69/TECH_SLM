import os
import torch
import argparse
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

def train_slm(args):
    dataset_files = {"train": args.train_path, "validation": args.val_path}
    raw_datasets = load_dataset("json", data_files=dataset_files)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running training on: {device.upper()}")

    if device == "cuda":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
    else:
        bnb_config = None

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    model_kwargs = {"quantization_config": bnb_config, "device_map": "auto"} if bnb_config else {"device_map": "cpu"}
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.float32 if device == "cpu" else torch.float16,
        **model_kwargs
    )

    def apply_chat_formatting(batch):
        texts = [tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=False) for conv in batch["messages"]]
        return {"text": texts}

    formatted_datasets = raw_datasets.map(apply_chat_formatting, batched=True, remove_columns=["messages"])

    if device == "cuda":
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=args.r,
        lora_alpha=args.alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

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
        fp16=(device == "cuda"),
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_dir=os.path.join(args.output_dir, "logs"),
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        save_total_limit=2,
        gradient_checkpointing=True if device == "cuda" else False,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=formatted_datasets["train"],
        eval_dataset=formatted_datasets["validation"],
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        tokenizer=tokenizer,
        args=training_args,
    )

    print("Booting training loop...")
    trainer.train()

    adapter_output_dir = os.path.join(args.output_dir, "final_adapter")
    trainer.model.save_pretrained(adapter_output_dir)
    tokenizer.save_pretrained(adapter_output_dir)
    print(f"Successfully saved fine-tuned adapter weights to {adapter_output_dir}")

if __name__ == "__main__":
    import sys
    class Args:
        model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        train_path = "data/train_dataset.jsonl"
        val_path = "data/val_dataset.jsonl"
        output_dir = "checkpoints/tech_tutor_slm"
        num_epochs = 5
        batch_size = 2
        grad_accum = 2
        learning_rate = 2e-4
        max_seq_length = 512
        r = 16
        alpha = 32
    train_slm(Args())
