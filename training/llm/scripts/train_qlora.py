"""
train_qlora.py
QLoRA fine-tune Qwen3-8B-Instruct on Urdu lab call center data.
Run this on RunPod / Vast.ai (RTX 4090 24GB or A40 48GB).

Usage:
  pip install transformers peft bitsandbytes trl datasets accelerate
  python train_qlora.py --train_data ../data/train.jsonl --eval_data ../data/eval.jsonl
"""

import argparse
import json
import os
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer


def load_jsonl(path: str) -> Dataset:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return Dataset.from_list(records)


def format_conversation(example: dict) -> dict:
    """Convert messages list to a single formatted text string for training."""
    messages = example["messages"]
    # Qwen3 chat format
    text = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "system":
            text += f"<|im_start|>system\n{content}<|im_end|>\n"
        elif role == "user":
            text += f"<|im_start|>user\n{content}<|im_end|>\n"
        elif role == "assistant":
            text += f"<|im_start|>assistant\n{content}<|im_end|>\n"
    return {"text": text}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-8B-Instruct")
    parser.add_argument("--train_data", default="../data/train.jsonl")
    parser.add_argument("--eval_data", default="../data/eval.jsonl")
    parser.add_argument("--output", default="../checkpoints")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument(
        "--lora_r", type=int, default=16
    )  # r=16 prevents overfitting on narrow domain; r=64 memorises
    parser.add_argument("--lora_alpha", type=int, default=32)  # keep alpha = 2 * r
    parser.add_argument(
        "--max_seq_len", type=int, default=1024
    )  # 512 truncates multi-turn Urdu convos
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    print(
        f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU (slow!)'}"
    )

    # 4-bit quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)

    # LoRA config
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load and format datasets
    train_ds = load_jsonl(args.train_data).map(format_conversation)
    eval_ds = load_jsonl(args.eval_data).map(format_conversation)
    print(f"Train: {len(train_ds)} | Eval: {len(eval_ds)}")

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        eval_strategy="steps",  # evaluation_strategy deprecated in transformers ≥4.45
        eval_steps=50,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=3,
        load_best_model_at_end=True,
        logging_steps=10,
        bf16=True,
        optim="paged_adamw_8bit",
        report_to="none",
        dataloader_num_workers=2,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        args=training_args,
        dataset_text_field="text",
        max_seq_length=args.max_seq_len,
        packing=False,
    )

    print("Starting training...")
    trainer.train()

    # Save final adapter
    final_path = os.path.join(args.output, "final")
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"\nTraining complete. Model saved to: {final_path}")
    print("Push to HuggingFace Hub:")
    print(f"  model.push_to_hub('your-username/qwen3-8b-urdu-lab')")


if __name__ == "__main__":
    main()
