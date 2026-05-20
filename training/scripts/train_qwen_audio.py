"""
Fine-tune Qwen2-Audio-7B-Instruct on Urdu call center data using LoRA.

Usage:
  python train_qwen_audio.py \
      --dataset_dir ./qwen_dataset \
      --output_dir ./qwen2-audio-urdu \
      --lora_rank 64 \
      --epochs 2

Requirements:
  pip install transformers datasets peft accelerate bitsandbytes trl soundfile librosa

Note: Requires 1× A100 80GB or 2× RTX 4090 (with DeepSpeed ZeRO-3).
      For RTX 4090 single GPU, use --load_in_4bit for QLoRA.
"""

import os, json, argparse, logging
import torch
from datasets import load_dataset, Audio
from transformers import (
    Qwen2AudioForConditionalGeneration,
    AutoProcessor,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def preprocess_function(examples, processor):
    conversations = examples["messages"]
    texts = []
    audios = []

    for conv in conversations:
        text = processor.apply_chat_template(conv, tokenize=False, add_generation_prompt=False)
        texts.append(text)

        audio_url = None
        for msg in conv:
            if isinstance(msg.get("content"), list):
                for item in msg["content"]:
                    if item.get("type") == "audio":
                        audio_url = item["audio_url"]
                        break
        audios.append(audio_url)

    inputs = processor(
        text=texts,
        audios=audios,
        return_tensors="pt",
        padding=True,
        sampling_rate=16000,
    )

    inputs["labels"] = inputs["input_ids"].clone()
    return inputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2-Audio-7B-Instruct")
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--output_dir", default="./qwen2-audio-urdu")
    parser.add_argument("--lora_rank", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--load_in_4bit", action="store_true", help="QLoRA: load base model in 4-bit")
    parser.add_argument("--load_in_8bit", action="store_true", help="Load base model in 8-bit")
    args = parser.parse_args()

    log.info(f"Loading model: {args.model}")

    quantization_config = None
    if args.load_in_4bit:
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    elif args.load_in_8bit:
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        args.model,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16 if not quantization_config else None,
        device_map="auto",
    )

    processor = AutoProcessor.from_pretrained(args.model)

    if quantization_config:
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    log.info(f"Loading dataset from: {args.dataset_dir}")
    data_files = {}
    for split in ["train", "validation", "test"]:
        fpath = os.path.join(args.dataset_dir, f"{split}.jsonl")
        if os.path.exists(fpath):
            data_files[split] = fpath

    dataset = load_dataset("json", data_files=data_files)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        warmup_ratio=0.03,
        num_train_epochs=args.epochs,
        gradient_checkpointing=True,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported() and torch.cuda.is_available(),
        eval_strategy="steps",
        eval_steps=500,
        save_steps=500,
        save_total_limit=3,
        load_best_model_at_end=True,
        logging_steps=10,
        dataloader_num_workers=4,
        report_to=["tensorboard"],
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("validation"),
        processing_class=processor,
    )

    log.info("Starting LoRA training...")
    trainer.train()

    log.info(f"Saving LoRA adapter to {args.output_dir}/best")
    model.save_pretrained(os.path.join(args.output_dir, "best"))
    processor.save_pretrained(os.path.join(args.output_dir, "best"))

    log.info("Training complete!")


if __name__ == "__main__":
    main()
