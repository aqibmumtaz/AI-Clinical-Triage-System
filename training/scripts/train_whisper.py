"""
Fine-tune Whisper large-v3 / large-v3-turbo on Urdu call center data.

Usage:
  # Whisper large-v3-turbo (recommended first — faster training)
  python train_whisper.py \
      --model openai/whisper-large-v3-turbo \
      --dataset_dir ./whisper_dataset \
      --output_dir ./whisper-large-v3-turbo-urdu \
      --batch_size 32 --epochs 3

  # Whisper large-v3 (best accuracy — slower)
  python train_whisper.py \
      --model openai/whisper-large-v3 \
      --dataset_dir ./whisper_dataset \
      --output_dir ./whisper-large-v3-urdu \
      --batch_size 16 --epochs 3

  # Resume from checkpoint
  python train_whisper.py \
      --model openai/whisper-large-v3-turbo \
      --dataset_dir ./whisper_dataset \
      --output_dir ./whisper-large-v3-turbo-urdu \
      --resume_from_checkpoint latest

Requirements:
  pip install transformers datasets evaluate jiwer accelerate soundfile librosa tensorboard
"""

import os, argparse, logging
import torch
import evaluate
import numpy as np
from dataclasses import dataclass
from typing import Any

from datasets import load_dataset, Audio
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features):
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


def prepare_dataset(batch, processor):
    audio = batch["audio"]
    batch["input_features"] = processor.feature_extractor(
        audio["array"], sampling_rate=audio["sampling_rate"]
    ).input_features[0]
    batch["labels"] = processor.tokenizer(batch["sentence"]).input_ids
    return batch


def compute_metrics(pred, processor, metric):
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="openai/whisper-large-v3-turbo")
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--output_dir", default="./whisper-urdu-finetuned")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--max_steps", type=int, default=50000)
    parser.add_argument("--epochs", type=int, default=None, help="If set, overrides max_steps")
    parser.add_argument("--eval_steps", type=int, default=1000)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--resume_from_checkpoint", default=None)
    parser.add_argument("--language", default="ur")
    parser.add_argument("--task", default="transcribe")
    args = parser.parse_args()

    log.info(f"Loading model: {args.model}")
    processor = WhisperProcessor.from_pretrained(args.model, language=args.language, task=args.task)
    model = WhisperForConditionalGeneration.from_pretrained(args.model)

    model.generation_config.language = args.language
    model.generation_config.task = args.task
    model.generation_config.forced_decoder_ids = None

    log.info(f"Loading dataset from: {args.dataset_dir}")

    data_files = {}
    for split in ["train", "validation", "test"]:
        meta = os.path.join(args.dataset_dir, split, "metadata.jsonl")
        if os.path.exists(meta):
            data_files[split] = meta

    dataset = load_dataset(
        "json",
        data_files=data_files,
    )

    for split in dataset:
        audio_dir = os.path.join(args.dataset_dir, split)
        dataset[split] = dataset[split].map(
            lambda x: {"audio": os.path.join(audio_dir, x["audio"])},
        )
        dataset[split] = dataset[split].cast_column("audio", Audio(sampling_rate=16000))

    log.info("Preprocessing dataset...")
    dataset = dataset.map(
        lambda batch: prepare_dataset(batch, processor),
        remove_columns=dataset["train"].column_names,
        num_proc=4,
    )

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    metric = evaluate.load("wer")

    training_args_dict = dict(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        gradient_checkpointing=True,
        fp16=torch.cuda.is_available(),
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        logging_steps=25,
        dataloader_num_workers=4,
        predict_with_generate=True,
        generation_max_length=225,
        report_to=["tensorboard"],
        push_to_hub=False,
        remove_unused_columns=False,
    )

    if args.epochs:
        training_args_dict["num_train_epochs"] = args.epochs
    else:
        training_args_dict["max_steps"] = args.max_steps

    training_args = Seq2SeqTrainingArguments(**training_args_dict)

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("validation"),
        data_collator=data_collator,
        compute_metrics=lambda pred: compute_metrics(pred, processor, metric),
        processing_class=processor.feature_extractor,
    )

    log.info("Starting training...")
    resume = args.resume_from_checkpoint
    if resume == "latest":
        resume = True
    trainer.train(resume_from_checkpoint=resume)

    log.info(f"Saving model to {args.output_dir}/best")
    trainer.save_model(os.path.join(args.output_dir, "best"))
    processor.save_pretrained(os.path.join(args.output_dir, "best"))

    if "test" in dataset:
        log.info("Evaluating on test set...")
        results = trainer.evaluate(dataset["test"])
        log.info(f"Test WER: {results['eval_wer']:.2f}%")

        with open(os.path.join(args.output_dir, "test_results.json"), "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    import json
    main()
