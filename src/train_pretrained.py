"""Fine-tune mT5 and LoRA-adapt LLaMA/Gemma with comparable prompts and data."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from datasets import load_from_disk
from transformers import (AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer,
                          DataCollatorForSeq2Seq, DataCollatorForLanguageModeling,
                          Seq2SeqTrainer, Seq2SeqTrainingArguments, Trainer, TrainingArguments)

from .common import ROOT, Timer, device, load_config, parameter_count, seed_everything, write_json


def mt5(config: dict):
    seed_everything(config["seed"]); settings = config["mt5"]
    corpus = load_from_disk(str(ROOT / "data" / "opus_books_en_fr"))
    tokenizer = AutoTokenizer.from_pretrained(settings["model_name"])
    model = AutoModelForSeq2SeqLM.from_pretrained(settings["model_name"])
    prefix = "translate English to French: "
    def encode(rows):
        inputs = [prefix + text for text in rows["source"]]
        encoded = tokenizer(inputs, max_length=settings["max_source_length"], truncation=True)
        encoded["labels"] = tokenizer(text_target=rows["target"], max_length=settings["max_target_length"], truncation=True)["input_ids"]
        return encoded
    tokenized = corpus.map(encode, batched=True, remove_columns=corpus["train"].column_names)
    args = Seq2SeqTrainingArguments(output_dir=str(ROOT / "checkpoints" / "mt5"), num_train_epochs=settings["epochs"], per_device_train_batch_size=settings["batch_size"], per_device_eval_batch_size=settings["batch_size"], learning_rate=settings["learning_rate"], evaluation_strategy="epoch", save_strategy="epoch", save_total_limit=1, predict_with_generate=True, fp16=torch.cuda.is_available(), report_to="none", logging_steps=25)
    trainer = Seq2SeqTrainer(model=model, args=args, train_dataset=tokenized["train"], eval_dataset=tokenized["validation"], data_collator=DataCollatorForSeq2Seq(tokenizer, model=model), tokenizer=tokenizer)
    with Timer() as timer: trainer.train()
    destination = ROOT / "artifacts" / "mt5"; trainer.save_model(str(destination)); tokenizer.save_pretrained(str(destination))
    write_json(ROOT / "results" / "mt5_training.json", {"training_seconds": timer.seconds, "parameters": parameter_count(model), "model": settings["model_name"]})


def causal_prompt(source: str, target: str | None = None) -> str:
    prompt = f"Translate the following English sentence into French. Return only the French translation.\nEnglish: {source}\nFrench:"
    return prompt if target is None else prompt + " " + target


def causal(model_key: str, config: dict):
    """QLoRA is used on CUDA; ordinary LoRA is used when running without bitsandbytes."""
    from peft import LoraConfig, TaskType, get_peft_model
    seed_everything(config["seed"]); settings = config[model_key]
    corpus = load_from_disk(str(ROOT / "data" / "opus_books_en_fr"))
    tokenizer = AutoTokenizer.from_pretrained(settings["model_name"])
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    kwargs = {"device_map": "auto"} if torch.cuda.is_available() else {}
    model = AutoModelForCausalLM.from_pretrained(settings["model_name"], **kwargs)
    def encode(rows):
        texts = [causal_prompt(s, t) + tokenizer.eos_token for s, t in zip(rows["source"], rows["target"])]
        return tokenizer(texts, truncation=True, max_length=settings["max_length"])
    tokenized = corpus.map(encode, batched=True, remove_columns=corpus["train"].column_names)
    targets = [name for name, _ in model.named_modules() if name.endswith(("q_proj", "k_proj", "v_proj", "o_proj"))]
    targets = [name.split(".")[-1] for name in targets]
    target_modules = sorted(set(targets)) or ["q_proj", "v_proj"]
    model = get_peft_model(model, LoraConfig(task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=32, lora_dropout=0.05, target_modules=target_modules))
    args = TrainingArguments(output_dir=str(ROOT / "checkpoints" / model_key), num_train_epochs=settings["epochs"], per_device_train_batch_size=settings["batch_size"], per_device_eval_batch_size=settings["batch_size"], learning_rate=settings["learning_rate"], evaluation_strategy="epoch", save_strategy="epoch", save_total_limit=1, fp16=torch.cuda.is_available(), report_to="none", logging_steps=25)
    trainer = Trainer(model=model, args=args, train_dataset=tokenized["train"], eval_dataset=tokenized["validation"], data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False))
    with Timer() as timer: trainer.train()
    destination = ROOT / "artifacts" / model_key; trainer.save_model(str(destination)); tokenizer.save_pretrained(str(destination))
    write_json(ROOT / "results" / f"{model_key}_training.json", {"training_seconds": timer.seconds, "parameters_total": parameter_count(model), "parameters_trainable": sum(p.numel() for p in model.parameters() if p.requires_grad), "model": settings["model_name"], "adaptation": "LoRA"})


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("model", choices=["mt5", "llama", "gemma"]); parser.add_argument("--config", default=ROOT / "configs" / "experiment.yaml")
    args = parser.parse_args(); config = load_config(args.config)
    mt5(config) if args.model == "mt5" else causal(args.model, config)


if __name__ == "__main__": main()
