"""Generate the same held-out translations, score BLEU/chrF, and save samples."""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import pandas as pd
import sacrebleu
import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

from .basic_transformer import TransformerMT, Vocabulary, translate
from .common import ROOT, device, load_config, parameter_count, write_json
from .train_pretrained import causal_prompt


def score(name, predictions, references, elapsed, parameters, training=None):
    outcome = {"model": name, "bleu": round(sacrebleu.corpus_bleu(predictions, [references]).score, 2), "chrf": round(sacrebleu.corpus_chrf(predictions, [references]).score, 2), "inference_seconds": round(elapsed, 3), "milliseconds_per_sentence": round(1000 * elapsed / len(predictions), 3), "parameters": parameters}
    if training: outcome.update(training)
    return outcome


def basic(data, config):
    saved = torch.load(ROOT / "artifacts" / "basic_transformer" / "model.pt", map_location=device(), weights_only=False)
    source_vocab, target_vocab = Vocabulary([], 1), Vocabulary([], 1); source_vocab.itos, target_vocab.itos = saved["source_vocab"], saved["target_vocab"]; source_vocab.stoi = {w:i for i,w in enumerate(source_vocab.itos)}; target_vocab.stoi = {w:i for i,w in enumerate(target_vocab.itos)}
    model = TransformerMT(len(source_vocab.itos), len(target_vocab.itos), saved["settings"]).to(device()); model.load_state_dict(saved["state_dict"])
    start = time.perf_counter(); outputs = [translate(model, x, source_vocab, target_vocab, saved["settings"]["max_length"], device()) for x in data["source"]]; elapsed = time.perf_counter() - start
    return outputs, score("Basic Transformer", outputs, data["target"], elapsed, parameter_count(model))


def pretrained(key, data, config):
    settings, folder = config[key], ROOT / "artifacts" / key
    tokenizer = AutoTokenizer.from_pretrained(folder)
    causal = key in {"llama", "gemma"}
    if causal:
        # LoRA training saves an adapter, not a standalone base checkpoint.
        from peft import PeftModel
        base = AutoModelForCausalLM.from_pretrained(settings["model_name"])
        model = PeftModel.from_pretrained(base, folder).to(device())
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(folder).to(device())
    model.eval()
    outputs = []; start = time.perf_counter()
    for source in data["source"]:
        prompt = causal_prompt(source) if causal else "translate English to French: " + source
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device())
        generated = model.generate(**inputs, max_new_tokens=config["generation"]["max_new_tokens"], num_beams=config["generation"]["num_beams"], do_sample=False)
        outputs.append(tokenizer.decode(generated[0][inputs["input_ids"].shape[1]:] if causal else generated[0], skip_special_tokens=True).strip())
    elapsed = time.perf_counter() - start
    return outputs, score({"mt5":"mT5-small", "llama":"TinyLlama (LLaMA)", "gemma":"Gemma-2-2B"}[key], outputs, data["target"], elapsed, parameter_count(model))


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--models", nargs="+", default=["basic", "mt5", "llama", "gemma"]); parser.add_argument("--limit", type=int, default=None); parser.add_argument("--config", default=ROOT / "configs" / "experiment.yaml")
    args = parser.parse_args(); config = load_config(args.config);     data = load_from_disk(str(ROOT / "data" / "opus_books_en_fr"))["test"]
    if args.limit: data = data.select(range(min(args.limit, len(data))))
    records, columns = [], {"source": data["source"], "reference": data["target"]}
    for key in args.models:
        predictions, record = basic(data, config) if key == "basic" else pretrained(key, data, config)
        records.append(record); columns[key] = predictions
    ROOT.joinpath("results", "samples").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns).to_csv(ROOT / "results" / "samples" / "test_translations.csv", index=False)
    pd.DataFrame(records).to_csv(ROOT / "results" / "metrics.csv", index=False)
    write_json(ROOT / "results" / "metrics.json", {"models": records, "test_examples": len(data)})
    print(pd.DataFrame(records).to_string(index=False))


if __name__ == "__main__": main()
