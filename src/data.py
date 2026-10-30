"""Download, clean, split and persist the public OPUS Books corpus."""
from __future__ import annotations

import argparse
import re
from pathlib import Path

from datasets import DatasetDict, load_dataset

from .common import ROOT, load_config, seed_everything


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\u00a0", " ")).strip()


def prepare(config: dict) -> DatasetDict:
    spec = config["dataset"]
    seed_everything(config["seed"])
    # OPUS Books is a public parallel corpus. Selecting before splitting keeps this
    # assignment practical while preserving a fixed, reproducible test set.
    raw = load_dataset(spec["name"], spec["config"], split="train")
    raw = raw.shuffle(seed=config["seed"])
    rows = []
    for example in raw:
        pair = example["translation"]
        source, target = clean(pair[spec["source_lang"]]), clean(pair[spec["target_lang"]])
        if source and target and len(source) <= spec["max_source_chars"] and len(target) <= spec["max_target_chars"]:
            rows.append({"source": source, "target": target})
        if len(rows) == spec["max_examples"]:
            break
    required = spec["train_size"] + spec["validation_size"] + spec["test_size"]
    if len(rows) < required:
        raise RuntimeError(f"Only {len(rows)} usable pairs found; expected {required}.")
    dataset = DatasetDict({
        "train": raw.from_list(rows[:spec["train_size"]]),
        "validation": raw.from_list(rows[spec["train_size"]:spec["train_size"] + spec["validation_size"]]),
        "test": raw.from_list(rows[spec["train_size"] + spec["validation_size"]:required]),
    })
    destination = ROOT / "data" / "opus_books_en_fr"
    dataset.save_to_disk(str(destination))
    print({name: len(split) for name, split in dataset.items()})
    return dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=ROOT / "configs" / "experiment.yaml")
    args = parser.parse_args()
    prepare(load_config(args.config))


if __name__ == "__main__":
    main()
