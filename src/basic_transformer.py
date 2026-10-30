"""A from-scratch encoder-decoder Transformer baseline (no pretrained weights)."""
from __future__ import annotations

import argparse
import math
import re
from collections import Counter
from pathlib import Path

import torch
from datasets import load_from_disk
from torch import Tensor, nn
from torch.utils.data import DataLoader

from .common import ROOT, Timer, device, load_config, parameter_count, seed_everything, write_json

SPECIALS = ["<pad>", "<unk>", "<bos>", "<eos>"]


def tokens(text: str) -> list[str]:
    return re.findall(r"\w+|[^\w\s]", text.lower(), flags=re.UNICODE)


class Vocabulary:
    def __init__(self, texts: list[str], min_frequency: int):
        counts = Counter(token for text in texts for token in tokens(text))
        self.itos = SPECIALS + sorted(word for word, count in counts.items() if count >= min_frequency)
        self.stoi = {word: index for index, word in enumerate(self.itos)}

    @property
    def pad(self): return 0
    @property
    def bos(self): return 2
    @property
    def eos(self): return 3
    def encode(self, text: str, maximum: int) -> list[int]:
        ids = [self.bos] + [self.stoi.get(word, 1) for word in tokens(text)][: maximum - 2] + [self.eos]
        return ids
    def decode(self, ids: list[int]) -> str:
        words = [self.itos[i] for i in ids if i < len(self.itos) and i not in (self.pad, self.bos, self.eos)]
        return re.sub(r"\s+([.,!?;:])", r"\1", " ".join(words)).strip()


def collate(batch, source_vocab, target_vocab, maximum):
    src = [source_vocab.encode(row["source"], maximum) for row in batch]
    tgt = [target_vocab.encode(row["target"], maximum) for row in batch]
    pad = lambda xs: torch.nn.utils.rnn.pad_sequence([torch.tensor(x) for x in xs], batch_first=True, padding_value=0)
    return pad(src), pad(tgt)


class PositionalEncoding(nn.Module):
    def __init__(self, dimension: int, dropout: float, maximum: int = 512):
        super().__init__()
        position = torch.arange(maximum).unsqueeze(1)
        scale = torch.exp(torch.arange(0, dimension, 2) * (-math.log(10000.0) / dimension))
        encoding = torch.zeros(maximum, dimension)
        encoding[:, 0::2], encoding[:, 1::2] = torch.sin(position * scale), torch.cos(position * scale)
        self.register_buffer("encoding", encoding.unsqueeze(0))
        self.dropout = nn.Dropout(dropout)
    def forward(self, x): return self.dropout(x + self.encoding[:, :x.size(1)])


class TransformerMT(nn.Module):
    def __init__(self, src_vocab: int, tgt_vocab: int, settings: dict):
        super().__init__()
        dimension = settings["d_model"]
        self.dimension = dimension
        self.src_embedding, self.tgt_embedding = nn.Embedding(src_vocab, dimension), nn.Embedding(tgt_vocab, dimension)
        self.position = PositionalEncoding(dimension, settings["dropout"])
        self.core = nn.Transformer(d_model=dimension, nhead=settings["nhead"], num_encoder_layers=settings["num_encoder_layers"], num_decoder_layers=settings["num_decoder_layers"], dim_feedforward=settings["dim_feedforward"], dropout=settings["dropout"], batch_first=True)
        self.output = nn.Linear(dimension, tgt_vocab)
    def forward(self, source, target):
        source_pad, target_pad = source.eq(0), target.eq(0)
        causal = nn.Transformer.generate_square_subsequent_mask(target.size(1), device=target.device)
        source = self.position(self.src_embedding(source) * math.sqrt(self.dimension))
        target = self.position(self.tgt_embedding(target) * math.sqrt(self.dimension))
        return self.output(self.core(source, target, tgt_mask=causal, src_key_padding_mask=source_pad, tgt_key_padding_mask=target_pad, memory_key_padding_mask=source_pad))


@torch.no_grad()
def translate(model, sentence, source_vocab, target_vocab, maximum, run_device):
    model.eval()
    source = torch.tensor([source_vocab.encode(sentence, maximum)], device=run_device)
    generated = torch.tensor([[target_vocab.bos]], device=run_device)
    for _ in range(maximum - 1):
        next_id = model(source, generated)[:, -1].argmax(dim=-1)
        generated = torch.cat((generated, next_id[:, None]), dim=1)
        if next_id.item() == target_vocab.eos: break
    return target_vocab.decode(generated[0].tolist())


def train(config: dict):
    seed_everything(config["seed"])
    settings = config["basic_transformer"]
    data = load_from_disk(str(ROOT / "data" / "opus_books_en_fr"))
    source_vocab = Vocabulary(data["train"]["source"], settings["min_frequency"])
    target_vocab = Vocabulary(data["train"]["target"], settings["min_frequency"])
    run_device = device()
    model = TransformerMT(len(source_vocab.itos), len(target_vocab.itos), settings).to(run_device)
    loader = DataLoader(data["train"], batch_size=settings["batch_size"], shuffle=True, collate_fn=lambda rows: collate(rows, source_vocab, target_vocab, settings["max_length"]))
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"])
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)
    with Timer() as timer:
        for epoch in range(settings["epochs"]):
            model.train(); total = 0.0
            for source, target in loader:
                source, target = source.to(run_device), target.to(run_device)
                optimizer.zero_grad()
                logits = model(source, target[:, :-1])
                loss = loss_fn(logits.reshape(-1, logits.size(-1)), target[:, 1:].reshape(-1))
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
                total += loss.item()
            print(f"epoch {epoch + 1}/{settings['epochs']} loss={total / len(loader):.4f}")
    output = ROOT / "artifacts" / "basic_transformer"
    output.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "source_vocab": source_vocab.itos, "target_vocab": target_vocab.itos, "settings": settings}, output / "model.pt")
    write_json(ROOT / "results" / "basic_transformer_training.json", {"training_seconds": timer.seconds, "parameters": parameter_count(model), "device": str(run_device)})


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--config", default=ROOT / "configs" / "experiment.yaml")
    args = parser.parse_args(); train(load_config(args.config))


if __name__ == "__main__": main()
