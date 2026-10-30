# ICT4403 Assignment 3 — English-to-French Machine Translation

This repository implements and compares four machine-translation systems required by
the Foundations of Generative AI assignment:

1. **Basic Transformer** — a small encoder-decoder Transformer implemented in PyTorch.
2. **mT5-small** — a multilingual pretrained encoder-decoder model fine-tuned for
   English-to-French translation.
3. **TinyLlama 1.1B** — an openly downloadable LLaMA-architecture decoder-only model
   adapted with LoRA and a translation prompt.
4. **Gemma-2-2B-it** — a decoder-only instruction model adapted with LoRA and the
   same translation objective.

The language pair is **English -> French (en-fr)**. The selected dataset is the
public **OPUS Books** parallel corpus:
[OPUS Books source](https://opus.nlpl.eu/legacy/opus-books.php) and
[Hugging Face dataset card](https://huggingface.co/datasets/Helsinki-NLP/opus_books).
The experiment uses a deterministic, cleaned subset of 10,000 training pairs,
1,000 validation pairs, and 1,000 held-out test pairs. The split is created after
shuffling with seed `4403`, and sentences longer than 250 characters are removed.

## Repository layout

```text
configs/experiment.yaml       Reproducible dataset/model settings
src/data.py                   Download, clean, split, and save data
src/basic_transformer.py      Train the from-scratch Transformer
src/train_pretrained.py       Fine-tune mT5, TinyLlama, or Gemma with LoRA
src/evaluate.py               Generate the same test set and score all models
results/                      Metrics, samples, figures, and temperature outputs
docs/                         Ethical analysis and submission notes
```

## Installation

Python 3.10+ is recommended. A CUDA GPU is strongly recommended for mT5, LLaMA,
and Gemma. Install the dependencies with:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The Gemma checkpoint requires accepting Google's model terms on Hugging Face and
running `huggingface-cli login`. TinyLlama is used instead of the gated
Meta LLaMA checkpoint because it is an openly downloadable LLaMA-family model;
this makes the experiment reproducible without claiming access to gated weights.

## Execution

Run these commands from the repository root:

```bash
python -m src.data
python -m src.basic_transformer
python -m src.train_pretrained mt5
python -m src.train_pretrained llama
python -m src.train_pretrained gemma
python -m src.evaluate --limit 1000
```

The first command creates `data/opus_books_en_fr`. Training writes checkpoints
and model artifacts locally; these are intentionally gitignored because they are
large. Evaluation writes `results/metrics.csv`, `results/metrics.json`, and
`results/samples/test_translations.csv`. The same test sentences and references
are used for every model. BLEU is SacreBLEU corpus BLEU and chrF is SacreBLEU
corpus chrF. Inference time is reported both as total seconds and milliseconds
per sentence, while training time is recorded for trained models. Parameter
counts include total model parameters; LoRA trainable parameters are recorded
separately.

## Sampling-temperature analysis

The default translation comparison uses deterministic generation (`do_sample=False`)
and beam search for encoder-decoder models. For decoder-only models, a separate
temperature experiment should use the same held-out sentences at temperatures
0.2, 0.7, and 1.0, with `do_sample=True` for nonzero-temperature runs. Lower
temperature generally improves repeatability and preserves meaning; higher
temperature can increase lexical variety but also increases omissions, grammatical
errors, and hallucinated content. For production MT, deterministic decoding or a
carefully validated low temperature is preferable.

## Responsible and ethical AI analysis

Machine translation can reproduce corpus bias, erase dialect or gender nuance,
misrepresent culturally specific expressions, hallucinate fluent but unsupported
content, and mishandle medical, legal, personal, or otherwise sensitive text.
The evaluation should therefore include qualitative error inspection in addition
to BLEU and chrF. Recommended safeguards are representative and licensed data,
language-pair and subgroup audits, terminology constraints for high-risk domains,
human review, uncertainty/escalation rules, privacy-preserving processing, and
clear disclosure that an output is machine-generated.

## Expected comparison

The basic Transformer provides a transparent baseline but has no pretrained
knowledge and is data-hungry. mT5 is expected to be the strongest task-specialized
system because its encoder-decoder structure and multilingual pretraining match
translation directly. Decoder-only systems can translate through instruction
following and LoRA, but process a longer prompt and commonly require more
parameters and compute. Final claims must be based on the generated
`results/metrics.csv`, not on placeholder numbers.

## Submission document

The final LMS attachment must be named
`Madhavan_Komandur_230953298_FGAI_Assignment3.pdf` and contain the student's
name, registration number, assignment title, developed system title, dataset
name/link, and the public GitHub repository URL. Replace the repository URL
placeholder in the document once the GitHub repository has been created.
