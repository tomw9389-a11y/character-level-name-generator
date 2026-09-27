# Character-Level Name Generator using LSTM

A small NLP portfolio project that trains a character-level recurrent neural network to learn spelling patterns from a name dataset and generate new names from an optional prefix.

## What it demonstrates

- Character-level text preprocessing
- Vocabulary construction and token encoding
- Sequence-to-sequence training for autoregressive generation
- LSTM model design in PyTorch
- Temperature-based probabilistic sampling
- Reproducible training and inference
- Unit tests for data loading, vocabulary and generation

## Project structure

```text
ai_lstm_name_generator/
├── data/names.txt
├── models/                 # generated model artifacts (gitignored)
├── src/name_generator.py
├── tests/test_name_generator.py
├── train.py
├── generate.py
└── requirements.txt
```

## Run

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python train.py
python -m src.name_generator --prefix "al" --samples 10
pytest -q
```

## Portfolio extension ideas

Add an evaluation script, a web UI, experiment tracking, a larger curated dataset, and a model-comparison notebook for GRU vs LSTM.

## Attribution

Inspired by the App Ideas repository's “Name Generation using RNNs” project specification. The implementation in this repository is independently structured and extended for portfolio use.
