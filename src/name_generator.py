from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import torch
from torch import nn

PAD = "<PAD>"
START = "<START>"
END = "<END>"


class CharLSTM(nn.Module):
    """Small character-level LSTM for autoregressive name generation."""

    def __init__(self, vocab_size: int, embedding_dim: int = 32, hidden_dim: int = 64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x: torch.Tensor, hidden=None):
        x = self.embedding(x)
        out, hidden = self.lstm(x, hidden)
        return self.output(out), hidden


def load_names(path: Path) -> list[str]:
    names = []
    for line in path.read_text(encoding="utf-8").splitlines():
        name = line.strip().lower()
        if name:
            names.append(name)
    return sorted(set(names))


def build_vocab(names: list[str]):
    chars = sorted(set("".join(names) + " -'"))
    itos = [PAD, START, END] + chars
    stoi = {ch: i for i, ch in enumerate(itos)}
    return stoi, itos


def make_examples(names: list[str], stoi: dict[str, int]):
    examples = []
    for name in names:
        ids = [stoi[START]] + [stoi[c] for c in name] + [stoi[END]]
        examples.append((ids[:-1], ids[1:]))
    return examples


def pad_batch(examples, pad_id: int):
    max_len = max(len(x[0]) for x in examples)
    xs, ys = [], []
    for x, y in examples:
        xs.append(x + [pad_id] * (max_len - len(x)))
        ys.append(y + [pad_id] * (max_len - len(y)))
    return torch.tensor(xs, dtype=torch.long), torch.tensor(ys, dtype=torch.long)


def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)


def train(data_path: Path, model_dir: Path, epochs: int = 120, lr: float = 0.01, seed: int = 42):
    set_seed(seed)
    names = load_names(data_path)
    stoi, itos = build_vocab(names)
    examples = make_examples(names, stoi)
    model = CharLSTM(len(itos))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=stoi[PAD])

    for epoch in range(1, epochs + 1):
        random.shuffle(examples)
        total_loss = 0.0
        for start in range(0, len(examples), 64):
            batch = examples[start:start + 64]
            x, y = pad_batch(batch, stoi[PAD])
            optimizer.zero_grad()
            logits, _ = model(x)
            loss = loss_fn(logits.reshape(-1, len(itos)), y.reshape(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += float(loss.item())

        if epoch == 1 or epoch % 20 == 0 or epoch == epochs:
            avg = total_loss / math.ceil(len(examples) / 64)
            print(f"epoch={epoch:03d} loss={avg:.4f}")

    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_dir / "model.pt")
    (model_dir / "vocab.json").write_text(json.dumps({"stoi": stoi, "itos": itos}, indent=2), encoding="utf-8")
    print(f"saved model to {model_dir}")


def load_model(model_dir: Path):
    payload = json.loads((model_dir / "vocab.json").read_text(encoding="utf-8"))
    stoi = {k: int(v) for k, v in payload["stoi"].items()}
    itos = payload["itos"]
    model = CharLSTM(len(itos))
    model.load_state_dict(torch.load(model_dir / "model.pt", map_location="cpu"))
    model.eval()
    return model, stoi, itos


def generate(model, stoi, itos, prefix: str = "", max_len: int = 15, temperature: float = 0.9, seed: int = 42):
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    torch.manual_seed(seed)
    prefix = "".join(ch for ch in prefix.lower() if ch in stoi and ch not in {START, END, PAD})
    ids = [stoi[START]] + [stoi[ch] for ch in prefix]
    hidden = None
    with torch.no_grad():
        logits, hidden = model(torch.tensor([ids], dtype=torch.long), hidden)
        current = torch.tensor([[ids[-1]]], dtype=torch.long)
        name = list(prefix)
        for _ in range(max_len - len(name)):
            step_logits, hidden = model(current, hidden)
            probs = torch.softmax(step_logits[:, -1, :] / temperature, dim=-1)
            next_id = int(torch.multinomial(probs, 1).item())
            token = itos[next_id]
            if token == END:
                break
            if token not in {PAD, START}:
                name.append(token)
            current = torch.tensor([[next_id]], dtype=torch.long)
    return "".join(name)


def main():
    parser = argparse.ArgumentParser(description="Character-level LSTM name generator")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--prefix", default="")
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=120)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    data_path = root / "data" / "names.txt"
    model_dir = root / "models"

    if args.train or not (model_dir / "model.pt").exists():
        train(data_path, model_dir, epochs=args.epochs)

    model, stoi, itos = load_model(model_dir)
    for i in range(args.samples):
        print(generate(model, stoi, itos, prefix=args.prefix, seed=100 + i))


if __name__ == "__main__":
    main()
