from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from name_generator import build_vocab, load_names, generate


def test_dataset_loads():
    path = Path(__file__).resolve().parents[1] / "data" / "names.txt"
    names = load_names(path)
    assert len(names) > 250
    assert all(name == name.lower() for name in names)


def test_vocab_has_special_tokens():
    stoi, itos = build_vocab(["tom", "alice"])
    assert "<START>" in stoi and "<END>" in stoi and "<PAD>" in stoi
    assert len(stoi) == len(itos)


def test_generation_after_training(tmp_path):
    import torch
    from name_generator import CharLSTM

    stoi, itos = build_vocab(["tom", "alice", "bob"])
    model = CharLSTM(len(itos))
    output = generate(model, stoi, itos, prefix="a", seed=1)
    assert output.startswith("a")
