from pathlib import Path
from src.name_generator import load_model, generate

root = Path(__file__).resolve().parent
model, stoi, itos = load_model(root / "models")

for i in range(10):
    print(generate(model, stoi, itos, prefix="", seed=200 + i))
