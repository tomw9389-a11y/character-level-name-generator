from pathlib import Path
from src.name_generator import train

root = Path(__file__).resolve().parent
train(root / "data" / "names.txt", root / "models")
