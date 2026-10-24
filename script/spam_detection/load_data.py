
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "enron_spam_data.csv"

print("Loading Enron-Spam dataset...")

df = pd.read_csv(DATA_PATH)

print("\nDataset shape:", df.shape)
print("\nColumn names:", df.columns.tolist())
print("\nFirst five rows:")
print(df.head().to_string())
