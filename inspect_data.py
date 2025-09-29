# inspect_data.py
from pathlib import Path
import pandas as pd

BASE = Path("data/fastf1_sessions")

q = pd.read_parquet(BASE / "all_qualifying.parquet")
r = pd.read_parquet(BASE / "all_races.parquet")

print("Qualifying shape:", q.shape)
print("Race shape:", r.shape)

print("\nQualifying columns:", q.columns.tolist())
print("\nRace columns:", r.columns.tolist())

print("\nSample Qualifying rows:")
print(q.head())

print("\nSample Race rows:")
print(r.head())
