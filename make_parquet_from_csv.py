import pandas as pd
from pathlib import Path

BASE = Path("data/fastf1_sessions")

for name in ["all_qualifying", "all_races"]:
    csv_file = BASE / f"{name}.csv"
    pq_file = BASE / f"{name}.parquet"

    if csv_file.exists():
        df = pd.read_csv(csv_file)
        df.to_parquet(pq_file, index=False)
        print(f"Converted {csv_file} -> {pq_file} ({df.shape})")
    else:
        print(f"CSV not found: {csv_file}")
