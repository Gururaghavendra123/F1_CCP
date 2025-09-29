# download_fastf1_safe.py
import fastf1
import pandas as pd
from pathlib import Path
import time
import sys

# CONFIG
OUT_DIR = Path("data/fastf1_sessions")
CACHE_DIR = Path("f1_cache")
SEASONS = [2025]           # change to [2023,2024,2025] when you're comfortable
SESSION_TYPES = ['Q', 'R'] # or ['FP1','FP2','FP3','Q','R']
MAX_SESSIONS = 50         # STOPPER: maximum sessions to process in one run
DELAY_BETWEEN = 0.2       # seconds between session loads (reduce server stress)
RESUME = True             # skip files already saved

OUT_DIR.mkdir(parents=True, exist_ok=True)
fastf1.Cache.enable_cache(str(CACHE_DIR))

processed = 0
rows_q = []
rows_r = []

try:
    for season in SEASONS:
        print(f"\n=== Season {season} ===")
        for rnd in range(1, 31):
            if processed >= MAX_SESSIONS:
                print(f"Reached MAX_SESSIONS ({MAX_SESSIONS}) — stopping early.")
                raise KeyboardInterrupt

            for s_type in SESSION_TYPES:
                fname = OUT_DIR / f"{season}_round{rnd:02d}_{s_type}.csv"
                if RESUME and fname.exists():
                    print(f"Skipping (exists): {fname.name}")
                    continue

                try:
                    session = fastf1.get_session(season, rnd, s_type)
                except Exception:
                    # no such session configured
                    continue

                print(f"Loading {season} R{rnd} {s_type} ...", end=" ")
                try:
                    session.load(laps=False, telemetry=False)
                except Exception as e:
                    print(f"FAILED to load: {e}")
                    continue

                if session.results is None or session.results.empty:
                    print("no results, skipping.")
                    continue

                df = session.results.reset_index(drop=True).copy()
                # add metadata
                df['season'] = season
                df['round'] = rnd
                df['session_type'] = s_type
                df['event_name'] = getattr(session, "event", {}).get("EventName") if hasattr(session, "event") else None
                df['circuit'] = getattr(session, "event", {}).get("Location") if hasattr(session, "event") else None
                # save per-session
                try:
                    cols_keep = [c for c in ['Abbreviation','TeamName','Position','Q1','Q2','Q3','season','round','session_type'] if c in df.columns]
                    df[cols_keep].to_csv(fname, index=False)
                    print(f"Saved {fname.name} ({len(df)} rows)")
                    processed += 1
                except Exception as e:
                    print(f"Failed to save {fname.name}: {e}")

                # accumulate small lists for combined export (optional)
                if s_type == 'Q': rows_q.append(df)
                if s_type == 'R': rows_r.append(df)

                time.sleep(DELAY_BETWEEN)

except KeyboardInterrupt:
    print("\nUser requested stop or reached MAX_SESSIONS. Exiting gracefully.")

# Optional: write combined CSVs for what we downloaded in this run
def combine_and_save(rows, out_csv: Path):
    if not rows:
        return
    combined = pd.concat(rows, ignore_index=True, sort=False)
    combined.to_csv(out_csv, index=False)
    print(f"Combined saved to {out_csv} ({len(combined)} rows)")

combine_and_save(rows_q, OUT_DIR / "all_qualifying_partial.csv")
combine_and_save(rows_r, OUT_DIR / "all_races_partial.csv")

print("Done. You can re-run to continue (RESUME=True).")
