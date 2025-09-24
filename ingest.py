import argparse
import pandas as pd
from pathlib import Path

def ingest(source: str, out_dir: str = "data/processed") -> str:
    df = pd.read_csv(source)
    # simple cleaning example
    df = df.dropna(subset=["y"]).reset_index(drop=True)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out_path = str(Path(out_dir) / "dataset_clean.csv")
    df.to_csv(out_path, index=False)
    return out_path

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True)
    p.add_argument("--out", default="data/processed")
    args = p.parse_args()
    saved = ingest(args.source, args.out)
    print(saved)
