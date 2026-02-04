import re
import pandas as pd
from pathlib import Path


def clean_text(s: str) -> str:
    if pd.isna(s):
        return ""
    s = str(s)
    s = s.replace("\u00a0", " ")          # non-breaking spaces
    s = re.sub(r"\s+", " ", s).strip()    # collapse whitespace
    return s


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]

    in_path = project_root / "data" / "raw" / "cpt_hcpcs_hf.csv"
    out_dir = project_root / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path)

    # ---- Standardize columns (adjust if your column names differ)
    df.columns = [c.strip().lower() for c in df.columns]

    # Required columns (common: code, title, description, category, type)
    for col in ["code", "title", "description", "category", "type"]:
        if col in df.columns:
            df[col] = df[col].apply(clean_text)

    df["code"] = df["code"].str.upper()
    df["type"] = df["type"].str.upper()

    # Optional: drop duplicates by code/type (keep first)
    df = df.drop_duplicates(subset=["code", "type"], keep="first")

    # Build search text for matching
    df["search_text"] = (df["title"] + " " + df["category"] + " " + df["description"]).str.strip()

    # Save
    out_parquet = out_dir / "cpt_hcpcs_clean.parquet"
    out_csv = out_dir / "cpt_hcpcs_clean.csv"

    df.to_parquet(out_parquet, index=False)
    df.to_csv(out_csv, index=False)

    print(f"[OK] Saved: {out_parquet}")
    print(f"[OK] Saved: {out_csv}")
    print(f"[INFO] Rows: {len(df):,} | Cols: {len(df.columns)}")


if __name__ == "__main__":
    main()
