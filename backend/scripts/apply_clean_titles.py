"""Apply patient-friendly clean titles to cpt_hcpcs_enriched.parquet.

Run this after prepare_cpt_hcpcs.py to patch in the manually curated titles
stored in data/processed/clean_titles_88.json. For codes that have a clean
title, search_text is rebuilt as:

    clean_title + category + description + keywords

This gives each code a distinct embedding rather than sharing the generic
CPT breadcrumb (e.g. "Breast Mammography.Diagnostic Radiology. Mammography").

Usage:
    python backend/scripts/apply_clean_titles.py
    python backend/scripts/apply_clean_titles.py --src cpt_hcpcs_clean.parquet
"""

from __future__ import annotations

import argparse
import json
import pandas as pd
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src",
        default="cpt_hcpcs_enriched.parquet",
        help="Source parquet filename under data/processed/ (default: cpt_hcpcs_enriched.parquet)",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    processed_dir = project_root / "data" / "processed"

    src_path = processed_dir / args.src
    titles_path = processed_dir / "clean_titles_88.json"
    out_path = processed_dir / "cpt_hcpcs_enriched.parquet"

    if not src_path.exists():
        raise FileNotFoundError(f"Source parquet not found: {src_path}")
    if not titles_path.exists():
        raise FileNotFoundError(f"Clean titles JSON not found: {titles_path}")

    df = pd.read_parquet(src_path)
    clean_titles: dict = json.loads(titles_path.read_text())

    # Normalise codes to uppercase for matching
    df["code"] = df["code"].str.upper()
    clean_titles = {k.upper(): v for k, v in clean_titles.items()}

    # Add / overwrite clean_title column
    df["clean_title"] = df["code"].map(clean_titles)

    # Rebuild search_text only for rows that have a clean title
    mask = df["clean_title"].notna()

    def _build_search_text(row) -> str:
        parts = [
            str(row.get("clean_title") or ""),
            str(row.get("category") or ""),
            str(row.get("description") or ""),
            str(row.get("keywords") or ""),
        ]
        return " ".join(p.strip() for p in parts if p.strip())

    df.loc[mask, "search_text"] = df[mask].apply(_build_search_text, axis=1)

    patched = int(mask.sum())
    df.to_parquet(out_path, index=False)

    print(f"[OK] Patched {patched} rows with clean titles")
    print(f"[OK] Saved → {out_path}")
    print(f"[INFO] Total rows: {len(df):,}")


if __name__ == "__main__":
    main()
