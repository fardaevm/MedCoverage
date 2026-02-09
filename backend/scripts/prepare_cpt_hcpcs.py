import argparse
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


def extract_keywords(
    descriptions: pd.Series,
    titles: pd.Series,
    top_n: int = 5,
) -> pd.Series:
    """Extract top-N keywords from each description using KeyBERT.

    Deduplicates against words already present in the title so the
    enriched search_text doesn't just repeat the same tokens.
    """
    from keybert import KeyBERT

    kw_model = KeyBERT("sentence-transformers/all-MiniLM-L6-v2")

    keywords_list = []
    total = len(descriptions)

    for idx, (desc, title) in enumerate(zip(descriptions, titles)):
        if idx % 500 == 0:
            print(f"  [keywords] {idx:,}/{total:,} …")

        desc = str(desc).strip() if pd.notna(desc) else ""
        title = str(title).strip() if pd.notna(title) else ""

        if not desc:
            keywords_list.append("")
            continue

        kws = kw_model.extract_keywords(
            desc,
            keyphrase_ngram_range=(1, 2),
            stop_words="english",
            top_n=top_n,
        )

        # Deduplicate against title words
        title_words = set(title.lower().split())
        unique = []
        for phrase, _score in kws:
            phrase_words = set(phrase.lower().split())
            if not phrase_words.issubset(title_words):
                unique.append(phrase)

        keywords_list.append(" ".join(unique))

    return pd.Series(keywords_list, index=descriptions.index)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare CPT/HCPCS data")
    parser.add_argument(
        "--no-keywords",
        action="store_true",
        help="Skip keyword extraction (baseline mode)",
    )
    args = parser.parse_args()

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

    # Remove retired/replaced CPT codes that pollute search results
    RETIRED_CODES = {
        "77053",  # Retired mammary ductogram, replaced by 77065-77067
        "77054",  # Retired mammary ductogram, replaced by 77065-77067
    }
    before = len(df)
    df = df[~df["code"].isin(RETIRED_CODES)]
    removed = before - len(df)
    if removed:
        print(f"[INFO] Removed {removed} retired codes: {RETIRED_CODES}")

    # Keyword extraction
    if args.no_keywords:
        print("[INFO] Skipping keyword extraction (--no-keywords)")
        df["keywords"] = ""
        df["search_text"] = (
            df["title"] + " " + df["category"] + " " + df["description"]
        ).str.strip()
    else:
        print("[INFO] Extracting keywords with KeyBERT …")
        df["keywords"] = extract_keywords(df["description"], df["title"])
        df["search_text"] = (
            df["title"]
            + " " + df["category"]
            + " " + df["description"]
            + " " + df["keywords"]
        ).str.strip()
        n_with_kw = (df["keywords"].str.len() > 0).sum()
        print(f"[INFO] Keywords extracted for {n_with_kw:,}/{len(df):,} rows")

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
