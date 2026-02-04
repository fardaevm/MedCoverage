import re
import pandas as pd
from pathlib import Path


def clean_code(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip().upper()


def money_to_float(x):
    """Convert strings like '$576.00', '---', '--' to float/None."""
    if pd.isna(x):
        return None
    s = str(x).strip()
    if s in {"", "---", "--"}:
        return None
    s = s.replace("$", "").replace(",", "")
    s = re.sub(r"[^0-9\.\-]", "", s)
    try:
        return float(s)
    except ValueError:
        return None


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]

    in_path = project_root / "data" / "raw" / "medi_cal_rates.csv"
    out_dir = project_root / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    # Skip the first copyright/title row
    df = pd.read_csv(in_path, skiprows=1)

    # Normalize column names
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", "_", regex=True)
        .str.replace("%", "pct", regex=False)
    )

    # Your file uses these column names after skiprows=1:
    # proc_code, procedure_description, basic_rate, etc.
    required_cols = {"proc_code", "procedure_description", "basic_rate"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing expected columns {missing}. Found columns: {df.columns.tolist()}"
        )

    # Clean code + rates
    df["proc_code"] = df["proc_code"].apply(clean_code)
    df["basic_rate_num"] = df["basic_rate"].apply(money_to_float)

    # Drop empty codes
    df = df[df["proc_code"] != ""]

    # Keep the “best” row per code for MVP
    # Prefer non-null basic_rate_num; otherwise keep first
    df = df.sort_values(by=["proc_code", "basic_rate_num"], ascending=[True, False])
    df = df.drop_duplicates(subset=["proc_code"], keep="first")

    # Save clean outputs
    out_parquet = out_dir / "medi_cal_rates_clean.parquet"
    out_csv = out_dir / "medi_cal_rates_clean.csv"
    df.to_parquet(out_parquet, index=False)
    df.to_csv(out_csv, index=False)

    print(f"[OK] Saved: {out_parquet}")
    print(f"[OK] Saved: {out_csv}")
    print(f"[INFO] Rows: {len(df):,} | Non-null basic rates: {df['basic_rate_num'].notna().sum():,}")


if __name__ == "__main__":
    main()
