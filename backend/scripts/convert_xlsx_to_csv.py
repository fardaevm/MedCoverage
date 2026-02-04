import pandas as pd
from pathlib import Path


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]

    input_path = project_root / "data" / "raw" / "rates_data.xlsx"
    output_path = project_root / "data" / "raw" / "medi_cal_rates.csv"

    # Load first sheet by default
    df = pd.read_excel(input_path)

    # Save as CSV
    df.to_csv(output_path, index=False)

    print(f"[OK] Converted XLSX → CSV: {output_path}")


if __name__ == "__main__":
    main()
