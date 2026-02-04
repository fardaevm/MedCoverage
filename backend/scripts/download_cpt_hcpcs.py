from datasets import load_dataset
from pathlib import Path


def main() -> None:
    """
    Download CPT/HCPCS dataset from Hugging Face and store it
    as a raw CSV file under data/raw/.
    """

    # Resolve project root (src/scripts -> project root)
    project_root = Path(__file__).resolve().parents[2]

    data_raw_dir = project_root / "data" / "raw"
    data_raw_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    dataset = load_dataset("atta00/cpt-hcpcs-codes")

    # Convert to pandas DataFrame
    df = dataset["train"].to_pandas()

    # Save to CSV
    output_path = data_raw_dir / "cpt_hcpcs_hf.csv"
    df.to_csv(output_path, index=False)

    print(f"[OK] CPT/HCPCS dataset saved to: {output_path}")


if __name__ == "__main__":
    main()
