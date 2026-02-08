from pathlib import Path
import pandas as pd


def _normalize_code(code: str) -> str:
    """Normalize a procedure code for lookup matching.

    Strips whitespace, uppercases, and removes leading zeros from
    purely-numeric codes so that Medi-Cal codes like '00100' match
    CPT metadata codes like '100'.
    """
    code = code.strip().upper()
    if code.isdigit():
        code = code.lstrip("0") or "0"
    return code


class PricingLookup:
    def __init__(self, project_root: Path):
        path = project_root / "data" / "processed" / "medi_cal_rates_clean.parquet"
        df = pd.read_parquet(path)

        self._rates: dict[str, dict] = {}
        for _, row in df.iterrows():
            key = _normalize_code(str(row["proc_code"]))
            self._rates[key] = {
                "basic_rate": float(row["basic_rate_num"]),
                "proc_type": row["proc_type"],
                "procedure_description": row["procedure_description"],
            }

    def lookup(self, code: str) -> dict | None:
        return self._rates.get(_normalize_code(code))

    def lookup_batch(self, codes: list[str]) -> list[dict | None]:
        return [self.lookup(c) for c in codes]
