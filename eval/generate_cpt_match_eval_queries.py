"""
Policy-grounded eval set generator for Phase 1 (CPT retrieval).

For each target CPT code, pulls the clean title + description from the enriched
parquet and retrieves relevant context from the indexed policy docs (LanceDB).
Uses GPT-4o-mini to generate diverse patient queries at multiple verbosity levels.

Outputs:
    data/sample_data/eval/policy_eval_set.csv

Usage:
    python -m eval.generate_policy_eval_set
    python -m eval.generate_policy_eval_set --queries-per-code 8
    python -m eval.generate_policy_eval_set --save data/sample_data/eval/policy_eval_set.csv
"""

import argparse
import json
import re
import sys
from pathlib import Path

import lancedb
import pandas as pd
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from openai import OpenAI

# ── All 91 policy-grounded CPT codes (from evwoman.pdf + mammography_extracted_pages.pdf) ──
POLICY_CODES = [
    "10004", "10005", "10006", "10007", "10008", "10011", "10012", "10021",
    "15600",
    "19000", "19001", "19081", "19082", "19083", "19084", "19085", "19086",
    "19100", "19101", "19120", "19125", "19126",
    "19281", "19282", "19283", "19284", "19285", "19286", "19287", "19288",
    "19300",
    "57452", "57454", "57455", "57456", "57500", "57505",
    "58100", "58110",
    "76098", "76641", "76642", "76942",
    "77046", "77047", "77048", "77049",
    "77053", "77061", "77062", "77063", "77065", "77066", "77067",
    "81025",
    "87624", "87625",
    "88141", "88142", "88143", "88164", "88172", "88173", "88174", "88175",
    "88177", "88305", "88307", "88331", "88332", "88341", "88342", "88360",
    "88364", "88365", "88366", "88367", "88368", "88369", "88373", "88374",
    "88377",
    "95852",
    "99070", "99202", "99203", "99204", "99211", "99212", "99213", "99214",
]

SYSTEM_PROMPT = """You are generating a test dataset for a medical cost estimator app.
A patient is trying to find out if a specific medical procedure is covered by Medi-Cal.
They type a plain-English description of what they need.

Your task: given a CPT code, its description, and policy context, generate realistic patient queries.

Rules:
- Write queries as a patient would type them, NOT as a doctor or biller
- No CPT codes in the queries
- No medical jargon the patient wouldn't know
- Vary the phrasing — don't repeat the same structure
- Each query must unambiguously point to THIS code, not a similar one

Verbosity levels to generate:
- direct: 2-5 words (e.g. "screening mammogram")
- short: one short sentence (e.g. "is a mammogram covered by medi-cal")
- conversational: 1-2 sentences with some personal context
- verbose: 2-4 sentences, rambling, with extra irrelevant detail

Return ONLY a JSON array of objects, no other text:
[
  {"query": "...", "verbosity": "direct"},
  {"query": "...", "verbosity": "short"},
  {"query": "...", "verbosity": "conversational"},
  {"query": "...", "verbosity": "verbose"}
]
"""


def load_code_metadata(project_root: Path) -> dict:
    """Load clean_title and description for all policy codes from enriched parquet."""
    meta_path = project_root / "data" / "processed" / "cpt_hcpcs_enriched.parquet"
    df = pd.read_parquet(meta_path)
    result = {}
    for code in POLICY_CODES:
        row = df[df["code"] == code]
        if len(row) == 0:
            continue
        r = row.iloc[0]
        clean_title = str(r.get("clean_title") or "").strip()
        if not clean_title or clean_title.startswith("["):
            clean_title = str(r.get("title") or "").strip().strip('"').rstrip(".")
        description = str(r.get("description") or "").strip().strip('"')
        result[code] = {"title": clean_title, "description": description}
    return result


def load_policy_context(project_root: Path) -> dict:
    """For each policy code, pull the most relevant chunk text from LanceDB."""
    db_path = project_root / "backend" / "data" / "sample-lancedb"
    if not db_path.exists():
        return {}

    db = lancedb.connect(str(db_path))
    if not db.table_names():
        return {}

    df = db.open_table(db.table_names()[0]).to_pandas()
    text_col = next((c for c in df.columns if c in ("content", "text")), None)
    if not text_col:
        return {}

    context_map = {}
    for code in POLICY_CODES:
        # Find chunks that explicitly mention this code
        matches = df[df[text_col].str.contains(code, na=False)]
        if len(matches) == 0:
            context_map[code] = ""
            continue
        # Concatenate up to 2 most relevant chunks, capped at 400 chars each
        snippets = []
        for _, row in matches.head(2).iterrows():
            text = str(row[text_col]).strip()
            # Extract sentence(s) around the code mention
            sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if code in s]
            if sentences:
                snippets.append(" ".join(sentences[:2])[:400])
            else:
                snippets.append(text[:400])
        context_map[code] = " | ".join(snippets)

    return context_map


def generate_queries(
    client: OpenAI,
    code: str,
    title: str,
    description: str,
    policy_context: str,
    n: int,
) -> list[dict]:
    """Generate n queries per code using GPT-4o-mini."""
    sets_needed = max(1, n // 4)

    user_msg = f"""CPT Code: {code}
Title: {title}
Description: {description[:300]}
Policy context: {policy_context[:400] if policy_context else "Covered under Medi-Cal women's health services."}

Generate {sets_needed} complete set(s) of 4 queries (one per verbosity level).
Return a flat JSON array of {sets_needed * 4} query objects."""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.8,
    )

    raw = response.choices[0].message.content.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    queries = json.loads(raw)
    return [{"code": code, "query": q["query"], "verbosity": q["verbosity"]} for q in queries]


def main():
    parser = argparse.ArgumentParser(description="Generate policy-grounded eval set")
    parser.add_argument("--queries-per-code", type=int, default=8, help="Queries per CPT code (multiple of 4)")
    parser.add_argument("--save", type=str, default="data/sample_data/eval/policy_eval_set.csv")
    args = parser.parse_args()

    client = OpenAI()

    print("Loading code metadata from enriched parquet...")
    code_meta = load_code_metadata(PROJECT_ROOT)
    print(f"  Found metadata for {len(code_meta)}/{len(POLICY_CODES)} codes")

    print("Loading policy context from LanceDB...")
    policy_context = load_policy_context(PROJECT_ROOT)
    grounded = sum(1 for v in policy_context.values() if v)
    print(f"  Found context for {grounded}/{len(POLICY_CODES)} codes")

    rows = []
    errors = []
    total = len(code_meta)

    for i, (code, meta) in enumerate(code_meta.items(), 1):
        print(f"[{i}/{total}] {code}: {meta['title'][:50]}...", end=" ", flush=True)
        try:
            queries = generate_queries(
                client=client,
                code=code,
                title=meta["title"],
                description=meta["description"],
                policy_context=policy_context.get(code, ""),
                n=args.queries_per_code,
            )
            rows.extend(queries)
            print(f"done ({len(queries)} queries)")
        except Exception as e:
            print(f"ERROR: {e}")
            errors.append(code)

    df = pd.DataFrame(rows)
    save_path = PROJECT_ROOT / args.save
    save_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_path, index=False)

    print(f"\nSaved {len(df)} queries -> {save_path}")
    print(f"Codes: {df['code'].nunique()} | Verbosities: {df['verbosity'].value_counts().to_dict()}")
    if errors:
        print(f"Failed codes: {errors}")

    print("\nSample (first query per code):")
    print(df.groupby("code").first()[["verbosity", "query"]].to_string())


if __name__ == "__main__":
    main()
