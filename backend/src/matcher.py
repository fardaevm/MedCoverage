from pathlib import Path
import re
from collections import Counter
import numpy as np
import pandas as pd
import faiss
from rank_bm25 import BM25Okapi
from symspellpy import SymSpell, Verbosity
from sentence_transformers import SentenceTransformer
from src.pricing import PricingLookup
import os
import cohere

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Blend weight: fraction of the final score coming from FAISS (semantic).
SEMANTIC_WEIGHT = 0.6


class Matcher:
    def __init__(self, project_root: Path, pricing_lookup: PricingLookup):
        idx_path = project_root / "data" / "embeddings" / "cpt_faiss.index"
        meta_path = project_root / "data" / "embeddings" / "cpt_meta.parquet"

        self.index = faiss.read_index(str(idx_path))
        self.meta = pd.read_parquet(meta_path)
        self.model = SentenceTransformer(MODEL_NAME)
        self.pricing = pricing_lookup

        self.co = None
        api_key = os.getenv("COHERE_API_KEY")
        if api_key:
            self.co = cohere.ClientV2(api_key=api_key)

        # Corpus texts used by both BM25 and the spell checker
        if "search_text" in self.meta.columns:
            corpus_texts = self.meta["search_text"].fillna("").tolist()
        else:
            corpus_texts = (
                self.meta.get("title", pd.Series([""] * len(self.meta))).fillna("").astype(str)
                + " "
                + self.meta.get("description", pd.Series([""] * len(self.meta))).fillna("").astype(str)
            ).tolist()

        # BM25 index (lexical)
        tokenized_corpus = [doc.lower().split() for doc in corpus_texts]
        self.bm25 = BM25Okapi(tokenized_corpus)

        # Spell checker built entirely from procedure vocabulary.
        # Using our own corpus means real medical terms are never "corrected away".
        self.spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
        word_counts: Counter = Counter()
        for text in corpus_texts:
            word_counts.update(re.findall(r"[a-z]{3,}", text.lower()))
        for word, count in word_counts.items():
            self.spell.create_dictionary_entry(word, count)

    # ------------------------------------------------------------------
    # Spell correction
    # ------------------------------------------------------------------

    def _correct_query(self, text: str) -> str:
        """Correct each pure-alphabetic token (4+ chars) using the corpus vocabulary.

        Tokens that look like codes (contain digits or are ≤3 chars) are left as-is
        so that queries like "77065" or "MRI" pass through unchanged.
        """
        tokens = text.lower().split()
        result = []
        for tok in tokens:
            if re.match(r"^[a-z]{4,}$", tok):
                suggestions = self.spell.lookup(tok, Verbosity.CLOSEST, max_edit_distance=2)
                result.append(suggestions[0].term if suggestions else tok)
            else:
                result.append(tok)
        corrected = " ".join(result)
        return corrected

    # ------------------------------------------------------------------
    # Reranking
    # ------------------------------------------------------------------

    def _rerank(self, query: str, candidates: list[dict], top_k: int) -> list[dict]:
        if not self.co or not candidates:
            for i, c in enumerate(candidates[:top_k]):
                c["rank"] = i + 1
            return candidates[:top_k]

        docs = [
            f"{c.get('title','')}\n{c.get('description','')}\nCode:{c.get('code','')}".strip()
            for c in candidates
        ]

        resp = self.co.rerank(
            model="rerank-v3.5",
            query=query,
            documents=docs,
            top_n=min(top_k, len(docs)),
        )

        out = []
        for r in resp.results:
            c = candidates[r.index]
            c["rank"] = len(out) + 1
            c["rerank_score"] = float(getattr(r, "relevance_score", 0.0))
            out.append(c)

        return out

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, text: str, top_k: int = 10):
        pre_k = min(max(top_k * 10, 50), 200)

        # Stage 0: spell-correct the raw query
        corrected = self._correct_query(text)

        # Stage 1a: FAISS semantic search (on corrected query)
        q = self.model.encode([corrected], normalize_embeddings=True).astype("float32")
        faiss_scores_raw, faiss_ids_raw = self.index.search(q, pre_k)
        faiss_score_map = {
            int(fid): float(s)
            for fid, s in zip(faiss_ids_raw[0], faiss_scores_raw[0])
            if fid >= 0
        }

        # Stage 1b: BM25 lexical search (on corrected query)
        tokens = corrected.lower().split()
        bm25_scores = self.bm25.get_scores(tokens)
        bm25_max = float(bm25_scores.max()) if bm25_scores.max() > 0 else 1.0
        bm25_norm = bm25_scores / bm25_max

        bm25_top_ids = np.argsort(bm25_scores)[::-1][:pre_k].tolist()
        bm25_top_ids = [i for i in bm25_top_ids if bm25_scores[i] > 0]

        # Stage 2: Hybrid union + blended score
        seen: set = set()
        candidate_ids: list = []
        for fid in list(faiss_score_map.keys()) + bm25_top_ids:
            if fid not in seen:
                seen.add(fid)
                candidate_ids.append(fid)

        scored = []
        for fid in candidate_ids:
            faiss_s = faiss_score_map.get(fid, 0.0)
            bm25_s = float(bm25_norm[fid])
            hybrid = SEMANTIC_WEIGHT * faiss_s + (1 - SEMANTIC_WEIGHT) * bm25_s
            scored.append((fid, hybrid))

        scored.sort(key=lambda x: x[1], reverse=True)
        top_ids = [fid for fid, _ in scored[:pre_k]]

        # Stage 3: Build candidate dicts
        candidates = []
        for fid in top_ids:
            row = self.meta.iloc[fid]
            code = row.get("code")
            pricing = self.pricing.lookup(code) if code else None

            raw_desc = row.get("description")
            desc = None if raw_desc is None else str(raw_desc).strip().strip('"')

            raw_title = row.get("title")
            title = None if raw_title is None else str(raw_title).strip().strip('"').rstrip(".")

            candidates.append({
                "faiss_id": fid,
                "code": code,
                "title": title,
                "category": row.get("category"),
                "description": desc,
                "basic_rate": pricing["basic_rate"] if pricing else None,
            })

        return self._rerank(corrected, candidates, top_k)
