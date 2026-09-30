# MedCoverage

**Medi-Cal procedure coverage eligibility engine — semantic search, RAG-powered decision trees, and deterministic evaluation.**

A production-grade system that helps Medi-Cal beneficiaries determine whether a medical procedure is covered under their plan. The user searches for a procedure in natural language, and the system walks them through a short eligibility questionnaire generated from actual policy documents — delivering a clear *covered* or *not covered* answer with reasoning.

Built as a full-stack application with a FastAPI backend, custom RAG pipeline, hybrid search (FAISS + BM25 + Cohere reranking), and a warm dark-themed frontend. Deployed on AWS with EC2, ElastiCache, and S3.

---

## Table of Contents

- [Why This Exists](#why-this-exists)
- [Architecture](#architecture)
- [How It Works](#how-it-works)
- [Technical Stack](#technical-stack)
- [Search Pipeline](#search-pipeline)
- [Eligibility Engine](#eligibility-engine)
- [RAG Pipeline](#rag-pipeline)
- [Confidence Scoring](#confidence-scoring)
- [Frontend](#frontend)
- [Infrastructure & Deployment](#infrastructure--deployment)
- [API Reference](#api-reference)
- [Local Development](#local-development)
- [Evaluation](#evaluation)
- [Project Structure](#project-structure)

---

## Why This Exists

Medi-Cal policy documents are dense, legalistic, and spread across hundreds of pages. A patient searching "is a mammogram covered?" shouldn't need to parse regulatory language to get an answer. MedCoverage translates that complexity into a few yes/no questions and a definitive result.

The system is designed around three principles:

1. **No guessing** — if coverage can't be determined, the system says so.
2. **Transparency** — every answer traces back to policy context.
3. **Minimal LLM dependency at runtime** — one LLM call generates the decision tree upfront; all subsequent turns are deterministic.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Client (Browser)                        │
│         Landing Page  ──▶  Search UI  ──▶  Eligibility Flow     │
└──────────────────────────────┬──────────────────────────────────┘
                               │ HTTP
┌──────────────────────────────▼──────────────────────────────────┐
│                        FastAPI Backend                           │
│                                                                  │
│  POST /match ──────────▶  Matcher                                │
│    ┌──────────────────────────────────────────────┐              │
│    │  Spell Correct ▶ FAISS (semantic)            │              │
│    │                 + BM25  (lexical)             │              │
│    │                 ▶ Hybrid Blend                │              │
│    │                 ▶ Cohere Rerank               │              │
│    └──────────────────────────────────────────────┘              │
│                                                                  │
│  POST /eligibility/start ──▶  RAG Pipeline ──▶ LLM (1 call)     │
│    ┌──────────────────────────────────────────────┐              │
│    │  Retrieve policy context (LanceDB + Cohere)  │              │
│    │  Generate pathways + questions (GPT-4o-mini)  │              │
│    │  Cache result in Redis                        │              │
│    └──────────────────────────────────────────────┘              │
│                                                                  │
│  POST /eligibility/next ──▶  Deterministic Evaluator (0 calls)   │
│    ┌──────────────────────────────────────────────┐              │
│    │  Evaluate answers against pathway questions   │              │
│    │  Smart question picker (max pathway kill)     │              │
│    │  Return: covered / not_covered / next_question│              │
│    └──────────────────────────────────────────────┘              │
│                                                                  │
├──────────────────────────────────────────────────────────────────┤
│  Redis (cache)    │  FAISS Index    │  LanceDB    │  S3 (data)   │
└──────────────────────────────────────────────────────────────────┘
```

---

## How It Works

**Step 1 — Search.** The user types a natural language query (e.g., "breast MRI", "is a mammogram covered?"). The Matcher runs hybrid search across 10,000+ CPT/HCPCS procedure codes using semantic embeddings (FAISS) blended with lexical matching (BM25), followed by Cohere reranking to surface the best matches.

**Step 2 — Pathway Generation.** When the user selects a procedure, the system retrieves relevant Medi-Cal policy excerpts via RAG (LanceDB vector store + Cohere reranking). A single LLM call analyzes the policy context and produces distinct coverage pathways — each with 2–4 yes/no eligibility questions. This result is cached in Redis so identical lookups are instant.

**Step 3 — Deterministic Evaluation.** Each subsequent yes/no answer is evaluated purely in code — zero LLM calls. The evaluator tracks which pathways are still alive, which are eliminated, and picks the next question using a greedy strategy that maximizes pathway elimination per question. If all questions for any pathway are answered YES → covered. If all pathways are dead → not covered.

---

## Technical Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **API** | FastAPI + Uvicorn | Async HTTP server with Pydantic validation |
| **Search — Semantic** | FAISS (IndexFlatIP) + all-MiniLM-L6-v2 | Cosine similarity over 384-dim embeddings |
| **Search — Lexical** | BM25Okapi (rank-bm25) | Token-level term frequency matching |
| **Search — Rerank** | Cohere rerank-v3.5 | Cross-encoder reranking of hybrid candidates |
| **Search — Spelling** | SymSpell | Corpus-vocabulary spell correction |
| **RAG — Vector Store** | LanceDB | Embedded vector DB for policy documents |
| **RAG — Chunking** | Docling (HybridChunker) | PDF → semantic chunks with heading context |
| **RAG — Embeddings** | OpenAI text-embedding-3-small | 1536-dim document embeddings |
| **RAG — Retrieval** | Cohere rerank-v3.5 | Reranked top-k context retrieval |
| **LLM** | GPT-4o-mini | Pathway + question generation (single call) |
| **Cache** | Redis 7 | TTL-based JSON caching for generated trees |
| **Frontend** | Vanilla JS + CSS | Dark-themed UI, no framework dependencies |
| **Infra** | AWS EC2 + ALB + ElastiCache + S3 + ECR | Production deployment via CloudFormation |
| **Container** | Docker (multi-stage) | Reproducible builds with Poetry |

---

## Search Pipeline

The Matcher implements a four-stage retrieval pipeline:

### Stage 0 — Spell Correction
Queries are corrected against the procedure vocabulary using SymSpell (edit distance ≤ 2). Only alphabetic tokens of 4+ characters are corrected — codes like `77067` and abbreviations like `MRI` pass through untouched. This prevents the common failure mode of medical spell checkers that "correct" valid clinical terms.

### Stage 1 — Hybrid Retrieval
Two parallel retrievals run on the corrected query:

- **FAISS semantic search** — the query is encoded with all-MiniLM-L6-v2 (normalized L2 → cosine via inner product). Top-200 candidates retrieved.
- **BM25 lexical search** — standard Okapi BM25 over tokenized `search_text` fields. Scores are min-max normalized.

### Stage 2 — Score Blending
Candidates from both sources are unioned and scored:

```
hybrid_score = 0.6 × faiss_score + 0.4 × bm25_score
```

The 60/40 semantic-lexical split balances the strengths of each: semantic captures intent ("is a mammogram covered?"), lexical captures exact terms ("77067").

### Stage 3 — Cohere Reranking
Top candidates are reranked by Cohere's cross-encoder (`rerank-v3.5`), which scores each (query, document) pair jointly. This final stage corrects ranking errors that slip through the lightweight first-pass retrievers.

---

## Eligibility Engine

The eligibility system is split into two phases by design — minimizing LLM dependency and latency at interaction time.

### Phase 1 — Generation (1 LLM call, cached)

A single prompt instructs GPT-4o-mini to:

1. Read the retrieved policy context
2. Identify distinct coverage pathways (e.g., "routine screening for 40+", "diagnostic follow-up after abnormal results", "high-risk with BRCA mutation")
3. For each pathway, generate 2–4 yes/no questions where YES = patient meets the requirement
4. Return structured JSON

The prompt enforces critical constraints:
- Each pathway is a single scenario with one set of requirements
- Alternative qualifying conditions get separate pathways (enabling independent evaluation)
- Shared conditions (e.g., "365+ days since last screening") are repeated across pathways
- No administrative questions (prior auth, referrals, billing)

Result is cached in Redis (TTL: 7 days) keyed on `(procedure_code, query)`.

### Phase 2 — Evaluation (0 LLM calls, deterministic)

```python
# Core algorithm in eligibility_evaluator.py

For each pathway:
    if ALL questions answered YES → COVERED (immediate return)
    if ANY question answered NO  → pathway eliminated
    if unanswered questions remain and no NO → pathway alive

If alive pathways exist:
    Pick the unanswered question that appears in the MOST alive pathways
    → One NO answer kills maximum branches simultaneously

If no alive pathways:
    → NOT COVERED
```

The question picker is a greedy set-cover heuristic. If "Has it been 365+ days since your last screening?" appears in 3 of 4 alive pathways, it's asked first — a single NO eliminates 3 pathways at once, minimizing the number of questions the patient must answer.

---

## RAG Pipeline

The RAG system indexes Medi-Cal policy PDFs and retrieves relevant context for the LLM.

**Indexing:**
1. PDFs are converted with Docling's `DocumentConverter`
2. Documents are chunked with `HybridChunker` (semantic boundaries + heading preservation)
3. Each chunk is embedded with OpenAI `text-embedding-3-small` (1536 dimensions)
4. Chunks are stored in LanceDB with `merge_insert` (upsert by source)

**Retrieval:**
1. Query is embedded with the same model
2. LanceDB vector search returns top-k × 3 candidates
3. Cohere `rerank-v3.5` reranks and returns the final top-k

This over-retrieve-then-rerank pattern consistently outperforms single-stage retrieval in practice.

---

## Confidence Scoring

The system computes a **Policy Clarity Score** — a measure of how clearly the policy rules apply to the case. This is explicitly *not* a probability of payment.

```
confidence = 0.15
            + 0.45 × match_strength      (semantic similarity, normalized)
            + 0.30 × answer_strength      (proportion of YES answers)
            + 0.10 × completeness         (proportion of questions answered)
            − contradiction_penalty       (0.35 if contradictory answers detected)

confidence ∈ [0, 1], capped at 0.55 if decision = uncertain
```

---

## Frontend

The UI is built with vanilla JavaScript and CSS — no framework, no build step, no dependencies. Two pages:

- **Landing page** (`landing.html`) — explains the product, links to the tool
- **App** (`index.html`) — search bar, result cards, inline eligibility flow

Design decisions:
- Dark warm theme (`#111110` base) with orange accent (`#D4782A`)
- DM Sans typeface
- Collapsible sidebar with search history (sessionStorage)
- Loading states with animated dot phrases
- Answer history rendered inline with color-coded yes/no indicators
- Outcome boxes (covered = green, not covered = red) with policy reasoning

---

## Infrastructure & Deployment

Production runs on AWS with a single EC2 instance behind an ALB:

```
Internet → ALB (port 80) → EC2 t3.medium (port 8000) → Docker container
                                    ↕
                              ElastiCache Redis
                                    ↕
                              S3 (model data)
```

**CloudFormation provisions:** VPC with public/private subnets, internet gateway, NAT routing, security groups, IAM roles with least-privilege, ElastiCache Redis, ALB with health checks, Auto Scaling Group (min=max=1), and EC2 with a UserData bootstrap script.

**Bootstrap sequence:** The EC2 instance installs Docker, authenticates to ECR, fetches API keys from Secrets Manager, pulls the container image, and starts it with Redis and S3 environment variables. The entrypoint script syncs FAISS indexes and LanceDB data from S3 before launching Uvicorn.

Deploy with one command:
```bash
./deploy.sh
```

---

## API Reference

### `POST /match`
Search procedures by natural language query.
```json
{ "text": "mammogram", "top_k": 5 }
```
Returns ranked candidates with codes, titles, descriptions, and rerank scores.

### `POST /eligibility/start`
Generate coverage pathways and the first eligibility question.
```json
{
  "user_text": "I need a mammogram",
  "selected": { "faiss_id": 42, "code": "77067", "title": "Screening mammography" },
  "top_k": 10
}
```
Returns pathways, questions per pathway, and the opening question.

### `POST /eligibility/next`
Submit an answer and get the next question or final decision. Zero LLM calls.
```json
{
  "selected": { "faiss_id": 42, "code": "77067", "title": "Screening mammography" },
  "pathways": ["..."],
  "questions_per_pathway": [["..."]],
  "qa_so_far": [{ "q": "Are you 40 or older?", "a": true }]
}
```
Returns `decision` (covered/not_covered/uncertain), `reason`, and optionally `next_question`.

### `GET /health`
Returns service status, Redis connectivity, and cache state.

---

## Local Development

### Prerequisites
- Python 3.12+
- Docker & Docker Compose
- OpenAI API key
- Cohere API key (optional, degrades gracefully)

### Setup
```bash
cd backend
cp .env.example .env  # add your API keys

# Install dependencies
poetry install

# Run data pipeline
python scripts/download_cpt_hcpcs.py
python scripts/prepare_cpt_hcpcs.py
python scripts/build_cpt_fiass_index.py
python scripts/convert_xlsx_to_csv.py
python scripts/prepare_medi_cal_rates.py

# Index policy documents
python -m src.rag_cli add

# Start with Docker Compose
cd ..
docker-compose up
```

App runs at `http://localhost:8000`.

### Running Without Docker
```bash
cd backend
uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```
Note: Redis caching requires a running Redis instance. Without it, the cache is silently disabled.

---

## Evaluation

The project includes a match evaluation suite that tests the search pipeline against known procedure-query mappings:

```bash
cd backend
python -m src.evaluation match --top_k 10
```

Metrics computed per query:
- **Hit@5** and **Hit@K** — was a correct code in the top results?
- **MRR** (Mean Reciprocal Rank) — how high did the first correct result rank?
- **Recall@K** — what fraction of expected codes were found?

The default test suite includes 25 cases covering exact code lookups, natural language queries, and conversational patient-style questions.

---

## Project Structure

```
├── backend/
│   ├── src/
│   │   ├── main.py                    # FastAPI app, routes, startup
│   │   ├── matcher.py                 # Hybrid search: FAISS + BM25 + rerank
│   │   ├── pricing.py                 # Medi-Cal rate lookups
│   │   ├── eligibility_evaluator.py   # Deterministic pathway evaluator
│   │   ├── cache.py                   # Redis JSON cache
│   │   ├── invoke_ai.py              # OpenAI wrapper
│   │   ├── rag_pipeline.py           # RAG orchestrator
│   │   ├── rag/
│   │   │   ├── datastore.py          # LanceDB vector store
│   │   │   ├── indexer.py            # PDF → chunks (Docling)
│   │   │   ├── retriever.py          # Vector search + rerank
│   │   │   └── response_generator.py # LLM prompts, pathway generation
│   │   └── rag_interface/            # Abstract base classes
│   │       ├── base_datastore.py
│   │       ├── base_indexer.py
│   │       ├── base_retriever.py
│   │       └── base_response_generator.py
│   ├── scripts/                       # Data pipeline scripts
│   ├── frontend/                      # Static HTML/CSS/JS
│   │   ├── landing.html
│   │   ├── index.html
│   │   ├── css/
│   │   └── js/
│   └── data/                          # LanceDB tables
├── data/
│   ├── raw/                           # Source datasets
│   ├── processed/                     # Cleaned parquets
│   └── embeddings/                    # FAISS index + metadata
├── infra/
│   ├── cloudformation.yml             # AWS infrastructure as code
│   └── entrypoint.sh                 # Container bootstrap
├── Dockerfile                         # Local development
├── Dockerfile.prod                    # Production (multi-stage, amd64)
├── docker-compose.yml                 # Local: API + Redis
└── deploy.sh                          # One-command AWS deploy
```

---

## License

This project is for portfolio and educational purposes.

---

MedCoverage was built as a UC Berkeley MIDS capstone project.
| Team member | Role & contributions |
|---|---|
| **Ali Fardaev** | • Developed the RAG system with LanceDB as the vector database<br>• Built hybrid FAISS + BM25 retrieval with Cohere reranking for the search engine<br>• Built the LLM question-planning agent for conversational intake<br>• Implemented the decision tree logic for deterministic eligibility decisions<br>• Merged sequential LLM calls to reduce latency and cost<br>• Implemented Redis caching<br>• Developed the frontend<br>• Handled full AWS deployment (Docker, EC2, ALB, ElastiCache Redis, CloudFormation, GitHub Actions CI/CD via SSM)<br>• Set up Git for version control and collaboration<br>• Contributed to presentations |
| **David Carlson** | • Matching evaluation<br>• RAG retrieval evaluation<br>• Coverage decision evaluation<br>• Pricing evaluation<br>• Built multiple iterative test sets<br>• Conducted user survey<br>• Performed EDA<br>• Contributed to weekly updates, presentations, and weekly meetings<br>• Assisted with final demo scenarios |
| **Zia Williams** | • Performed EDA<br>• Optimized match performance and capabilities, including keyword extraction<br>• Implemented and improved evaluation capabilities, including using the decision tree as ground truth<br>• Contributed to weekly meetings and presentations |
| **Alec Heyde** | • Developed version 1 of the prototype<br>• Contributed to ideation of incremental product changes<br>• Served as subject matter expert and co-project manager<br>• Found the Medi-Cal documents needed for the project<br>• Minor slide creation; contributed to weekly meetings |
| **Umair Habib** | • Performed EDA<br>• Found datasets and researched documentation for scoping<br>• Stress-tested the RAG model and tree logic to find gaps<br>• Created most slides for each presentation<br>• Contributed to weekly meetings |
