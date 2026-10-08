# FinResearch AI

**AI-Powered Financial Research & Analysis Platform**

Upload annual reports, quarterly results, earnings-call transcripts and financial statements, then ask questions in plain English. Every answer is backed by evidence retrieved from your documents and cites the exact page it came from. Every ratio, growth rate and comparison is **calculated in Python**, never by the language model.

> FinResearch AI is an analytical research assistant, not a financial advisor. Its output may contain errors and should be independently verified. It does not give investment advice or guarantee future performance.

---

## Contents

- [Problem](#problem) · [Key features](#key-features) · [Screenshots](#screenshots)
- [Architecture](#architecture) · [RAG pipeline](#rag-pipeline) · [Agent architecture](#agent-architecture) · [Data model](#data-model)
- [Tech stack](#tech-stack) · [Project structure](#project-structure)
- [Local setup](#local-setup) · [Docker](#docker) · [Environment variables](#environment-variables)
- [API](#api) · [Evaluation](#evaluation) · [Testing](#testing)
- [Security](#security) · [Observability](#observability) · [Deployment](#deployment)
- [Verification status and known limitations](#verification-status-and-known-limitations) · [Future improvements](#future-improvements)

---

## Problem

"Chat with a PDF" breaks down on financial documents:

| Failure of naive RAG | What this project does instead |
|---|---|
| The model does arithmetic and gets ratios wrong | Statement tables are parsed into structured facts; ratios, CAGR, trends and comparisons are computed by a tested Python engine and handed to the model as citeable values |
| Fixed-size chunking cuts tables in half | Layout-aware chunking keeps every table whole, repeats headers when a table must be split, and never crosses a section boundary |
| Embeddings miss exact terms ("Note 14", "FY2024-25") and tables | Hybrid retrieval (dense + BM25), cross-encoder reranking, and a structured lane that guarantees the statement tables behind a metric are retrieved |
| Answers contain invented numbers and citations | Every answer is validated after generation: unknown citations are removed and each figure is checked against the evidence and the computed values |
| Every question goes to the LLM | A deterministic router classifies intent; the LLM is only called to write the final answer (and to break ties when the rules are unsure) |

## Key features

- **Document intelligence** — PDF (PyMuPDF + pdfplumber), DOCX, XLSX, TXT/MD. Detects company, document type, fiscal year/quarter, currency and unit. Recovers ruled *and* unruled tables, strips running headers/footers, handles two-column layouts, OCR for scanned pages when Tesseract is present.
- **Financial table understanding** — maps row labels to a canonical taxonomy (revenue, EBITDA, borrowings, …) and columns to reporting periods. Conservative by design: an unmapped row is preferred over a wrong number.
- **Hybrid RAG** — configurable weighted fusion (default 0.7 semantic / 0.3 keyword) or reciprocal-rank fusion, then reranking to the top 6–8 contexts.
- **Agentic routing** — a LangGraph supervisor dispatches to research, financial, comparison, risk and summary agents based on nine query intents.
- **Financial engine** — 18 ratios across liquidity, solvency, profitability, efficiency and growth; CAGR; YoY trends; rule-based risk signals; illustrative extrapolation (clearly labelled as such).
- **Citations** — document, page, section and chunk id; clickable in the UI with a full-passage preview.
- **Anti-hallucination** — strict prompt contract, post-generation citation and numeric-grounding validation, a fixed "insufficient evidence" response, and "Not available" for missing data.
- **Guardrails** — prompt injection (direct and planted inside documents), market-manipulation requests, guaranteed-return questions, off-topic requests, advice language in outputs.
- **Research reports** — 13-section Markdown report with globally numbered sources.
- **Evaluation** — Recall@K, Precision@K, MRR, context relevance, faithfulness, answer relevance, citation correctness; optional RAGAS.
- **Works with no infrastructure and no API key** — SQLite, an on-disk vector store and an *extractive* answer mode are the defaults, so the whole pipeline is demonstrable offline. PostgreSQL, Qdrant, Redis and any OpenAI-compatible LLM plug in through environment variables.

## Screenshots

Captured from the running app with the synthetic sample filings (fictitious companies).

| Dashboard | Financial analysis | Research chat |
|---|---|---|
| ![Dashboard](docs/screenshots/dashboard-dark.png) | ![Financial analysis](docs/screenshots/financial-analysis.png) | ![Research chat](docs/screenshots/research-chat-comparison.png) |

---

## Architecture

```mermaid
flowchart LR
    UI["Next.js UI<br/>Dashboard · Chat · Analysis · Reports"] -- "REST + SSE" --> API

    subgraph Backend["FastAPI backend"]
        API["API layer<br/>auth · rate limit · validation"] --> SVC["Services<br/>documents · chat · analysis · reports"]
        SVC --> AG["Agent graph<br/>(LangGraph supervisor)"]
        SVC --> ING["Ingestion pipeline"]
        AG --> RET["Hybrid retriever"]
        AG --> FIN["Financial engine<br/>(pure Python)"]
        AG --> LLMC["LLM client<br/>(provider abstraction)"]
        AG --> GR["Guardrails +<br/>citation validator"]
    end

    ING --> PG[("PostgreSQL / SQLite<br/>documents · chunks · facts")]
    ING --> VS[("Qdrant / local store<br/>embeddings")]
    RET --> VS
    RET --> PG
    FIN --> PG
    LLMC --> EXT["Groq · OpenAI · local model"]
    API -.-> RD[("Redis<br/>cache · rate limits")]
```

Layering rule: `api/` only translates HTTP; `services/` holds use-cases; `agents/`, `rag/`, `documents/`, `financial/` and `guardrails/` hold the logic and know nothing about FastAPI. `services/container.py` is the single composition root.

## RAG pipeline

**Ingestion**

```mermaid
flowchart LR
    A[Upload] --> B["Validate<br/>extension · magic bytes · size"] --> C["Parse<br/>layout elements"] --> D["Metadata<br/>company · type · period · unit"]
    D --> E["Financial-aware<br/>chunking"] --> F["Table → facts<br/>metric × period"] --> G[Embed] --> H["Vector store<br/>+ BM25"] --> I[Ready]
```

**Query**

```mermaid
flowchart TD
    Q[User query] --> G1["Input guard"] --> U["Query understanding<br/>company · period · metrics · intent"] --> R{Router}
    R -->|evidence| S["Semantic top-20"] & K["BM25 top-20"] & T["Statement lane<br/>tables for named metrics"]
    R -->|numbers| C["Financial tools<br/>ratios · trends · comparisons"]
    S & K --> F["Fusion<br/>0.7 semantic + 0.3 keyword"]
    F & T --> RR[Reranker] --> CB["Context builder<br/>S1…Sn sources"]
    C --> CB2["Computed blocks<br/>C1…Cn · T1…Tn"]
    CB & CB2 --> L["LLM reasoning<br/>(or extractive composer)"] --> V["Citation + numeric<br/>verification"] --> O["Output guard"] --> A[Final answer]
```

Design notes:

- **Chunk types** — `financial_statement`, `table`, `risk_factor`, `management_commentary`, `earnings_commentary`, `accounting_policy`, `notes`, `narrative`. Each chunk stores company, document, type, fiscal year, page range and section path (e.g. `Management Discussion and Analysis › Outlook`).
- **Provenance everywhere** — the embedding, the BM25 index and the reranker all see `company | document title | section` alongside the passage, because a balance sheet never repeats the company's name.
- **Statement lane** — when a question names a metric or ratio, `financial_facts` is used to find the tables those figures were extracted from, and up to two evidence slots are reserved for them. Cross-encoders rank Markdown tables poorly; this keeps the numbers citeable.
- **Graceful degradation** — vector store down → keyword-only with a note; reranker model unavailable → lexical reranker; LLM down or unconfigured → extractive answer.

## Agent architecture

```mermaid
flowchart TD
    START((start)) --> GUARD[guard] -->|blocked| END1((end))
    GUARD --> UND[understand] --> SUP{supervisor}
    SUP --> RES["Research agent<br/>retrieves evidence"] --> SUP
    SUP --> FA["Financial agent<br/>ratios · metrics · trends"] --> SUP
    SUP --> CMP["Comparison agent<br/>companies · periods"] --> SUP
    SUP --> RISK["Risk agent<br/>quantitative risk signals"] --> SUP
    SUP --> SUM["Summary agent<br/>writes the answer"] --> VER["verify<br/>citations · figures · output guard"] --> END2((end))
```

| Intent | Plan |
|---|---|
| `DOCUMENT_QA`, `MANAGEMENT_ANALYSIS` | research (+ financial first if the question names a metric) |
| `FINANCIAL_CALCULATION`, `TREND_ANALYSIS`, `SUMMARY` | financial → research |
| `COMPANY_COMPARISON`, `PERIOD_COMPARISON` | comparison → research |
| `RISK_ANALYSIS` | risk → research |
| `GENERAL_FINANCE` | summary only (no retrieval; answer is labelled as general knowledge) |

Only the summary agent calls the LLM to produce text. Only the financial, comparison and risk agents produce numbers.

## Data model

```mermaid
erDiagram
    COMPANY ||--o{ DOCUMENT : has
    DOCUMENT ||--o{ CHUNK : "split into"
    DOCUMENT ||--o{ FINANCIAL_FACT : yields
    CHUNK ||--o{ FINANCIAL_FACT : "source of"
    COMPANY ||--o{ REPORT : about
    CONVERSATION ||--o{ MESSAGE : contains

    DOCUMENT { string id PK  string document_type  int fiscal_year  int quarter  string status  string sha256 }
    CHUNK { string id PK  text text  int page_start  int page_end  string section  string chunk_type }
    FINANCIAL_FACT { string metric  string period_label  float value  float scale  string unit  string currency  int page  float confidence }
    MESSAGE { string role  text content  json payload }
    QUERY_TRACE { string intent  float total_ms  json data }
```

Vectors live in the vector store keyed by chunk id, with `company_id`, `document_id`, `fiscal_year` and `document_type` as filterable payload.

---

## Tech stack

| Layer | Choice |
|---|---|
| API | Python 3.10+ (3.12 in Docker), FastAPI, Pydantic v2, Uvicorn |
| Orchestration | LangGraph |
| Retrieval | BGE embeddings + MiniLM cross-encoder via fastembed (ONNX, no PyTorch); `rank-bm25`; Qdrant over REST |
| Documents | PyMuPDF, pdfplumber, python-docx, openpyxl |
| Storage | SQLAlchemy 2 → PostgreSQL or SQLite; Redis (optional) |
| LLM | Any OpenAI-compatible endpoint: Groq, OpenAI, Ollama/vLLM/LM Studio |
| Frontend | Next.js 15 (App Router), React 19, TypeScript, Tailwind CSS, Recharts |
| Ops | Docker, Docker Compose, Render blueprint, Vercel-ready frontend |

Swappable by configuration: LLM provider, embedding provider (`fastembed`, `sentence_transformers`, `openai`, `hash`), reranker (`fastembed`, `sentence_transformers`, `lexical`, `none`), vector store (`qdrant`, `local`, `memory`), database and cache.

## Project structure

```
backend/
  app/
    main.py, config.py
    api/            deps.py (auth, RBAC, rate limit) · routes/{documents,chat,analysis,companies,reports,health}.py
    agents/         supervisor.py · router.py · research/financial/comparison/risk/summarization agents · prompts.py
    rag/            ingestion · chunking · embeddings · vector_store · bm25 · hybrid_search · reranker · retriever
    documents/      parser.py · table_extractor.py · metadata.py
    financial/      metrics · periods · calculations · ratios · trends · risk_signals · forecasting
    guardrails/     input_guard · output_guard · citation_validator
    evaluation/     retrieval_eval · answer_eval · ragas_eval · runner
    llm/            base · openai_compatible · factory
    services/       container (composition root) · documents · chat · analysis · comparison · financials · reports · dashboard
    models/         db.py (ORM) · schemas.py (Pydantic) · enums.py · database.py
    utils/          errors · logging · security · cache · rate_limit · observability · formatting · text
  tests/            366 tests
frontend/           app/ (8 pages) · components/ · lib/ · hooks/
scripts/            generate_sample_data.py · seed_demo.py · run_evaluation.py
evaluation/         dataset.jsonl
data/samples/       synthetic sample filings
docker-compose.yml · render.yaml · .env.example
```

---

## Local setup

No Docker, database or API key is required.

**Backend**

```bash
cd backend
python -m venv .venv
# Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate
pip install -r requirements-dev.txt
cp ../.env.example ../.env             # optional; every value has a default
uvicorn app.main:app --reload --port 8000
```

API docs: <http://localhost:8000/api/docs>. The first upload downloads the embedding and reranker models (~200 MB) into `data/models`.

**Frontend**

```bash
cd frontend
npm install
cp .env.example .env.local            # NEXT_PUBLIC_API_URL=http://localhost:8000
npm run dev                           # http://localhost:3000
```

**Sample data** (fictitious companies, generated locally)

```bash
python scripts/generate_sample_data.py     # writes data/samples/*.pdf and evaluation/dataset.jsonl
python scripts/seed_demo.py                # uploads them to the running backend
```

Then try: *"Why did Aurora's operating margin decline?"*, *"What is Aurora's debt-to-equity ratio?"*, *"Compare Aurora with Borealis."*

**Enable generated answers** — set `LLM_API_KEY` in `.env` (a Groq key starting with `gsk_` is auto-detected) and restart. Without a key, answers are *extractive*: computed metrics plus the most relevant passages quoted verbatim.

<details>
<summary>Troubleshooting</summary>

- **Ports 8000/3000 already in use** — run `uvicorn … --port 8765` and set `NEXT_PUBLIC_API_URL` and `CORS_ORIGINS` to match.
- **Windows: "An Application Control policy has blocked this file"** when importing `charset_normalizer` (a pdfplumber dependency) — install its pure-Python build: `pip install --force-reinstall --no-deps --no-binary charset-normalizer charset-normalizer`. The same policy can block the `psycopg` binary driver; use SQLite locally and PostgreSQL in Docker.
- **Project inside OneDrive/Dropbox** — set `DATA_DIR` and `MODEL_CACHE` to a folder outside the synced directory to avoid syncing the database, index and models.
- **Changed the embedding model** — existing vectors have a different dimension. Use a new `QDRANT_COLLECTION` (or delete `data/vectors`) and re-upload.
</details>

## Docker

```bash
cp .env.example .env        # set POSTGRES_PASSWORD; LLM_API_KEY is optional
docker compose up --build
```

Starts `backend`, `frontend`, `postgres`, `qdrant` and `redis`. Only the UI (3000) and API (8000) are published to the host.

## Environment variables

Full list with comments in [.env.example](.env.example). The most important:

| Variable | Default | Purpose |
|---|---|---|
| `LLM_PROVIDER` / `LLM_API_KEY` / `LLM_MODEL` / `LLM_BASE_URL` | `auto` / – / `llama-3.3-70b-versatile` / – | LLM selection. `auto` picks Groq or OpenAI from the key; `local` targets any OpenAI-compatible server; `none` forces extractive mode |
| `EMBEDDING_PROVIDER` / `EMBEDDING_MODEL` | `fastembed` / `BAAI/bge-small-en-v1.5` | Embedding backend |
| `RERANKER_PROVIDER` / `RERANKER_MODEL` | `fastembed` / `Xenova/ms-marco-MiniLM-L-6-v2` | Reranker |
| `SEMANTIC_WEIGHT` / `KEYWORD_WEIGHT` / `FUSION_METHOD` / `RERANK_TOP_N` | `0.7` / `0.3` / `weighted` / `6` | Retrieval tuning |
| `VECTOR_STORE` / `QDRANT_URL` / `QDRANT_API_KEY` | `auto` / – / – | `auto` → Qdrant when a URL is set, else the local store |
| `DATABASE_URL` | SQLite in `DATA_DIR` | e.g. `postgresql+psycopg://user:pass@host/db` |
| `REDIS_URL` | – | Cache and rate-limit counters; in-process fallback |
| `AUTH_ENABLED` / `API_KEYS` | `false` / – | `key:role` pairs; roles `viewer`, `analyst`, `admin` |
| `LANGCHAIN_TRACING_V2` / `LANGCHAIN_API_KEY` / `OTEL_EXPORTER_OTLP_ENDPOINT` | off | LangSmith and OpenTelemetry export |

---

## API

Interactive docs at `/api/docs`. Errors share one envelope: `{"error": {"code", "message", "details"}}`.

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/documents/upload` | Multipart upload (1–10 files; optional `company`, `document_type`, `fiscal_year`, `quarter`). Returns `202`; processing continues in the background |
| `GET` | `/api/documents` · `/api/documents/{id}` | List / status |
| `GET` | `/api/documents/{id}/facts` | Structured facts extracted from the document's tables |
| `GET` | `/api/documents/chunks/{chunk_id}` | Full text of a cited passage |
| `DELETE` | `/api/documents/{id}` | Remove file, chunks, facts and vectors |
| `GET` | `/api/companies` | Companies with covered periods |
| `POST` | `/api/chat` | Ask a question. `stream: true` → Server-Sent Events; `false` → JSON |
| `GET` `DELETE` | `/api/chat/conversations[/{id}]` | Conversation history |
| `POST` | `/api/analyze` | KPIs, period snapshots, trend charts, risk signals |
| `POST` | `/api/compare` | ≥2 companies → company comparison; 1 → period comparison |
| `POST` | `/api/financial-ratios` | Ratios from stored statements or from raw `values` |
| `POST` | `/api/reports/generate` · `GET /api/reports[/{id}]` | Research reports |
| `GET` | `/api/dashboard` · `/api/system` · `/api/observability/traces` | Aggregates, non-secret config, query traces |
| `GET` | `/api/health` | Liveness and component status |

**Chat stream events** — `conversation` → `status` / `meta` (intent, plan) → `artifacts` (calculations, tables, charts) → `sources` → `token`… → `final`. The `final` event carries the verified answer and replaces the streamed draft.

```bash
curl -s localhost:8000/api/chat -H "Content-Type: application/json" \
  -d '{"message": "What is Aurora'\''s debt-to-equity ratio?", "stream": false}'
```

```jsonc
{
  "intent": "FINANCIAL_CALCULATION",
  "mode": "extractive",                       // "generative" when an LLM is configured
  "calculations": [{
    "id": "C1", "name": "Debt-to-Equity", "period": "FY2025", "display": "0.64x",
    "formula": "Total Debt / Shareholders' Equity",
    "inputs": [
      {"name": "Total Debt", "display": "INR 3,500 crore", "derived": true,
       "formula": "Short-Term Borrowings + Long-Term Borrowings"},
      {"name": "Shareholders' Equity", "display": "INR 5,500 crore",
       "sources": [{"document_title": "Annual Report FY2025", "page": 5}]}
    ]
  }],
  "citations": [{"id": "S1", "document_title": "Annual Report FY2025", "page": 5, "section": "Consolidated Balance Sheet"}],
  "validation": {"status": "grounded", "grounding_score": 1.0}
}
```

## Evaluation

```bash
python scripts/run_evaluation.py --offline      # hash embeddings, lexical reranker, no LLM
python scripts/run_evaluation.py --isolated     # models and LLM from .env, isolated temp store
python scripts/run_evaluation.py                # whatever is already indexed
python scripts/run_evaluation.py --isolated --ragas   # adds LLM-judged RAGAS metrics (optional deps)
```

Dataset format (`evaluation/dataset.jsonl`): `question`, `expected_answer`, `relevant_document`, `relevant_page`, `key_facts`.

Measured on the 13-question synthetic set, k = 5, **no LLM (extractive answers)**:

| Metric | Offline stack (hash + lexical) | BGE-small + MiniLM cross-encoder |
|---|---|---|
| Recall@5 / hit rate | 1.00 | 1.00 |
| MRR | 0.83 | 0.77 |
| Precision@5 | 0.34 | 0.35 |
| Faithfulness | 1.00 | 1.00 |
| Key-fact recall | 0.86 | 0.83 |
| Citation precision / page hit | 0.67 / 1.00 | 0.71 / 1.00 |

Read these with care: the corpus is four short synthetic documents, there is one labelled page per question, and the generation metrics are lexical proxies measured on extractive answers (which quote sources, so faithfulness is 1.0 by construction). They are a regression baseline, not a benchmark. Before the statement lane was added, Recall@5 with the real models was 0.69 — the misses were all questions answered by a statement table.

## Testing

```bash
cd backend && pytest            # 366 tests, ~1 minute, fully offline
cd frontend && npm run typecheck && npm run build
```

Coverage by area: financial engine, period and label parsing, table extraction, parsers (PDF/DOCX/XLSX/text, invalid and empty files), metadata, chunking, embeddings, vector stores (local store end to end; Qdrant against a stubbed HTTP transport), fusion, reranking, retriever, intent routing, guardrails, citation validation, LLM client (retries, streaming, error mapping), auth/RBAC, rate limiting, log redaction, the agent graph with a scripted LLM, every API endpoint, and the evaluation runner.

## Security

- **Uploads** — extension allow-list, magic-byte and Office-container checks, streamed size limit, SHA-256 de-duplication. Files are stored under server-generated names; the client's filename is only a sanitised display label, and every path is resolved through a traversal guard.
- **Auth-ready** — `AUTH_ENABLED=true` requires `X-API-Key` or a Bearer token. Three roles: `viewer` (read, chat), `analyst` (+ upload, reports), `admin` (+ delete, traces). Conversations are scoped to their owner. Swapping API keys for JWT/OIDC means replacing one dependency in `api/deps.py`.
- **Rate limiting** — per-client fixed window, backed by Redis or in-process counters.
- **Prompt injection** — direct attempts are blocked before retrieval; passages containing embedded instructions are dropped before they reach the model; the system prompt treats sources as untrusted data.
- **Secrets** — only from environment variables. Model-provider keys never reach the browser. Logs are JSON with credential redaction; provider error bodies are never surfaced to clients.
- **Network** — CORS allow-list, `nosniff` / `X-Frame-Options` headers, non-root containers, databases unexposed in Compose.

## Observability

Each query produces a trace (table `query_traces`, `GET /api/observability/traces`, and the Settings page): intent, agent plan, latency per stage (understanding, retrieval, embedding, vector search, BM25, rerank, LLM, verification), token usage, retrieved chunk and document ids, validation status and errors. Query text is stored only as a hash unless `TRACE_QUERY_TEXT=true`; document text is never logged. LangGraph runs export to LangSmith via `LANGCHAIN_TRACING_V2`; spans export to any OTLP collector via `OTEL_EXPORTER_OTLP_ENDPOINT` (optional dependencies).

## Deployment

| Component | Target | Notes |
|---|---|---|
| Frontend | Vercel | Root directory `frontend`; set `NEXT_PUBLIC_API_URL` to the API's public URL |
| Backend | Render / Railway / AWS (ECS, App Runner) | `backend/Dockerfile`; honours `$PORT`; health check `/api/health`; mount a volume at `/data`. [render.yaml](render.yaml) is a starting blueprint |
| Vectors | Qdrant Cloud | `QDRANT_URL` + `QDRANT_API_KEY` |
| Database | Managed PostgreSQL | `DATABASE_URL` (`postgres://` and `postgresql://` forms are normalised) |
| Cache | Managed Redis | Optional |

Production checklist: `AUTH_ENABLED=true` with strong keys; `CORS_ORIGINS` set to the frontend origin; TLS at the load balancer; persistent volume for `/data`; run one backend worker per container (BM25 is in-process) and scale horizontally.

---

## Verification status and known limitations

What was actually run while building this, and what was not:

| Verified | Not verified |
|---|---|
| Backend test suite (366 passing) on Python 3.10 / Windows | `docker compose up` — the compose file passes `docker compose config`, but images were not built or run (Docker was not running on the build machine) |
| Full stack running locally: real BGE + cross-encoder models, background ingestion, persisted local vector store, SSE chat, UI in light and dark themes | PostgreSQL — the code path is standard SQLAlchemy, but only SQLite was exercised |
| Frontend type-check and production build | Qdrant against a real server — the REST client is tested against a stubbed transport only |
| Evaluation script in offline and real-model modes | A live LLM provider — generation, streaming, retries and fallbacks are tested with a scripted model and a mocked HTTP transport |
| | The RAGAS adapter, OpenTelemetry export, OCR, and the Render blueprint |

Known limitations:

- **Table extraction is heuristic.** It was developed against synthetic samples and one real Ind AS annual report (two-page spreads, wrapped rows, standalone + consolidated statements). It expects a recognisable period header and standard line-item labels. Unusual layouts, merged multi-level headers, or scanned tables may yield few or no facts; qualitative Q&A still works, and missing figures are reported as "Not available".
- **Interim periods.** Quarterly columns are stored but ratios and trends currently use annual figures only.
- **No currency conversion.** Cross-company comparisons show amounts in each company's reporting currency and say so; compare ratios and margins.
- **Period-end balances.** ROA, ROE and turnover ratios use period-end rather than average balances (stated in each formula).
- **Guardrails are pattern-based.** They catch common attacks and are not a substitute for provider-side safety or human review.
- **Schema management.** Tables are created at start-up; there are no migrations yet.
- **Single-process indexes.** BM25 lives in memory and is rebuilt when the chunk set changes — fine for thousands of documents, not for millions of chunks.

## Future improvements

- Alembic migrations; background job queue (Celery/RQ) for ingestion with progress events
- Vision-model table extraction for scanned and complex statements; XBRL ingestion
- Quarterly and trailing-twelve-month analytics; average-balance ratios; FX normalisation
- Sparse vectors in Qdrant (native hybrid search) instead of in-process BM25
- OIDC/JWT authentication, per-tenant data isolation, audit log
- LLM-judged evaluation in CI with a larger, real-filing dataset
- PDF page-image preview with the cited region highlighted
