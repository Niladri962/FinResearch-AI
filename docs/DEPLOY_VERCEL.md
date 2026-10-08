# Deploying FinResearch AI entirely on Vercel

The app deploys as **two Vercel projects from the same repository**: the Next.js frontend and the FastAPI backend (as a Python serverless function). A managed PostgreSQL database holds all data, including the search vectors, so no other service is required.

```
Browser ──► Vercel project 1: frontend (Next.js)
               │  NEXT_PUBLIC_API_URL
               ▼
            Vercel project 2: backend (FastAPI function)
               │
               ├──► PostgreSQL (Neon, via Vercel Storage)   documents, passages, facts, vectors, chats
               ├──► LLM API (e.g. Groq)                      answer writing            (optional)
               └──► Embeddings API (OpenAI-compatible)       semantic search           (optional)
```

## What changes in serverless mode

The backend detects Vercel automatically (the platform sets `VERCEL=1`) and adapts:

| Topic | On a server / Docker | On Vercel |
|---|---|---|
| Uploads | Processed in the background, any size up to `MAX_UPLOAD_MB` | Processed during the upload request; **4 MB per file** (Vercel's request limit is 4.5 MB) |
| Embeddings | BGE model running locally | An embeddings API if `EMBEDDING_API_KEY` is set, otherwise a built-in keyword-style embedder |
| Reranker | Cross-encoder model running locally | Lexical reranker |
| Vectors | Local disk or Qdrant | The PostgreSQL database (or Qdrant if `QDRANT_URL` is set) |
| Ruled-table detection | pdfplumber | Tables are read from text only |
| Disk | `data/` | `/tmp` (scratch only — nothing is kept there) |

Large reports such as a 29 MB annual report **cannot be uploaded on Vercel**. For those, run the backend on a container host (see the main README) and keep only the frontend on Vercel.

## Step 1 — Backend project

1. In Vercel: **Add New → Project**, import the repository.
2. **Root Directory:** `backend`. Framework preset: **Other**. Leave build and output settings empty.
3. Open the project's **Storage** tab → **Create Database → Neon (Postgres)** and connect it to the project. This adds `DATABASE_URL` automatically.
4. Under **Settings → Environment Variables** add:

   | Variable | Value | Needed |
   |---|---|---|
   | `LLM_API_KEY` | Your Groq (`gsk_…`) or OpenAI key | For written answers; without it answers are extractive |
   | `EMBEDDING_API_KEY` | An OpenAI-compatible embeddings key | For semantic search; without it search is keyword-based |
   | `EMBEDDING_BASE_URL` | Only if not OpenAI, e.g. another provider's `/v1` URL | Optional |
   | `EMBEDDING_MODEL` | e.g. `text-embedding-3-small` | Optional |
   | `AUTH_ENABLED` | `true` | Recommended — the URL is public |
   | `API_KEYS` | `<long-random-string>:admin` | With `AUTH_ENABLED` |
   | `CORS_ORIGINS` | Leave empty until step 3 | — |

5. Deploy. Open `https://<backend>.vercel.app/api/health`. It should show `"status": "ok"`, and `"storage": {"status": "ok"}` (if it says `ephemeral`, the database is not connected).

`backend/vercel.json` routes every path to the function and sets a 300-second limit with 1 GB of memory. `backend/requirements.txt` is the slim dependency set Vercel installs.

## Step 2 — Frontend project

1. **Add New → Project**, import the same repository again.
2. **Root Directory:** `frontend`. Framework preset: **Next.js** (detected).
3. Environment variable: `NEXT_PUBLIC_API_URL` = `https://<backend>.vercel.app` (no trailing slash).
4. Deploy.

## Step 3 — Connect them

In the **backend** project's environment variables set, then redeploy it:

```
CORS_ORIGINS=https://<frontend>.vercel.app
CORS_ORIGIN_REGEX=https://<frontend>(-[a-z0-9-]+)?\.vercel\.app
```

Open the frontend, go to **Settings**, paste the key from `API_KEYS`, and save. Then upload a document (under 4 MB) on the Documents page.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| Frontend shows a red "no backend configured" banner | `NEXT_PUBLIC_API_URL` was not set when the frontend was built. Set it and redeploy the frontend |
| "Cannot reach the API" | The frontend's origin is missing from `CORS_ORIGINS` on the backend |
| Documents disappear after a while | `DATABASE_URL` is not set; health shows `storage: ephemeral` |
| Backend build fails with a size error | The function exceeds Vercel's 250 MB limit. Confirm Root Directory is `backend` so only `requirements.txt` is installed |
| Upload fails with HTTP 413 | The file is larger than 4 MB |
| Upload times out | The document is too large to process within 300 seconds; use a container host for it |
| HTTP 401 everywhere | `AUTH_ENABLED=true` and no key saved on the Settings page |
| First request is slow | Cold start: the function loads and connects to the database |

## What has and has not been verified

The serverless code paths are covered by automated tests run locally (database-backed vectors, start-up without lifespan events, in-request processing, the 4 MB limit, slim-install fallbacks). The Vercel deployment itself has **not** been run: the bundle size against Vercel's limit, response streaming through the Python runtime, and the Neon connection are untested until you deploy.
