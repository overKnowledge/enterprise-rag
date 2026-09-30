# Enterprise RAG & Knowledge Intelligence Platform

A document question-answering service: upload PDFs, DOCX files or web pages, then ask
questions and get answers grounded in those documents, with citations, or an explicit
"I don't know" when the documents don't support an answer.

Built as a learning project in production-style AI engineering, with an emphasis on
**measuring** every change rather than assuming it helped. The evaluation harness caught
a real hallucination bug and a real reranking regression — both are documented, not
hidden. Full write-up: [docs/evaluation.md](docs/evaluation.md).

## Results at a glance

| | |
|---|---|
| Retrieval (baseline, no rerank) | **100% Hit Rate, 0.950 MRR** on a 27-question set |
| Refusal correctness | **100%** on 5 adversarial / unanswerable questions |
| Found via evaluation | A hallucinated claim with a false citation — root-caused and fixed |
| Reranking A/B test | Tested, found it *underperformed* baseline, kept optional rather than forced |

## How it works

```
INGESTION
  PDF / DOCX / URL -> loader -> chunker (450 tokens, 65 overlap)
                                   |-> embed (bge-small-en-v1.5) -> ChromaDB (vectors + text)
                                   '-> SQLite (documents + chunk metadata, dedup by SHA-256)

QUERY
  question -> embed -> ChromaDB top-5 -> grounded prompt -> Groq LLM -> answer + citations
```

| Layer | Choice |
|---|---|
| API | FastAPI, Pydantic settings, API-key auth |
| Parsing | PyMuPDF (PDF), python-docx (DOCX), trafilatura (web) |
| Chunking | Fixed-size, token-based (tiktoken), 450 tokens with 65 overlap |
| Embeddings | `BAAI/bge-small-en-v1.5` (384-dim), local, CPU |
| Vector store | ChromaDB (persistent, cosine) |
| Metadata store | SQLite via SQLAlchemy |
| Generation | Groq API, `openai/gpt-oss-120b` by default (configurable) |
| Reranking (optional) | Cross-encoder `ms-marco-MiniLM-L-6-v2` with a confidence threshold |
| Deployment | Docker + Docker Compose, persistent volumes |

The pipeline is written directly rather than through LangChain/LangGraph, so each stage
stays visible. Embedding and LLM backends sit behind small interfaces, so swapping
providers is a contained change.

## Quick start

Requires Python 3.12 (or Docker) and a Groq API key ([console.groq.com](https://console.groq.com)).

```bash
git clone https://github.com/overKnowledge/enterprise-rag.git
cd enterprise-rag
cp .env.example .env        # then fill in GROQ_API_KEY and API_KEY
```

Generate an API key for the service:

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

**With Docker (recommended):**

```bash
docker compose up --build
```

**Locally:**

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000/docs. The first `/v1/chat` call is slower because the
embedding model loads; later calls are fast. Data lives in `./data`
(SQLite + ChromaDB) and survives container restarts.

> The source documents used to build the evaluation set (four papers/textbook excerpts
> plus one web article) are not included in this repo, since one is a copyrighted
> textbook. Upload your own PDFs/DOCX/URLs via `POST /v1/documents` to try it.

### Configuration (`.env`)

| Variable | Required | Purpose |
|---|---|---|
| `GROQ_API_KEY` | yes | Groq API access |
| `API_KEY` | yes (unless auth is disabled) | Value clients send in the `x-api-key` header |
| `GROQ_MODEL` | no | Defaults to `openai/gpt-oss-120b` |
| `LOG_LEVEL`, `ENVIRONMENT`, `APP_NAME` | no | Logging and display |

## API

All `/v1` endpoints require an `x-api-key` header.

| Endpoint | Purpose |
|---|---|
| `GET /health` | Liveness check (no auth) |
| `POST /v1/documents` | Upload a `.pdf` or `.docx` (max 25 MB); chunks, embeds and indexes it |
| `POST /v1/documents/url` | Ingest a web page by URL |
| `GET /v1/documents` | List ingested documents |
| `POST /v1/chat` | Ask a question; returns `answer` and `sources` (file, pages, distance) |

Uploads are validated by size and by file signature, not just extension. Re-uploading
identical content is a no-op (documents are keyed by content hash). Unexpected errors
return a generic 500; details are logged server-side only. When the model answers
"I don't know based on the available documents.", `sources` is empty.

## Evaluation

A golden set of 27 hand-written questions (easy to adversarial, including paraphrased,
negated and vocabulary-avoidant ones, plus deliberately unanswerable ones) scores
retrieval (Hit Rate, MRR) and refusal correctness. Full methodology, the three bugs it
caught, and the reranking A/B comparison: **[docs/evaluation.md](docs/evaluation.md)**.

Run it yourself (needs your own ingested documents and a Groq key):
```bash
python evaluation/run_eval.py            # baseline
python evaluation/run_eval_reranked.py   # with reranking, for comparison
```

## Project layout

```
app/
  api/            routes (documents, chat, health), shared dependencies
  core/           settings, auth, logging, error handling
  db/             SQLAlchemy models and session
  embeddings/     embedding interface + Hugging Face implementation
  generation/     LLM interface, Groq provider, grounding prompt, answerer
  ingestion/      loaders, chunker, validation, pipeline
  retrieval/      cross-encoder reranker + reranking retriever
  vectorstore/    ChromaDB wrapper
evaluation/       golden dataset, metrics, eval runners
scripts/          ingest, embed, list and remove utilities
tests/            unit tests
docs/             detailed evaluation write-up
```

## Known limitations and future work

- Embedding runs inside the upload request; large files block it. Background jobs are
  the fix.
- Reranking is implemented and evaluated but not wired into `/v1/chat` by default (see
  evaluation doc for why).
- Hybrid (BM25 + vector) search was scoped but not built.
- No rate limiting on the API, no CI pipeline, and test coverage is minimal — the
  evaluation harness is the main quality check.
- No `DELETE` endpoint; `scripts/remove_document.py` does it manually.
- Fixed-size chunking ignores document structure.
- Single-turn Q&A: no conversation memory or query rewriting.
- Free-tier Groq quotas (per-minute and per-day tokens) limit how often the full
  evaluation can be run in one sitting. 