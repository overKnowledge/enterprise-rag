# Enterprise RAG & Knowledge Intelligence Platform

A document question-answering service: ingest PDFs, DOCX files and web pages, then ask
questions and get answers grounded in those documents, with citations, or an explicit
"I don't know" when the documents don't support an answer.

Built step by step as a learning project in production-style AI engineering. The emphasis
is on measurement: every retrieval and prompting change was checked against a golden
question set, and the results (including the regressions) are documented below.

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
| Chunking | Fixed-size, token-based (tiktoken `cl100k_base`), 450 tokens with 65 overlap |
| Embeddings | `BAAI/bge-small-en-v1.5` (384-dim) via sentence-transformers, runs locally on CPU |
| Vector store | ChromaDB (persistent, cosine) |
| Metadata store | SQLite via SQLAlchemy |
| Generation | Groq API, `openai/gpt-oss-120b` by default (configurable) |
| Reranking (optional) | Cross-encoder `ms-marco-MiniLM-L-6-v2` with a confidence threshold |
| Deployment | Docker + Docker Compose, persistent volumes |

The pipeline is written directly rather than through LangChain/LangGraph, to keep each
stage visible. Embedding and LLM backends sit behind small interfaces, so swapping
providers is a contained change.

## Quick start

Requires Python 3.12 (or Docker) and a Groq API key.

```bash
git clone <this repo> && cd enterprise-rag
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

Open http://127.0.0.1:8000/docs. The first `/v1/chat` call is slow because the embedding
model is downloaded and loaded; later calls are fast. Data lives in `./data`
(SQLite + ChromaDB) and survives container restarts.

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

Behaviour worth knowing:
- Re-uploading identical content is a no-op (documents are keyed by content hash).
- Uploads are validated by size and by file signature, not just extension.
- Unexpected errors return a generic 500; details are logged server-side only.
- If the model answers "I don't know based on the available documents.", `sources` is empty.

## Evaluation

A golden set of 27 questions over four PDFs and one web page: 20 scored answerable
questions (easy to hard, including paraphrased, negated and vocabulary-avoidant ones),
2 cross-document questions (logged, not scored), and 5 unanswerable or adversarial
questions that must be refused. Run it with `python evaluation/run_eval.py`
(resumable, rate-limit aware).

Hit Rate and MRR are measured at the source-document level (was the right document
retrieved, and at what rank). Page ranges are recorded in the dataset but not scored.

| Configuration | Hit Rate | MRR | False refusals | Correct refusals (of 5) |
|---|---|---|---|---|
| **Vector search only (default)** | 100% | 0.950 | 0 | 5 |
| + cross-encoder rerank, no threshold | 100% | 0.950 | 0 | 4 |
| + rerank, absolute threshold (score >= 0) | 90% | 0.850 | 2 | 5 |
| + rerank, top-score threshold (>= -1.0) | 95% | 0.900 | 1 | 5 |

Note: in the default-configuration run, q01 and q02 were carried over from an earlier
`gpt-oss-20b` run via the resume feature; Hit Rate and MRR don't depend on the LLM.

### What the evaluation caught

1. **Hallucination with a false citation.** Asked "What model won the WMT 2014
   competition outright?", the model answered that the big Transformer "won the
   competition outright" and cited a page. The paper says it outperformed prior published
   results; it never describes a competition or a win. Tightening the prompt (concrete
   negative examples, a citation-integrity rule) removed the invented framing on
   `gpt-oss-20b`, but only `gpt-oss-120b` fully refused the false-premise question.
   Prompt fixes and model capability each closed part of the gap.
2. **A reranking regression.** Adding reranking made an unanswerable question ("the exact
   GitHub URL for the CRAG implementation") get answered with an invented URL. Inspection
   showed every candidate chunk scored negative, a signal the pipeline was discarding.
3. **A miscalibrated threshold.** Filtering chunks below score 0 fixed that case but
   caused false refusals, because correct chunks for indirectly-worded questions
   legitimately score below zero. A relative rule (reject the result set only if even the
   best candidate is weak) fixed most of it. One vocabulary-avoidant question (q23) still
   fails; this is documented rather than hidden.

Decision: reranking is implemented and evaluated, but it did not beat plain vector search
on this corpus, so it is not enabled in `/v1/chat`.

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
scripts/          one-off utilities (embedding, inspection, cleanup)
tests/            unit tests
```

## Known limitations and future work

- Embedding runs inside the upload request, so very large files block it; move to
  background jobs.
- Reranking is not wired into `/v1/chat`; hybrid (BM25 + vector) search was not built.
- No rate limiting on the API, no CI pipeline, and test coverage is minimal (the
  evaluation harness is the main quality check).
- No `DELETE` endpoint; `scripts/remove_document.py` does it manually.
- `scripts/ingest_all.py` records metadata only and `scripts/embed_all.py` indexes only;
  they predate API-side indexing and should be unified.
- Fixed-size chunking ignores document structure. Structure-aware chunking is untested.
- Single-turn Q&A: no conversation memory or query rewriting.
- The evaluation set is small (27 questions) and scored at document level.
- `requirements.lock` is a Windows snapshot, not a cross-platform lock; Docker installs
  from the pinned `requirements.txt`.
- Handler logs may not appear in the `uvicorn --reload` terminal (root cause not found;
  the handler itself is verified working).
- Free-tier Groq quotas (per-minute and per-day tokens) limit how often the full
  evaluation can be run.