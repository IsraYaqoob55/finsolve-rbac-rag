# FinSolve RBAC RAG Chatbot

An internal chatbot for FinSolve Technologies. It uses Retrieval-Augmented Generation (RAG) with **role-based access control**: each user sees only the data their role permits, and every answer cites its source document.

**Stack:** Python · FastAPI · Qdrant · Streamlit · LLM (pluggable)

> 🚧 Work in progress. See [PROJECT_PLAN.md](PROJECT_PLAN.md) for the full roadmap.

## Roles
| Role | Data access |
|---|---|
| Employee | General (handbook, policies, FAQs) |
| Engineering | Engineering + General |
| Marketing | Marketing + General |
| Finance | Finance + Marketing + General |
| HR | HR + General |
| C-Level | Everything |

## Getting started (development)

**Requirements:** Python 3.11+, Git.

```bash
# 1. Clone and enter the project
git clone https://github.com/IsraYaqoob55/finsolve-rbac-rag.git
cd finsolve-rbac-rag

# 2. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

# 3. Install the app + developer tools
pip install -e ".[dev]"
pre-commit install

# 4. Configure environment
copy .env.example .env          # Windows  (cp on macOS / Linux)

# 5. Run the API
uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000/health or the interactive API docs at http://127.0.0.1:8000/docs.

### Index the documents
```bash
python -m app.ingestion.pipeline --dry-run --show 3   # preview chunking, no model needed
python -m app.ingestion.pipeline                      # embed + index (incremental)
python -m app.ingestion.pipeline --rebuild            # wipe and re-index everything
```
The first run downloads the embedding model (~130 MB) into `.cache/`. Later runs only re-index files that changed.

### Try a search
```bash
python -m app.retrieval.search "What was revenue in Q3 2024?" --departments finance
python -m app.retrieval.search "leave policy" --departments general -k 3
```

### Quality checks
```bash
ruff check .        # lint
ruff format .       # auto-format
mypy app            # static types
pytest --cov=app    # tests + coverage
```
The same checks run automatically on every commit (pre-commit) and on every push (GitHub Actions).

## Project structure
```
app/
├── api/         HTTP routes (FastAPI)
├── auth/        login, password hashing, JWT
├── rbac/        role → data permission policy
├── ingestion/   load, chunk, embed documents
├── retrieval/   vector + keyword search, reranking
├── nlp/         query rewriting, routing, clarification
├── structured/  read-only SQL over HR data
├── llm/         LLM providers + prompts
├── services/    chat orchestration
└── core/        settings, logging, audit
```

## How ingestion works
```
data/<department>/<file>  ──load──▶  Document  ──chunk──▶  Chunks  ──embed──▶  Qdrant
```
- **Department = folder name.** `data/finance/*` is tagged `department=finance`; this tag is what access control filters on.
- **Structure-aware chunking.** Splits at headings (both `## Heading` and underlined `Heading:` styles), never cuts tables mid-row (oversized tables repeat their header), merges tiny sibling sections, and overlaps chunks by whole paragraphs.
- **Contextual headers.** Each chunk is embedded with its document title and section path, so "Revenue: $2.4B" is still found by "Q3 2024 revenue".
- **Time metadata.** Quarter/year come from the filename, or from section headings for multi-quarter reports, enabling filters like `quarter=Q3`.
- **CSV rows become readable records** ("Department: Finance") instead of raw comma-separated lines.
- **Incremental and idempotent.** File SHA-256 hashes detect changes; chunk ids are deterministic UUIDs; new versions are written before stale chunks are deleted, so a document never disappears mid-update.

| Component | Choice | Why |
|---|---|---|
| Embeddings | `BAAI/bge-small-en-v1.5` via fastembed (ONNX) | Strong quality for its size, runs locally on CPU, no PyTorch, no API cost |
| Vector DB | Qdrant (embedded locally, server in production) | Fast payload filtering, which RBAC depends on |
