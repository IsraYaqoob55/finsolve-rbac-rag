# FinSolve RBAC RAG Chatbot

An internal chatbot for FinSolve Technologies. It uses Retrieval-Augmented Generation (RAG) with **role-based access control**: each user sees only the data their role permits, and every answer cites its source document.

**Stack:** Python · FastAPI · Qdrant · Streamlit · LLM (pluggable)

> 🚧 Work in progress. See [PROJECT_PLAN.md](PROJECT_PLAN.md) for the full roadmap.

## Roles
Defined in [`config/roles.yaml`](config/roles.yaml); adding a role or department is a config change, not a code change.

| Role | Data access | Demo user |
|---|---|---|
| Employee | General (handbook, policies, FAQs) | `alice_emp` |
| Engineering | Engineering + General | `ravi_eng` |
| Marketing | Marketing + General | `meera_mkt` |
| Finance | Finance + Marketing + General | `kabir_fin` |
| HR | HR + General | `sara_hr` |
| C-Level | Everything | `tony_ceo` |

All demo users share the password set in `SEED_USER_PASSWORD` (see `.env.example`).

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

# 5. Index documents and create demo users
python -m app.ingestion.pipeline
python -m app.auth.manage seed

# 6. Run the API
uvicorn app.main:app --reload
```
> Embedded Qdrant allows one process at a time: stop the API before re-running the ingestion pipeline (a Qdrant server, used in Docker, has no such limit).

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

## Security model
```
login ─▶ bcrypt check ─▶ JWT ─▶ every request: verify token ─▶ reload user from DB ─▶ role ─▶ AccessScope
                                                                                         │
             Qdrant search with filter department ∈ scope ◀───────────────────────────────┘
                              │
                     RBAC guard re-checks every chunk ─▶ response
```
- **Access control happens in retrieval, not in the prompt.** Out-of-scope chunks are filtered inside the vector database, so they never reach the LLM and can't be leaked by prompt injection.
- **Defence in depth.** A guard re-checks each retrieved chunk's department and logs a security event if anything slips through.
- **Deny by default.** Unknown roles get HTTP 403, `"*"` expands to an explicit department list, and the policy is validated at startup (typos in department names fail fast).
- **Instant revocation.** The user record is re-read on every request, so disabling a user or changing their role applies even to tokens already issued.
- **Credential hygiene.** bcrypt hashes; the same error for unknown user and wrong password, with equal timing; pinned JWT algorithm (no `alg: none`); the app refuses to start in production with the placeholder JWT secret.
- **Tested.** 54 leak tests (9 adversarial queries × 6 roles) check the access matrix from the problem statement, written independently of the config.

## API
Open http://127.0.0.1:8000/docs, click **Authorize**, and log in as a demo user.

| Method | Path | Description |
|---|---|---|
| POST | `/auth/login` | Username + password → bearer token |
| GET | `/auth/me` | Your profile, role and accessible departments |
| POST | `/search` | Semantic search limited to your role's departments, with source references |
| GET | `/health` | Liveness check |

User admin: `python -m app.auth.manage {seed,list,create,set-role,disable,enable}`.
