# FinSolve RBAC RAG Chatbot

An internal chatbot for FinSolve Technologies. It uses Retrieval-Augmented Generation (RAG) with **role-based access control**: each user sees only the data their role permits, and every answer cites its source document.

**Stack:** Python · FastAPI · Qdrant (hybrid dense + BM25) · fastembed (BGE embeddings, Jina reranker) · Groq `gpt-oss-120b` (any OpenAI-compatible LLM) · Streamlit

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

# 3. Install the app, the UI and developer tools
pip install -e ".[dev,ui]"
pre-commit install

# 4. Configure environment
copy .env.example .env          # Windows  (cp on macOS / Linux)
# then set LLM_API_KEY (free key: https://console.groq.com/keys)

# 5. Index documents and create demo users
python -m app.ingestion.pipeline
python -m app.auth.manage seed

# 6. Run the API
uvicorn app.main:app --reload
```
Then, in a second terminal, start the chat UI and open http://localhost:8501:
```bash
streamlit run ui/streamlit_app.py
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

### Measure retrieval quality
```bash
python -m eval.retrieval_eval --compare --details
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
| POST | `/chat` | Ask a question, get a grounded answer with `[n]` citations and its sources |
| POST | `/chat/stream` | Same, streamed as Server-Sent Events (`token` … `done`, or `error`) |
| POST | `/search` | Semantic search limited to your role's departments, with source references |
| GET | `/health` | Liveness check |

User admin: `python -m app.auth.manage {seed,list,create,set-role,disable,enable}`.

## How an answer is generated
```
question ─▶ SecureRetriever (top-k chunks, filtered by role) ─▶ prompt: rules + numbered <source> blocks + question
         ─▶ LLM ─▶ answer with [n] citations ─▶ map [n] back to documents/sections ─▶ response
```
- **Grounded:** the model may only use the numbered sources and must say "not found" rather than guess. If retrieval returns nothing the role may see, no LLM call is made.
- **Cited:** every fact carries `[n]`; the API marks which sources were actually cited.
- **Injection-aware:** source text is wrapped in `<source>` tags and declared to be data. Clients can send only `user`/`assistant` history, never `system`.
- **Provider-agnostic:** one OpenAI-compatible client covers Groq, OpenAI and Ollama; switching is a `.env` change (`LLM_PROVIDER`, `LLM_MODEL`, `LLM_BASE_URL`).

## Chat UI
A Streamlit app ([`ui/`](ui/)) that talks to the API over HTTP only. It never imports backend code, so UI and API deploy and scale separately (`API_BASE_URL` points it at the backend).

- Login with friendly errors; the sidebar shows your role and the departments you can access.
- Suggested questions per role (from `roles.yaml`) on an empty chat.
- Streamed answers, follow-up questions with conversation history, and a **Sources** panel per answer (document, section, department, quarter, relevance, excerpt). A toggle also shows retrieved-but-uncited sources.
- Expired sessions return to login. The session lives in the browser tab: refreshing the page logs you out (by design; the token is never put in the URL).

## Retrieval quality (measured)
The four quarterly marketing reports share one template, and finance reports cover four quarters with identical section names, so pure semantic search often returned the right *kind* of section from the *wrong* quarter. Phase 5 adds layers on top of semantic search, each measured on a 25-question golden set ([`eval/retrieval_golden.yaml`](eval/retrieval_golden.yaml)) with C-level scope (all departments competing):

| Retrieval mode | Hit@1 | Hit@3 | Hit@6 | MRR |
|---|---|---|---|---|
| Dense only (Phases 1-4) | 64% | 80% | 84% | 0.730 |
| + quarter boost | 76% | 84% | 88% | 0.807 |
| Hybrid (dense + BM25) | 80% | 92% | 96% | 0.870 |
| Hybrid + quarter boost | 84% | 88% | 96% | 0.877 |
| **Hybrid + boost + rerank (default)** | **88%** | **100%** | **100%** | **0.940** |

*Hit@k: share of questions whose correct section is in the top k (6 chunks go to the LLM). MRR: mean of 1/rank of the first correct chunk.*

- **Hybrid search.** Each chunk has a dense (meaning) vector and a sparse BM25 (keyword) vector. Qdrant runs both and merges them with Reciprocal Rank Fusion, so exact terms like "DSO" or "FINEMP1003" are found too.
- **Quarter boost (soft).** Quarters in the question ("Q3", "third quarter") add extra candidate lists restricted to those quarters. Matching chunks get more votes but nothing is excluded, so "recommendations for Q1 2025" still finds them in the Q4 report.
- **Reranking.** A cross-encoder reads question and chunk *together* and re-scores the top candidates. Chosen by benchmark: `jina-reranker-v1-turbo-en` on 10 candidates (MRR 0.940, ~1 s on CPU) beat `ms-marco-MiniLM-L-12` on 30 candidates (MRR 0.873, ~6.7 s).
- **RBAC on every list.** Each dense, BM25 and quarter-boosted candidate list carries the department filter; the leak tests run through this full pipeline.

### Conversational behaviour
- **Follow-ups** ("And Q4?") are rewritten by the LLM into standalone searches ("net income and marketing spend Q4 2024"); the UI shows what was searched.
- **Out-of-scope questions** get a clear answer ("your role (Employee) does not have access… contact HR") instead of a vague "not found". Only department *names and descriptions* are given to the model, never their contents.
- **Ambiguous questions** get a clarifying question (e.g. two employees named Isha Chowdhury → "which one?").
- **Vague questions** get a short overview plus suggested follow-up questions.
