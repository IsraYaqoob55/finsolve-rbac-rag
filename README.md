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
