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
