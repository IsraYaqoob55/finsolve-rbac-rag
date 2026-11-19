"""UI settings, from environment variables (the UI doesn't share the backend's config)."""

import os

API_BASE_URL = os.environ.get("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
# How many previous messages to send with each question, for follow-ups.
HISTORY_MESSAGES = int(os.environ.get("UI_HISTORY_MESSAGES", "6"))

# Colours for department badges (Streamlit badge colour names).
DEPARTMENT_COLORS = {
    "general": "gray",
    "engineering": "blue",
    "finance": "green",
    "marketing": "orange",
    "hr": "violet",
}

DEMO_ACCOUNTS = [
    ("alice_emp", "Employee"),
    ("ravi_eng", "Engineering"),
    ("meera_mkt", "Marketing"),
    ("kabir_fin", "Finance"),
    ("sara_hr", "HR"),
    ("tony_ceo", "C-Level"),
]
