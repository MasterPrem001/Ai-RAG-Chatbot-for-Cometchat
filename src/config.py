"""
Configuration loader for Aster & Row RAG Support Agent.

Reads settings from environment variables / .env file.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

# --- Paths ---
PROJECT_ROOT = _PROJECT_ROOT
KNOWLEDGE_BASE_DIR = _PROJECT_ROOT / "knowledge-base"
ORDERS_FILE = _PROJECT_ROOT / "data" / "orders.json"
CHROMA_PERSIST_DIR = _PROJECT_ROOT / ".chroma_db"
LOG_DIR = _PROJECT_ROOT / "logs" / "traces"

# --- LLM configuration ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
GEMINI_BACKUP_MODEL = os.getenv("GEMINI_BACKUP_MODEL", "")


# --- Embeddings (local) ---
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# --- Agent settings ---
DEBUG = os.getenv("DEBUG", "false").lower() == "true"
MAX_HISTORY_TURNS = int(os.getenv("MAX_HISTORY_TURNS", "10"))
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "10"))
FINAL_TOP_K = int(os.getenv("FINAL_TOP_K", "5"))

# --- Ensure directories exist ---
LOG_DIR.mkdir(parents=True, exist_ok=True)
