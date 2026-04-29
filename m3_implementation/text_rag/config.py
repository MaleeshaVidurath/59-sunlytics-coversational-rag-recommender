# m3_implementation/text_rag/config.py
#
# Configuration for all Text RAG components.
# All settings read from environment variables with sensible defaults.

import os
from dotenv import load_dotenv
load_dotenv()

# ── PostgreSQL ─────────────────────────────────────────────────────────────────
POSTGRES_HOST     = os.getenv("POSTGRES_HOST",     "localhost")
POSTGRES_PORT     = int(os.getenv("POSTGRES_PORT", "5432"))
POSTGRES_DB       = os.getenv("POSTGRES_DB",       "sunlytics")
POSTGRES_USER     = os.getenv("POSTGRES_USER",     "postgres")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "postgres")
POSTGRES_URL      = (
    f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}"
    f"@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
)

# ── Qdrant ─────────────────────────────────────────────────────────────────────
QDRANT_HOST            = os.getenv("QDRANT_HOST",       "localhost")
QDRANT_PORT            = int(os.getenv("QDRANT_PORT",   "6333"))
QDRANT_COLLECTION      = os.getenv("QDRANT_COLLECTION", "articles")
QDRANT_VECTOR_SIZE     = 384   # all-MiniLM-L6-v2 output dimension
QDRANT_DISTANCE        = "Cosine"

# ── LLM Provider (Groq or Ollama) ──────────────────────────────────────────────
# Set LLM_PROVIDER=groq to use Groq cloud API (recommended)
# Set LLM_PROVIDER=ollama to use local Ollama (default fallback)
LLM_PROVIDER           = os.getenv("LLM_PROVIDER", "groq")

# Groq settings
GROQ_API_KEY           = os.getenv("GROQ_API_KEY", "")
GROQ_BASE_URL          = "https://api.groq.com/openai/v1"
GROQ_MODEL             = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

# Ollama settings (fallback if LLM_PROVIDER=ollama)
OLLAMA_HOST            = os.getenv("OLLAMA_HOST",  "http://localhost:11434")
OLLAMA_RAG_MODEL       = os.getenv("OLLAMA_RAG_MODEL", "llama3.1:8b")

# Unified model name for logging
ACTIVE_MODEL = GROQ_MODEL if LLM_PROVIDER == "groq" else OLLAMA_RAG_MODEL

# ── NLI Hallucination Checker ──────────────────────────────────────────────────
NLI_MODEL_NAME              = os.getenv("NLI_MODEL_NAME", "cross-encoder/nli-deberta-v3-base")
NLI_CONTRADICTION_THRESHOLD = float(os.getenv("NLI_CONTRADICTION_THRESHOLD", "0.65"))
NLI_ENTAILMENT_THRESHOLD    = float(os.getenv("NLI_ENTAILMENT_THRESHOLD",    "0.20"))
MAX_REGENERATION_ATTEMPTS   = int(os.getenv("MAX_REGENERATION_ATTEMPTS",     "3"))

# ── Data paths ─────────────────────────────────────────────────────────────────
BASE_DIR         = os.path.join(os.path.dirname(__file__), '..', '..', 'shared', 'main_data_set')
ARTICLES_CSV     = os.path.join(BASE_DIR, 'sample_articles.csv')
TRANSACTIONS_CSV = os.path.join(BASE_DIR, 'sample_transactions.csv')

# ── Embedding model (shared with memory module) ────────────────────────────────
EMBEDDING_MODEL  = "all-MiniLM-L6-v2"

# ── RAG settings ───────────────────────────────────────────────────────────────
MAX_RECOMMENDATIONS = 2       # always recommend exactly 2 items
MAX_EVIDENCE_ITEMS  = 5       # max evidence pieces passed to LLM
PRICE_SCALE         = 595.08  # multiply normalised price to get £
