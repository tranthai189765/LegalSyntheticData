import os
from dotenv import load_dotenv

load_dotenv()

# ── LLM ──────────────────────────────────────────────────────────────────────
FPT_API_KEY   = os.getenv("FPT_API_KEY", "")
FPT_API_URL   = os.getenv("FPT_API_URL", "https://mkp-api.fptcloud.com/v1/chat/completions")
FPT_MODEL     = os.getenv("FPT_MODEL", "gpt-oss-120b")
FPT_BASE_URL  = FPT_API_URL.replace("/chat/completions", "")

# ── Neo4j (HTTP API) ─────────────────────────────────────────────────────────
NEO4J_URL      = os.getenv("NEO4J_URL", "http://localhost:7474")
NEO4J_USER     = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "12345678")

# ── Pipeline ─────────────────────────────────────────────────────────────────
BTC_PRIORITY_RATIO  = float(os.getenv("BTC_PRIORITY_RATIO", "0.85"))   # % samples from BTC docs
MAX_LEGAL_UNITS     = int(os.getenv("MAX_LEGAL_UNITS", "5"))            # articles per context block
KG_EXPANSION_HOPS   = int(os.getenv("KG_EXPANSION_HOPS", "2"))         # graph traversal depth
MAX_RETRIES         = int(os.getenv("MAX_RETRIES", "3"))                # QA regeneration attempts
LLM_CONCURRENCY     = int(os.getenv("LLM_CONCURRENCY", "3"))           # parallel pipeline slots
LLM_TEMPERATURE     = float(os.getenv("LLM_TEMPERATURE", "0.7"))
CHECKER_PASS_THRESHOLD = float(os.getenv("CHECKER_PASS_THRESHOLD", "0.7"))  # min confidence to accept

# ── Output ───────────────────────────────────────────────────────────────────
OUTPUT_DIR  = os.getenv("OUTPUT_DIR", "output")
OUTPUT_FILE = os.getenv("OUTPUT_FILE", "output/synthetic_data.jsonl")
LOG_FILE    = os.getenv("LOG_FILE", "output/pipeline.log")
