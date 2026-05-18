import os
from dotenv import load_dotenv

load_dotenv()

# ── Active LLM provider ───────────────────────────────────────────────────────
# Priority: if OPENAI_API_KEY is set, use OpenAI; otherwise use FPT Cloud.
# Override by setting ACTIVE_PROVIDER=fpt or ACTIVE_PROVIDER=openai explicitly.

_fpt_key    = os.getenv("FPT_API_KEY", "")
_openai_key = os.getenv("OPENAI_API_KEY", "")
_provider   = os.getenv("ACTIVE_PROVIDER", "").lower()

if not _provider:
    _provider = "openai" if _openai_key else "fpt"

if _provider == "openai":
    ACTIVE_API_KEY  = _openai_key
    _openai_url     = os.getenv("OPENAI_API_URL", "https://api.openai.com/v1/chat/completions")
    ACTIVE_BASE_URL = _openai_url.replace("/chat/completions", "")
    ACTIVE_MODEL    = os.getenv("OPENAI_MODEL", "gpt-4o")
    IS_REASONING_MODEL = False   # gpt-4o supports response_format=json_object
else:
    ACTIVE_API_KEY  = _fpt_key
    _fpt_url        = os.getenv("FPT_API_URL", "https://mkp-api.fptcloud.com/v1/chat/completions")
    ACTIVE_BASE_URL = _fpt_url.replace("/chat/completions", "")
    ACTIVE_MODEL    = os.getenv("FPT_MODEL", "gpt-oss-120b")
    IS_REASONING_MODEL = True    # gpt-oss-120b puts CoT in reasoning_content

# Legacy aliases kept for any code that still imports them directly
FPT_API_KEY  = _fpt_key
FPT_BASE_URL = os.getenv("FPT_API_URL", "https://mkp-api.fptcloud.com/v1/chat/completions").replace("/chat/completions", "")
FPT_MODEL    = os.getenv("FPT_MODEL", "gpt-oss-120b")

# ── Neo4j (HTTP API) ─────────────────────────────────────────────────────────
NEO4J_URL      = os.getenv("NEO4J_URL", "http://localhost:7474")
NEO4J_USER     = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "12345678")

# ── Pipeline ─────────────────────────────────────────────────────────────────
BTC_PRIORITY_RATIO     = float(os.getenv("BTC_PRIORITY_RATIO", "0.85"))
MAX_LEGAL_UNITS        = int(os.getenv("MAX_LEGAL_UNITS", "5"))
KG_EXPANSION_HOPS      = int(os.getenv("KG_EXPANSION_HOPS", "2"))
MAX_RETRIES            = int(os.getenv("MAX_RETRIES", "3"))
LLM_CONCURRENCY        = int(os.getenv("LLM_CONCURRENCY", "3"))
LLM_TEMPERATURE        = float(os.getenv("LLM_TEMPERATURE", "0.7"))
CHECKER_PASS_THRESHOLD = float(os.getenv("CHECKER_PASS_THRESHOLD", "0.7"))

# ── Output ───────────────────────────────────────────────────────────────────
OUTPUT_DIR  = os.getenv("OUTPUT_DIR", "output")
OUTPUT_FILE = os.getenv("OUTPUT_FILE", "output/synthetic_data.jsonl")
LOG_FILE    = os.getenv("LOG_FILE", "output/pipeline.log")
