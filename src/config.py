import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
INDEX_DIR = Path(os.getenv("INDEX_DIR", str(DATA_DIR / "indices")))

DATA_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR.mkdir(parents=True, exist_ok=True)

# API Keys
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Embedding Model
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-m3")
EMBEDDING_DIM = 1024

# LLM Configurations
PRIMARY_LLM_PROVIDER = os.getenv("PRIMARY_LLM_PROVIDER", "groq")
PRIMARY_LLM_MODEL = os.getenv("PRIMARY_LLM_MODEL", "llama-3.3-70b-versatile")
FALLBACK_LLM_MODEL = os.getenv("FALLBACK_LLM_MODEL", "llama-3.1-8b-instant")

# Retrieval & Guardrail Thresholds
MAX_RETRIEVAL_RESULTS = int(os.getenv("MAX_RETRIEVAL_RESULTS", "5"))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.35"))
RRF_K = 60
DENSE_WEIGHT = 0.65
LEXICAL_WEIGHT = 0.35

def get_bool_env(name: str, default: bool = False) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in ("true", "1", "yes", "on")

# Mock & Fallback Configuration Toggles
FORCE_STT_MOCK = get_bool_env("FORCE_STT_MOCK", False)
ALLOW_STT_FALLBACK = get_bool_env("ALLOW_STT_FALLBACK", True)

FORCE_LLM_MOCK = get_bool_env("FORCE_LLM_MOCK", False)
ALLOW_LLM_FALLBACK = get_bool_env("ALLOW_LLM_FALLBACK", True)

FORCE_SAMPLE_CORPUS = get_bool_env("FORCE_SAMPLE_CORPUS", False)
ALLOW_DATASET_FALLBACK = get_bool_env("ALLOW_DATASET_FALLBACK", True)

# Supported Languages in MSMARCO-XI
SUPPORTED_LANGUAGES = ["en", "hi", "ta"]
DEFAULT_LANG = "en"
