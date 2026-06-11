# config.py  —  Colab-friendly configuration for SICD Test
# All keys are read from environment variables (set in the notebook).
import os
from pathlib import Path

# Load .env file if present
try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).resolve().parent / ".env"
    if _env_path.exists():
        load_dotenv(_env_path)
except ImportError:
    pass

ANTHROPIC_API_KEY  = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY     = os.getenv("OPENAI_API_KEY", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
GOOGLE_API_KEY     = os.getenv("GOOGLE_API_KEY", "")

UMLS_API_KEY       = os.getenv("UMLS_API_KEY", "")
UMLS_USERNAME      = os.getenv("UMLS_USERNAME", "")
UMLS_LOCAL_DB_PATH = os.getenv("UMLS_LOCAL_DB_PATH", "")

UMLS_REST_ENDPOINT = os.getenv("UMLS_REST_ENDPOINT", "https://uts-ws.nlm.nih.gov")
UMLS_AUTH_ENDPOINT = os.getenv("UMLS_AUTH_ENDPOINT", "https://utslogin.nlm.nih.gov")

# App (not used in notebook mode, but kept for import compatibility)
APP_HOST          = os.getenv("APP_HOST", "127.0.0.1")
APP_PORT          = int(os.getenv("APP_PORT", "5005"))
AUTO_OPEN_BROWSER = os.getenv("AUTO_OPEN_BROWSER", "0") == "1"

RUNS_DIR    = os.getenv("RUNS_DIR", "runs")
REPORTS_DIR = os.getenv("REPORTS_DIR", "reports")
LOG_LEVEL   = os.getenv("LOG_LEVEL", "INFO")

# NLI
NLI_MODEL_NAME = os.getenv("NLI_MODEL_NAME", "Bam3752/PubMedBERT-BioNLI-LoRA")
NLI_MAX_LEN    = int(os.getenv("NLI_MAX_LEN", "256"))
NLI_BATCH_SIZE = int(os.getenv("NLI_BATCH_SIZE", "16"))
NLI_TEMP       = float(os.getenv("NLI_TEMP", "1.2"))

# Calibration
CALIBRATION_PATH = os.getenv(
    "CALIBRATION_PATH",
    f"{RUNS_DIR}/pubmedbert_nli_lora/calibration/isotonic.pkl"
)
USE_CALIBRATION = os.getenv("USE_CALIBRATION", "1") == "1"
CONFORMAL_ALPHA = float(os.getenv("CONFORMAL_ALPHA", "0.1"))

CONTRADICTION_THRESHOLD: float = 0.25
RELATION_AWARE_OVERRIDE: bool  = True

USE_ENSEMBLE_EXTRACTOR = os.getenv("USE_ENSEMBLE_EXTRACTOR", "1") == "1"
TOP_K_LINKING          = int(os.getenv("TOP_K_LINKING", "5"))

LR     = float(os.getenv("LR", "2e-5"))
EPOCHS = int(os.getenv("EPOCHS", "4"))

SHOW_FLAGS_DEFAULT = os.getenv("SHOW_FLAGS_DEFAULT", "0") == "1"
ENABLE_HEATMAP     = os.getenv("ENABLE_HEATMAP", "1") == "1"
ENABLE_GRAPH       = os.getenv("ENABLE_GRAPH", "1") == "1"
