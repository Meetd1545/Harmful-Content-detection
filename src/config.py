"""Central configuration for the project. Change values here, not inside the scripts."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_MANUAL = ROOT / "data" / "manual"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
for p in (DATA_RAW, DATA_MANUAL, DATA_PROCESSED, MODELS, RESULTS):
    p.mkdir(parents=True, exist_ok=True)

SEED = 42
DIMENSIONS = ["malicious", "deceptive", "manipulative"]
UNKNOWN = -1  # label value when a source does not annotate a dimension

# Reduced technique set (Section 3.3.2 of the report)
TECHNIQUES = [
    "urgency", "scarcity", "fear_threat", "false_authority", "impersonation",
    "reward_reciprocity", "social_proof", "loaded_language", "false_dilemma",
    "instruction_override",
]

# ---- Data sources (downloaded by prepare_data.py) ----
SMS_SPAM_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
LIAR_URL = "https://www.cs.ucsb.edu/~william/data/liar_dataset.zip"
NAZARIO_INDEX_URL = "https://monkey.org/~jose/phishing/"
NAZARIO_MAX_EMAILS = 1500
PROMPT_INJECTION_HF = "deepset/prompt-injections"
# SemEval-2023 Task 3 data needs registration with the organisers. If you obtain it,
# place the English training folder at data/raw/semeval2023/ (see README).
SEMEVAL_DIR = DATA_RAW / "semeval2023"

# ---- Transformer screener (M2) ----
ENCODER_NAME = "google/muril-base-cased"   # alternative: "xlm-roberta-base"
MAX_LEN = 128
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
EPOCHS = 3
WEIGHT_DECAY = 0.01

# ---- LLM analyser (M3) ----
# backend: "hf" (Hugging Face Transformers, needs a GPU) or "ollama" (laptop, uses the GPU if present)
LLM_BACKEND = "hf"
LLM_HF_MODEL = "Qwen/Qwen2.5-3B-Instruct"
LLM_OLLAMA_MODEL = "qwen2.5:3b"
LLM_OLLAMA_URL = "http://localhost:11434/api/chat"
LLM_MAX_NEW_TOKENS = 400
LLM_TEMPERATURE = 0.0
FEW_SHOT = True

# ---- Cascade (M4) ----
LOW_THRESHOLD = 0.20     # if all calibrated screener scores are below this, item is "likely safe"
DECISION_THRESHOLD = 0.50
W_SCREENER, W_LLM = 0.5, 0.5
EVIDENCE_BOOST = 0.20    # added to malicious and deceptive scores when a look-alike URL is found
RISK_HIGH, RISK_MEDIUM = 0.70, 0.40
AGREEMENT_HIGH = 0.30    # |screener - llm| below this on all dimensions -> High confidence

# ---- Evaluation ----
LLM_EVAL_SAMPLE = 400    # items from the test set evaluated with M3 / M4 (LLM is slow)
ROBUSTNESS_SAMPLE = 100  # harmful test items used to create each adversarial variant

# Official domains used by the look-alike URL check (extend as needed)
OFFICIAL_DOMAINS = [
    "sbi.co.in", "onlinesbi.sbi", "hdfcbank.com", "icicibank.com", "axisbank.com", "kotak.com",
    "pnbindia.in", "bankofbaroda.in", "paytm.com", "phonepe.com", "npci.org.in", "bhimupi.org.in",
    "incometax.gov.in", "uidai.gov.in", "india.gov.in", "indiapost.gov.in", "cybercrime.gov.in",
    "epfindia.gov.in", "amazon.in", "flipkart.com", "irctc.co.in", "airtel.in", "jio.com",
    "mahadiscom.in", "tatapower.com", "adanielectricity.com", "bescom.co.in", "google.com", "microsoft.com",
]
BRAND_KEYWORDS = ["sbi", "hdfc", "icici", "axis", "kotak", "pnb", "paytm", "phonepe", "upi", "npci",
                  "incometax", "aadhaar", "uidai", "indiapost", "epf", "amazon", "flipkart", "irctc",
                  "airtel", "jio", "bijli", "electricity", "mseb", "kyc"]
URL_SHORTENERS = ["bit.ly", "tinyurl.com", "t.co", "goo.gl", "cutt.ly", "rb.gy", "is.gd", "ow.ly",
                  "shorturl.at", "tiny.cc", "s.id", "rebrand.ly"]
