from pathlib import Path
import os
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROC_DIR = DATA_DIR / "processed"
REPORT_DIR = ROOT / "reports"

load_dotenv(ROOT / ".env")

ALPHAVANTAGE_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY", "")
if not ALPHAVANTAGE_API_KEY:
    # You can still run code that doesn't hit the API,
    # but fetching will error until you set this.
    pass

for d in [DATA_DIR, RAW_DIR, PROC_DIR, REPORT_DIR]:
    d.mkdir(parents=True, exist_ok=True)
