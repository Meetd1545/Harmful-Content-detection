"""Check one message from the command line (demo without the web app).

    python check_message.py "Aapka bijli connection aaj raat kaat diya jayega. Turant call karein 98XXXXXX10"
    python check_message.py            (asks you to type a message)

Run  python run_easy.py  once first, so that the model exists.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import joblib  # noqa: E402
import requests  # noqa: E402
import baseline  # noqa: E402,F401
from config import MODELS, DIMENSIONS  # noqa: E402
from cascade import Cascade, load_temperatures  # noqa: E402
from evidence import PatternIndex  # noqa: E402


def build():
    if (MODELS / "transformer" / "model.pt").exists():
        from cascade import TransformerScreener
        screener = TransformerScreener()
    else:
        from evaluate import BaselineAsScreener
        if not (MODELS / "baseline.joblib").exists():
            sys.exit("No model found. Run  python run_easy.py  first.")
        screener = BaselineAsScreener(joblib.load(MODELS / "baseline.joblib"))
    llm = None
    try:
        requests.get("http://localhost:11434/api/tags", timeout=3)
        from llm_analyser import LLMAnalyser
        llm = LLMAnalyser("ollama")
    except Exception:
        print("(Ollama not running: showing the result without the LLM explanation)\n")
    return Cascade(screener, llm, load_temperatures(), PatternIndex())


def show(r, text):
    print("=" * 70)
    print("MESSAGE :", text)
    print("LANGUAGE:", r["language"])
    for d in DIMENSIONS:
        bar = "#" * int(r[d] * 20)
        print(f"{d.capitalize():<13} {int(r[d] * 100):>3}%  {bar}")
    print(f"OVERALL : {r['risk']} RISK   (confidence: {r['confidence']})")
    if r["techniques"]:
        print("TECHNIQUES:")
        for t in r["techniques"]:
            print(f"   - {t['name'].replace('_', ' ')}: \"{t['span']}\"")
    for u in r["evidence"]["urls"]:
        if u["reason"]:
            print(f"LINK    : {u['domain']} -> {u['reason']}")
    for m in r["evidence"]["pattern_matches"]:
        print(f"SIMILAR : known scam pattern '{m['scam_type'].replace('_', ' ')}'")
    print("WHY     :", r["explanation"])
    print("ADVICE  :", r["advice"])
    print(f"(time: {r['latency_s']} s)")
    print("=" * 70)


if __name__ == "__main__":
    text = " ".join(sys.argv[1:]) or input("Paste a message and press Enter:\n> ")
    show(build().analyse(text), text)
