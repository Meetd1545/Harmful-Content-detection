"""ONE-COMMAND RUN for a normal laptop (no GPU needed).

    python run_easy.py            quick run  (about 15-40 minutes with the LLM, 2-5 minutes without)
    python run_easy.py --full     larger samples (slower, better for the final report)
    python run_easy.py --no-llm   skip the LLM even if Ollama is installed

At the end, open results/REPORT_OUTPUT.txt and copy the tables into Chapter 4 of the report.
The LLM part needs Ollama (free, ollama.com). If Ollama is not installed, everything else still runs.
"""
import argparse
import importlib.util
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
PY = sys.executable
OLLAMA_MODEL = "qwen2.5:3b"
LITE_PACKAGES = {"sklearn": "scikit-learn", "pandas": "pandas", "numpy": "numpy", "scipy": "scipy",
                 "matplotlib": "matplotlib", "joblib": "joblib", "requests": "requests"}

try:  # Hindi text in the Windows terminal
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def step(n, msg):
    print(f"\n========== STEP {n}: {msg} ==========", flush=True)


def run(script, *args):
    t0 = time.time()
    r = subprocess.run([PY, str(SRC / script), *args], cwd=str(ROOT))
    if r.returncode != 0:
        sys.exit(f"\n[ERROR] {script} failed. Take a screenshot of the error above and send it for help.")
    print(f"   done in {time.time() - t0:.0f} s")


def ensure_packages():
    missing = [pip for mod, pip in LITE_PACKAGES.items() if importlib.util.find_spec(mod) is None]
    if missing:
        print("Installing:", " ".join(missing))
        subprocess.run([PY, "-m", "pip", "install", "--quiet", *missing], check=True)
    else:
        print("   all packages already installed")


def ollama_ready():
    import requests
    try:
        tags = requests.get("http://localhost:11434/api/tags", timeout=5).json()
    except Exception:
        print("   Ollama is not running or not installed -> the LLM part (M3, M4) will be skipped.")
        print("   To include it: install Ollama from https://ollama.com, open it, then run this script again.")
        return False
    names = [m.get("name", "") for m in tags.get("models", [])]
    if not any(n.startswith(OLLAMA_MODEL) for n in names):
        exe = shutil.which("ollama")
        if not exe:
            print(f"   Ollama is running but the model is missing. Run:  ollama pull {OLLAMA_MODEL}")
            return False
        print(f"   downloading the LLM {OLLAMA_MODEL} (about 2 GB, one time only) ...")
        if subprocess.run([exe, "pull", OLLAMA_MODEL]).returncode != 0:
            return False
    print(f"   Ollama ready with {OLLAMA_MODEL}")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--no-llm", action="store_true")
    a = ap.parse_args()
    if sys.version_info < (3, 9):
        sys.exit("Python 3.9 or newer is needed. Download it from python.org.")
    t_all = time.time()

    step(1, "checking Python packages"); ensure_packages()
    step(2, "downloading and preparing the datasets (needs internet)"); run("prepare_data.py")
    step(3, "training M1 (TF-IDF + logistic regression)"); run("baseline.py")

    use_llm = False
    if not a.no_llm:
        step(4, "checking the LLM (Ollama)")
        use_llm = ollama_ready()

    if (ROOT / "models" / "transformer" / "model.pt").exists():
        print("\n   Found a trained transformer (M2) - it will be used as the screener.")
    else:
        print("\n   No transformer (M2) found - the cascade uses M1 as its screener (normal for the laptop run).")

    step(5, "running the experiments")
    sizes = ["--sample", "300", "--rob", "50", "--gen", "20"] if a.full else ["--sample", "80", "--rob", "15", "--gen", "6"]
    if use_llm:
        print("   The LLM checks one message at a time; on a laptop this takes a while. Keep this window open.")
        run("evaluate.py", "--llm-backend", "ollama", *sizes)
    else:
        run("evaluate.py", "--skip-llm", *sizes)

    out = ROOT / "results" / "REPORT_OUTPUT.txt"
    print(f"\nALL DONE in {(time.time() - t_all) / 60:.1f} minutes.")
    print(f"Open this file and copy the tables into the report:\n   {out}")
    print("Charts (insert as figures):  results/chart_dataset.png, results/chart_f1_by_approach.png, results/chart_robustness.png")
    print("Open the web app:            python webapp.py")
    print("Try a single message:        python check_message.py \"your message here\"")


if __name__ == "__main__":
    main()
