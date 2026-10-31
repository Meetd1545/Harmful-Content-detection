"""FULL RUN with an NVIDIA GPU: trains M2 (MuRIL) and runs M1-M4 and the robustness tests.

    python run_full.py

Before running:
  1. Run  python run_easy.py  once (prepares the data and trains M1).
  2. Install Ollama from https://ollama.com and open it (the LLM, qwen2.5:3b, is downloaded automatically).
The first time, this script installs PyTorch with GPU support (about 2.5 GB) and downloads MuRIL (about 1 GB).
At the end, results/REPORT_OUTPUT.txt contains every table for Chapter 4.
"""
import importlib.util
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
PY = sys.executable
sys.path.insert(0, str(ROOT))

try:
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
    print(f"   done in {(time.time() - t0) / 60:.1f} min")


def cuda_ok():
    code = "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)"
    return subprocess.run([PY, "-c", code], capture_output=True).returncode == 0


def ensure_gpu_packages():
    if importlib.util.find_spec("torch") is None or not cuda_ok():
        print("   installing PyTorch with GPU (CUDA 12.4) support - about 2.5 GB, one time only ...")
        extra = ["--force-reinstall", "--no-deps"] if importlib.util.find_spec("torch") else []
        subprocess.run([PY, "-m", "pip", "install", "--upgrade", *extra, "torch", "--index-url",
                        "https://download.pytorch.org/whl/cu124"], check=True)
        if extra:  # make sure torch's own dependencies are present
            subprocess.run([PY, "-m", "pip", "install", "torch", "--index-url", "https://download.pytorch.org/whl/cu124"], check=True)
    if importlib.util.find_spec("transformers") is None:
        subprocess.run([PY, "-m", "pip", "install", "transformers>=4.41", "accelerate>=0.30"], check=True)
    if not cuda_ok():
        sys.exit("[ERROR] PyTorch cannot see the NVIDIA GPU. Update the NVIDIA driver (nvidia.com/drivers) and try again.")
    out = subprocess.run([PY, "-c", "import torch;print(torch.cuda.get_device_name(0))"], capture_output=True, text=True)
    print("   GPU:", out.stdout.strip())


def main():
    t_all = time.time()
    if not (ROOT / "data" / "processed" / "test.csv").exists() or not (ROOT / "models" / "baseline.joblib").exists():
        sys.exit("Run  python run_easy.py  once first (it prepares the data and trains M1).")

    step(1, "checking PyTorch and the GPU"); ensure_gpu_packages()

    step(2, "training M2 (MuRIL transformer) on the GPU")
    if (ROOT / "models" / "transformer" / "model.pt").exists():
        print("   already trained - delete models/transformer to train again")
    else:
        run("transformer_model.py")

    step(3, "checking the LLM (Ollama)")
    import run_easy
    if not run_easy.ollama_ready():
        sys.exit("[ERROR] Ollama is needed for M3 and M4. Install it from https://ollama.com, open it, and run this script again.")

    step(4, "running all experiments (M1-M4 and robustness) - about 30-60 minutes, keep this window open")
    run("evaluate.py", "--llm-backend", "ollama", "--sample", "300", "--rob", "50", "--gen", "20")

    print(f"\nALL DONE in {(time.time() - t_all) / 60:.1f} minutes.")
    print("Results: results/REPORT_OUTPUT.txt (tell Claude 'done' so the report can be updated)")
    print("Web app with M2 + LLM:  python webapp.py")


if __name__ == "__main__":
    main()
