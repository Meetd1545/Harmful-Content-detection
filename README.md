# NLP and LLM-Based Methodology for Identifying Malicious, Deceptive, and Manipulative Content

Design Experience I (VIP) project — Meet Dodiya (F070), MPSTME, SVKM's NMIMS, 2026–27.
Mentor: Dr. Ruchi Sharma.

The system checks a piece of text (SMS, WhatsApp, email, social post, or web text read by an AI assistant)
for three kinds of harm — **malicious**, **deceptive**, **manipulative** — names the manipulation techniques,
highlights the exact words, explains the result in plain language and suggests a safe action.

## How it works
```
text ─► pre-processing ─► transformer screener (MuRIL, 3 scores)
                         │        │ any score ≥ 0.20 or suspicious link
                         │        ▼
                         │   LLM analyser (Qwen2.5-3B-Instruct, JSON: scores, techniques, spans, explanation)
                         ├─► evidence checks (look-alike URLs, shorteners, known scam patterns)
                         ▼
                      fusion (calibrated) ─► web app: highlights, risk, explanation, advice, feedback
```
Four approaches are compared: **M1** TF-IDF + logistic regression, **M2** fine-tuned MuRIL,
**M3** prompted LLM, **M4** hybrid cascade (the proposed system).

## Folder structure
| Path | Contents |
|---|---|
| `src/config.py` | All settings (models, thresholds, data URLs) |
| `src/preprocess.py` | Language ID (English / Hindi / Hinglish), de-obfuscation, URL / phone / amount extraction |
| `src/prepare_data.py` | Downloads public datasets, maps labels, creates splits |
| `src/baseline.py` | M1 |
| `src/transformer_model.py` | M2 (multi-label, masked loss for unannotated labels) |
| `src/llm_analyser.py` | M3: prompt, prompt-injection defence, JSON parsing, span verification |
| `src/evidence.py` | Look-alike URL check, shortener check, known-scam-pattern search |
| `src/calibration.py` | Temperature scaling, expected calibration error |
| `src/cascade.py` | M4: routing and fusion |
| `src/robustness.py` | Obfuscation, LLM paraphrase / generation, embedded-instruction tests |
| `src/evaluate.py` | Runs all experiments, writes `results/` |
| `webapp.py`, `web/index.html` | Web application (message checker), runs with no extra install |
| `src/cues.py` | Rule-based cue words, used for highlights when the LLM is off |
| `src/survey_analysis.py` | Survey CSV (Google Forms export) → findings + chart for Section 3.1.2 |
| `data/manual/` | Indian collection (seed + your collected messages), known scam patterns |

## Run on a laptop (no GPU needed)
See **RUN_ON_LAPTOP.txt**. In short: `python run_easy.py`, then open `results/REPORT_OUTPUT.txt`.
Open the web app: `python webapp.py` (opens http://localhost:8501).
Try one message in the terminal: `python check_message.py "your message"`.

## Full run with an NVIDIA GPU (M2 transformer + LLM)
Install Ollama (ollama.com), then: `python run_full.py`. It installs PyTorch with GPU support, trains M2 and runs M1–M4 and the robustness tests.

## Advanced: step by step
```bash
pip install -r requirements.txt
bash run_all.sh                 # the transformer needs an NVIDIA GPU
# LLM on a laptop via Ollama instead of Hugging Face:
#   install Ollama, run `ollama pull qwen2.5:3b`, then set LLM_BACKEND = "ollama" in src/config.py
python webapp.py                # web app at http://localhost:8501
```
Quick test without a GPU (results are **not valid** for the report):
`python src/prepare_data.py && python src/baseline.py && python src/evaluate.py --mock-llm`

## Data
| Source | Use | Access |
|---|---|---|
| SMS Spam Collection (Almeida et al., 2011) | malicious / manipulative vs. benign SMS | downloaded automatically (UCI) |
| Nazario phishing corpus | phishing emails | downloaded automatically |
| LIAR (Wang, 2017) | deceptive vs. true claims | downloaded automatically |
| deepset/prompt-injections | instructions aimed at AI systems | downloaded automatically (Hugging Face) |
| SemEval-2023 Task 3 (optional) | persuasion techniques in news | register with the organisers, place files in `data/raw/semeval2023/` |
| `data/manual/indian_messages_seed.csv` | 36 **author-written** example messages (English, Hindi, Hinglish) for testing | included |
| `data/manual/indian_messages_collected.csv` | **your** anonymised real messages | collect with consent; mask names, numbers, account details |

Labels: `1` yes, `0` no, `-1` not annotated by that source (ignored in training and evaluation).

## Ethics
- Collect messages only with consent; remove names, phone numbers, account numbers and real links before saving.
- `results/adversarial_variants_PRIVATE.csv` contains generated scam variants for testing only. Do not publish it.
- The app is advisory: it never blocks, deletes or reports anything automatically.
