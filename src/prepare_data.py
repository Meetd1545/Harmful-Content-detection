"""Download public datasets, map them to the three dimensions and create train / validation / test splits.

Label values: 1 = yes, 0 = no, -1 = not annotated by this source (ignored in training and evaluation).
Run:  python src/prepare_data.py
Output: data/processed/{train,val,test}.csv and results/dataset_stats.json
"""
import io
import json
import mailbox
import re
import zipfile
from email.header import decode_header, make_header
from html import unescape

import numpy as np
import pandas as pd
import requests
from sklearn.model_selection import train_test_split

from config import (DATA_RAW, DATA_MANUAL, DATA_PROCESSED, RESULTS, SEED, DIMENSIONS, UNKNOWN,
                    SMS_SPAM_URL, LIAR_URL, NAZARIO_INDEX_URL, NAZARIO_MAX_EMAILS, PROMPT_INJECTION_HF, SEMEVAL_DIR)
from preprocess import detect_language

COLS = ["id", "text", "language", *DIMENSIONS, "techniques", "spans", "category", "source"]


def _get(url, timeout=60):
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.content


def row(i, text, mal, dec, man, source, category="", techniques="", spans="", language=None):
    text = re.sub(r"\s+", " ", str(text)).strip()
    return {"id": i, "text": text, "language": language or detect_language(text), "malicious": mal, "deceptive": dec,
            "manipulative": man, "techniques": techniques, "spans": spans, "category": category, "source": source}


def load_sms():
    z = zipfile.ZipFile(io.BytesIO(_get(SMS_SPAM_URL)))
    lines = z.read("SMSSpamCollection").decode("utf-8", errors="ignore").splitlines()
    out = []
    for k, ln in enumerate(lines):
        if "\t" not in ln:
            continue
        lab, txt = ln.split("\t", 1)
        if lab == "spam":   # spam in this corpus is mostly premium-rate / prize scams
            out.append(row(f"sms{k}", txt, 1, UNKNOWN, 1, "sms_spam_collection", "sms_spam"))
        else:
            out.append(row(f"sms{k}", txt, 0, 0, 0, "sms_spam_collection", "sms_ham"))
    return out


def load_liar():
    z = zipfile.ZipFile(io.BytesIO(_get(LIAR_URL)))
    out = []
    for name in ["train.tsv", "valid.tsv", "test.tsv"]:
        df = pd.read_csv(io.BytesIO(z.read(name)), sep="\t", header=None, quoting=3)
        for _, r in df.iterrows():
            lab, stmt = str(r[1]), str(r[2])
            if lab in ("false", "pants-fire"):
                out.append(row(f"liar{r[0]}", stmt, 0, 1, UNKNOWN, "liar", "false_claim"))
            elif lab in ("true", "mostly-true"):
                out.append(row(f"liar{r[0]}", stmt, 0, 0, UNKNOWN, "liar", "true_claim"))
    return out


def _email_text(msg):
    subj = str(make_header(decode_header(msg.get("Subject", "") or "")))
    body = ""
    parts = msg.walk() if msg.is_multipart() else [msg]
    for p in parts:
        ctype = p.get_content_type()
        if ctype in ("text/plain", "text/html"):
            try:
                payload = p.get_payload(decode=True) or b""
                t = payload.decode(p.get_content_charset() or "utf-8", errors="ignore")
            except Exception:
                continue
            if ctype == "text/html":
                t = unescape(re.sub(r"<[^>]+>", " ", t))
            body += " " + t
            if ctype == "text/plain":
                break
    return (subj + ". " + body)[:1000]


def load_nazario():
    index = _get(NAZARIO_INDEX_URL).decode("utf-8", errors="ignore")
    files = [f for f in re.findall(r'href="([^"?/][^"]*)"', index) if f.endswith(".mbox")]  # skip yearly folders and text files
    out = []
    for f in sorted(files, reverse=True):          # newest files first
        if len(out) >= NAZARIO_MAX_EMAILS:
            break
        try:
            path = DATA_RAW / f"nazario_{f}"
            if not path.exists():
                path.write_bytes(_get(NAZARIO_INDEX_URL + f, timeout=120))
            for k, msg in enumerate(mailbox.mbox(str(path))):
                t = _email_text(msg)
                if len(t) > 40:
                    out.append(row(f"naz_{f}_{k}", t, 1, 1, 1, "nazario_phishing", "phishing_email"))
                if len(out) >= NAZARIO_MAX_EMAILS:
                    break
        except Exception as e:
            print("  skipped", f, e)
    return out


def load_prompt_injections():
    """Fetch rows through the Hugging Face dataset-viewer API (no extra library needed)."""
    out = []
    for split in ("train", "test"):
        offset = 0
        while True:
            url = (f"https://datasets-server.huggingface.co/rows?dataset={PROMPT_INJECTION_HF}"
                   f"&config=default&split={split}&offset={offset}&length=100")
            rows = requests.get(url, timeout=60).json().get("rows", [])
            if not rows:
                break
            for r in rows:
                k, rr = r["row_idx"], r["row"]
                if int(rr["label"]) == 1:
                    out.append(row(f"pi_{split}_{k}", rr["text"], 1, UNKNOWN, UNKNOWN, "deepset_prompt_injections",
                                   "prompt_injection", "instruction_override"))
                else:
                    out.append(row(f"pi_{split}_{k}", rr["text"], 0, 0, 0, "deepset_prompt_injections", "benign_prompt"))
            offset += len(rows)
    return out


def load_semeval():
    """Optional: SemEval-2023 Task 3 English subtask-3 files (paragraph-level technique labels)."""
    lab_files = list(SEMEVAL_DIR.glob("**/train-labels-subtask-3.txt")) + list(SEMEVAL_DIR.glob("**/dev-labels-subtask-3.txt"))
    out = []
    for lf in lab_files:
        art_dir = lf.parent / lf.name.replace("-labels-subtask-3.txt", "-articles-subtask-3")
        for ln in lf.read_text(encoding="utf-8").splitlines():
            parts = ln.split("\t")
            if len(parts) < 2:
                continue
            art, para = parts[0], int(parts[1])
            techs = parts[2] if len(parts) > 2 else ""
            f = art_dir / f"article{art}.txt"
            if not f.exists():
                continue
            lines = f.read_text(encoding="utf-8").splitlines()
            if para - 1 < len(lines) and len(lines[para - 1].strip()) > 20:
                out.append(row(f"sem{art}_{para}", lines[para - 1], 0, UNKNOWN, 1 if techs.strip() else 0,
                               "semeval2023_task3", "news_paragraph", techs.replace(",", ";")))
    return out


def load_manual():
    out = []
    for f in sorted(DATA_MANUAL.glob("indian_messages*.csv")):
        if "TEMPLATE" in f.name:
            continue
        df = pd.read_csv(f).fillna("")
        for _, r in df.iterrows():
            out.append(row(r["id"], r["text"], int(r["malicious"]), int(r["deceptive"]), int(r["manipulative"]),
                           r.get("source", "indian_collection") or "indian_collection", r.get("category", ""),
                           r.get("techniques", ""), r.get("spans", ""), r.get("language") or None))
    return out


def main():
    loaders = [("SMS Spam Collection", load_sms), ("LIAR", load_liar), ("Nazario phishing", load_nazario),
               ("Prompt injections", load_prompt_injections), ("SemEval-2023 (optional)", load_semeval),
               ("Indian collection", load_manual)]
    rows = []
    for name, fn in loaders:
        try:
            r = fn()
            print(f"{name}: {len(r)} items")
            rows += r
        except Exception as e:
            print(f"{name}: could not load ({type(e).__name__}) - continuing without it")
    df = pd.DataFrame(rows, columns=COLS).drop_duplicates(subset="text")
    df = df[df["text"].str.len() > 5].reset_index(drop=True)

    # Indian collection is small: split 40 / 20 / 40 so that per-language test results are meaningful
    is_manual = df["source"].str.contains("indian|volunteer|author", case=False)
    man, pub = df[is_manual], df[~is_manual]

    def split(part, test_size, val_share):
        if len(part) < 10:
            return part.iloc[:0], part.iloc[:0], part.iloc[:0]
        part = part.copy()
        part["strat"] = part["source"] + "_" + part[DIMENSIONS].astype(str).agg("".join, axis=1)
        counts = part["strat"].value_counts()
        part.loc[part["strat"].map(counts) < 6, "strat"] = part["source"]
        if part["strat"].value_counts().min() < 3:
            part["strat"] = None
        tr, rest = train_test_split(part, test_size=test_size, random_state=SEED, stratify=part["strat"])
        strat = rest["strat"] if rest["strat"].notna().all() and rest["strat"].value_counts().min() >= 2 else None
        va, te = train_test_split(rest, test_size=1 - val_share, random_state=SEED, stratify=strat)
        return tr, va, te

    tr1, va1, te1 = split(pub, 0.30, 0.5)
    parts = [(tr1, va1, te1)]
    if len(man) >= 10:
        parts.append(split(man, 0.60, 1 / 3))
    train = pd.concat([p[0] for p in parts]); val = pd.concat([p[1] for p in parts]); test = pd.concat([p[2] for p in parts])
    for name, part in (("train", train), ("val", val), ("test", test)):
        part.drop(columns="strat", errors="ignore").to_csv(DATA_PROCESSED / f"{name}.csv", index=False)

    stats = {"total": int(len(df)), "splits": {"train": len(train), "val": len(val), "test": len(test)},
             "by_source": df["source"].value_counts().to_dict(), "by_language": df["language"].value_counts().to_dict(),
             "positives": {d: int((df[d] == 1).sum()) for d in DIMENSIONS},
             "negatives": {d: int((df[d] == 0).sum()) for d in DIMENSIONS},
             "unannotated": {d: int((df[d] == UNKNOWN).sum()) for d in DIMENSIONS}}
    (RESULTS / "dataset_stats.json").write_text(json.dumps(stats, indent=2))
    try:  # Fig. 4.1: label counts per dimension and language
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), dpi=200)
        pos = [stats["positives"][d] for d in DIMENSIONS]; neg = [stats["negatives"][d] for d in DIMENSIONS]
        x = np.arange(len(DIMENSIONS))
        axes[0].bar(x - 0.2, pos, 0.4, label="Yes (1)", color="#C00000"); axes[0].bar(x + 0.2, neg, 0.4, label="No (0)", color="#2F5597")
        axes[0].set_xticks(x); axes[0].set_xticklabels([d.capitalize() for d in DIMENSIONS]); axes[0].legend(frameon=False, fontsize=8)
        axes[0].set_title("Labels per dimension", fontsize=9)
        langs = stats["by_language"]
        axes[1].bar(list(langs), list(langs.values()), color="#548235"); axes[1].set_title("Items per language", fontsize=9)
        for ax in axes:
            ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout(); fig.savefig(RESULTS / "chart_dataset.png"); plt.close(fig)
    except Exception as e:
        print("chart skipped:", e)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
