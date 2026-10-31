"""Run all experiments (RQ1, RQ2, technique and calibration results) and save every number for the report.

Run after prepare_data.py, baseline.py and transformer_model.py:
    python src/evaluate.py                 # full run (needs GPU for the LLM)
    python src/evaluate.py --skip-llm      # only M1 and M2
Outputs: results/results.json, results/tables.md, results/*.png, results/llm_outputs.jsonl
"""
import argparse
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

from config import (DATA_PROCESSED, MODELS, RESULTS, DIMENSIONS, UNKNOWN, SEED, TECHNIQUES, LLM_EVAL_SAMPLE,
                    ROBUSTNESS_SAMPLE, LOW_THRESHOLD, DECISION_THRESHOLD, LLM_BACKEND, LLM_HF_MODEL, LLM_OLLAMA_MODEL, ENCODER_NAME)
from preprocess import preprocess
from evidence import gather_evidence, PatternIndex
from calibration import fit_temperatures, apply_temperatures, ece, sigmoid
from cascade import fuse
import robustness as rb
import baseline  # noqa: F401  (needed to load the pickled baseline model)

TH = DECISION_THRESHOLD


# ---------------- metrics ----------------
def dim_metrics(probs, df):
    out = {}
    for j, d in enumerate(DIMENSIONS):
        m = (df[d] != UNKNOWN).values
        y, p = df.loc[m, d].astype(int).values, probs[m, j]
        if len(y) == 0:
            continue
        pr, rc, f1, _ = precision_recall_fscore_support(y, (p >= TH).astype(int), average="binary", zero_division=0)
        auc = roc_auc_score(y, p) if len(set(y)) == 2 else float("nan")
        out[d] = {"n": int(m.sum()), "precision": round(pr, 4), "recall": round(rc, 4), "f1": round(f1, 4), "auroc": round(float(auc), 4)}
    out["macro_f1"] = round(float(np.mean([out[d]["f1"] for d in DIMENSIONS if d in out])), 4)
    benign = (df[DIMENSIONS] == 0).all(axis=1).values
    out["fpr_genuine"] = round(float((probs[benign].max(axis=1) >= TH).mean()), 4) if benign.any() else float("nan")
    out["n_genuine"] = int(benign.sum())
    return out


def detection_rate(probs):
    return round(float((np.asarray(probs).max(axis=1) >= TH).mean()), 4) if len(probs) else float("nan")


def technique_f1(pred_lists, gold_strings):
    tp = fp = fn = 0
    for pred, gold in zip(pred_lists, gold_strings):
        g = {t.strip() for t in str(gold).split(";") if t.strip() in TECHNIQUES}
        p = {t["name"] for t in pred}
        tp += len(p & g); fp += len(p - g); fn += len(g - p)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4), "tp": tp, "fp": fp, "fn": fn}


# ---------------- mock LLM (ONLY for testing the code without a GPU) ----------------
class MockLLM:
    """Keyword heuristic that imitates the LLM interface. Results produced with it are NOT valid for the report."""
    CUES = {"urgency": ["today", "now", "immediately", "turant", "aaj", "within", "24", "तुरंत", "आज"],
            "fear_threat": ["blocked", "block", "band", "kaat", "arrest", "suspend", "बंद"],
            "reward_reciprocity": ["won", "prize", "earn", "refund", "lottery", "jeeti", "रिफंड"],
            "instruction_override": ["ignore all previous", "ignore previous", "developer mode", "disregard"]}

    def chat(self, messages, max_new_tokens=200, temperature=0.0, **kw):
        text = messages[-1]["content"]
        return text.split("Message:")[-1].strip() if "Message:" in text else "Your XYZ Bank account is blocked, verify now at example-link.in"

    def analyse(self, text):
        low = text.lower()
        techs = [{"name": n, "span": next(c for c in cues if c in low)} for n, cues in self.CUES.items() if any(c in low for c in cues)]
        # crude span: recover original casing
        for t in techs:
            i = low.find(t["span"]); t["span"] = text[i:i + len(t["span"])]
        s = min(1.0, 0.25 * len(techs) + (0.3 if "http" in low or ".in" in low or ".com" in low else 0))
        return {"valid": True, "malicious": s, "deceptive": s * 0.9, "manipulative": min(1.0, 0.3 * len(techs)),
                "techniques": techs, "explanation": "mock", "advice": "mock", "rejected_spans": 0, "latency_s": 0.0}


class BaselineAsScreener:
    """Used only in --mock-llm test mode when no trained transformer exists."""
    def __init__(self, base):
        self.base = base

    def logits(self, texts):
        p = np.clip(self.base.predict_proba(texts), 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p))


# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-llm", action="store_true")
    ap.add_argument("--mock-llm", action="store_true", help="test the pipeline without a GPU (results not valid)")
    ap.add_argument("--llm-backend", default=LLM_BACKEND, choices=["hf", "ollama"])
    ap.add_argument("--sample", type=int, default=LLM_EVAL_SAMPLE)
    ap.add_argument("--rob", type=int, default=ROBUSTNESS_SAMPLE, help="harmful items per robustness variant")
    ap.add_argument("--gen", type=int, default=40, help="new scam messages generated by the LLM")
    args = ap.parse_args()
    rng = np.random.RandomState(SEED)
    val = pd.read_csv(DATA_PROCESSED / "val.csv").fillna({"techniques": "", "spans": ""})
    test = pd.read_csv(DATA_PROCESSED / "test.csv").fillna({"techniques": "", "spans": ""})
    R = {"mode": "MOCK - NOT VALID FOR REPORT" if args.mock_llm else "real", "test_items": len(test),
         "config": {"encoder": ENCODER_NAME, "llm": LLM_HF_MODEL if args.llm_backend == "hf" else LLM_OLLAMA_MODEL,
                    "llm_backend": args.llm_backend, "low_threshold": LOW_THRESHOLD, "decision_threshold": TH}}

    # ---- M1 baseline ----
    base = joblib.load(MODELS / "baseline.joblib")
    t0 = time.time(); p1_full = base.predict_proba(test["text"]); lat1 = (time.time() - t0) / len(test)

    # ---- M2 transformer (+ temperature scaling on validation) ----
    has_m2 = (MODELS / "transformer" / "model.pt").exists()
    if has_m2:
        from cascade import TransformerScreener
        screener = TransformerScreener()
    else:  # easy / CPU mode: the TF-IDF baseline acts as the screener of the cascade
        screener = BaselineAsScreener(base)
    sc = "M2" if has_m2 else "M1"
    R["screener"] = "M2 (fine-tuned MuRIL)" if has_m2 else "M1 (TF-IDF baseline) - transformer not trained in this run"
    vlog = screener.logits(val["text"].tolist())
    temps = fit_temperatures(vlog, val[DIMENSIONS].values)
    (MODELS / "temperatures.json").write_text(json.dumps({"temperatures": temps.tolist()}))
    t0 = time.time(); tlog = screener.logits(test["text"].tolist()); lat2 = (time.time() - t0) / len(test)
    p2_raw, p2_full = sigmoid(tlog), apply_temperatures(tlog, temps)
    R["calibration"] = {"temperatures": dict(zip(DIMENSIONS, np.round(temps, 3).tolist())),
                        "ece_before": {d: round(ece(p2_raw[:, j], test[d].values), 4) for j, d in enumerate(DIMENSIONS)},
                        "ece_after": {d: round(ece(p2_full[:, j], test[d].values), 4) for j, d in enumerate(DIMENSIONS)}}
    R["full_test"] = {"M1": dim_metrics(p1_full, test)}
    if has_m2:
        R["full_test"]["M2"] = dim_metrics(p2_full, test)

    # ---- evaluation sample for LLM-based approaches (all manual items + random public items) ----
    manual = test["source"].str.contains("indian|volunteer|author", case=False).values
    pub_idx = np.where(~manual)[0]
    k = max(0, min(len(pub_idx), args.sample - manual.sum()))
    S_idx = np.sort(np.concatenate([np.where(manual)[0], rng.choice(pub_idx, k, replace=False)])).astype(int)
    S = test.iloc[S_idx].reset_index(drop=True)
    p1, p2 = p1_full[S_idx], p2_full[S_idx]
    R["sample_size"] = int(len(S))
    approaches = {"M1": p1, "M2": p2} if has_m2 else {"M1": p1}
    lat = {"M1": lat1, "M2": lat2} if has_m2 else {"M1": lat1}

    if not args.skip_llm:
        if args.mock_llm:
            llm = an = MockLLM()
        else:
            from llm_analyser import LLMAnalyser
            an = LLMAnalyser(args.llm_backend); llm = an.llm
        index = PatternIndex()
        llm_res, evs, routed = [], [], []
        cached = {}  # reuse LLM replies from an earlier, interrupted run with the same sample
        if not args.mock_llm and (RESULTS / "llm_outputs.jsonl").exists():
            for ln in (RESULTS / "llm_outputs.jsonl").read_text(encoding="utf-8").splitlines():
                try:
                    o = json.loads(ln); cached[str(o.pop("id"))] = o
                except Exception:
                    pass
            if cached:
                print(f"  reusing {len(cached)} LLM replies saved by the previous run")
        with open(RESULTS / "llm_outputs.jsonl", "w", encoding="utf-8") as fo:
            for i, text in enumerate(S["text"]):
                r = cached.get(str(S.loc[i, "id"])) or an.analyse(text)
                llm_res.append(r)
                fo.write(json.dumps({"id": S.loc[i, "id"], **{k2: v for k2, v in r.items() if k2 != "raw"}}, ensure_ascii=False) + "\n")
                evs.append(gather_evidence(preprocess(text), index))
                if (i + 1) % 10 == 0:
                    print(f"  LLM analysed {i + 1}/{len(S)}")
        valid = np.array([r["valid"] for r in llm_res])
        p3 = np.array([[r.get(d, 0.0) if r["valid"] else p2[i, j] for j, d in enumerate(DIMENSIONS)] for i, r in enumerate(llm_res)])
        p4, m4_tech = [], []
        for i in range(len(S)):
            go = p2[i].max() >= LOW_THRESHOLD or evs[i]["suspicious_url"]
            routed.append(go)
            f = fuse(p2[i], llm_res[i] if go else None, evs[i])
            p4.append([f[d] for d in DIMENSIONS])
            m4_tech.append(llm_res[i]["techniques"] if go and llm_res[i]["valid"] else [])
        p4 = np.array(p4)
        llm_lat = np.array([r["latency_s"] for r in llm_res])
        approaches.update({"M3": p3, "M4": p4})
        lat["M3"] = float(llm_lat.mean()); lat["M4"] = float(lat2 + llm_lat[np.array(routed)].sum() / len(S))
        R["llm"] = {"valid_json_rate": round(float(valid.mean()), 4), "rejected_spans": int(sum(r.get("rejected_spans", 0) for r in llm_res)),
                    "routed_to_llm_rate": round(float(np.mean(routed)), 4), "suspicious_url_items": int(sum(e["suspicious_url"] for e in evs))}
        has_tech = (S["techniques"].astype(str).str.len() > 0).values | ((S[DIMENSIONS] == 0).all(axis=1).values & manual[S_idx])
        R["technique_f1"] = {"items": int(has_tech.sum()),
                             "M3": technique_f1([llm_res[i]["techniques"] for i in np.where(has_tech)[0]], S.loc[has_tech, "techniques"]),
                             "M4": technique_f1([m4_tech[i] for i in np.where(has_tech)[0]], S.loc[has_tech, "techniques"])}

    R["sample"] = {a: dim_metrics(p, S) for a, p in approaches.items()}
    R["latency_s_per_item"] = {a: round(v, 4) for a, v in lat.items()}
    best = max(R["sample"], key=lambda a: R["sample"][a]["macro_f1"])
    R["best_approach"] = best
    R["by_language"] = {}
    for lang in ["en", "hi", "hinglish"]:
        m = (S["language"] == lang).values
        if m.sum() >= 3:
            R["by_language"][lang] = {**{k2: v for k2, v in dim_metrics(approaches[best][m], S[m]).items()
                                         if k2 in ("macro_f1", "fpr_genuine", "n_genuine")}, "items": int(m.sum())}

    # ---- RQ2 robustness ----
    if not args.skip_llm:
        harm_idx = np.where((test[DIMENSIONS] == 1).any(axis=1).values)[0]
        H = test.iloc[rng.choice(harm_idx, min(args.rob, len(harm_idx)), replace=False)].reset_index(drop=True)
        sets = {"original": H["text"].tolist(), "obfuscated": [rb.obfuscate(t) for t in H["text"]]}
        var_file = RESULTS / "adversarial_variants_PRIVATE.json"
        saved = json.loads(var_file.read_text(encoding="utf-8")) if (var_file.exists() and not args.mock_llm) else {}
        if saved.get("original") == sets["original"] and saved.get("llm_paraphrased") and saved.get("llm_generated"):
            print("  reusing paraphrased and generated messages from the previous run")
            sets["llm_paraphrased"], sets["llm_generated"] = saved["llm_paraphrased"], saved["llm_generated"]
        else:
            print("  creating paraphrases ..."); sets["llm_paraphrased"] = [rb.paraphrase(llm, t) for t in H["text"]]
            print("  generating new messages ..."); sets["llm_generated"] = rb.generate(llm, args.gen)
            if not args.mock_llm:
                var_file.write_text(json.dumps({k2: sets[k2] for k2 in ("original", "llm_paraphrased", "llm_generated")},
                                               ensure_ascii=False), encoding="utf-8")
        sets["embedded_instructions"] = [rb.inject(t, i) for i, t in enumerate(H["text"])]
        rob, cache = {}, {}
        rob_file = RESULTS / "llm_robustness_cache.jsonl"
        rcache = {}
        if not args.mock_llm and rob_file.exists():
            for ln in rob_file.read_text(encoding="utf-8").splitlines():
                try:
                    o = json.loads(ln); rcache[o.pop("text")] = o
                except Exception:
                    pass

        def rob_analyse(t):
            if t in rcache:
                return rcache[t]
            r = {k2: v for k2, v in an.analyse(t).items() if k2 != "raw"}
            if not args.mock_llm:
                rcache[t] = r
                with open(rob_file, "a", encoding="utf-8") as fr:
                    fr.write(json.dumps({"text": t, **r}, ensure_ascii=False) + "\n")
            return r

        for name, texts in sets.items():
            if not texts:
                continue
            q2 = apply_temperatures(screener.logits(texts), temps)
            res3 = [rob_analyse(t) for t in texts]
            q3 = np.array([[r.get(d, 0.0) if r["valid"] else q2[i, j] for j, d in enumerate(DIMENSIONS)] for i, r in enumerate(res3)])
            q4 = []
            for i, t in enumerate(texts):
                ev = gather_evidence(preprocess(t), index)
                go = q2[i].max() >= LOW_THRESHOLD or ev["suspicious_url"]
                f = fuse(q2[i], res3[i] if go else None, ev); q4.append([f[d] for d in DIMENSIONS])
            cache[name] = q3
            rob[name] = {"items": len(texts), sc: detection_rate(q2), "M3": detection_rate(q3), "M4": detection_rate(np.array(q4))}
            print(f"  robustness set '{name}' done")
        if "embedded_instructions" in cache:
            o = cache["original"].max(axis=1) >= TH
            inj = cache["embedded_instructions"].max(axis=1) >= TH
            rob["embedded_instructions"]["analyser_followed_instruction_rate"] = round(float((o & ~inj).sum() / max(o.sum(), 1)), 4)
        R["robustness_detection_rate"] = rob
        pd.DataFrame({k2: pd.Series(v) for k2, v in sets.items()}).to_csv(RESULTS / "adversarial_variants_PRIVATE.csv", index=False)

    (RESULTS / "results.json").write_text(json.dumps(R, indent=2, ensure_ascii=False), encoding="utf-8")
    write_tables(R)
    make_charts(R)
    txt = write_report_text(R)
    print(txt)


def write_tables(R):
    L = [f"# Results ({R['mode']})", "", f"Evaluation sample: {R['sample_size']} items", "",
         "| Approach | Malicious F1 | Deceptive F1 | Manipulative F1 | Macro-F1 | FPR genuine | Latency (s) |", "|---|---|---|---|---|---|---|"]
    for a, m in R["sample"].items():
        L.append(f"| {a} | " + " | ".join(str(m.get(d, {}).get("f1", "-")) for d in DIMENSIONS) +
                 f" | {m['macro_f1']} | {m['fpr_genuine']} | {R['latency_s_per_item'][a]} |")
    L += ["", f"Best approach: {R['best_approach']}", "", "| Language | Items | Macro-F1 | FPR genuine |", "|---|---|---|---|"]
    for lang, m in R["by_language"].items():
        L.append(f"| {lang} | {m['items']} | {m['macro_f1']} | {m['fpr_genuine']} |")
    if "robustness_detection_rate" in R:
        L += ["", "| Test set | Items | Screener | M3 | M4 | Analyser followed instruction |", "|---|---|---|---|---|---|"]
        for n, m in R["robustness_detection_rate"].items():
            scr = m.get("M2", m.get("M1"))
            L.append(f"| {n} | {m['items']} | {scr} | {m['M3']} | {m['M4']} | {m.get('analyser_followed_instruction_rate', '-')} |")
    (RESULTS / "tables.md").write_text("\n".join(L))


def make_charts(R):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = list(R["sample"])
    fig, ax = plt.subplots(figsize=(7, 3.6), dpi=200)
    w = 0.2
    for j, d in enumerate(DIMENSIONS):
        ax.bar(np.arange(len(names)) + (j - 1) * w, [R["sample"][a].get(d, {}).get("f1", 0) for a in names], w, label=d.capitalize())
    ax.set_xticks(np.arange(len(names))); ax.set_xticklabels(names); ax.set_ylim(0, 1); ax.set_ylabel("F1-score")
    ax.legend(frameon=False, ncol=3, fontsize=8); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(RESULTS / "chart_f1_by_approach.png"); plt.close(fig)
    if "robustness_detection_rate" in R:
        rob = R["robustness_detection_rate"]; sets = list(rob)
        fig, ax = plt.subplots(figsize=(7, 3.6), dpi=200)
        keys = [k for k in ["M1", "M2", "M3", "M4"] if k in rob[sets[0]]]
        for j, a in enumerate(keys):
            ax.bar(np.arange(len(sets)) + (j - 1) * w, [rob[s][a] for s in sets], w, label=a)
        ax.set_xticks(np.arange(len(sets))); ax.set_xticklabels([s.replace("_", "\n") for s in sets], fontsize=8)
        ax.set_ylim(0, 1); ax.set_ylabel("Detection rate"); ax.legend(frameon=False, ncol=3, fontsize=8)
        ax.spines[["top", "right"]].set_visible(False); fig.tight_layout(); fig.savefig(RESULTS / "chart_robustness.png"); plt.close(fig)



# ---------------- plain-text output for the report ----------------
def pc(x):
    return "-" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.1f}"


def write_report_text(R):
    names = {"M1": "M1 TF-IDF + LR", "M2": "M2 Transformer (MuRIL)", "M3": "M3 LLM (few-shot)", "M4": "M4 Hybrid cascade"}
    langs = {"en": "English", "hi": "Hindi (Devanagari)", "hinglish": "Hinglish (Roman)"}
    L = ["RESULTS FOR THE REPORT  (copy each table into the matching table in Chapter 4)",
         f"Run mode: {R['mode']}", f"Cascade screener: {R.get('screener', '')}",
         f"LLM: {R['config']['llm']} via {R['config']['llm_backend']}" if "llm" in R else "LLM: not run", ""]
    st = RESULTS / "dataset_stats.json"
    if st.exists():
        d = json.loads(st.read_text())
        L += ["== Table 4.1: Items per source after removing duplicates ==",
              *[f"  {k}: {v}" for k, v in d["by_source"].items()],
              f"  TOTAL: {d['total']}   (train {d['splits']['train']}, validation {d['splits']['val']}, test {d['splits']['test']})",
              "  By language: " + ", ".join(f"{langs.get(k, k)} {v}" for k, v in d["by_language"].items()), ""]
    L += [f"== Table 4.6: Detection performance on the common evaluation sample ({R['sample_size']} test items). F1 and FPR in % ==",
          "  Approach | Malicious F1 | Deceptive F1 | Manipulative F1 | FPR genuine | Latency (s)"]
    for a, m in R["sample"].items():
        L.append(f"  {names[a]} | " + " | ".join(pc(m.get(dd, {}).get("f1")) for dd in DIMENSIONS) +
                 f" | {pc(m['fpr_genuine'])} | {R['latency_s_per_item'][a]:.3f}")
    L.append(f"  Macro-F1: " + ", ".join(f"{a} {pc(m['macro_f1'])}" for a, m in R["sample"].items()) + f"   Best: {R['best_approach']}")
    L += ["", f"== Table 4.7: Results by language for the best approach ({R['best_approach']}) ==", "  Language | Items | Macro-F1 (%) | FPR genuine (%)"]
    for k, m in R["by_language"].items():
        L.append(f"  {langs.get(k, k)} | {m['items']} | {pc(m['macro_f1'])} | {pc(m['fpr_genuine'])}")
    if "robustness_detection_rate" in R:
        L += ["", "== Table 4.8: Detection rate (%) under evasion and embedded instructions ==",
              "  Test set | Items | Screener | M3 | M4 | LLM followed injected instruction (%)"]
        for n, m in R["robustness_detection_rate"].items():
            scr = m.get("M2", m.get("M1"))
            f = m.get("analyser_followed_instruction_rate")
            L.append(f"  {n} | {m['items']} | {pc(scr)} | {pc(m['M3'])} | {pc(m['M4'])} | {pc(f) if f is not None else '-'}")
    c = R.get("calibration", {})
    if c:
        L += ["", "== Calibration of the screener (Section 4.1.4) ==",
              "  Temperatures: " + ", ".join(f"{k} {v}" for k, v in c["temperatures"].items()),
              "  ECE before: " + ", ".join(f"{k} {v}" for k, v in c["ece_before"].items()),
              "  ECE after:  " + ", ".join(f"{k} {v}" for k, v in c["ece_after"].items())]
    if "llm" in R:
        L += ["", "== LLM analyser checks ==",
              f"  Valid JSON: {pc(R['llm']['valid_json_rate'])}%   Rejected (invented) spans: {R['llm']['rejected_spans']}",
              f"  Items sent to the LLM by the cascade: {pc(R['llm']['routed_to_llm_rate'])}%",
              f"  Technique-level F1 ({R['technique_f1']['items']} items): M3 {pc(R['technique_f1']['M3']['f1'])}, M4 {pc(R['technique_f1']['M4']['f1'])}"]
    L += ["", "Charts to insert: results/chart_dataset.png (Fig. 4.1), results/chart_f1_by_approach.png, results/chart_robustness.png"]
    txt = "\n".join(L)
    (RESULTS / "REPORT_OUTPUT.txt").write_text(txt, encoding="utf-8")
    return txt


if __name__ == "__main__":
    main()
