"""M4 - Hybrid cascade: transformer screener -> (if flagged or uncertain) LLM analyser -> evidence -> fusion."""
import json
import time

import numpy as np

from config import (DIMENSIONS, MODELS, LOW_THRESHOLD, W_SCREENER, W_LLM, EVIDENCE_BOOST, RISK_HIGH, RISK_MEDIUM,
                    AGREEMENT_HIGH, DECISION_THRESHOLD)
from preprocess import preprocess
from evidence import gather_evidence, PatternIndex
from calibration import apply_temperatures


def fuse(screener_p, llm_res, evidence):
    """screener_p: calibrated probabilities (3,), llm_res: dict from LLMAnalyser or None, evidence: dict."""
    p = np.array(screener_p, dtype=float)
    used_llm = llm_res is not None and llm_res.get("valid", False)
    if used_llm:
        lp = np.array([llm_res[d] for d in DIMENSIONS])
        p = W_SCREENER * p + W_LLM * lp
        agree = np.all(np.abs(np.array(screener_p) - lp) < AGREEMENT_HIGH)
    else:
        agree = None
    if evidence.get("suspicious_url"):
        p[0] = min(1.0, p[0] + EVIDENCE_BOOST)   # malicious
        p[1] = min(1.0, p[1] + EVIDENCE_BOOST)   # deceptive
    top = float(p.max())
    risk = "HIGH" if top >= RISK_HIGH else "MEDIUM" if top >= RISK_MEDIUM else "LOW"
    if agree is None:
        conf = "High" if (top < 0.1 or top > 0.9) else "Medium"
    else:
        conf = "High" if agree else "Low"
    return {**{d: round(float(v), 3) for d, v in zip(DIMENSIONS, p)}, "risk": risk, "confidence": conf,
            "harmful": bool(top >= DECISION_THRESHOLD), "used_llm": used_llm}


class Cascade:
    def __init__(self, screener, llm=None, temps=None, index=None):
        """screener: object with .logits(texts) -> (n,3) array; llm: LLMAnalyser or None."""
        self.screener, self.llm = screener, llm
        self.temps = temps if temps is not None else np.ones(len(DIMENSIONS))
        self.index = index if index is not None else PatternIndex()

    def analyse(self, text: str) -> dict:
        t0 = time.time()
        pre = preprocess(text)
        sp = apply_temperatures(self.screener.logits([pre["clean"]]), self.temps)[0]
        ev = gather_evidence(pre, self.index)
        llm_res = None
        if self.llm is not None and (sp.max() >= LOW_THRESHOLD or ev["suspicious_url"]):
            llm_res = self.llm.analyse(text)
        out = fuse(sp, llm_res, ev)
        out.update({"language": pre["language"], "screener": [round(float(x), 3) for x in sp], "evidence": ev,
                    "techniques": (llm_res or {}).get("techniques", []),
                    "explanation": (llm_res or {}).get("explanation", "") or default_explanation(out, ev),
                    "advice": (llm_res or {}).get("advice", "") or default_advice(out),
                    "latency_s": round(time.time() - t0, 2)})
        return out


def default_explanation(out, ev):
    if not out["harmful"]:
        return "No strong signs of scam, deception or manipulation were found."
    parts = [d for d in DIMENSIONS if out[d] >= DECISION_THRESHOLD]
    names = parts if len(parts) < 2 else [", ".join(parts[:-1]) + " and " + parts[-1]]
    msg = "The message looks " + names[0] + "."
    reasons = [u["reason"] for u in ev["urls"] if u["reason"]]
    if reasons:
        msg += " Link check: " + "; ".join(reasons) + "."
    return msg


def default_advice(out):
    if out["harmful"]:
        return "Do not click links, call numbers or share OTPs or PINs. Verify through the official app or website, and report fraud on 1930 or cybercrime.gov.in."
    return "No action needed, but stay careful with links and payment requests."


class TransformerScreener:
    def __init__(self):
        from transformer_model import load_trained, predict_logits
        self.model, self.tok = load_trained()
        self._pl = predict_logits

    def logits(self, texts):
        return self._pl(self.model, self.tok, texts)


def load_temperatures():
    f = MODELS / "temperatures.json"
    return np.array(json.loads(f.read_text())["temperatures"]) if f.exists() else None
