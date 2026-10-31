"""Evidence checks: look-alike URLs, URL shorteners and similarity to known scam patterns."""
import re
from difflib import SequenceMatcher
from urllib.parse import urlparse

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config import OFFICIAL_DOMAINS, BRAND_KEYWORDS, URL_SHORTENERS, ROOT

PATTERNS_FILE = ROOT / "data" / "manual" / "known_scam_patterns.csv"


def domain_of(url: str) -> str:
    u = url if re.match(r"https?://", url, re.I) else "http://" + url
    host = (urlparse(u).hostname or "").lower()
    return host[4:] if host.startswith("www.") else host


def check_url(url: str) -> dict:
    d = domain_of(url)
    res = {"url": url, "domain": d, "official": False, "shortener": False, "lookalike": False, "reason": ""}
    if not d:
        return res
    if any(d == o or d.endswith("." + o) for o in OFFICIAL_DOMAINS):
        res["official"] = True
        return res
    if any(d == s or d.endswith("." + s) for s in URL_SHORTENERS):
        res["shortener"] = True
        res["reason"] = "shortened link hides the real destination"
        return res
    base = d.split(".")[0] if d.count(".") <= 1 else ".".join(d.split(".")[:-1])
    for o in OFFICIAL_DOMAINS:
        ob = o.split(".")[0]
        if len(ob) >= 4 and SequenceMatcher(None, base.replace("-", ""), ob).ratio() >= 0.8:
            res["lookalike"] = True
            res["reason"] = f"domain resembles official site {o}"
            return res
    for kw in BRAND_KEYWORDS:
        if re.search(rf"(^|[\-.]){kw}([\-.]|$)|{kw}[\-]", d):
            res["lookalike"] = True
            res["reason"] = f"uses the name '{kw}' but is not an official domain"
            return res
    return res


class PatternIndex:
    """Character n-gram TF-IDF search over short descriptions of known scam patterns."""

    def __init__(self, path=PATTERNS_FILE):
        self.df = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=["pattern_id", "scam_type", "pattern"])
        self.vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), lowercase=True)
        self.mat = self.vec.fit_transform(self.df["pattern"].astype(str)) if len(self.df) else None

    def search(self, text: str, top_k: int = 1, min_sim: float = 0.40):
        if self.mat is None:
            return []
        sims = cosine_similarity(self.vec.transform([text]), self.mat)[0]
        idx = sims.argsort()[::-1][:top_k]
        return [{"scam_type": self.df.iloc[i]["scam_type"], "pattern": self.df.iloc[i]["pattern"], "similarity": round(float(sims[i]), 3)}
                for i in idx if sims[i] >= min_sim]


def gather_evidence(pre: dict, index: PatternIndex = None) -> dict:
    url_checks = [check_url(u) for u in pre.get("urls", [])]
    matches = index.search(pre["clean"]) if index is not None else []
    return {
        "urls": url_checks,
        "suspicious_url": any(c["lookalike"] or c["shortener"] for c in url_checks),
        "pattern_matches": matches,
    }


if __name__ == "__main__":
    for u in ["bijli-update-pay.in/kyc", "https://onlinesbi.sbi/login", "sbi-kyc-verify.in", "bit.ly/3xYz", "hdfcbnk.com", "example.org"]:
        print(check_url(u))
