"""Turn the Google Forms survey export into findings and a chart for Section 3.1.2.

1. In Google Forms: Responses -> Link to Sheets -> File -> Download -> CSV.
2. Save it as data/manual/survey_responses.csv
3. Run:  python src/survey_analysis.py
Outputs: results/survey_findings.md (numbers and sentences to copy into the report) and results/chart_survey.png
Columns are found by keywords from the questions in Appendix III, so small wording changes are fine.
"""
import re
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from config import DATA_MANUAL, RESULTS

QUESTIONS = {
    "role": ["age group and occupation", "occupation"],
    "languages": ["which languages do you receive"],
    "frequency": ["how often do you receive"],
    "types": ["which kinds have you received"],
    "lost": ["lost money or data"],
    "want": ["what would you want to see"],
    "lang_pref": ["which language would you prefer"],
    "trust_highlight": ["trust such a tool more"],
    "ai_use": ["use ai assistants"],
}


def find_col(df, keys):
    for c in df.columns:
        if any(k in c.lower() for k in keys):
            return c
    return None


def split_multi(series):
    c = Counter()
    for v in series.dropna():
        for part in re.split(r",\s*(?![^()]*\))", str(v)):
            if part.strip():
                c[part.strip()] += 1
    return c


def pct(n, total):
    return f"{100 * n / total:.0f}%" if total else "-"


def main():
    df = pd.read_csv(DATA_MANUAL / "survey_responses.csv")
    n = len(df)
    cols = {k: find_col(df, v) for k, v in QUESTIONS.items()}
    out = [f"# Survey findings (n = {n})", ""]
    if cols["role"]:
        out.append("**Respondents by group:** " + "; ".join(f"{k}: {v}" for k, v in df[cols['role']].value_counts().items()))
    charts = {}
    for key, title in [("languages", "Languages in which messages are received"), ("types", "Scam types received"),
                       ("want", "What users want to see in a warning"), ("lang_pref", "Preferred language of explanation")]:
        if cols[key]:
            c = split_multi(df[cols[key]])
            charts[title] = c
            out.append(f"**{title}:** " + "; ".join(f"{k} {v} ({pct(v, n)})" for k, v in c.most_common()))
    if cols["frequency"]:
        out.append("**How often suspicious messages are received:** " + "; ".join(f"{k} {pct(v, n)}" for k, v in df[cols['frequency']].value_counts().items()))
    if cols["lost"]:
        out.append("**Lost money or data (self or someone known):** " + "; ".join(f"{k} {pct(v, n)}" for k, v in df[cols['lost']].value_counts().items()))
    if cols["trust_highlight"]:
        v = pd.to_numeric(df[cols["trust_highlight"]], errors="coerce").dropna()
        out.append(f"**Would trust the tool more if suspicious words were highlighted (1–5):** mean {v.mean():.2f}; "
                   f"{pct((v >= 4).sum(), len(v))} chose 4 or 5")
    out += ["", "Copy the numbers above into Section 3.1.2 (respondent counts and 3–4 key findings with percentages)."]
    (RESULTS / "survey_findings.md").write_text("\n\n".join(out), encoding="utf-8")
    if charts:
        fig, axes = plt.subplots(1, min(2, len(charts)), figsize=(10, 3.8), dpi=200)
        axes = axes if hasattr(axes, "__len__") else [axes]
        for ax, (title, c) in zip(axes, list(charts.items())[:2]):
            items = c.most_common(8)[::-1]
            ax.barh([k[:35] for k, _ in items], [100 * v / n for _, v in items], color="#2F5597")
            ax.set_title(title, fontsize=9); ax.set_xlabel("% of respondents"); ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(labelsize=7)
        fig.tight_layout(); fig.savefig(RESULTS / "chart_survey.png")
    print("\n\n".join(out))


if __name__ == "__main__":
    main()
