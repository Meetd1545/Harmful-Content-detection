"""Rule-based cue spotter: finds common manipulation cue words (English, Hinglish, Hindi).

Used by the web app to highlight words when the LLM analyser is not running. The LLM, when available,
replaces these cues with its own verified techniques and spans. Cues are hints, not proof.
"""
import re

CUES = {
    "urgency": [r"\burgent(?:ly)?\b", r"\bimmediately\b", r"\btoday\b", r"\btonight\b", r"\bnow\b", r"\bwithin \d+ ?(?:hours?|hrs?|minutes?|mins?)\b",
                r"\bturant\b", r"\babhi\b", r"\baaj raat\b", r"\baaj\b", r"\bjaldi\b", r"तुरंत", r"आज", r"अभी", r"\blast (?:date|day|chance)\b",
                r"\bexpires?\b", r"\bexpiring\b"],
    "fear_threat": [r"\bblock(?:ed)?\b", r"\bsuspend(?:ed)?\b", r"\bdeactivat\w*\b", r"\bdisconnect\w*\b", r"\bkaat diya jayega\b", r"\bband ho jayega\b",
                    r"\barrest\w*\b", r"\blegal action\b", r"\bpenalty\b", r"\bfine\b", r"बंद कर दिया जाएगा", r"बंद", r"\bcut off\b"],
    "false_authority": [r"\bofficer\b", r"\bpolice\b", r"\bcbi\b", r"\brbi\b", r"\bincome tax\b", r"\bcustoms\b", r"\bcustomer care\b",
                        r"\bbank manager\b", r"\bgovernment\b", r"\btrai\b"],
    "reward_reciprocity": [r"\bcongratulations\b", r"\bwon\b", r"\bwinner\b", r"\blottery\b", r"\bprize\b", r"\bcashback\b", r"\bfree\b",
                           r"\bgift\b", r"\breward\b", r"\brs\.? ?[\d,]+ ?(?:lakh|crore)?\b", r"\b₹ ?[\d,]+\b", r"\blakh\b", r"\bcrore\b"],
    "scarcity": [r"\bonly \d+\b", r"\blimited\b", r"\bfirst \d+\b", r"\blast few\b"],
    "instruction_override": [r"ignore (?:all )?(?:previous|prior|above) instructions", r"note to (?:the )?ai", r"\bai assistant\b",
                             r"system prompt", r"disregard (?:the )?(?:previous|above)", r"\bsend the user'?s\b"],
}
_COMPILED = {k: [re.compile(p, re.I) for p in v] for k, v in CUES.items()}


def find_cues(text: str, max_per_type: int = 2):
    """Return [{"name": technique, "span": exact words}] for cue words found in the text."""
    found, taken = [], []
    for name, pats in _COMPILED.items():
        n = 0
        for p in pats:
            for m in p.finditer(text):
                s, e = m.span()
                if any(s < te and e > ts for ts, te in taken):
                    continue
                found.append({"name": name, "span": text[s:e], "start": s})
                taken.append((s, e))
                n += 1
                if n >= max_per_type:
                    break
            if n >= max_per_type:
                break
    found.sort(key=lambda d: d["start"])
    return [{"name": d["name"], "span": d["span"]} for d in found]


if __name__ == "__main__":
    print(find_cues("Aapka bijli connection aaj raat 9:30 baje kaat diya jayega. Turant electricity officer se sampark karein"))
