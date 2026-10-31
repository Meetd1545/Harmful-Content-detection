"""Pre-processing: language identification, normalisation, de-obfuscation and extraction."""
import re
import unicodedata

DEVANAGARI = re.compile(r"[ऀ-ॿ]")
URL_RE = re.compile(r"((?:https?://|www\.)\S+|\b[a-z0-9][a-z0-9\-]{0,62}(?:\.[a-z0-9\-]+)*\.(?:com|in|net|org|info|xyz|top|online|site|live|shop|link|club|app|me|io|biz|ly|gl|co|cc|id|at|sbi|ai|tk|ml|ga|cf|gq|ws|su|ru|cn)\b(?:/\S*)?)", re.I)
PHONE_RE = re.compile(r"(?:\+91[\-\s]?)?\b[6-9]\d{4}[\s\-]?\d{5}\b|\b[6-9]\d[X\d]{6,8}\d{1,2}\b", re.I)
AMOUNT_RE = re.compile(r"(?:rs\.?|inr|₹)\s?\d[\d,]*(?:\.\d+)?|\d[\d,]*\s?(?:rupees|rs)", re.I)
ZERO_WIDTH = re.compile(r"[​-‏⁠﻿]")

# Common Hindi words written in Roman script (used to detect Hinglish)
HINGLISH_WORDS = {
    "hai", "hain", "aap", "aapka", "aapke", "aapki", "apna", "apne", "karein", "karen", "kare", "karo", "nahi",
    "nahin", "kya", "ka", "ki", "ke", "se", "mein", "ko", "par", "turant", "jaldi", "abhi", "aaj", "kal", "raat",
    "baje", "diya", "jayega", "jayegi", "hoga", "hogi", "liye", "bhi", "aur", "ya", "wala", "wali", "paise",
    "paisa", "bijli", "kripya", "dhanyavad", "sampark", "naukri", "ghar", "lakh", "hazaar", "milega", "milegi",
}
LEET = str.maketrans({"@": "a", "$": "s", "0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "!": "i"})


def detect_language(text: str) -> str:
    """Return 'hi' (Devanagari), 'hinglish' (Roman-script Hindi) or 'en'."""
    if not text:
        return "en"
    deva = len(DEVANAGARI.findall(text))
    letters = sum(ch.isalpha() for ch in text) or 1
    if deva / letters > 0.3:
        return "hi"
    words = re.findall(r"[a-z]+", text.lower())
    hits = sum(w in HINGLISH_WORDS for w in words)
    if words and (hits >= 3 or hits / len(words) >= 0.15):
        return "hinglish"
    return "en"


def _deleet_token(tok: str) -> str:
    # only touch tokens that mix letters with digits/symbols, e.g. "p@yment", "acc0unt"
    if re.search(r"[a-zA-Z]", tok) and re.search(r"[@$0-9!]", tok) and not re.fullmatch(r"[\d,.:/\-]+", tok):
        if URL_RE.search(tok) or PHONE_RE.fullmatch(tok) or "/" in tok:
            return tok
        return tok.translate(LEET)
    return tok


def normalise(text: str) -> str:
    """Undo common obfuscation while keeping the message readable."""
    t = unicodedata.normalize("NFKC", str(text))
    t = ZERO_WIDTH.sub("", t)
    # join letters separated by single spaces or dots: "p a y m e n t" -> "payment"
    t = re.sub(r"\b(?:[A-Za-z][ .]){3,}[A-Za-z]\b", lambda m: re.sub(r"[ .]", "", m.group(0)), t)
    t = " ".join(_deleet_token(tok) for tok in t.split())
    t = re.sub(r"(.)\1{3,}", r"\1\1", t)  # "urgentttttt" -> "urgentt"
    return t.strip()


def extract(text: str) -> dict:
    return {
        "urls": [u.rstrip(".,)") for u in URL_RE.findall(text)],
        "phones": PHONE_RE.findall(text),
        "amounts": AMOUNT_RE.findall(text),
    }


def preprocess(text: str) -> dict:
    clean = normalise(text)
    out = {"original": text, "clean": clean, "language": detect_language(clean)}
    out.update(extract(clean))
    return out


if __name__ == "__main__":
    for s in ["Aapka bijli connection aaj raat 9:30 baje kaat diya jayega. Pay at bijli-update-pay.in/kyc",
              "Your acc0unt will be bl0cked. Update K Y C now: bit.ly/x1",
              "आपका खाता आज बंद हो जाएगा", "Meeting moved to 4 pm tomorrow."]:
        print(preprocess(s))
