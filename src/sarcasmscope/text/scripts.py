"""Unicode-script utilities for Indian-language text.

Script detection is exact for native-script text. It cannot see code-mixing that is written
entirely in the Latin script (for example Hinglish), so the code-mix index below is a
*script-based proxy*, not a language-identification result.
"""
from __future__ import annotations

import unicodedata
from collections import Counter

SCRIPT_RANGES = [
    ("Devanagari", 0x0900, 0x097F), ("Devanagari", 0xA8E0, 0xA8FF),
    ("Bengali", 0x0980, 0x09FF),
    ("Gurmukhi", 0x0A00, 0x0A7F),
    ("Gujarati", 0x0A80, 0x0AFF),
    ("Odia", 0x0B00, 0x0B7F),
    ("Tamil", 0x0B80, 0x0BFF),
    ("Telugu", 0x0C00, 0x0C7F),
    ("Kannada", 0x0C80, 0x0CFF),
    ("Malayalam", 0x0D00, 0x0D7F),
    ("Arabic", 0x0600, 0x06FF), ("Arabic", 0x0750, 0x077F),
    ("Arabic", 0xFB50, 0xFDFF), ("Arabic", 0xFE70, 0xFEFF),
]


def script_of(ch: str) -> str | None:
    """Script name of a letter or combining mark; None for digits, punctuation, emoji, etc."""
    if unicodedata.category(ch)[0] not in ("L", "M"):
        return None
    cp = ord(ch)
    for name, lo, hi in SCRIPT_RANGES:
        if lo <= cp <= hi:
            return name
    if 0x41 <= cp <= 0x5A or 0x61 <= cp <= 0x7A or 0xC0 <= cp <= 0x24F:
        return "Latin"
    return "Other"


def tokenize_words(text: str) -> list[str]:
    """Whitespace tokens with punctuation/symbols/emoji trimmed from both edges.

    Python's ``\\w`` splits Indic words at vowel signs, so we avoid regex word tokens.
    """
    out: list[str] = []
    for raw in str(text).split():
        i, j = 0, len(raw)
        while i < j and unicodedata.category(raw[i])[0] in "PSC":
            i += 1
        while j > i and unicodedata.category(raw[j - 1])[0] in "PSC":
            j -= 1
        if j > i:
            out.append(raw[i:j])
    return out


def script_profile(text: str) -> Counter:
    """Count of letters (and marks) per script."""
    c: Counter = Counter()
    for ch in str(text):
        s = script_of(ch)
        if s:
            c[s] += 1
    return c


def dominant_script(text: str) -> str:
    prof = script_profile(text)
    return prof.most_common(1)[0][0] if prof else "None"


def token_script(tok: str) -> str | None:
    prof = script_profile(tok)
    return prof.most_common(1)[0][0] if prof else None


def code_mix_index(text: str) -> dict:
    """Script-based code-mix index (0 = one script, higher = more mixed)."""
    scripts = Counter()
    for tok in tokenize_words(text):
        s = token_script(tok)
        if s:
            scripts[s] += 1
    n = sum(scripts.values())
    cmi = 100.0 * (1.0 - max(scripts.values()) / n) if n else 0.0
    return {"cmi": round(cmi, 2), "n_tokens": n, "token_scripts": dict(scripts)}
