"""Heuristic script/language guess for display and sanity checks.

This is NOT a research-grade language identifier. For experiments use the gold language
column of the dataset. For Roman-script text the guess relies on tiny function-word lists
and is always reported with low or medium confidence.
"""
from __future__ import annotations

from .scripts import code_mix_index, dominant_script, script_profile, tokenize_words

SCRIPT_TO_LANG = {
    "Bengali": "bn", "Gurmukhi": "pa", "Gujarati": "gu", "Odia": "or",
    "Tamil": "ta", "Telugu": "te", "Kannada": "kn", "Malayalam": "ml", "Arabic": "ur",
}

HINDI_DEV = {"है", "हैं", "और", "नहीं", "क्या", "मैं", "मुझे", "बहुत", "फिर", "वाह", "यह", "था", "थी", "आप", "हम"}
MARATHI_DEV = {"आहे", "आहेत", "आणि", "नाही", "काय", "मी", "मला", "खूप", "पुन्हा", "व्वा", "छान", "आम्ही", "तुम्ही", "होता"}

ROMAN_SETS = {
    "hi": {"hai", "hain", "nahi", "nahin", "kya", "yaar", "bhai", "kuch", "bahut", "mein", "phir", "abhi",
           "accha", "achha", "bohot", "wah", "waah", "aaj", "mera", "tum", "main", "hoon", "raha", "rahi",
           "matlab", "mast", "kaise", "ghar", "baarish", "gaye", "gayi"},
    "mr": {"aahe", "ahe", "kay", "mala", "tula", "amhi", "tumhi", "khup", "chan", "majha", "tujha",
           "kasa", "kuthe", "ata", "ani", "zala", "zali", "pan"},
    "te": {"ledu", "undi", "unnaru", "enti", "ela", "chala", "bagundi", "baagundi", "nenu", "meeru",
           "avunu", "ayyo", "sare", "malli"},
    "ta": {"illa", "enna", "romba", "nalla", "naan", "nee", "vanakkam", "epdi", "iruku", "irukku",
           "machan", "seri", "paaru"},
    "bn": {"ami", "tumi", "achi", "kemon", "bhalo", "korcho", "hobe", "ekdom", "darun", "abar"},
}


def guess_language(text: str) -> dict:
    prof = script_profile(text)
    dom = dominant_script(text)
    cmi = code_mix_index(text)
    toks = [t.lower() for t in tokenize_words(text)]
    out = {
        "script": dom,
        "scripts_present": sorted(prof),
        "language": "unknown",
        "romanized": False,
        "confidence": "low",
        "code_mixed_by_script": len(cmi["token_scripts"]) > 1,
        "cmi": cmi["cmi"],
        "note": "Heuristic guess for display only; use gold language labels in experiments.",
    }
    if dom == "None":
        return out
    if dom == "Devanagari":
        hi = sum(t in HINDI_DEV for t in toks)
        mr = sum(t in MARATHI_DEV for t in toks)
        if hi > mr:
            out.update(language="hi", confidence="medium" if hi >= 2 else "low")
        elif mr > hi:
            out.update(language="mr", confidence="medium" if mr >= 2 else "low")
        else:
            out.update(language="hi-or-mr", confidence="low")
        return out
    if dom in SCRIPT_TO_LANG:
        out.update(language=SCRIPT_TO_LANG[dom], confidence="high" if dom != "Arabic" else "medium")
        return out
    if dom == "Latin":
        hits = {lang: sum(t in words for t in toks) for lang, words in ROMAN_SETS.items()}
        best = max(hits, key=hits.get)
        if hits[best] == 0:
            out.update(language="en", confidence="low")
        else:
            ranked = sorted(hits.values(), reverse=True)
            tie = len(ranked) > 1 and ranked[0] == ranked[1]
            out.update(language=("indic-roman" if tie else best), romanized=True,
                       confidence="medium" if hits[best] >= 2 and not tie else "low")
        return out
    out.update(language="other", confidence="low")
    return out
