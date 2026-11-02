"""Text normalisation that keeps what matters for sarcasm (emoji, '!!!', '...') and removes
what leaks the label (hashtags such as #sarcasm).

ZWJ/ZWNJ (U+200D/U+200C) are kept on purpose: they are meaningful in Indic scripts.
"""
from __future__ import annotations

import re
import unicodedata

from .emoji import strip_emojis

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.I)
MENTION_RE = re.compile(r"(?<!\w)@\w+")
HASHTAG_RE = re.compile(r"#(\w+)")
ZERO_WIDTH_RE = re.compile("[\u200b\u200e\u200f\u2060\ufeff]")
REPEAT_RE = re.compile(r"(\w)\1{2,}")
WS_RE = re.compile(r"\s+")
PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)

# Hashtags that directly reveal the label. Extend this list for your datasets
# (add native-script equivalents after checking them with a native speaker).
LABEL_HASHTAGS = {
    "sarcasm", "sarcastic", "sarcasmdetection", "irony", "ironic", "not", "sarcasmalert", "sarcasticmode",
}


def normalize(
    text: str,
    *,
    lowercase: bool = True,
    keep_emoji: bool = True,
    mask_urls: bool = True,
    mask_mentions: bool = True,
    hashtags: str = "strip_symbol",   # "strip_symbol" | "keep" | "remove"
    squeeze_repeats: bool = True,
    label_hashtags: set[str] | None = None,
) -> str:
    t = unicodedata.normalize("NFC", str(text))
    t = ZERO_WIDTH_RE.sub("", t)
    if mask_urls:
        t = URL_RE.sub(" <url> ", t)
    if mask_mentions:
        t = MENTION_RE.sub(" <user> ", t)
    banned = LABEL_HASHTAGS if label_hashtags is None else label_hashtags

    def _tag(m: re.Match) -> str:
        word = m.group(1)
        if word.lower() in banned:
            return " "
        if hashtags == "remove":
            return " "
        if hashtags == "keep":
            return m.group(0)
        return " " + word + " "

    t = HASHTAG_RE.sub(_tag, t)
    if not keep_emoji:
        t = strip_emojis(t)
    if squeeze_repeats:
        t = REPEAT_RE.sub(r"\1\1", t)
    if lowercase:
        t = t.lower()
    return WS_RE.sub(" ", t).strip()


def dedup_key(text: str) -> str:
    """Aggressive key used only to detect duplicates (ignores case, emoji, punctuation, links)."""
    t = normalize(text, keep_emoji=False, hashtags="remove", mask_urls=True, mask_mentions=True)
    t = t.replace("<url>", " ").replace("<user>", " ")
    t = PUNCT_RE.sub(" ", t)
    return WS_RE.sub(" ", t).strip()
