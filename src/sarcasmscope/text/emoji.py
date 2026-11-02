"""Emoji helpers (regex based; no external dependency)."""
from __future__ import annotations

import re

_E = (
    "[\U0001F300-\U0001FAFF\U0001F000-\U0001F2FF\u2600-\u27BF\u2B00-\u2BFF\U0001F1E6-\U0001F1FF]"
)
EMOJI_RE = re.compile(f"{_E}\uFE0F?(?:\u200D{_E}\uFE0F?)*")

IRONY_EMOJI = set("🙄😏😒🙃🤡😑😬")
POSITIVE_EMOJI = set("😂😍😊😁😀👍❤🙏🎉😘🥰😄")
NEGATIVE_EMOJI = set("😡😢😭👎💔😠😞😩😤")


def extract_emojis(text: str) -> list[str]:
    return EMOJI_RE.findall(str(text))


def strip_emojis(text: str) -> str:
    return EMOJI_RE.sub(" ", str(text))


def emoji_features(text: str) -> dict:
    ems = extract_emojis(text)
    flat = "".join(ems)
    return {
        "n_emoji": len(ems),
        "n_irony_emoji": sum(ch in IRONY_EMOJI for ch in flat),
        "n_positive_emoji": sum(ch in POSITIVE_EMOJI for ch in flat),
        "n_negative_emoji": sum(ch in NEGATIVE_EMOJI for ch in flat),
    }
