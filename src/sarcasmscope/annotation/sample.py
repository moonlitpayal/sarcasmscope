"""Draw a pilot set, a RANDOM set and an ENRICHED set of sentences from a large corpus for human labelling.

Why three sets: sarcasm is rare in raw text, so a purely random sample contains very few positives. The enriched set is
chosen with a transparent cue filter (editable lists below). Items are flagged by subset so that results can always
be reported on the random subset (natural rate) and on the enriched subset (harder cases) separately.
"""
from __future__ import annotations

import csv
import random
import re
from pathlib import Path

import pandas as pd

from ..data.audit import near_duplicate_pairs
from ..text.emoji import IRONY_EMOJI
from ..text.normalize import dedup_key
from ..text.scripts import dominant_script, tokenize_words

# Cue lists used ONLY to pick candidates for the enriched set. They are heuristics: Marathi/Hindi readers on the team
# should extend and correct them. They never decide a label.
PRAISE = {"wah", "waah", "vah", "wow", "great", "super", "mast", "masta", "chan", "chhan", "ekdum", "perfect",
          "wonderful", "amazing", "best", "awesome", "thanks", "व्वा", "वाह", "छान", "मस्त", "सुपर", "भारी"}
NEGATIVE = {"traffic", "paus", "pavsa", "rain", "late", "delay", "bore", "boring", "waste", "worst", "bekar",
            "kharab", "problem", "trass", "बोअर", "त्रास", "उशीर", "पाऊस", "ट्रॅफिक", "वाया", "खराब", "प्रॉब्लेम",
            "power", "cut", "bandh", "fail", "failed"}
URL_RE = re.compile(r"https?://|www\.|@\w+")
TEXT_COLUMNS = ("text", "sentence", "tweet", "comment")


def _iter_lines(path: Path):
    suffix = path.suffix.lower()
    if suffix in (".csv", ".tsv"):
        with open(path, encoding="utf-8-sig", newline="") as f:
            reader = csv.reader(f, delimiter="\t" if suffix == ".tsv" else ",")
            first = next(reader, None)
            if first is None:
                return
            low = [h.strip().lower() for h in first]
            col = next((low.index(n) for n in TEXT_COLUMNS if n in low), None)
            if col is None:                      # no recognisable header: the first row is data
                col = 0
                yield first[0]
            for row in reader:
                if row and len(row) > col:
                    yield row[col]
    else:
        with open(path, encoding="utf-8-sig") as f:
            for line in f:
                yield line.rstrip("\n")


def reservoir_sample_lines(path: str | Path, k: int = 200_000, seed: int = 42) -> list[str]:
    """Uniform random sample of k lines from a file of any size, read once, constant memory."""
    rng = random.Random(seed)
    out: list[str] = []
    n = 0
    for line in _iter_lines(Path(path)):
        line = line.strip()
        if not line:
            continue
        n += 1
        if len(out) < k:
            out.append(line)
        else:
            j = rng.randint(0, n - 1)
            if j < k:
                out[j] = line
    return out


def is_candidate(s: str, min_words: int = 5, max_words: int = 40) -> bool:
    toks = tokenize_words(s)
    if not (min_words <= len(toks) <= max_words) or URL_RE.search(s):
        return False
    return sum(ch.isalpha() for ch in s) >= 0.5 * len(s.replace(" ", ""))


def cue_hits(s: str) -> list[str]:
    toks = {t.lower() for t in tokenize_words(s)}
    hits = []
    if any(ch in IRONY_EMOJI for ch in s):
        hits.append("irony_emoji")
    if toks & PRAISE and toks & NEGATIVE:
        hits.append("praise_near_negative")
    elif toks & PRAISE and ("!" in s or "..." in s):
        hits.append("praise_with_emphasis")
    return hits


def make_sample(sentences: list[str], n_random: int = 450, n_enriched: int = 450, n_pilot: int = 100,
                seed: int = 42, near_dup_threshold: float = 0.7) -> tuple[pd.DataFrame, dict]:
    """Return (frame with item_id, text, subset, script, cues; report). subset is pilot | random | enriched."""
    rng = random.Random(seed)
    cands, seen = [], set()
    for s in sentences:
        if not is_candidate(s):
            continue
        k = dedup_key(s)
        if k and k not in seen:
            seen.add(k)
            cands.append(s)
    keys = [dedup_key(s) for s in cands]
    drop = {j for _, j, _ in near_duplicate_pairs(keys, near_dup_threshold)}
    cands = [s for i, s in enumerate(cands) if i not in drop]
    report = {"candidates_after_filters": len(cands), "near_duplicates_removed": len(drop)}

    rng.shuffle(cands)
    if len(cands) < n_pilot + n_random:
        raise ValueError(f"Only {len(cands)} usable sentences; need at least {n_pilot + n_random}. "
                         f"Use a larger source or lower the counts.")
    enriched_pool = [s for s in cands if cue_hits(s)]
    enriched = enriched_pool[:n_enriched]
    taken = set(enriched)
    plain = [s for s in cands if s not in taken]
    pilot, rand = plain[:n_pilot], plain[n_pilot:n_pilot + n_random]
    report.update({"enriched_available": len(enriched_pool), "enriched_taken": len(enriched),
                   "random_taken": len(rand), "pilot_taken": len(pilot)})
    if len(enriched) < n_enriched:
        report["warning"] = (f"Only {len(enriched)} enriched sentences found (wanted {n_enriched}). Extend the "
                             f"PRAISE/NEGATIVE lists with Marathi/Hindi cue words or use a larger source.")
    df = pd.DataFrame([(t, "pilot") for t in pilot] + [(t, "random") for t in rand] + [(t, "enriched") for t in enriched],
                      columns=["text", "subset"])
    df["script"] = df["text"].map(dominant_script)
    df["cues"] = df["text"].map(lambda s: "+".join(cue_hits(s)))
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    df.insert(0, "item_id", [f"I{i:04d}" for i in range(1, len(df) + 1)])
    return df, report
