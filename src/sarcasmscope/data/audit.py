"""Dataset audit: duplicates, near-duplicates, label/language confounds, and label-correlated cues.

The goal is to find reasons why a reported score might be inflated BEFORE any model is trained.
"""
from __future__ import annotations

import re
import zlib

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, mannwhitneyu

from ..text.emoji import extract_emojis
from ..text.normalize import dedup_key
from ..text.scripts import code_mix_index, dominant_script

_SHINGLE = 4


def _shingles(s: str, k: int = _SHINGLE) -> set[int]:
    if len(s) < k:
        return {zlib.crc32(s.encode("utf-8"))}
    return {zlib.crc32(s[i:i + k].encode("utf-8")) for i in range(len(s) - k + 1)}


def near_duplicate_pairs(keys: list[str], threshold: float = 0.7, n_perm: int = 64,
                         bands: int = 16, min_len: int = 12, seed: int = 0) -> list[tuple[int, int, float]]:
    """MinHash + LSH near-duplicate search on character 4-grams (deterministic).

    Returns (i, j, jaccard) with i < j. Texts shorter than `min_len` characters are only
    matched exactly (handled elsewhere) because tiny strings give meaningless Jaccard scores.
    """
    rng = np.random.RandomState(seed)
    P = (1 << 31) - 1
    a = rng.randint(1, P, size=n_perm).astype(np.int64)
    b = rng.randint(0, P, size=n_perm).astype(np.int64)
    rows = n_perm // bands
    sets: dict[int, set[int]] = {}
    buckets: dict[tuple, list[int]] = {}
    for idx, k in enumerate(keys):
        if len(k) < min_len:
            continue
        sh = _shingles(k)
        sets[idx] = sh
        h = np.fromiter(sh, dtype=np.int64)
        sig = ((a[:, None] * h[None, :] + b[:, None]) % P).min(axis=1)
        for band in range(bands):
            key = (band, tuple(sig[band * rows:(band + 1) * rows].tolist()))
            buckets.setdefault(key, []).append(idx)
    seen: set[tuple[int, int]] = set()
    pairs: list[tuple[int, int, float]] = []
    for members in buckets.values():
        if len(members) < 2 or len(members) > 200:
            continue
        for x in range(len(members)):
            for y in range(x + 1, len(members)):
                i, j = members[x], members[y]
                if (i, j) in seen:
                    continue
                seen.add((i, j))
                s1, s2 = sets[i], sets[j]
                jac = len(s1 & s2) / len(s1 | s2)
                if jac >= threshold:
                    pairs.append((i, j, round(jac, 3)))
    return sorted(pairs)


class _UF:
    def __init__(self, n: int):
        self.p = list(range(n))

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def duplicate_clusters(df: pd.DataFrame, near_dup_threshold: float | None = 0.7,
                       group_col: str | None = None, merge_exact: bool = True) -> tuple[np.ndarray, dict]:
    """Cluster id per row: exact duplicates, near-duplicates and optional user groups share an id."""
    keys = [dedup_key(t) for t in df["text"]]
    uf = _UF(len(df))
    first: dict[str, int] = {}
    n_exact = 0
    for i, k in enumerate(keys):
        if not merge_exact:
            break
        if k in first:
            uf.union(first[k], i)
            n_exact += 1
        else:
            first[k] = i
    n_near = 0
    if near_dup_threshold:
        for i, j, _ in near_duplicate_pairs(keys, near_dup_threshold):
            if uf.find(i) != uf.find(j):
                n_near += 1
            uf.union(i, j)
    if group_col and group_col in df.columns:
        gfirst: dict[str, int] = {}
        for i, g in enumerate(df[group_col].astype(str)):
            if g in ("", "nan", "None"):
                continue
            if g in gfirst:
                uf.union(gfirst[g], i)
            else:
                gfirst[g] = i
    roots = np.array([uf.find(i) for i in range(len(df))])
    return roots, {"exact_duplicate_rows": n_exact, "near_duplicate_links": n_near}


def _cue_table(df: pd.DataFrame) -> pd.DataFrame:
    t = df["text"].astype(str)
    feats = pd.DataFrame({
        "has_emoji": t.map(lambda s: len(extract_emojis(s)) > 0),
        "has_hashtag": t.str.contains(r"#\w+", regex=True),
        "has_url": t.str.contains(r"https?://|www\.", regex=True),
        "has_mention": t.str.contains(r"(?<!\w)@\w+", regex=True),
        "has_exclamation": t.str.contains("!", regex=False),
        "has_question": t.str.contains(r"\?", regex=True),
        "has_ellipsis": t.str.contains(r"\.\.\.|…", regex=True),
        "has_quote_marks": t.str.contains(r"[\"“”]", regex=True),
        "has_code_mix_by_script": t.map(lambda s: len(code_mix_index(s)["token_scripts"]) > 1),
    })
    rows = []
    y = df["label"].values
    for col in feats.columns:
        x = feats[col].values.astype(bool)
        r1, r0 = x[y == 1].mean() if (y == 1).any() else np.nan, x[y == 0].mean() if (y == 0).any() else np.nan
        try:
            p = chi2_contingency(pd.crosstab(y, x))[1] if len(set(x)) > 1 and len(set(y)) > 1 else 1.0
        except ValueError:
            p = 1.0
        rows.append({"cue": col, "rate_sarcastic": r1, "rate_not_sarcastic": r0, "diff": r1 - r0, "p_value": p})
    length = t.str.len().values
    try:
        p = mannwhitneyu(length[y == 1], length[y == 0]).pvalue
    except ValueError:
        p = 1.0
    rows.append({"cue": "mean_chars", "rate_sarcastic": length[y == 1].mean(), "rate_not_sarcastic": length[y == 0].mean(),
                 "diff": length[y == 1].mean() - length[y == 0].mean(), "p_value": p})
    return pd.DataFrame(rows)


def audit(df: pd.DataFrame, near_dup_threshold: float = 0.7, max_near_dup_rows: int = 60000) -> dict:
    """Return a dict with all audit findings and a list of human-readable flags."""
    n = len(df)
    flags: list[dict] = []
    res: dict = {"n_rows": n}
    res["sources"] = df["source"].value_counts().to_dict() if "source" in df else {}
    share = float(df["label"].mean()) if n else float("nan")
    res["share_sarcastic"] = share
    if n and (share < 0.2 or share > 0.8):
        flags.append({"severity": "warn", "item": "label_balance",
                      "message": f"Strong class imbalance ({share:.1%} sarcastic). Report macro-F1, not accuracy."})

    by_lang = df.groupby("lang")["label"].agg(n="size", share_sarcastic="mean").reset_index()
    res["by_language"] = by_lang.to_dict("records")
    for r in by_lang.itertuples():
        if r.n >= 30 and (r.share_sarcastic < 0.3 or r.share_sarcastic > 0.7):
            flags.append({"severity": "warn", "item": f"language_confound:{r.lang}",
                          "message": f"Language '{r.lang}' has {r.share_sarcastic:.1%} sarcastic rows. A model can learn "
                                     f"'language = label'. Use per-language metrics and stratify splits by language."})
        if r.n < 100:
            flags.append({"severity": "info", "item": f"small_language:{r.lang}",
                          "message": f"Only {r.n} rows for '{r.lang}'. Treat its scores as indicative only."})

    keys = [dedup_key(t) for t in df["text"]]
    empty_after = sum(1 for k in keys if k == "")
    res["empty_after_cleaning"] = empty_after
    key_series = pd.Series(keys)
    dup_mask = key_series.duplicated(keep="first")
    res["exact_duplicate_rows"] = int(dup_mask.sum())
    conflict = df.assign(_k=keys).groupby("_k")["label"].nunique()
    res["conflicting_label_texts"] = int((conflict > 1).sum())
    if res["exact_duplicate_rows"] / max(n, 1) > 0.01:
        flags.append({"severity": "warn", "item": "exact_duplicates",
                      "message": f"{res['exact_duplicate_rows']} duplicate rows ({res['exact_duplicate_rows']/n:.1%}). "
                                 f"Duplicates across train/test inflate scores. Splits here remove them."})
    if res["conflicting_label_texts"] > 0:
        flags.append({"severity": "warn", "item": "conflicting_labels",
                      "message": f"{res['conflicting_label_texts']} texts appear with both labels (annotation noise)."})

    if n <= max_near_dup_rows and near_dup_threshold:
        pairs = near_duplicate_pairs(keys, near_dup_threshold)
        res["near_duplicate_pairs"] = len(pairs)
        if n and len(pairs) / n > 0.02:
            flags.append({"severity": "warn", "item": "near_duplicates",
                          "message": f"{len(pairs)} near-duplicate pairs (character 4-gram Jaccard >= {near_dup_threshold}). "
                                     f"Keep them in the same split."})
    else:
        res["near_duplicate_pairs"] = None

    cues = _cue_table(df)
    res["cues"] = cues.round(4).to_dict("records")
    for r in cues.itertuples():
        if r.cue == "mean_chars":
            continue
        if abs(r.diff) >= 0.10 and r.p_value < 0.01:
            flags.append({"severity": "warn", "item": f"cue:{r.cue}",
                          "message": f"'{r.cue}' differs by label ({r.rate_sarcastic:.1%} vs {r.rate_not_sarcastic:.1%}, "
                                     f"p={r.p_value:.1e}). It may act as a shortcut; run the shortcut baselines."})

    res["dominant_scripts"] = df["text"].map(dominant_script).value_counts().to_dict()
    res["mean_script_cmi"] = float(df["text"].map(lambda s: code_mix_index(s)["cmi"]).mean()) if n else 0.0
    res["flags"] = flags
    return res


def audit_markdown(res: dict) -> str:
    L = ["# Dataset audit", "",
         f"- Rows: **{res['n_rows']}**   |   Share sarcastic: **{res['share_sarcastic']:.1%}**",
         f"- Exact duplicate rows: **{res['exact_duplicate_rows']}**   |   Texts with conflicting labels: **{res['conflicting_label_texts']}**",
         f"- Near-duplicate pairs: **{res['near_duplicate_pairs']}**   |   Mean script code-mix index: **{res['mean_script_cmi']:.1f}**", "",
         "## By language", "", "| lang | rows | share sarcastic |", "|---|---|---|"]
    for r in res["by_language"]:
        L.append(f"| {r['lang']} | {r['n']} | {r['share_sarcastic']:.1%} |")
    L += ["", "## Label-correlated cues (possible shortcuts)", "", "| cue | sarcastic | not sarcastic | diff | p |", "|---|---|---|---|---|"]
    for r in res["cues"]:
        L.append(f"| {r['cue']} | {r['rate_sarcastic']:.3f} | {r['rate_not_sarcastic']:.3f} | {r['diff']:+.3f} | {r['p_value']:.1e} |")
    L += ["", "## Flags", ""]
    if not res["flags"]:
        L.append("No flags raised.")
    for f in res["flags"]:
        L.append(f"- **[{f['severity']}] {f['item']}**: {f['message']}")
    return "\n".join(L) + "\n"
