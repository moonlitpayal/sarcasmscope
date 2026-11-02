"""Leakage-safe train/val/test splits.

Steps: drop exact duplicates and texts with conflicting labels, cluster near-duplicates (and
optional conversation/user groups), then assign WHOLE clusters to splits so that related rows
can never sit on both sides of the train/test boundary. Assignment is stratified by label x
language and fully determined by the seed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..text.normalize import dedup_key
from .audit import duplicate_clusters

SPLITS = ("train", "val", "test")


def _greedy_assign(strata: np.ndarray, clusters: np.ndarray, ratios, seed: int) -> dict:
    rng = np.random.RandomState(seed)
    df = pd.DataFrame({"s": strata, "c": clusters})
    totals = df["s"].value_counts().to_dict()
    target = {sp: {s: totals[s] * r for s in totals} for sp, r in zip(SPLITS, ratios)}
    current = {sp: {s: 0 for s in totals} for sp in SPLITS}
    per_cluster = df.groupby("c")["s"].agg(lambda x: x.value_counts().to_dict())
    sizes = df.groupby("c").size()
    order = sorted(per_cluster.index, key=lambda c: (-int(sizes[c]), rng.rand()))
    assignment = {}
    for c in order:
        counts = per_cluster[c]
        best_sp, best_score = None, -1e18
        for sp in SPLITS:
            score = sum((target[sp][s] - current[sp][s]) / max(target[sp][s], 1.0) * k for s, k in counts.items())
            score += rng.rand() * 1e-6
            if score > best_score:
                best_sp, best_score = sp, score
        assignment[c] = best_sp
        for s, k in counts.items():
            current[best_sp][s] += k
    return assignment


def make_split(df: pd.DataFrame, seed: int = 42, ratios=(0.7, 0.15, 0.15), group_col: str | None = None,
               near_dup_threshold: float | None = 0.7, drop_conflicts: bool = True,
               dedup: str = "drop") -> tuple[pd.DataFrame, dict]:
    """Return (frame with 'split' and 'cluster' columns, report).

    dedup="drop" (tweets, comments): remove exact duplicates and conflicting-label texts, merge near-duplicates.
    dedup="keep" (dialogues): keep every row, because identical short utterances can be legitimately sarcastic in one
    context and literal in another; related rows are kept together through `group_col` instead. The report then states
    how many distinct texts appear in more than one split, so that overlap is never hidden.
    """
    assert dedup in ("drop", "keep"), "dedup must be 'drop' or 'keep'"
    assert abs(sum(ratios) - 1.0) < 1e-6, "ratios must sum to 1"
    d = df.reset_index(drop=True).copy()
    n0 = len(d)
    d["_k"] = [dedup_key(t) for t in d["text"]]
    d = d[d["_k"] != ""].copy()
    n_empty = n0 - len(d)
    n_conf = n_dup = 0
    if dedup == "drop":
        if drop_conflicts:
            nun = d.groupby("_k")["label"].transform("nunique")
            n_conf = int((nun > 1).sum())
            d = d[nun == 1].copy()
        before = len(d)
        d = d.drop_duplicates("_k", keep="first").copy()
        n_dup = before - len(d)
    keys_kept = d["_k"].values
    d = d.drop(columns="_k").reset_index(drop=True)

    clusters, info = duplicate_clusters(d, near_dup_threshold if dedup == "drop" else None, group_col,
                                        merge_exact=(dedup == "drop"))
    d["cluster"] = clusters
    strata = (d["label"].astype(str) + "|" + d["lang"].astype(str)).values
    assignment = _greedy_assign(strata, clusters, ratios, seed)
    d["split"] = d["cluster"].map(assignment)

    g = d.groupby("cluster")["split"].nunique()
    assert (g == 1).all(), "internal error: a cluster was split across partitions"
    shared = pd.Series(d["split"].values).groupby(keys_kept).nunique()
    n_shared = int((shared > 1).sum())
    sizes = d.groupby("cluster").size()
    largest = float(sizes.max() / len(d)) if len(d) else 0.0
    report = {
        "seed": seed, "dedup": dedup, "distinct_texts_in_multiple_splits": n_shared,
        "largest_cluster_share": round(largest, 4),
        "warning": ("One near-duplicate/group cluster holds more than 5% of the data; split ratios may be off. "
                    "Raise near_dup_threshold or check for templated text.") if largest > 0.05 else "", "rows_in": n0, "rows_out": len(d),
        "dropped_empty": n_empty, "dropped_conflicting_label_rows": n_conf, "dropped_exact_duplicates": n_dup,
        "near_duplicate_links_merged": info["near_duplicate_links"],
        "counts": d["split"].value_counts().to_dict(),
        "share_sarcastic": d.groupby("split")["label"].mean().round(4).to_dict(),
        "languages": d.groupby("split")["lang"].value_counts().unstack(fill_value=0).to_dict("index"),
    }
    return d, report
