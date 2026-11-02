import pandas as pd
import pytest

from sarcasmscope.data.audit import audit, near_duplicate_pairs
from sarcasmscope.data.schema import standardize
from sarcasmscope.data.splits import make_split
from sarcasmscope.data.toy import make_toy
from sarcasmscope.text.normalize import dedup_key


def test_tc09_standardize_maps_labels_and_drops_bad_rows():
    raw = pd.DataFrame({"Tweet": ["a b c", "d e f", "", "g h i", "j k l"],
                        "Class": ["sarcastic", "non-sarcastic", "sarcastic", "maybe", "1"]})
    df, rep = standardize(raw, {"text": "Tweet", "label": "Class"}, source="X", lang="hi")
    assert list(df["label"]) == [1, 0, 1] and set(df["lang"]) == {"hi"}
    assert rep["dropped_bad_label_or_empty"] == 2 and "maybe" in rep["unrecognised_label_values"]


def test_tc10_standardize_missing_column_gives_clear_error():
    with pytest.raises(ValueError, match="Missing required column"):
        standardize(pd.DataFrame({"a": [1]}), None)


def test_tc11_near_duplicates_found():
    a = "wah kya baat hai phir se baarish aa gayi"
    b = "wah kya baat hai phir se baarish aa gayii"
    c = "bilkul alag sa ek vaakya jo kisi se match nahi karta"
    pairs = near_duplicate_pairs([a, b, c], threshold=0.7)
    assert [(i, j) for i, j, _ in pairs] == [(0, 1)]


def test_tc12_audit_flags_duplicates_conflicts_and_emoji_cue():
    df = make_toy(n_per_lang=60, seed=1, leak=1.0, n_dups=20)
    df = pd.concat([df, df.iloc[[0]].assign(label=1 - df.iloc[0]["label"], id="conflict")], ignore_index=True)
    s = pd.Series(df["lang"])
    df["source"] = "toy"
    res = audit(df)
    items = {f["item"] for f in res["flags"]}
    assert res["exact_duplicate_rows"] >= 20
    assert res["conflicting_label_texts"] >= 1
    assert "cue:has_emoji" in items
    assert set(res["dominant_scripts"]) >= {"Devanagari", "Telugu", "Tamil", "Bengali", "Latin"}


def _prep(seed=0, **kw):
    kw.setdefault("suffix", True)
    df = make_toy(n_per_lang=120, seed=seed, **kw)
    df["source"] = "toy"
    df["context"] = ""
    df["group"] = ""
    return df


def test_tc13_split_has_no_leakage_and_is_stratified():
    df = _prep(seed=3, n_dups=30)
    out, rep = make_split(df, seed=7, near_dup_threshold=None)
    assert rep["dropped_exact_duplicates"] >= 30
    keys = {sp: set(dedup_key(t) for t in out.loc[out.split == sp, "text"]) for sp in ("train", "val", "test")}
    assert not keys["train"] & keys["val"] and not keys["train"] & keys["test"] and not keys["val"] & keys["test"]
    frac = out["split"].value_counts(normalize=True)
    assert abs(frac["train"] - 0.7) < 0.06 and abs(frac["val"] - 0.15) < 0.06 and abs(frac["test"] - 0.15) < 0.06
    for sp in ("train", "val", "test"):
        part = out[out.split == sp]
        assert 0.3 < part["label"].mean() < 0.7
        assert part["lang"].nunique() == df["lang"].nunique()


def test_tc14_split_is_deterministic_and_seed_dependent():
    df = _prep(seed=5)
    a, _ = make_split(df, seed=11)
    b, _ = make_split(df, seed=11)
    c, _ = make_split(df, seed=12)
    assert (a["split"].values == b["split"].values).all()
    assert (a["split"].values != c["split"].values).any()


def test_tc15_groups_stay_together():
    df = _prep(seed=2)
    df["group"] = [f"conv{i // 5}" for i in range(len(df))]
    out, _ = make_split(df, seed=1, group_col="group")
    assert (out.groupby("group")["split"].nunique() == 1).all()


def test_tc16_near_duplicates_never_cross_the_train_test_boundary():
    base = _prep(seed=9)
    base = base.iloc[::3].reset_index(drop=True)             # keep the sample diverse
    variants = base.sample(40, random_state=0).copy()
    variants["text"] = variants["text"] + "a"                 # tiny edit -> near-duplicate
    variants["id"] = [f"v{i}" for i in range(len(variants))]
    df = pd.concat([base, variants], ignore_index=True)
    out, rep = make_split(df, seed=4, near_dup_threshold=0.7)
    assert rep["near_duplicate_links_merged"] + rep["dropped_exact_duplicates"] >= 40
    assert (out.groupby("cluster")["split"].nunique() == 1).all()
