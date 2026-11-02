import os
import pickle

import numpy as np
import pandas as pd
import pytest

from sarcasmscope.data.prepare import prepare_masac, safe_read_pickle
from sarcasmscope.data.splits import make_split
from sarcasmscope.eval.shortcuts import shortcut_report


def _obj(df):
    """Plain object-dtype frame (what the original MaSaC pickle contains; avoids pandas 3's string array types)."""
    df = df.astype(object)
    df.columns = pd.Index(list(df.columns), dtype=object)
    df.index = pd.RangeIndex(len(df))
    return df


def _fake_repo(tmp_path, n_dialogues=12, turns=6):
    rows, raw, eps = [], [], []
    spk = ["Sahil", "Indu", "Maya"]
    for d in range(n_dialogues):
        for t in range(turns):
            s = spk[t % 3]
            text = f"utterance {d} {t} {'wah' if t % 2 else 'nahin'}"
            rows.append({"Speaker": s, "text": "noisy " + text, "Context": [], "Sarcasm": int(t == 3),
                         "Humour": int(t == 2), "Episode_label": f"E{d}-D1", "Audio_Filename": f"a{d}-{t}.mp3"})
            raw.append(f"{s}: {text}")
            eps.append(f"E{d}-D1")
    pk = _obj(pd.DataFrame(rows))
    base = tmp_path / "repo" / "Data"
    (base / "Pickles").mkdir(parents=True)
    (base / "Raw_Utterances").mkdir(parents=True)
    pk.to_pickle(base / "Pickles" / "Dataset.p")
    (base / "Raw_Utterances" / "Utterances.txt").write_text("\r\n".join(raw), encoding="utf-8")
    (base / "Raw_Utterances" / "Episodewise_dialoguelabels.txt").write_text("Episode_label\r\n" + "\r\n".join(eps), encoding="utf-8")
    return tmp_path / "repo"


def test_tc39_dedup_keep_policy_keeps_everything_and_reports_overlap():
    rng = np.random.RandomState(0)
    rows = []
    for d in range(60):
        for t in range(6):
            rows.append({"id": f"{d}-{t}", "text": "nahin!" if t == 0 else f"line {d} {t}", "label": int(t == 0 and d % 2 == 0),
                         "lang": "hi-en", "source": "x", "context": "", "group": f"dlg{d}"})
    df = pd.DataFrame(rows)
    out, rep = make_split(df, seed=1, group_col="group", near_dup_threshold=None, dedup="keep")
    assert len(out) == len(df) and rep["dropped_exact_duplicates"] == 0 and rep["dropped_conflicting_label_rows"] == 0
    assert (out.groupby("group")["split"].nunique() == 1).all()                 # dialogues stay together
    assert rep["distinct_texts_in_multiple_splits"] >= 1                         # 'nahin!' overlap is reported, not hidden
    out2, rep2 = make_split(df, seed=1, group_col="group", near_dup_threshold=None, dedup="drop")
    assert len(out2) < len(df)                                                   # the default policy drops repeats/conflicts


def test_tc40_prepare_masac_aligns_checks_and_builds_context(tmp_path):
    repo = _fake_repo(tmp_path)
    info = prepare_masac(repo, tmp_path / "out" / "m.csv", context_turns=2)
    d = pd.read_csv(tmp_path / "out" / "m.csv")
    assert info["rows"] == 72 and info["speaker_match"] == 1.0 and info["sarcastic"] == 12
    first = d[d.dialogue.str.startswith("E0-D1")].iloc[0]
    assert pd.isna(first["context"])                                  # first turn has no context
    third = d[d.dialogue.str.startswith("E0-D1")].iloc[2]
    assert "utterance 0 0" in third["context"] and "utterance 0 1" in third["context"] and "utterance 0 2" not in third["context"]
    assert set(d["lang"]) == {"hi-en"}
    # misaligned files must be refused
    (repo / "Data" / "Raw_Utterances" / "Utterances.txt").write_text("Zed: x\r\n" * 72, encoding="utf-8")
    with pytest.raises(ValueError, match="not aligned"):
        prepare_masac(repo, tmp_path / "bad.csv")


def test_tc41_unsafe_pickle_is_refused(tmp_path):
    class Evil:
        def __reduce__(self):
            return (os.system, ("echo pwned > " + str(tmp_path / "pwned.txt"),))
    p = tmp_path / "evil.p"
    p.write_bytes(pickle.dumps(Evil()))
    with pytest.raises(pickle.UnpicklingError, match="Refusing to load"):
        safe_read_pickle(p)
    assert not (tmp_path / "pwned.txt").exists()
    ok = tmp_path / "ok.p"
    _obj(pd.DataFrame({"a": [1, 2]})).to_pickle(ok)
    assert list(safe_read_pickle(ok)["a"]) == [1, 2]


def test_tc42_extra_column_shortcut_is_detected():
    rng = np.random.RandomState(0)
    n = 1200
    speaker = rng.choice(["A", "B", "C", "D"], n)
    label = (np.isin(speaker, ["A", "B"]) & (rng.rand(n) < 0.9)).astype(int)
    df = pd.DataFrame({"text": [f"t{i}" for i in range(n)], "label": label, "lang": "hi-en", "source": "x", "speaker": speaker})
    rep = shortcut_report(df.iloc[:800], df.iloc[800:], extra_cols=["speaker"]).set_index("shortcut")
    assert bool(rep.loc["speaker_tag_only", "flag"])
