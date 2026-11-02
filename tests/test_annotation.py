import numpy as np
import pandas as pd
import pytest
from openpyxl import load_workbook

from sarcasmscope.annotation.agreement import adjudicate, agreement_report, collect, fleiss_kappa, landis_koch
from sarcasmscope.annotation.sample import cue_hits, make_sample, reservoir_sample_lines
from sarcasmscope.annotation.workbook import read_annotator_workbook, write_annotator_workbook
from sarcasmscope.cli import main


def _sentences(n=3000, seed=0, cue_share=0.2):
    rng = np.random.RandomState(seed)
    letters = np.array(list("abcdefghijklmnopqrstuvwxyz"))
    vocab = ["".join(rng.choice(letters, 6)) for _ in range(600)]
    out = []
    for i in range(n):
        words = list(rng.choice(vocab, 8))
        if rng.rand() < cue_share:
            words[2], words[5] = "wah", "traffic"
        out.append(" ".join(words))
    return out


def test_tc43_sampler_subsets_are_disjoint_reproducible_and_cued():
    s = _sentences()
    a, rep = make_sample(s, n_random=200, n_enriched=150, n_pilot=50, seed=1)
    b, _ = make_sample(s, n_random=200, n_enriched=150, n_pilot=50, seed=1)
    assert a["text"].tolist() == b["text"].tolist()
    assert a["subset"].value_counts().to_dict() == {"random": 200, "enriched": 150, "pilot": 50}
    assert a["text"].is_unique and a["item_id"].is_unique
    assert all(cue_hits(t) for t in a.loc[a.subset == "enriched", "text"])
    assert rep["enriched_taken"] == 150
    w, rep2 = make_sample(s, n_random=100, n_enriched=5000, n_pilot=20, seed=1)
    assert "warning" in rep2
    with pytest.raises(ValueError, match="usable sentences"):
        make_sample(s[:100], n_random=200, n_enriched=10, n_pilot=20)


def test_tc44_annotator_workbook_hides_subset_and_has_dropdown(tmp_path):
    items, _ = make_sample(_sentences(), n_random=60, n_enriched=40, n_pilot=10, seed=2)
    p = write_annotator_workbook(items, "Asha", tmp_path / "Asha_final.xlsx")
    wb = load_workbook(p)
    ws = wb["Label"]
    header = [c.value for c in ws[1]]
    assert header[:3] == ["item_id", "text", "label"] and "subset" not in " ".join(map(str, header)).lower()
    assert ws.data_validations.dataValidation and "sarcastic" in ws.data_validations.dataValidation[0].formula1
    back = read_annotator_workbook(p)
    assert len(back) == len(items) and (back["label"] == "").all()
    assert "cues" not in str(wb["Guidelines"]["A1"].value).lower()


def test_tc45_kappa_matches_published_examples():
    # Fleiss' kappa: the standard worked example with 10 items, 14 raters, 5 categories (kappa = 0.210)
    table = [[0, 0, 0, 0, 14], [0, 2, 6, 4, 2], [0, 0, 3, 5, 6], [0, 3, 9, 2, 0], [2, 2, 8, 1, 1],
             [7, 7, 0, 0, 0], [3, 2, 6, 3, 0], [2, 5, 3, 2, 2], [6, 5, 2, 1, 0], [0, 2, 2, 3, 7]]
    assert fleiss_kappa(table) == pytest.approx(0.210, abs=0.001)
    assert fleiss_kappa([[3, 0], [0, 3], [3, 0], [0, 3]]) == pytest.approx(1.0)
    assert [landis_koch(x) for x in (-0.1, 0.1, 0.3, 0.5, 0.7, 0.9)] == ["poor", "slight", "fair", "moderate", "substantial", "almost perfect"]


def _fill(path, truth, rng, accuracy=0.9):
    wb = load_workbook(path)
    ws = wb["Label"]
    for i in range(2, ws.max_row + 1):
        t = truth[ws.cell(row=i, column=1).value]
        r = rng.rand()
        lab = "cannot tell" if r < 0.03 else (t if r < 0.03 + accuracy else ("not sarcastic" if t == "sarcastic" else "sarcastic"))
        ws.cell(row=i, column=3, value=lab)
    wb.save(path)


def test_tc46_end_to_end_sampling_labelling_agreement(tmp_path):
    src = tmp_path / "corpus.txt"
    src.write_text("\n".join(_sentences()), encoding="utf-8")
    out = tmp_path / "ann"
    main(["annotate-sample", "--source", str(src), "--out", str(out), "--annotators", "Asha", "Bimal", "Chitra",
          "--n-random", "120", "--n-enriched", "80", "--pilot", "10", "--pool", "3000"])
    key = pd.read_csv(out / "master_key.csv")
    assert (out / "pilot" / "Asha_pilot.xlsx").exists() and len(key) == 210
    rng = np.random.RandomState(0)
    truth = {r.item_id: ("sarcastic" if rng.rand() < 0.3 else "not sarcastic") for r in key.itertuples()}
    filled = tmp_path / "filled"
    filled.mkdir()
    for name in ("Asha", "Bimal", "Chitra"):
        dst = filled / f"{name}_final_filled.xlsx"
        dst.write_bytes((out / "final" / f"{name}_final.xlsx").read_bytes())
        _fill(dst, truth, rng)
    rep_dir = tmp_path / "rep"
    main(["annotate-agreement", "--dir", str(filled), "--out", str(rep_dir), "--key", str(out / "master_key.csv")])
    wide = collect(sorted(filled.glob("*.xlsx")))
    res = agreement_report(wide)
    assert res["annotators"] == ["Asha", "Bimal", "Chitra"] and res["items_fully_labelled"] == 200
    assert 0.5 < res["fleiss_kappa_3way"] < 1.0
    adj = pd.read_csv(rep_dir / "adjudication.csv")
    fin = pd.read_csv(rep_dir / "final_labels.csv")
    assert {"text", "subset"} <= set(adj.columns) and len(fin) >= 150
    todo = adj[adj["how"] == "needs_discussion"]
    if len(todo):
        pd.DataFrame({"item_id": todo["item_id"], "label": "not sarcastic"}).to_csv(tmp_path / "disc.csv", index=False)
        main(["annotate-agreement", "--dir", str(filled), "--out", str(rep_dir), "--key", str(out / "master_key.csv"),
              "--discussed", str(tmp_path / "disc.csv")])
        assert len(pd.read_csv(rep_dir / "final_labels.csv")) == 200
    assert (rep_dir / "agreement.md").read_text(encoding="utf-8").startswith("# Inter-annotator agreement")


def test_tc47_reservoir_sampling_and_csv_headers(tmp_path):
    lines = [f"sentence number {i} with some words" for i in range(500)]
    t = tmp_path / "a.txt"; t.write_text("\n".join(lines), encoding="utf-8")
    s1, s2 = reservoir_sample_lines(t, k=50, seed=3), reservoir_sample_lines(t, k=50, seed=3)
    assert s1 == s2 and len(s1) == 50 and len(set(s1)) == 50 and set(s1) <= set(lines)
    c1 = tmp_path / "h.csv"; pd.DataFrame({"id": range(5), "text": lines[:5]}).to_csv(c1, index=False)
    assert reservoir_sample_lines(c1, k=10) == lines[:5]
    c2 = tmp_path / "n.csv"; c2.write_text("\n".join(lines[:5]), encoding="utf-8")
    assert reservoir_sample_lines(c2, k=10) == lines[:5]
    assert adjudicate(pd.DataFrame({"item_id": ["a"], "x": ["sarcastic"], "y": ["not sarcastic"]})).iloc[0]["how"] == "needs_discussion"
