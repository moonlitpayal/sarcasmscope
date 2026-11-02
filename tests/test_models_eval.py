import numpy as np
import pandas as pd
import pytest

from sarcasmscope.data.splits import make_split
from sarcasmscope.data.toy import make_toy
from sarcasmscope.eval.metrics import binary_metrics, bootstrap_ci, macro_f1, per_group
from sarcasmscope.eval.robustness import PERTURBATIONS, clusters, perturb_all, robustness_table
from sarcasmscope.eval.shortcuts import shortcut_report
from sarcasmscope.eval.stats import mcnemar, paired_bootstrap
from sarcasmscope.eval.transfer import family_summary, transfer_matrix
from sarcasmscope.explain.occlusion import occlusion
from sarcasmscope.models.calibration import (TemperatureScaler, abstain_mask, aurc, crossfit_temperature,
                                             expected_calibration_error, risk_coverage,
                                             threshold_for_selective_accuracy)
from sarcasmscope.models.tfidf import TfidfLR
from sarcasmscope.text.normalize import normalize


@pytest.fixture(scope="module")
def split_df():
    df = make_toy(n_per_lang=150, seed=21, leak=0.0, n_dups=0, suffix=True)
    df["source"] = "toy"; df["context"] = ""; df["group"] = ""
    out, _ = make_split(df, seed=3, near_dup_threshold=None)
    out["text_norm"] = [normalize(t) for t in out["text"]]
    return out


def test_tc17_metrics_match_hand_computation():
    y = np.array([1, 1, 0, 0, 1, 0]); p = np.array([1, 0, 0, 1, 1, 0])
    m = binary_metrics(y, p)
    assert (m["tp"], m["tn"], m["fp"], m["fn"]) == (2, 2, 1, 1)
    assert m["accuracy"] == pytest.approx(4 / 6) and m["f1"] == pytest.approx(2 / 3)
    assert macro_f1(y, p) == pytest.approx(2 / 3)
    lo, hi = bootstrap_ci(y, p, n_boot=200)
    assert 0 <= lo <= hi <= 1


def test_tc18_mcnemar_exact():
    y = np.zeros(20, int)
    a = np.zeros(20, int); b = a.copy(); b[:8] = 1       # B wrong on 8 items A got right
    r = mcnemar(y, a, b)
    assert (r["a_only_correct"], r["b_only_correct"]) == (8, 0) and r["p_value"] == pytest.approx(2 * 0.5 ** 8)
    assert mcnemar(y, a, a)["p_value"] == 1.0
    d = paired_bootstrap(y, a, b, n_boot=100)
    assert d["mean_diff"] <= 0


def test_tc19_tfidf_learns_toy_data_and_beats_majority(split_df):
    tr, te = split_df[split_df.split == "train"], split_df[split_df.split == "test"]
    m = TfidfLR().fit(tr["text_norm"], tr["label"])
    pred = (m.predict_proba(te["text_norm"])[:, 1] >= 0.5).astype(int)
    assert macro_f1(te["label"].values, pred) > 0.70


def test_tc20_temperature_scaling_fixes_overconfidence():
    rng = np.random.RandomState(0)
    y = rng.randint(0, 2, 4000)
    right = rng.rand(4000) < 0.75                       # the model is right 75% of the time ...
    p = np.where(right == (y == 1), 0.97, 0.03)          # ... but always claims 97% confidence
    before = expected_calibration_error(p, y)
    scaler = TemperatureScaler().fit(p, y)
    after = expected_calibration_error(scaler.transform(p), y)
    assert scaler.T > 1.0 and after < before
    cf = crossfit_temperature(p, y)
    assert expected_calibration_error(cf, y) < before


def test_tc21_selective_prediction_improves_accuracy():
    rng = np.random.RandomState(1)
    y = rng.randint(0, 2, 3000)
    conf = rng.uniform(0.5, 1.0, 3000)
    correct = rng.rand(3000) < conf                       # confidence is informative
    p = np.where(correct == (y == 1), conf, 1 - conf)     # P(sarcastic)
    rc = risk_coverage(p, y)
    assert rc["selective_accuracy"].iloc[0] > rc["selective_accuracy"].iloc[-1]
    tau, cov = threshold_for_selective_accuracy(p, y, target=0.9)
    assert tau is not None and 0.1 <= cov < 1.0
    assert abstain_mask(p, tau).mean() == pytest.approx(1 - cov, abs=0.02)
    assert 0 <= aurc(p, y) <= 1


def test_tc22_perturbations_are_deterministic_and_safe():
    text = "वाह, क्या बात है फिर से बारिश 🙄 #life"
    a, b = perturb_all(text, seed=5), perturb_all(text, seed=5)
    assert a == b and set(a) == set(PERTURBATIONS)
    assert "🙄" not in a["remove_emoji"]
    assert all(t.startswith("#") is False for t in a["drop_tags"].split())
    assert clusters("क्षत्रिय") == ["क्ष", "त्रि", "य"] or len(clusters("क्षत्रिय")) >= 3
    for k, v in a.items():
        assert isinstance(v, str)


def test_tc23_robustness_table_shape_and_baseline(split_df):
    tr, te = split_df[split_df.split == "train"], split_df[split_df.split == "test"]
    m = TfidfLR().fit(tr["text_norm"], tr["label"])
    tab = robustness_table(lambda xs: m.predict_proba(xs)[:, 1], te["text"], te["label"], prep=normalize)
    assert tab.iloc[0]["perturbation"] == "none" and len(tab) == 1 + len(PERTURBATIONS)
    assert tab["prediction_flip_rate"].between(0, 1).all()
    shuffled = tab.set_index("perturbation").loc["shuffle_words"]
    assert shuffled["macro_f1"] <= tab.iloc[0]["macro_f1"] + 1e-9


def test_tc24_shortcut_baselines_flag_planted_emoji_shortcut():
    df = make_toy(n_per_lang=200, seed=2, leak=1.0, n_dups=0, suffix=True)
    df["source"] = "toy"
    tr, te = df.iloc[: len(df) // 2], df.iloc[len(df) // 2:]
    rep = shortcut_report(tr, te).set_index("shortcut")
    assert bool(rep.loc["emoji_only", "flag"]) and rep.loc["emoji_only", "macro_f1"] > 0.6
    clean = make_toy(n_per_lang=200, seed=2, leak=0.0, n_dups=0, suffix=True); clean["source"] = "toy"
    rep2 = shortcut_report(clean.iloc[:700], clean.iloc[700:]).set_index("shortcut")
    assert not bool(rep2.loc["emoji_only", "flag"])


def test_tc25_transfer_matrix_and_family_summary(split_df):
    long, matrix = transfer_matrix(split_df, TfidfLR, eval_split="test", min_train=30, min_eval=10)
    assert not long.empty and set(matrix.index) <= set(split_df.lang)
    diag = np.mean([matrix.loc[l, l] for l in matrix.index if l in matrix.columns])
    off = np.nanmean(matrix.values[~np.eye(len(matrix), len(matrix.columns), dtype=bool)]) if matrix.shape[0] == matrix.shape[1] else 0
    assert diag > 0.6
    assert "same language" in set(family_summary(long)["relation"])


def test_tc26_occlusion_finds_the_planted_trigger():
    class Toy:
        def __call__(self, xs):
            return np.array([0.9 if "zzz" in x.split() else 0.2 for x in xs])
    out = occlusion(Toy(), "aaa bbb zzz ccc")
    assert out["top_sarcastic"][0]["token"] == "zzz"
    assert out["p_sarcastic"] == pytest.approx(0.9)


def test_tc27_per_group_metrics():
    df = pd.DataFrame({"label": [1, 0, 1, 0], "pred": [1, 0, 0, 0], "lang": ["hi", "hi", "te", "te"]})
    g = per_group(df).set_index("lang")
    assert g.loc["hi", "macro_f1"] == 1.0 and g.loc["te", "macro_f1"] < 1.0
