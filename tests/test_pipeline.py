import json

import pandas as pd
import pytest

from sarcasmscope.bundle import ModelBundle, demo_bundle
from sarcasmscope.data.toy import make_toy
from sarcasmscope.pipeline import DEFAULT_CONFIG, load_config, run


@pytest.fixture()
def cfg(tmp_path):
    toy = make_toy(n_per_lang=160, seed=4, leak=0.3, n_dups=15, suffix=True)
    path = tmp_path / "toy.csv"
    toy.to_csv(path, index=False)
    c = json.loads(json.dumps(DEFAULT_CONFIG))
    c.update({"run_name": "t", "output_dir": str(tmp_path / "runs"), "seeds": [1, 2], "models": ["tfidf"]})
    c["data"] = {"files": [{"path": str(path), "source": "toy", "mapping": {"text": "text", "label": "label", "lang": "lang"}}]}
    c["split"] = {"ratios": [0.7, 0.15, 0.15], "near_dup_threshold": None, "group_col": None}
    return c


def test_tc28_pipeline_end_to_end_writes_all_artifacts(cfg, tmp_path):
    out = run(cfg, log=lambda *_: None)
    d = tmp_path / "runs" / "t"
    for f in ["audit.md", "report.md", "tables/main_results.csv", "tables/per_language.csv", "tables/shortcuts.csv",
              "tables/robustness.csv", "tables/transfer_family.csv", "splits/seed_1.csv", "splits/seed_2.csv",
              "bundles/tfidf_seed1.joblib"]:
        assert (d / f).exists(), f
    assert out["eval_split"] == "val" and not (d / "FINAL_LOCK.json").exists()
    assert out["summary"].iloc[0]["macro_f1_mean"] > 0.6
    assert "Evaluation split: **VAL**" in (d / "report.md").read_text(encoding="utf-8")


def test_tc29_test_set_is_locked_after_final_run(cfg, tmp_path):
    run(cfg, final=True, log=lambda *_: None)
    assert (tmp_path / "runs" / "t" / "FINAL_LOCK.json").exists()
    with pytest.raises(RuntimeError, match="test-set peeking"):
        run(cfg, final=True, log=lambda *_: None)
    run(cfg, final=True, force=True, log=lambda *_: None)      # explicit override is possible


def test_tc30_bundle_roundtrip_and_abstention(cfg, tmp_path):
    run(cfg, log=lambda *_: None)
    b = ModelBundle.load(str(tmp_path / "runs" / "t" / "bundles" / "tfidf_seed1.joblib"))
    p = b.predict_p(["wah, phir se baarish 🙄"])
    assert p.shape == (1,) and 0 <= p[0] <= 1
    d = b.decide(float(p[0]))
    assert set(d) == {"label", "p_sarcastic", "confidence", "abstain", "threshold"}


def test_tc31_demo_bundle_is_flagged_as_demo():
    b = demo_bundle()
    assert b.meta["demo"] is True and b.temperature > 0


def test_tc32_config_loader_merges_defaults(tmp_path):
    y = tmp_path / "c.yaml"
    y.write_text("run_name: x\nseeds: [7]\nsplit:\n  near_dup_threshold: 0.8\n", encoding="utf-8")
    c = load_config(y)
    assert c["run_name"] == "x" and c["seeds"] == [7]
    assert c["split"]["near_dup_threshold"] == 0.8 and c["split"]["ratios"] == [0.7, 0.15, 0.15]


def test_tc38_context_option_runs_and_validates(cfg, tmp_path):
    import pandas as pd
    toy = pd.read_csv(cfg["data"]["files"][0]["path"])
    toy["prev"] = ["pichla vaakya " + str(i % 7) for i in range(len(toy))]
    toy.to_csv(cfg["data"]["files"][0]["path"], index=False)
    # no mapping for context -> clear error
    bad = json.loads(json.dumps(cfg)); bad["use_context"] = True; bad["run_name"] = "ctx_bad"
    with pytest.raises(ValueError, match="no row has a context"):
        run(bad, log=lambda *_: None)
    ok = json.loads(json.dumps(cfg)); ok["use_context"] = True; ok["run_name"] = "ctx_ok"
    ok["data"]["files"][0]["mapping"]["context"] = "prev"
    out = run(ok, log=lambda *_: None)
    assert out["summary"].iloc[0]["macro_f1_mean"] > 0.5
    assert not (tmp_path / "runs" / "ctx_ok" / "tables" / "robustness.csv").exists()
