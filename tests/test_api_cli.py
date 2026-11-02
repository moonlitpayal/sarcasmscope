import json

import pytest
from fastapi.testclient import TestClient

from sarcasmscope.app.api import create_app
from sarcasmscope.cli import main


@pytest.fixture(scope="module")
def client():
    return TestClient(create_app())


def test_tc33_health_and_ui(client):
    h = client.get("/health").json()
    assert h["status"] == "ok" and h["demo_model"] is True
    page = client.get("/")
    assert page.status_code == 200 and "SarcasmScope" in page.text


def test_tc34_predict_returns_decision_language_emoji(client):
    r = client.post("/predict", json={"text": "वाह, फिर से बारिश! 🙄"}).json()
    assert 0 <= r["p_sarcastic"] <= 1 and r["label"] in ("sarcastic", "not sarcastic")
    assert r["language"]["script"] == "Devanagari" and r["emoji"]["n_irony_emoji"] == 1
    assert isinstance(r["abstain"], bool) and r["demo_model"] is True


def test_tc35_explain_and_robustness(client):
    e = client.post("/explain", json={"text": "wah, traffic mein fas gaye"}).json()
    assert len(e["attributions"]) == 5 and "occlusion" in e["method"]
    r = client.post("/robustness", json={"text": "wah, traffic mein fas gaye"}).json()
    assert len(r["results"]) == 8 and all("flipped" in x for x in r["results"])


def test_tc36_batch_and_input_validation(client):
    b = client.post("/predict_batch", json={"texts": ["wah!", "ధన్యవాదాలు", "   "]}).json()
    assert b["n"] == 3 and "error" in b["results"][2]
    assert client.post("/predict", json={"text": ""}).status_code == 422
    assert client.post("/predict", json={"text": "x" * 5000}).status_code == 422
    assert client.post("/predict_batch", json={"texts": ["a"] * 500}).status_code == 422
    assert client.post("/predict", json={"text": "#sarcasm"}).status_code == 422     # empty after cleaning


def test_tc37_cli_init_config_and_demo_data(tmp_path):
    main(["init-config", "--out", str(tmp_path / "c.yaml")])
    main(["demo-data", "--out", str(tmp_path / "t.csv"), "--n", "20"])
    assert (tmp_path / "c.yaml").exists() and (tmp_path / "t.csv").stat().st_size > 100
