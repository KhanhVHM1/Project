import os
os.environ["ENV"] = "TEST"  # isolation hint for any conditional logic

from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

def test_predict_all_candidate():
    body = {"features": {"x1": 2, "x2": 0.3}, "country": "all", "model": "candidate"}
    r = client.post("/predict", json=body)
    assert r.status_code == 200
    payload = r.json()
    assert payload["country"] == "all"
    assert "prediction" in payload

def test_predict_specific_country_baseline():
    body = {"features": {"x1": 1, "x2": 0.7}, "country": "US", "model": "baseline"}
    r = client.post("/predict", json=body)
    assert r.status_code == 200
    assert r.json()["country"] == "US"

def test_metrics_endpoint():
    r = client.get("/metrics")
    assert r.status_code == 200
    assert b"api_requests_total" in r.content
