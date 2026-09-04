"""FastAPI endpoint tests (mock mode)."""

from __future__ import annotations

from fastapi.testclient import TestClient

from src.api.app import app

client = TestClient(app)


def _auth_header() -> dict[str, str]:
    token = client.post("/login", json={"username": "doctor", "password": "x"}).json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def test_health() -> None:
    assert client.get("/health").status_code == 200
    assert client.get("/health").json()["status"] == "ok"


def test_login_returns_token() -> None:
    r = client.post("/login", json={"username": "doctor", "password": "x"})
    assert r.status_code == 200
    assert "access_token" in r.json()


def test_analyze_requires_auth() -> None:
    r = client.post("/analyze", json={"query": "test"})
    assert r.status_code == 401


def test_analyze_end_to_end() -> None:
    h = _auth_header()
    r = client.post(
        "/analyze",
        json={"query": "45-year-old woman with dysuria and urinary frequency"},
        headers=h,
    )
    assert r.status_code == 200
    body = r.json()
    assert body["primary_diagnosis"] == "uncomplicated_uti"
    assert body["decision"] in ("answer", "ask_followup", "escalate")
    assert "differential" in body
    assert "evidence" in body


def test_metrics_endpoint() -> None:
    h = _auth_header()
    body = client.get("/metrics", headers=h).json()
    assert body["method"] == "aeb"
    assert body["accuracy"] >= 0.9
