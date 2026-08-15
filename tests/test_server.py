"""API tests for the FastAPI backend (no heavy training jobs)."""

import pytest

pytestmark = pytest.mark.smoke


def test_health():
    from fastapi.testclient import TestClient

    from oneclick_distill.server.app import app

    with TestClient(app) as client:
        r = client.get("/api/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"


def test_hardware_endpoint():
    from fastapi.testclient import TestClient

    from oneclick_distill.server.app import app

    with TestClient(app) as client:
        r = client.get("/api/hardware")
        assert r.status_code == 200
        assert r.json()["strategy"]["backend"] in {"transformers", "unsloth"}


def test_index_served():
    from fastapi.testclient import TestClient

    from oneclick_distill.server.app import app

    with TestClient(app) as client:
        r = client.get("/")
        assert r.status_code == 200
        assert "OneClick" in r.text


def test_invalid_job_spec():
    from fastapi.testclient import TestClient

    from oneclick_distill.server.app import app

    with TestClient(app) as client:
        r = client.post("/api/jobs", json={"size": "huge", "teacher": {"name": "none"}})
        assert r.status_code == 422
