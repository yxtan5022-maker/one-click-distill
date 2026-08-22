"""API tests for the FastAPI backend (no heavy training jobs)."""

import importlib
import time
from pathlib import Path

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


def _wait_terminal(mgr, job_id, timeout=10.0):
    deadline = time.time() + timeout
    state = mgr.get(job_id)
    while time.time() < deadline and state and state.status.value not in ("done", "failed", "cancelled"):
        time.sleep(0.05)
        state = mgr.get(job_id)
    return state


def test_cancel_running_job(monkeypatch):
    """DELETE /api/jobs/{id} cooperatively stops a running pipeline."""
    from oneclick_distill.schema import Stage
    mgr_mod = importlib.import_module("oneclick_distill.server.manager")
    from oneclick_distill.server.app import app

    def slow_run(spec, cb):
        for i in range(1000):  # ~5s if never cancelled
            cb(Stage.PREPARE, i / 1000, f"step {i}")
            time.sleep(0.005)
        return {}

    monkeypatch.setattr(mgr_mod, "run_pipeline", slow_run)

    spec = mgr_mod.JobSpec.from_dict({"task": "llm", "teacher": {"name": "none"}})
    state = mgr_mod.manager.submit(spec)

    from fastapi.testclient import TestClient

    with TestClient(app) as client:
        r = client.delete(f"/api/jobs/{state.id}")
        assert r.status_code == 200
        final = _wait_terminal(mgr_mod.manager, state.id)
        assert final is not None
        assert final.status.value == "cancelled"
        # Cancelling an already-terminal job is a no-op that returns its state.
        r2 = client.delete(f"/api/jobs/{state.id}")
        assert r2.status_code == 200
        assert r2.json()["status"] == "cancelled"
        # Unknown job id -> 404.
        assert client.delete("/api/jobs/nonexistent").status_code == 404


def test_export_completed_job(tmp_path, monkeypatch):
    """POST /api/export re-quantizes a completed job's model to GGUF."""
    from fastapi.testclient import TestClient

    import oneclick_distill.quantize.llama_cpp as qmod
    mgr_mod = importlib.import_module("oneclick_distill.server.manager")
    from oneclick_distill.server.app import app

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "adapter_config.json").write_text("{}", encoding="utf-8")

    def fake_run(spec, cb):
        cb("prepare", 1.0, "ok")
        return {"model_dir": str(model_dir)}

    monkeypatch.setattr(mgr_mod, "run_pipeline", fake_run)

    captured = {}

    def fake_quantize(md, out_dir, quant_type="Q4_K_M", progress=None):
        captured["model_dir"] = str(md)
        captured["quant_type"] = quant_type
        out = Path(out_dir) / f"model-{quant_type.lower()}.gguf"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"\x00" * 4096)
        return out

    monkeypatch.setattr(qmod, "quantize", fake_quantize)

    spec = mgr_mod.JobSpec.from_dict({"task": "llm", "teacher": {"name": "none"}})
    state = mgr_mod.manager.submit(spec)
    final = _wait_terminal(mgr_mod.manager, state.id)
    assert final is not None and final.status.value == "done"

    with TestClient(app) as client:
        r = client.post(
            "/api/export",
            json={"job_id": state.id, "format": "gguf", "quantization": "q4_k_m"},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["job_id"] == state.id
        assert data["gguf"] == str(Path(data["gguf"]))
        assert Path(data["gguf"]).exists()
        assert data["size_mb"] >= 0
        assert captured["quant_type"] == "Q4_K_M"
        assert Path(captured["model_dir"]) == model_dir

        # Unsupported format / quantization are rejected.
        assert client.post("/api/export", json={"format": "onnx"}).status_code == 422
        assert (
            client.post(
                "/api/export",
                json={"format": "gguf", "quantization": "q9_zzz"},
            ).status_code
            == 422
        )
        # Unknown job id -> 404.
        assert client.post("/api/export", json={"job_id": "nope"}).status_code == 404
