"""E2E smoke tests: the whole pipeline (data → train) on a tiny model.

Uses hf-internal-testing/tiny-random-gpt2 (~2 MB) so the run finishes in
under a minute on CPU. Network is only needed to download the tiny model
on the first run (cached afterwards by HF_HOME).
"""

import json
import os

import pytest

from oneclick_distill.config import SAMPLE_DATA_PATH, settings
from oneclick_distill.data import chunk_text, load_text_files
from oneclick_distill.hardware import probe
from oneclick_distill.runner import run_pipeline
from oneclick_distill.schema import JobSpec, TeacherConfig

pytestmark = pytest.mark.smoke


def test_hardware_probe():
    report = probe()
    assert report["device"] in {"cpu", "cuda"}
    assert report["python"].startswith("3")
    assert report["strategy"]["backend"] in {"transformers", "unsloth"}


def test_data_loading_and_chunking():
    texts = load_text_files([str(SAMPLE_DATA_PATH)])
    assert texts
    chunks = chunk_text(texts[0], max_chars=100)
    assert all(len(c) <= 100 for c in chunks)
    assert all(c for c in chunks)


def test_schema_roundtrip():
    spec = JobSpec(source="cli", size="ultra", teacher=TeacherConfig(name="deepseek"))
    d = spec.to_dict()
    spec2 = JobSpec.from_dict(d)
    assert spec2.size == spec.size
    assert spec2.teacher.name == "deepseek"
    assert "api_key" not in json.dumps(d) or "***" in d["teacher"]["api_key"] or d["teacher"]["api_key"] == ""


def test_pipeline_end_to_end():
    spec = JobSpec(
        source="test",
        data_paths=[str(SAMPLE_DATA_PATH)],
        teacher=TeacherConfig(name="none"),
        model=settings.models.get("smoke", "hf-internal-testing/tiny-random-gpt2"),
        size="smoke",
        max_steps=2,
        smoke=True,
        quantize=False,
        out_dir="runs/test-smoke",
    )
    events: list[dict] = []

    def cb(stage, progress, message, metrics=None):
        events.append({"stage": stage, "progress": progress})

    result = run_pipeline(spec, cb)
    assert result["model_dir"]
    assert os.path.exists(os.path.join(result["model_dir"], "config.json"))
    assert events, "进度回调必须被触发"
    assert events[-1]["stage"].value == "done"
