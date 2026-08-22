"""FastAPI + WebSocket backend and built-in Web UI for OneClick Distill."""

from __future__ import annotations

import asyncio
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from ..config import WEB_DIR, settings
from ..hardware import probe
from ..schema import JobSpec, Status
from ..teacher import TeacherClient, TeacherConfig
from .manager import manager


@asynccontextmanager
async def lifespan(_: FastAPI):
    manager.set_loop(asyncio.get_running_loop())
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    runs_dir = Path("runs")
    if runs_dir.exists():
        app.mount("/runs", StaticFiles(directory=str(runs_dir)), name="runs")
    yield


app = FastAPI(title="OneClick Distill", version="0.1.0", lifespan=lifespan)
UPLOADS_DIR = Path("uploads")


@app.get("/")
async def index():
    return FileResponse(WEB_DIR / "index.html")


@app.get("/api/health")
async def health():
    return {"status": "ok", "time": time.time()}


@app.get("/api/hardware")
async def hardware():
    return probe()


@app.get("/api/jobs")
async def jobs():
    return manager.list()


@app.get("/api/jobs/{job_id}")
async def job_detail(job_id: str):
    state = manager.get(job_id)
    if not state:
        return JSONResponse({"error": "job not found"}, status_code=404)
    return state.to_dict()


@app.delete("/api/jobs/{job_id}")
async def cancel_job(job_id: str):
    """Cooperative cancel: the pipeline stops at its next progress callback."""
    state = manager.cancel(job_id)
    if not state:
        return JSONResponse({"error": "job not found"}, status_code=404)
    return state.to_dict()


@app.post("/api/jobs")
async def create_job(spec: dict):
    size = spec.get("size", "ultra")
    if size not in ("ultra", "balanced", "smoke"):
        return JSONResponse({"error": f"无效的规格: {size}（可选 ultra/balanced/smoke）"}, status_code=422)
    try:
        job_spec = JobSpec.from_dict(fill_teacher_defaults(spec))
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"error": f"invalid spec: {e}"}, status_code=422)
    state = manager.submit(job_spec)
    return state.to_dict()


def fill_teacher_defaults(spec: dict) -> dict:
    """Fill missing teacher base_url/model from presets.yaml / .env so that
    UI clients only need to send {name, model?, api_key?}."""
    teacher = spec.get("teacher") or {}
    name = str(teacher.get("name", "") or "")
    if name and name != "none":
        preset = settings.teacher_config(name)
        if not teacher.get("base_url"):
            teacher["base_url"] = preset["base_url"]
        if not teacher.get("model"):
            teacher["model"] = preset["model"]
        spec["teacher"] = teacher
    return spec


@app.post("/api/files")
async def upload_file(file: UploadFile):
    safe_name = (file.filename or "upload.bin").replace("\\", "/").rsplit("/", 1)[-1] or "upload.bin"
    path = UPLOADS_DIR / f"{uuid.uuid4().hex}_{safe_name}"
    path.write_bytes(await file.read())
    return {"name": file.filename, "path": str(path), "size": path.stat().st_size}


@app.websocket("/ws/jobs/{job_id}")
async def ws_jobs(websocket: WebSocket, job_id: str):
    await websocket.accept()
    q = manager.subscribe(job_id)
    try:
        state = manager.get(job_id)
        if state:
            await websocket.send_json(state.to_dict())
        while True:
            payload = await asyncio.wait_for(q.get(), timeout=30)
            await websocket.send_json(payload)
    except (asyncio.TimeoutError, WebSocketDisconnect):
        pass
    finally:
        manager.unsubscribe(job_id, q)


# ---------------------------------------------------------------------------
# Playground: A/B comparison between teacher and the distilled student model
# ---------------------------------------------------------------------------
_student_cache: dict[str, tuple[Any, Any]] = {}


def _load_student(model_dir: str):
    if model_dir in _student_cache:
        return _student_cache[model_dir]
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_dir)
    model.eval()
    _student_cache[model_dir] = (model, tok)
    return _student_cache[model_dir]


def _student_answer(model_dir: str, question: str, max_new: int = 128) -> dict[str, Any]:
    import torch

    model, tok = _load_student(model_dir)
    t0 = time.time()
    prompt = f"问题：{question}\n回答："
    inputs = tok(prompt, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new, do_sample=True, top_p=0.9, temperature=0.7)
    answer = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    return {"answer": answer, "latency_ms": round((time.time() - t0) * 1000, 1)}


def _teacher_answer(question: str, max_new: int = 256) -> dict[str, Any]:
    cfg = TeacherConfig(
        name=settings.env("TEACHER_NAME", "deepseek"),
        model=settings.env("TEACHER_MODEL", ""),
        base_url=settings.env("TEACHER_BASE_URL", ""),
        api_key=settings.env("TEACHER_API_KEY", ""),
    )
    if not cfg.base_url or not cfg.api_key:
        return {"answer": "未配置教师模型 API（请填写 .env 中的 TEACHER_*）", "latency_ms": None}
    client = TeacherClient(cfg)
    t0 = time.time()
    content = client.chat([{"role": "user", "content": question}], max_tokens=max_new)
    return {"answer": content, "latency_ms": round((time.time() - t0) * 1000, 1)}


@app.post("/api/playground/ask")
async def playground_ask(body: dict):
    question = str(body.get("question", "")).strip()
    if not question:
        return JSONResponse({"error": "question required"}, status_code=422)
    model_dir = body.get("model_dir") or ""
    if not model_dir:
        for job in manager.list():
            if job["status"] == Status.DONE.value and job.get("result", {}).get("model_dir"):
                model_dir = job["result"]["model_dir"]
                break
    if not model_dir or not Path(model_dir).exists():
        return JSONResponse(
            {"error": "没有可用的蒸馏模型。请先运行一个蒸馏任务。"}, status_code=422
        )
    try:
        student = _student_answer(model_dir, question)
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"error": f"学生模型推理失败: {e}"}, status_code=500)
    teacher = _teacher_answer(question)
    return {"student": student, "teacher": teacher, "model_dir": model_dir}


def start_server(host: str = "127.0.0.1", port: int = 8080) -> None:
    import uvicorn

    print(f"* OneClick Distill server: http://{host}:{port}")
    print(f"* WebSocket: ws://{host}:{port}/ws/jobs/{{job_id}}")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import sys

    host = sys.argv[1] if len(sys.argv) > 1 else "127.0.0.1"
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 8080
    start_server(host, port)


@app.post("/api/ollama")
async def ollama_import(body: dict):
    """Generate an Ollama Modelfile for a GGUF and print the import command."""
    gguf = body.get("gguf", "")
    if not gguf or not Path(gguf).exists():
        return JSONResponse({"error": "GGUF 文件不存在"}, status_code=404)
    name = body.get("name") or Path(gguf).stem.replace("model-", "")
    modelfile = Path(gguf).with_suffix(".Modelfile")
    modelfile.write_text(f"FROM {gguf}\n", encoding="utf-8")
    return {"modelfile": str(modelfile), "command": f"ollama create {name} -f {modelfile}"}


# ---------------------------------------------------------------------------
# Local API node (llama.cpp llama-server, OpenAI-compatible /v1)
# ---------------------------------------------------------------------------
@app.get("/api/server")
async def server_status():
    from ..serve_model import list_servers

    return {"servers": list_servers()}


@app.post("/api/server/start")
async def server_start(body: dict):
    from ..serve_model import start_server

    gguf = body.get("gguf", "")
    if not gguf or not Path(gguf).exists():
        return JSONResponse({"error": "GGUF 文件不存在"}, status_code=404)
    try:
        info = start_server(gguf, port=int(body.get("port", 8123)), ctx_size=int(body.get("ctx_size", 2048)))
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"error": f"启动失败: {e}"}, status_code=500)
    return info


@app.post("/api/server/stop")
async def server_stop(body: dict):
    from ..serve_model import stop_server

    return stop_server(int(body.get("port", 8123)))


# ---------------------------------------------------------------------------
# Live system metrics (drives the desktop VRAM/CPU chart)
# ---------------------------------------------------------------------------
@app.get("/api/metrics")
async def metrics():
    try:
        import psutil

        mem = psutil.virtual_memory()
        cpu = psutil.cpu_percent(interval=None)
    except Exception:  # noqa: BLE001
        return {"cpu_percent": 0.0, "ram_used_gb": 0.0, "ram_total_gb": 0.0, "vram_used_gb": 0.0, "device": "cpu"}
    try:
        import torch

        vram_used = 0.0
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            vram_used = round(torch.cuda.memory_allocated(0) / 1024**3, 3)
    except Exception:  # noqa: BLE001
        vram_used, device = 0.0, "cpu"
    return {
        "cpu_percent": round(cpu, 1),
        "ram_used_gb": round(mem.used / 1024**3, 2),
        "ram_total_gb": round(mem.total / 1024**3, 1),
        "vram_used_gb": vram_used,
        "device": device,
        "time": time.time(),
    }


# ---------------------------------------------------------------------------
# A/B evaluation (latency + consistency), shared with CLI `eval`
# ---------------------------------------------------------------------------
@app.post("/api/eval")
async def run_eval(body: dict):
    from ..eval import evaluate

    student = body.get("student", "")
    teacher = body.get("teacher", "")
    questions = body.get("questions") or []
    if not student or not teacher:
        return JSONResponse({"error": "需要 student 与 teacher 后端"}, status_code=422)
    if not questions:
        return JSONResponse({"error": "需要 questions 列表"}, status_code=422)
    try:
        result = evaluate(student, teacher, list(questions), max_tokens=int(body.get("max_tokens", 128)))
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"error": f"评测失败: {e}"}, status_code=500)
    return result


# ---------------------------------------------------------------------------
# Export: (re-)quantize a completed job's model to GGUF
# ---------------------------------------------------------------------------
_QUANT_TYPES = {"f16", "q8_0", "q5_k_m", "q4_k_m", "q3_k_s"}


def _gguf_url(path: Path) -> str | None:
    try:
        rel = path.resolve().relative_to(Path.cwd().resolve())
        return "/" + str(rel).replace("\\", "/")
    except ValueError:
        return None


def _pick_done_job(job_id: str):
    if job_id:
        s = manager.get(job_id)
        if s and s.status == Status.DONE and s.result.get("model_dir"):
            return s
        return None
    for j in manager.list():  # newest first
        if j["status"] == Status.DONE.value and j.get("result", {}).get("model_dir"):
            return manager.get(j["id"])
    return None


@app.post("/api/export")
async def export_model(body: dict):
    from ..quantize.llama_cpp import quantize

    fmt = str(body.get("format", "gguf")).lower()
    if fmt != "gguf":
        return JSONResponse({"error": f"暂仅支持 GGUF 导出（收到 {fmt}）"}, status_code=422)
    q = str(body.get("quantization", "q4_k_m")).lower()
    if q not in _QUANT_TYPES:
        return JSONResponse({"error": f"不支持的量化格式: {q}"}, status_code=422)

    state = _pick_done_job(str(body.get("job_id", "")).strip())
    if not state:
        return JSONResponse(
            {"error": "没有可导出的已完成蒸馏任务，请先运行一次蒸馏"},
            status_code=404,
        )

    model_dir = Path(state.result["model_dir"])
    out_dir = model_dir.parent / "export"
    existing = sorted(out_dir.glob(f"model-{q}.gguf"))
    if existing:
        gguf = existing[-1]
    else:
        def _do():
            return quantize(model_dir, out_dir, quant_type=q.upper())

        try:
            gguf = await asyncio.to_thread(_do)
        except Exception as e:  # noqa: BLE001
            return JSONResponse({"error": f"导出失败: {e}"}, status_code=500)
        if not gguf:
            return JSONResponse(
                {"error": "量化工具不可用（llama.cpp 工具未就绪，可查看后端日志）"},
                status_code=500,
            )

    size_mb = round(gguf.stat().st_size / 1024**2, 1)
    return {"job_id": state.id, "gguf": str(gguf), "url": _gguf_url(gguf), "size_mb": size_mb}
