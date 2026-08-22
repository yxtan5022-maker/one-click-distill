"""Local model serving via llama.cpp's llama-server (OpenAI-compatible /v1).

The distilled GGUF can be started as a local OpenAI-compatible API node with a
few clicks (or one CLI command), then consumed by any OpenAI client — which is
also what the A/B evaluator and the desktop shell use as their "teacher".

Registration is persisted to a state directory (one JSON file per port) so the
node survives the launching process: `list_servers` / `stop_server` work from
any process, exactly what the CLI daemon lifecycle requires.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

from .config import settings

STATE_DIR = Path(os.environ.get("OCD_STATE_DIR", Path.home() / ".oneclick-distill" / "nodes"))


def _ensure_llama_server() -> Path:
    """Make sure llama-server exists next to the other llama.cpp tools."""
    from .quantize.llama_cpp import ensure_tools

    tools = settings.tools_dir
    tools.mkdir(parents=True, exist_ok=True)
    exe = tools / ("llama-server.exe" if sys.platform == "win32" else "llama-server")
    if exe.exists():
        return exe
    ensure_tools()  # extracts the full win-cpu-x64 zip, including llama-server.exe
    if not exe.exists():
        raise RuntimeError("llama-server 不在工具目录，且自动下载失败")
    return exe


def _health(base_url: str, timeout: float = 3.0) -> bool:
    try:
        with urllib.request.urlopen(base_url + "/health", timeout=timeout) as resp:
            return resp.status == 200
    except Exception:  # noqa: BLE001
        return False


def _alive(pid: int) -> bool:
    if sys.platform == "win32":
        try:
            subprocess.run(["tasklist", "/FI", f"PID eq {pid}"], capture_output=True, timeout=5)
        except Exception:  # noqa: BLE001
            return False
        try:
            out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}"], capture_output=True, timeout=5)
            return str(pid) in out.stdout.decode(errors="replace")
        except Exception:  # noqa: BLE001
            return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _state_file(port: int) -> Path:
    return STATE_DIR / f"{port}.json"


def _load(port: int) -> dict[str, Any] | None:
    p = _state_file(port)
    if not p.exists():
        return None
    try:
        info = json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None
    if not _alive(info.get("pid", -1)):
        p.unlink(missing_ok=True)
        return None
    return info


def list_servers() -> list[dict[str, Any]]:
    """Live snapshot of running local API nodes (across processes)."""
    out = []
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    for f in sorted(STATE_DIR.glob("*.json")):
        if not f.stem.isdigit():
            continue
        info = _load(int(f.stem))
        if not info:
            continue
        out.append(
            {
                "port": info["port"],
                "base_url": info["base_url"],
                "gguf": info["gguf"],
                "pid": info["pid"],
                "started_at": info["started_at"],
                "healthy": _health(info["base_url"]),
            }
        )
    return out


def start_server(
    gguf: str | Path,
    port: int = 8123,
    host: str = "127.0.0.1",
    ctx_size: int = 2048,
    wait_health: bool = True,
) -> dict[str, Any]:
    """Start a llama-server on <gguf>. Returns status JSON. Raises on failure."""
    gguf = Path(gguf).resolve()
    if not gguf.exists():
        raise FileNotFoundError(f"GGUF 文件不存在: {gguf}")
    if _load(port):
        raise RuntimeError(f"端口 {port} 已有节点在运行")

    base_url = f"http://{host}:{port}"
    if _health(base_url):
        raise RuntimeError(f"{base_url} 已被占用（已有服务在监听）")

    exe = _ensure_llama_server()
    args = [
        str(exe),
        "-m", str(gguf),
        "--host", host,
        "--port", str(port),
        "--ctx-size", str(ctx_size),
    ]
    proc = subprocess.Popen(
        args,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        cwd=str(settings.tools_dir),
    )
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    _state_file(port).write_text(
        json.dumps(
            {"port": port, "base_url": base_url, "gguf": str(gguf), "pid": proc.pid, "started_at": time.time()}
        ),
        encoding="utf-8",
    )

    if wait_health:
        deadline = time.time() + 90
        while time.time() < deadline:
            if proc.poll() is not None:
                _state_file(port).unlink(missing_ok=True)
                raise RuntimeError(f"llama-server 提前退出，退出码 {proc.returncode}")
            if _health(base_url):
                return {
                    "port": port,
                    "base_url": base_url,
                    "gguf": str(gguf),
                    "pid": proc.pid,
                    "started_at": _load(port)["started_at"],
                    "healthy": True,
                }
            time.sleep(0.5)
        _state_file(port).unlink(missing_ok=True)
        proc.kill()
        raise RuntimeError(f"等待 llama-server 健康检查超时（{base_url}/health）")
    return {"port": port, "base_url": base_url, "pid": proc.pid}


def stop_server(port: int) -> dict[str, Any]:
    info = _load(port)
    if not info:
        return {"port": port, "stopped": False, "message": "该端口没有本地节点"}
    pid = info["pid"]
    _state_file(port).unlink(missing_ok=True)
    if sys.platform == "win32":
        try:
            subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, timeout=15)
        except Exception:  # noqa: BLE001
            pass
    else:
        try:
            os.kill(pid, 15)
        except OSError:
            pass
    return {"port": port, "stopped": True, "pid": pid}
