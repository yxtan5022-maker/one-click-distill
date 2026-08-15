"""GGUF export via llama.cpp's official tooling.

Flow:  HF model dir --convert_hf_to_gguf.py--> base.gguf --llama-quantize--> quantized.gguf

The two tools (convert script + prebuilt llama-quantize.exe) are auto-downloaded
into oneclick_distill/tools/ on first use. If the download fails the stage is
skipped gracefully (the trained HF model is still produced).
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

from ..config import settings
from ..schema import ProgressCallback, Stage

LLAMA_CPP_PIN = "b6067"  # pinned llama.cpp tag for reproducibility
RELEASE_API = f"https://api.github.com/repos/ggerganov/llama.cpp/releases/tags/{LLAMA_CPP_PIN}"
CONVERT_URL = f"https://raw.githubusercontent.com/ggerganov/llama.cpp/{LLAMA_CPP_PIN}/convert_hf_to_gguf.py"


class QuantizeError(Exception):
    pass


def _tools_dir() -> Path:
    tools = settings.tools_dir
    tools.mkdir(parents=True, exist_ok=True)
    return tools


def _download(url: str, dest: Path, timeout: int = 300):
    req = urllib.request.Request(url, headers={"User-Agent": "oneclick-distill/0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as resp, dest.open("wb") as f:
        shutil.copyfileobj(resp, f)


def ensure_tools(progress: ProgressCallback | None = None) -> tuple[Path | None, Path | None]:
    """Ensure convert script and quantize binary exist. Returns (script, binary)."""
    tools = _tools_dir()
    script = tools / "convert_hf_to_gguf.py"
    binary = tools / ("llama-quantize.exe" if sys.platform == "win32" else "llama-quantize")
    binary = settings.llama_quantize if settings.llama_quantize and settings.llama_quantize.exists() else binary

    if script.exists() and binary.exists():
        return script, binary

    if progress:
        progress(Stage.QUANTIZE, 0.0, "下载 llama.cpp 量化工具（首次使用需要网络）")
    try:
        if not script.exists():
            _download(CONVERT_URL, script)
        if not binary.exists():
            _download_release_binary(binary)
    except Exception as e:  # noqa: BLE001
        if progress:
            progress(Stage.QUANTIZE, 0.0, f"量化工具下载失败，跳过 GGUF 导出: {e}")
        return None, None
    return script, binary


def _download_release_binary(dest: Path):
    import json as _json

    tools = _tools_dir()
    url = None
    with urllib.request.urlopen(RELEASE_API, timeout=60) as resp:
        data = _json.load(resp)
    for asset in data.get("assets", []):
        name = asset.get("name", "")
        if "win-cpu-x64" in name and name.endswith(".zip"):
            url = asset["browser_download_url"]
            break
    if not url:
        raise QuantizeError("未在 llama.cpp 发布中找到 Windows CPU 工具包")
    zip_path = dest.with_suffix(".zip")
    _download(url, zip_path, timeout=600)
    try:
        # extract everything so DLLs (llama.dll etc.) sit next to the exe
        with zipfile.ZipFile(zip_path) as z:
            for member in z.namelist():
                if member.endswith("/"):
                    continue
                if member.endswith("llama-quantize.exe"):
                    target = dest
                else:
                    target = tools / member.rsplit("/", 1)[-1]
                with z.open(member) as src, target.open("wb") as f:
                    shutil.copyfileobj(src, f)
            if not dest.exists():
                raise QuantizeError("zip 内未找到 llama-quantize.exe")
    finally:
        zip_path.unlink(missing_ok=True)


def quantize(
    model_dir: Path,
    out_dir: Path,
    quant_type: str = "Q4_K_M",
    progress: ProgressCallback | None = None,
) -> Path | None:
    """Convert a HF model directory to a quantized GGUF file. Returns path or None."""
    script, binary = ensure_tools(progress)
    if not script or not binary:
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    base_gguf = out_dir / "model-f16.gguf"
    out_gguf = out_dir / f"model-{quant_type.lower()}.gguf"
    python = sys.executable

    if progress:
        progress(Stage.QUANTIZE, 0.2, "转换 HF → GGUF（f16）")
    try:
        subprocess.run(
            [python, str(script), str(model_dir), "--outfile", str(base_gguf), "--outtype", "f16"],
            check=True,
            capture_output=True,
            timeout=1800,
        )
    except subprocess.CalledProcessError as e:
        raise QuantizeError(f"convert_hf_to_gguf 失败: {e.stderr.decode(errors='replace')[-500:]}") from e
    except subprocess.TimeoutExpired as e:
        raise QuantizeError("GGUF 转换超时") from e

    if progress:
        progress(Stage.QUANTIZE, 0.7, f"量化到 {quant_type}")
    try:
        subprocess.run(
            [str(binary), str(base_gguf), str(out_gguf), quant_type],
            check=True,
            capture_output=True,
            timeout=1800,
        )
    except subprocess.CalledProcessError as e:
        out = e.stdout.decode(errors="replace")[-500:]
        err = e.stderr.decode(errors="replace")[-500:]
        detail = (out or err or "未知错误").strip()
        raise QuantizeError(f"llama-quantize 失败: {detail}") from e
    except subprocess.TimeoutExpired as e:
        raise QuantizeError("量化超时") from e

    if progress:
        size_mb = round(out_gguf.stat().st_size / 1024**2, 1)
        progress(Stage.QUANTIZE, 1.0, f"量化完成：{out_gguf.name}（{size_mb} MB）", {"size_mb": size_mb})
    return out_gguf
