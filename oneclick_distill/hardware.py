"""Hardware probe: VRAM / RAM / disk detection and the '保命防爆' strategy.

Reads live free VRAM via torch.cuda.mem_get_info(), falls back to CPU-only
reporting when no NVIDIA GPU is present. Produces the exact hyper-parameter
strategy shown to the user in the GUI ("已检测到 ... 自动配置保命防爆模式").
"""

from __future__ import annotations

import os
import shutil
import sys
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Strategy:
    backend: str = "transformers"          # transformers | unsloth
    quantized: str = "none"                # none | 4bit
    optimizer: str = "adamw"               # adamw | paged_adamw_8bit
    batch_size: int = 1
    grad_accumulation_steps: int = 8
    gradient_checkpointing: bool = True
    max_seq_len: int = 512
    dtype: str = "fp32"
    notes: list[str] = field(default_factory=list)


def _torch() -> Any:
    import torch  # lazy: torch import is slow

    return torch


def probe() -> dict[str, Any]:
    """Return a hardware report dict."""
    total_ram = _total_ram_gb()
    free_ram = _free_ram_gb()
    device = "cpu"
    total_vram = 0.0
    free_vram = 0.0
    device_name = "CPU"

    try:
        torch = _torch()
        cuda_available = torch.cuda.is_available()
    except ImportError:
        # torch not installed (e.g. server-only deployment): report CPU honestly.
        torch = None
        cuda_available = False

    if torch is not None and cuda_available:
        device = "cuda"
        props = torch.cuda.get_device_properties(0)
        device_name = props.name
        total_vram = round(props.total_memory / 1024**3, 1)
        free_vram = round(torch.cuda.mem_get_info(0)[0] / 1024**3, 1)

    report = {
        "device": device,
        "device_name": device_name,
        "total_vram_gb": total_vram,
        "free_vram_gb": free_vram,
        "total_ram_gb": round(total_ram, 1),
        "free_ram_gb": round(free_ram, 1),
        "python": sys.version.split()[0],
        "torch": getattr(torch, "__version__", "not installed"),
        "cuda_available": cuda_available,
        "os": sys.platform,
        "disk_free_gb": round(_disk_free_gb(), 1),
    }
    report["strategy"] = choose_strategy(report).__dict__
    return report


def _total_ram_gb() -> float:
    if sys.platform == "win32":
        return _win_total_ram_gb()
    try:
        import psutil

        return psutil.virtual_memory().total / 1024**3
    except Exception:
        return 0.0


def _win_total_ram_gb() -> float:
    try:
        import ctypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        m = MEMORYSTATUSEX()
        m.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        return m.ullTotalPhys / 1024**3
    except Exception:
        return 0.0


def _free_ram_gb() -> float:
    if sys.platform != "win32":
        try:
            import psutil

            return psutil.virtual_memory().available / 1024**3
        except Exception:
            return 0.0
    try:
        import ctypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        m = MEMORYSTATUSEX()
        m.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        return m.ullAvailPhys / 1024**3
    except Exception:
        return 0.0


def _disk_free_gb() -> float:
    try:
        total, used, free = shutil.disk_usage(str(os.getcwd()))
        return free / 1024**3
    except Exception:
        return 0.0


def choose_strategy(report: dict[str, Any]) -> Strategy:
    """VRAM-aware strategy, mirroring the design doc's 保命防爆 rules."""
    device = report.get("device", "cpu")
    free_vram = float(report.get("free_vram_gb", 0.0))
    s = Strategy()

    if device == "cpu":
        s.backend = "transformers"
        s.quantized = "none"
        s.optimizer = "adamw"
        s.batch_size = 1
        s.grad_accumulation_steps = 8
        s.gradient_checkpointing = True
        s.max_seq_len = 512
        s.dtype = "fp32"
        s.notes.append("未检测到 NVIDIA 显卡，使用 CPU 保底后端（适合 1B 以下小模型）")
        return s

    # NVIDIA GPU present
    if free_vram >= 16:
        s.backend = "unsloth"
        s.quantized = "4bit"
        s.optimizer = "paged_adamw_8bit"
        s.batch_size = 2
        s.grad_accumulation_steps = 4
        s.max_seq_len = 1024
        s.dtype = "bf16"
        s.notes.append(f"检测到 {report['device_name']} 显存充足，启用 Unsloth 4-bit 极速模式")
    elif free_vram >= 8:
        s.backend = "transformers"
        s.quantized = "4bit"
        s.optimizer = "paged_adamw_8bit"
        s.batch_size = 1
        s.grad_accumulation_steps = 8
        s.max_seq_len = 512
        s.dtype = "bf16"
        s.notes.append(f"检测到 {report['device_name']}，启用 QLoRA 保命模式（batch=1, grad_accum=8）")
    else:
        s.backend = "transformers"
        s.quantized = "none"
        s.optimizer = "adamw"
        s.batch_size = 1
        s.grad_accumulation_steps = 8
        s.max_seq_len = 512
        s.dtype = "fp32"
        s.notes.append(f"检测到 {report['device_name']} 显存紧张，自动降级为 CPU 兼容配置（防爆模式）")
    s.notes.append("开启 gradient_checkpointing，节省约 60% 激活显存")
    return s
