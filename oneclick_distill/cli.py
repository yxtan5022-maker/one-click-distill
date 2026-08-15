"""OneClick Distill CLI — the agent-friendly entry point.

Designed to be invoked by humans AND by AI agents (Codex / OpenCode / etc.)
with plain arguments. JSON progress is emitted to stderr so machine output
stays on stdout.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .config import settings
from .schema import JobSpec, ProgressCallback, Stage, TeacherConfig

STAGE_LABEL = {
    Stage.PREPARE: "初始化",
    Stage.DATA: "数据",
    Stage.SYNTHETIC: "合成",
    Stage.TRAIN: "训练",
    Stage.QUANTIZE: "量化",
    Stage.DONE: "完成",
}


def _print_json(obj) -> None:
    print(json.dumps(obj, ensure_ascii=False))


def _progress_cb(state: dict) -> ProgressCallback:
    def cb(stage: Stage, progress: float, message: str, metrics: dict | None = None):
        state.update(stage=stage.value, progress=progress, message=message, metrics=metrics or {})
        _print_json(state)
        print(f"[{STAGE_LABEL.get(stage, stage.value)}] {message}", file=sys.stderr)

    return cb


def cmd_hardware(args) -> int:
    from .hardware import probe

    _print_json(probe())
    return 0


def cmd_doctor(args) -> int:
    from .hardware import probe

    checks = []
    try:
        report = probe()
        checks.append(("Python", f"{report['python']}", True))
        checks.append(("PyTorch", report["torch"], True))
        checks.append(("设备", report["device_name"], True))
        if report["device"] == "cuda":
            checks.append(("显存", f"{report['free_vram_gb']}GB 可用", True))
        checks.append(("磁盘空间", f"{report['disk_free_gb']}GB", report["disk_free_gb"] > 2))
        from .data import load_text_files  # noqa: F401

        checks.append(("数据模块", "OK", True))
        import fastapi  # noqa: F401

        checks.append(("FastAPI", fastapi.__version__, True))
        try:
            import torch

            torch_ok = True
        except Exception:
            torch_ok = False
        checks.append(("Torch 可用", "OK" if torch_ok else "缺失", torch_ok))
    except Exception as e:  # noqa: BLE001
        checks.append(("运行环境", f"异常: {e}", False))

    for name, value, ok in checks:
        print(f"{'✓' if ok else '✗'} {name}: {value}")
    if not all(ok for _, _, ok in checks):
        return 1
    return 0


def cmd_pipeline(args) -> int:
    from .runner import run_pipeline

    teacher = TeacherConfig(
        name=args.teacher,
        model=args.teacher_model,
        base_url=args.teacher_base_url,
        api_key=args.teacher_api_key,
    )
    if args.teacher == "none":
        teacher = TeacherConfig(name="none")
    spec = JobSpec(
        source=args.source,
        data_paths=args.data,
        teacher=teacher,
        model=args.model,
        size=args.size,
        max_steps=args.max_steps,
        smoke=args.smoke,
        dry_run=args.dry_run,
        quantize=args.quantize,
        out_dir=args.out_dir,
    )
    state: dict = {"status": "running", "stage": "prepare", "progress": 0.0, "message": "start"}
    try:
        result = run_pipeline(spec, _progress_cb(state))
        state.update(status="done", progress=1.0, message="蒸馏完成")
        _print_json(state)
        _print_json(result)
        return 0
    except Exception as e:  # noqa: BLE001
        state.update(status="failed", message=str(e), error=str(e))
        _print_json(state)
        print(f"失败: {e}", file=sys.stderr)
        return 1


def cmd_demo(args) -> int:
    args.teacher = "none"
    args.data = []
    args.teacher_model = ""
    args.teacher_base_url = ""
    args.teacher_api_key = ""
    args.smoke = True
    args.dry_run = False
    args.max_steps = args.max_steps or 5
    args.model = ""
    args.size = "smoke"
    args.quantize = False
    args.source = "cli"
    args.out_dir = ""
    return cmd_pipeline(args)


def cmd_serve(args) -> int:
    from .server.app import start_server

    start_server(host=args.host, port=args.port)
    return 0


def cmd_mcp(args) -> int:
    from .mcp.server import main as mcp_main

    return mcp_main()


def cmd_ollama(args) -> int:
    """Write an Ollama Modelfile and print the import command for a GGUF."""
    gguf = Path(args.gguf).resolve()
    if not gguf.exists():
        print(f"未找到 GGUF 文件: {gguf}", file=sys.stderr)
        return 1
    name = args.name or gguf.stem.replace("model-", "")
    modelfile = gguf.with_suffix(".Modelfile")
    modelfile.write_text(f"FROM {gguf}\n", encoding="utf-8")
    print(json.dumps({"modelfile": str(modelfile), "command": f"ollama create {name} -f {modelfile}"}))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="oneclick-distill",
        description="OneClick Distill — 开源一键模型蒸馏工具",
    )
    p.add_argument("--version", action="version", version=f"oneclick-distill {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    hw = sub.add_parser("hardware", help="硬件自检（显存/RAM/磁盘/策略）")
    hw.set_defaults(func=cmd_hardware)

    doc = sub.add_parser("doctor", help="运行环境自检")
    doc.set_defaults(func=cmd_doctor)

    pipe = sub.add_parser("pipeline", help="运行完整蒸馏流程")
    pipe.add_argument("--data", nargs="*", default=[], help="数据文件/目录（.txt/.md/.json/.jsonl/.pdf）")
    pipe.add_argument("--teacher", default="none", choices=["none", "deepseek", "openai", "local"],
                      help="教师模型（用于合成数据）")
    pipe.add_argument("--teacher-model", default="")
    pipe.add_argument("--teacher-base-url", default="")
    pipe.add_argument("--teacher-api-key", default="")
    pipe.add_argument("--model", default="", help="学生模型 HF ID（默认按规格选）")
    pipe.add_argument("--size", default="ultra", choices=["ultra", "balanced", "smoke"], help="目标规格")
    pipe.add_argument("--max-steps", type=int, default=None, help="最大训练步数（冒烟/试运行用）")
    pipe.add_argument("--smoke", action="store_true", help="冒烟模式：超小模型 + 少量步数")
    pipe.add_argument("--dry-run", action="store_true", help="试运行探路：1 步验证显存，不真正训练")
    pipe.add_argument("--no-quantize", action="store_true", help="跳过 GGUF 导出")
    pipe.add_argument("--out-dir", default="", help="输出目录")
    pipe.add_argument("--source", default="cli", help="任务来源标记（cli/mcp/ui）")
    pipe.set_defaults(func=cmd_pipeline)

    demo = sub.add_parser("demo", help="一键演示：内置数据跑通冒烟流程")
    demo.add_argument("--max-steps", type=int, default=None)
    demo.set_defaults(func=cmd_demo)

    serve = sub.add_parser("serve", help="启动 FastAPI + WebSocket 后端（含内置 Web UI）")
    serve.add_argument("--host", default=settings.host)
    serve.add_argument("--port", type=int, default=settings.port)
    serve.set_defaults(func=cmd_serve)

    mcp = sub.add_parser("mcp", help="启动 MCP stdio server（供 Codex/OpenCode 调用）")
    mcp.set_defaults(func=cmd_mcp)

    olla = sub.add_parser("ollama", help="为 GGUF 生成 Ollama 导入命令")
    olla.add_argument("--gguf", required=True)
    olla.add_argument("--name", default="")
    olla.set_defaults(func=cmd_ollama)

    return p


def main(argv: list[str] | None = None) -> int:
    if sys.platform == "win32":
        for stream in (sys.stdout, sys.stderr):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
            except Exception:
                pass
    settings.load_presets()
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
