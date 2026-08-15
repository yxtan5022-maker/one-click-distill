"""MCP (Model Context Protocol) stdio server.

Lets AI agents (Codex, OpenCode, OpenManus, ...) drive OneClick Distill:
  - probe hardware
  - start a distillation pipeline
  - poll job status

Run with:  oneclick-distill mcp
"""

from __future__ import annotations

import sys


def _build_server():
    try:
        from mcp.server import MCPServer
    except ImportError as e:
        raise SystemExit(
            "MCP server 需要 mcp 包（>=2.0）：pip install 'mcp>=2.0'\n"
            f"(原始错误: {e})"
        ) from e

    server = MCPServer(
        name="oneclick-distill",
        title="OneClick Distill",
        version="0.1.0",
    )

    @server.tool(name="hardware", description="检测本机硬件并给出推荐的蒸馏策略（显存/RAM/磁盘/防爆配置）。")
    def hardware() -> dict:
        from ..hardware import probe

        return probe()

    @server.tool(
        name="start_pipeline",
        description=(
            "启动一个模型蒸馏任务。data_paths 为数据文件/目录列表；"
            "teacher 可选 deepseek/openai/local（用于合成数据）；size 为 ultra/balanced；"
            "smoke=True 用超小模型快速验证流程。返回 job id，可轮询 pipeline_status。"
        ),
    )
    def start_pipeline(
        data_paths: list[str],
        teacher: str = "none",
        model: str = "",
        size: str = "ultra",
        max_steps: int | None = None,
        smoke: bool = False,
        quantize: bool = True,
    ) -> dict:
        from ..schema import JobSpec
        from ..server.manager import manager

        spec = {
            "source": "mcp",
            "data_paths": list(data_paths or []),
            "teacher": {"name": teacher, "model": model, "base_url": "", "api_key": ""},
            "model": model,
            "size": size,
            "max_steps": max_steps,
            "smoke": bool(smoke),
            "dry_run": False,
            "quantize": bool(quantize),
            "out_dir": "",
        }
        state = manager.submit(JobSpec.from_dict(spec))
        return state.to_dict()

    @server.tool(name="pipeline_status", description="查询任务状态：status/stage/progress/message/result。")
    def pipeline_status(job_id: str) -> dict:
        from ..server.manager import manager

        state = manager.get(job_id)
        if not state:
            return {"error": f"job {job_id} not found"}
        return state.to_dict()

    @server.tool(
        name="local_server_start",
        description="把 GGUF 启动成本地 OpenAI 兼容 API 节点（llama.cpp llama-server），返回 base_url。",
    )
    def local_server_start(gguf: str, port: int = 8123, ctx_size: int = 2048) -> dict:
        from ..serve_model import start_server

        try:
            return start_server(gguf, port=port, ctx_size=ctx_size)
        except Exception as e:  # noqa: BLE001
            return {"error": str(e)}

    @server.tool(name="local_server_stop", description="停止指定端口的本地 API 节点。")
    def local_server_stop(port: int = 8123) -> dict:
        from ..serve_model import stop_server

        return stop_server(port)

    @server.tool(name="local_server_status", description="列出运行中的本地 API 节点（跨进程）。")
    def local_server_status() -> dict:
        from ..serve_model import list_servers

        return {"servers": list_servers()}

    @server.tool(
        name="evaluate",
        description=(
            "A/B 评测教师 vs 学生：跑同一组问题，输出时延分布（avg/p95）、吞吐和回答一致性"
            "（exact_match + ROUGE-L F1）。student/teacher 语法：transformers:<目录> 或 openai:<url>#<模型>。"
        ),
    )
    def evaluate(student: str, teacher: str, questions: list[str], max_tokens: int = 128) -> dict:
        from ..eval import evaluate as _evaluate

        try:
            return _evaluate(student, teacher, list(questions), max_tokens=max_tokens)
        except Exception as e:  # noqa: BLE001
            return {"error": str(e)}

    return server


def main() -> int:
    server = _build_server()
    server.run(transport="stdio")
    return 0


if __name__ == "__main__":
    sys.exit(main())
