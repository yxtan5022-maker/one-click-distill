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

    return server


def main() -> int:
    server = _build_server()
    server.run(transport="stdio")
    return 0


if __name__ == "__main__":
    sys.exit(main())
