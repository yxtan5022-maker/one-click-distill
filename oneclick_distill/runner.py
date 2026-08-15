"""Pipeline runner: orchestrates data → synthetic → train → quantize.

Given a JobSpec, executes the full distillation pipeline and reports progress
through a ProgressCallback (used by CLI, REST/WS server and MCP alike).
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

from .config import SAMPLE_DATA_PATH, settings
from .data import chunk_all, load_text_files
from .hardware import choose_strategy, probe
from .quantize import quantize
from .schema import JobSpec, ProgressCallback, Stage, TeacherConfig
from .train import train as train_model
from .teacher import TeacherError, synthesize


class PipelineError(Exception):
    pass


class Pipeline:
    def __init__(self, spec: JobSpec, progress: ProgressCallback | None = None):
        self.spec = spec
        self.progress = progress
        self.timings: dict[str, float] = {}
        self.result: dict = {}

    def run(self) -> dict:
        spec = self.spec
        out_dir = Path(spec.out_dir or f"runs/{spec.id}")
        out_dir.mkdir(parents=True, exist_ok=True)

        # ---- prepare --------------------------------------------------
        t0 = time.time()
        self._emit(Stage.PREPARE, 0.0, "初始化")
        spec.model = settings.resolve_model(spec.model, spec.size)
        report = probe()
        strategy = choose_strategy(report)
        self._emit(
            Stage.PREPARE, 1.0, "硬件自检完成：" + "；".join(strategy.notes),
            {"hardware": {k: report[k] for k in ("device", "device_name", "free_vram_gb", "free_ram_gb")},
             "strategy": strategy.__dict__},
        )
        self.timings["prepare"] = time.time() - t0

        # ---- data -----------------------------------------------------
        t1 = time.time()
        self._emit(Stage.DATA, 0.0, "导入并清洗数据")
        data_paths = spec.data_paths or [str(SAMPLE_DATA_PATH)]
        try:
            texts = load_text_files(data_paths)
        except Exception as e:  # DataLoadError
            raise PipelineError(str(e)) from e
        chunks = chunk_all(texts)
        self._emit(Stage.DATA, 0.6, f"数据清洗完成：{len(texts)} 段 → {len(chunks)} 个分片", {"chunks": len(chunks)})

        # ---- synthetic ------------------------------------------------
        jsonl_path = out_dir / "train.jsonl"
        if spec.teacher.name != "none":
            self._emit(Stage.SYNTHETIC, 0.0, f"调用教师模型 {spec.teacher.name} 合成训练数据")
            try:
                synthesize(
                    chunks,
                    spec.teacher,
                    jsonl_path,
                    pairs_per_chunk=2,
                    progress=self.progress,
                    dry_run=spec.dry_run,
                )
            except TeacherError as e:
                raise PipelineError(f"数据合成失败: {e}") from e
        else:
            pairs = self._existing_pairs(data_paths, chunks)
            if not pairs:
                raise PipelineError(
                    "未配置教师模型且数据中没有 {prompt,response} 问答对。"
                    "请配置教师模型（--teacher deepseek --api-key ...），或提供 Q&A 格式的 JSON/JSONL 数据。"
                )
            jsonl_path.write_text(
                "\n".join(json.dumps(p, ensure_ascii=False) for p in pairs), encoding="utf-8"
            )
            self._emit(Stage.SYNTHETIC, 1.0, f"使用现有问答数据：{len(pairs)} 条", {"pairs": len(pairs)})
        self.timings["data"] = time.time() - t1

        # ---- train ----------------------------------------------------
        t2 = time.time()
        self._emit(Stage.TRAIN, 0.0, f"开始微调（{spec.model}，规格 {spec.size}）")
        try:
            metrics = train_model(spec, jsonl_path, out_dir / "model", strategy, self.progress)
        except Exception as e:
            if "out of memory" in str(e).lower():
                raise PipelineError(
                    f"训练触发显存不足（OOM）。{str(e)}。建议：换更小规格（极速版）或换更大显存的机器。"
                ) from e
            raise PipelineError(f"训练失败: {e}") from e
        self.timings["train"] = time.time() - t2

        # ---- quantize ---------------------------------------------------
        gguf_path = None
        if spec.quantize:
            t3 = time.time()
            self._emit(Stage.QUANTIZE, 0.0, "导出 GGUF")
            try:
                gguf_path = quantize(out_dir / "model", out_dir / "export", progress=self.progress)
            except Exception as e:  # QuantizeError etc.
                self._emit(Stage.QUANTIZE, 0.0, f"GGUF 导出跳过：{e}")
            if gguf_path:
                self.timings["quantize"] = time.time() - t3

        # ---- done -------------------------------------------------------
        gguf_url = None
        if gguf_path:
            try:
                rel = gguf_path.resolve().relative_to(Path.cwd().resolve())
                if str(rel).startswith("runs"):
                    gguf_url = "/" + str(rel).replace("\\", "/")
            except ValueError:
                pass
        self.result = {
            "id": spec.id,
            "model": spec.model,
            "size": spec.size,
            "model_dir": str(out_dir / "model"),
            "train_data": str(jsonl_path),
            "gguf": str(gguf_path) if gguf_path else None,
            "gguf_url": gguf_url,
            "metrics": metrics,
            "timings": {k: round(v, 1) for k, v in self.timings.items()},
            "hardware": report.get("device", ""),
            "strategy_notes": strategy.notes,
        }
        self._emit(Stage.DONE, 1.0, "蒸馏完成 ✓", {"result": self.result})
        return self.result

    # ---- helpers --------------------------------------------------------
    def _emit(self, stage: Stage, progress: float, message: str, metrics: dict | None = None):
        if self.progress:
            self.progress(stage, progress, message, metrics or {})

    def _existing_pairs(self, data_paths: list[str], chunks: list[str]) -> list[dict]:
        pairs: list[dict] = []
        for p in data_paths:
            path = Path(p)
            if path.is_dir():
                continue
            if path.suffix.lower() not in {".json", ".jsonl"}:
                continue
            try:
                if path.suffix.lower() == ".jsonl":
                    for line in path.read_text(encoding="utf-8").splitlines():
                        try:
                            obj = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if isinstance(obj, dict) and "prompt" in obj and "response" in obj:
                            pairs.append(obj)
                else:
                    data = json.loads(path.read_text(encoding="utf-8"))
                    items = data if isinstance(data, list) else [data]
                    for obj in items:
                        if isinstance(obj, dict) and "prompt" in obj and "response" in obj:
                            pairs.append(obj)
            except Exception:
                continue
        return pairs


def run_pipeline(spec: JobSpec, progress: ProgressCallback | None = None) -> dict:
    return Pipeline(spec, progress).run()
