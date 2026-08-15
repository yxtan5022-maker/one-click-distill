"""Teacher synthesis: generate Q&A training pairs from source chunks via an
OpenAI-compatible chat completions API (DeepSeek / OpenAI / local Ollama)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import requests

from .schema import ProgressCallback, Stage, TeacherConfig

SYNTH_PROMPT = """你是数据合成助手。请根据下面提供的参考资料，生成 {n} 条高质量的中文知识问答对（Q&A），用于训练一个小型语言模型。

要求：
- 严格基于参考资料内容，不要编造。
- 每条包含 "prompt"（问题）和 "response"（答案）两个字段。
- 答案要完整、准确、自包含，不超过 200 字。
- 问题应覆盖不同角度与难度。

参考资料：
{context}

只输出 JSON 数组，例如：
[{{"prompt": "...", "response": "..."}}]
"""


class TeacherError(Exception):
    pass


class TeacherClient:
    def __init__(self, cfg: TeacherConfig):
        if not cfg.base_url or not cfg.model:
            raise TeacherError("教师模型未配置（需要 base_url 和 model）")
        self.base_url = cfg.base_url.rstrip("/")
        self.model = cfg.model
        self.api_key = cfg.api_key

    def chat(self, messages: list[dict], max_tokens: int = 512) -> str:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.7,
        }
        try:
            resp = requests.post(f"{self.base_url}/chat/completions", headers=headers, json=payload, timeout=120)
        except requests.RequestException as e:
            raise TeacherError(f"请求教师模型失败: {e}") from e
        if resp.status_code != 200:
            raise TeacherError(f"教师模型返回 {resp.status_code}: {resp.text[:300]}")
        data = resp.json()
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            raise TeacherError(f"教师模型响应格式异常: {data}") from e


def synthesize(
    chunks: list[str],
    cfg: TeacherConfig,
    out_path: Path,
    pairs_per_chunk: int = 2,
    max_chunks: int = 50,
    max_tokens: int = 512,
    progress: ProgressCallback | None = None,
    dry_run: bool = False,
) -> list[dict]:
    """Generate Q&A pairs from chunks and write them as JSONL."""
    if not chunks:
        raise TeacherError("没有可用于合成的数据分片")
    if dry_run:
        pairs = [{"prompt": "示例问题", "response": "示例答案"}]
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            for p in pairs:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
        return pairs

    client = TeacherClient(cfg)
    all_pairs: list[dict] = []
    total = min(len(chunks), max_chunks)
    for i, chunk in enumerate(chunks[:total]):
        if progress:
            progress(Stage.SYNTHETIC, i / total, f"合成数据 {i + 1}/{total}")
        prompt = SYNTH_PROMPT.format(n=pairs_per_chunk, context=chunk[:3000])
        content = client.chat([{"role": "user", "content": prompt}], max_tokens=max_tokens)
        pairs = _parse_json_array(content)
        all_pairs.extend(pairs[:pairs_per_chunk])

    if not all_pairs:
        raise TeacherError("教师模型未产出任何有效问答对")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for p in all_pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    if progress:
        progress(Stage.SYNTHETIC, 1.0, f"合成完成，共 {len(all_pairs)} 条问答对", {"pairs": len(all_pairs)})
    return all_pairs


def _parse_json_array(content: str) -> list[dict]:
    content = content.strip()
    content = re.sub(r"```(?:json)?", "", content).strip()
    start = content.find("[")
    end = content.rfind("]")
    if start == -1 or end == -1 or end <= start:
        raise TeacherError("教师模型未返回 JSON 数组")
    try:
        data = json.loads(content[start : end + 1])
    except json.JSONDecodeError as e:
        raise TeacherError(f"教师模型返回非法 JSON: {e}") from e
    pairs = []
    for item in data:
        if isinstance(item, dict) and "prompt" in item and "response" in item:
            pairs.append({"prompt": str(item["prompt"]), "response": str(item["response"])})
    return pairs
