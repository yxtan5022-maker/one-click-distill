"""A/B evaluation: latency + answer-consistency between teacher and student.

Runs the same question set through both backends and reports hard numbers:
latency distribution (avg/p95), throughput (tokens per second) and answer
consistency (exact-match rate + ROUGE-L F1 via LCS).

Backend syntax:
  transformers:<model_dir>            — local HF model dir (the distilled student)
  openai:<base_url>#<model>[:<key>]   — any OpenAI-compatible /v1 endpoint
                                        (llama-server, Ollama, remote API…)
"""

from __future__ import annotations

import json
import re
import statistics
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_QUESTIONS = [
    "用一句话解释什么是知识蒸馏",
    "LoRA 和全量微调有什么区别",
    "什么是量化，为什么要量化模型",
    "在 CPU 上训练大模型需要注意什么",
    "开源模型和闭源模型的取舍是什么",
]


def normalize(text: str) -> str:
    t = text.lower().strip()
    t = re.sub(r"[\s\u3000]+", " ", t)
    t = re.sub(r"[^\w\u4e00-\u9fff%\.\-]", "", t)
    return t


def tokenize(text: str) -> list[str]:
    """CJK chars as single tokens, latin words kept whole — enables meaningful
    ROUGE-L / overlap scoring for mixed Chinese-English answers."""
    t = normalize(text)
    return re.findall(r"[\u4e00-\u9fff]|[a-z0-9%\.\-]+", t)


def exact_match(a: str, b: str) -> bool:
    na, nb = normalize(a), normalize(b)
    return bool(na) and na == nb


def _lcs_len(a: list[str], b: list[str]) -> int:
    m, n = len(a), len(b)
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        cur = [0] * (n + 1)
        ai = a[i - 1]
        for j in range(1, n + 1):
            if ai == b[j - 1]:
                cur[j] = prev[j - 1] + 1
            else:
                cur[j] = max(prev[j], cur[j - 1])
        prev = cur
    return prev[n]


def rouge_l_f1(a: str, b: str) -> float:
    """ROUGE-L F1 computed on character/word tokens (LCS based)."""
    A = tokenize(a)
    B = tokenize(b)
    if not A or not B:
        return 0.0
    l = _lcs_len(A, B)
    prec = l / len(B)
    rec = l / len(A)
    if prec + rec == 0:
        return 0.0
    return round(2 * prec * rec / (prec + rec), 4)


@dataclass
class Backend:
    kind: str            # transformers | openai
    model: str = ""      # HF dir or model name for the /v1 API
    base_url: str = ""   # for openai
    api_key: str = ""

    @classmethod
    def parse(cls, spec: str) -> "Backend":
        if spec.startswith("transformers:"):
            return cls(kind="transformers", model=spec.split(":", 1)[1].strip())
        if spec.startswith("openai:"):
            rest = spec.split(":", 1)[1].strip()
            if "#" in rest:
                base_url, model = rest.split("#", 1)
                model_key = model
                key = ""
                if ":" in model:
                    model_key, key = model.split(":", 1)
            else:
                base_url, model_key, key = rest, "default", ""
            return cls(kind="openai", base_url=base_url.rstrip("/"), model=model_key, api_key=key)
        raise ValueError(f"未知后端: {spec}（支持 transformers:<目录> 或 openai:<url>#<模型>）")


def call_openai(base_url: str, model: str, question: str, max_tokens: int, api_key: str = "") -> dict[str, Any]:
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": question}],
            "max_tokens": max_tokens,
            "temperature": 0.0,
        }
    ).encode()
    req = urllib.request.Request(
        base_url + "/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=180) as resp:
        data = json.load(resp)
    latency = (time.time() - t0) * 1000
    content = data["choices"][0]["message"]["content"].strip()
    tokens = data.get("usage", {}).get("completion_tokens", 0) or 0
    return {"answer": content, "latency_ms": round(latency, 1), "tokens": tokens}


def call_transformers(model_dir: str, question: str, max_tokens: int) -> dict[str, Any]:
    import os

    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_dir)
    model.eval()
    prompt = f"问题：{question}\n回答："
    inputs = tok(prompt, return_tensors="pt")
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_tokens, do_sample=False,
        )
    latency = (time.time() - t0) * 1000
    answer = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    tokens = int(out.shape[1]) - int(inputs["input_ids"].shape[1])
    return {"answer": answer, "latency_ms": round(latency, 1), "tokens": tokens}


def run_one(backend: Backend, question: str, max_tokens: int) -> dict[str, Any]:
    if backend.kind == "openai":
        return call_openai(backend.base_url, backend.model, question, max_tokens, backend.api_key)
    return call_transformers(backend.model, question, max_tokens)


def evaluate(
    student_spec: str,
    teacher_spec: str,
    questions: list[str],
    max_tokens: int = 128,
) -> dict[str, Any]:
    student = Backend.parse(student_spec)
    teacher = Backend.parse(teacher_spec)
    samples = []
    student_lat, teacher_lat, student_tok = [], [], []
    for q in questions:
        s = run_one(student, q, max_tokens)
        t = run_one(teacher, q, max_tokens)
        samples.append(
            {
                "question": q,
                "student_answer": s["answer"],
                "teacher_answer": t["answer"],
                "student_latency_ms": s["latency_ms"],
                "teacher_latency_ms": t["latency_ms"],
                "student_tokens": s.get("tokens", 0),
                "rouge_l_f1": rouge_l_f1(t["answer"], s["answer"]),
                "exact_match": exact_match(t["answer"], s["answer"]),
            }
        )
        student_lat.append(s["latency_ms"])
        teacher_lat.append(t["latency_ms"])
        student_tok.append(s.get("tokens", 0))

    def dist(vals: list[float]) -> dict[str, float]:
        if not vals:
            return {"avg_ms": 0.0, "p95_ms": 0.0, "min_ms": 0.0}
        s = sorted(vals)
        return {
            "avg_ms": round(statistics.mean(vals), 1),
            "p95_ms": round(s[min(len(s) - 1, int(0.95 * (len(s) - 1)))], 1),
            "min_ms": round(s[0], 1),
        }

    em = sum(1 for x in samples if x["exact_match"]) / len(samples)
    rl = statistics.mean(x["rouge_l_f1"] for x in samples)
    speedups = []
    for x in samples:
        if x["teacher_latency_ms"] and x["student_latency_ms"]:
            speedups.append(x["teacher_latency_ms"] / x["student_latency_ms"])
    return {
        "n_questions": len(questions),
        "student": {"backend": student_spec, **dist(student_lat),
                    "tokens_total": sum(student_tok),
                    "tokens_per_sec": round(sum(student_tok) / (sum(student_lat) / 1000), 2) if student_lat and sum(student_lat) else 0.0},
        "teacher": {"backend": teacher_spec, **dist(teacher_lat)},
        "consistency": {
            "exact_match_rate": round(em, 4),
            "rouge_l_f1": round(rl, 4),
        },
        "relative_speedup_x": round(statistics.mean(speedups), 2) if speedups else 0.0,
        "samples": samples,
    }


def load_questions(source: str, default: list[str]) -> list[str]:
    """Load questions from N (auto-generate count), a JSONL file, or text lines."""
    if not source:
        return default
    if source.isdigit():
        n = int(source)
        return DEFAULT_QUESTIONS[:n] if n <= len(DEFAULT_QUESTIONS) else DEFAULT_QUESTIONS + [
            f"自定义问题 {i}" for i in range(1, n - len(DEFAULT_QUESTIONS) + 1)
        ]
    p = Path(source)
    if p.exists():
        if p.suffix == ".jsonl":
            rows = [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
            return [r.get("question") or r.get("instruction") for r in rows]
        return [l.strip() for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    return [source]
