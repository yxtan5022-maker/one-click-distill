"""Data ingestion: load and clean TXT / Markdown / JSON / JSONL / PDF files."""

from __future__ import annotations

import json
from pathlib import Path

from ..config import PDF_EXTENSIONS, SUPPORTED_DATA_EXTENSIONS


class DataLoadError(Exception):
    pass


def load_text_files(paths: list[str]) -> list[str]:
    """Load one or more files (or directories) into a list of text chunks."""
    files = _expand(paths)
    if not files:
        raise DataLoadError("没有找到可导入的数据文件（支持 .txt/.md/.json/.jsonl/.pdf）")
    chunks: list[str] = []
    for f in files:
        chunks.extend(_read_file(f))
    cleaned = [_clean(t) for t in chunks]
    cleaned = [c for c in cleaned if c]
    if not cleaned:
        raise DataLoadError("数据清洗后为空，请检查输入内容")
    return cleaned


def _expand(paths: list[str]) -> list[Path]:
    out: list[Path] = []
    for p in paths:
        path = Path(p)
        if path.is_dir():
            for f in sorted(path.rglob("*")):
                if f.is_file() and f.suffix.lower() in SUPPORTED_DATA_EXTENSIONS | PDF_EXTENSIONS:
                    out.append(f)
        elif path.is_file():
            out.append(path)
    return out


def _read_file(path: Path) -> list[str]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _read_pdf(path)
    text = path.read_text(encoding="utf-8", errors="replace")
    if suffix in {".json", ".jsonl"}:
        return _json_to_texts(text, suffix)
    return _split_markdown(text)


def _json_to_texts(text: str, suffix: str) -> list[str]:
    out: list[str] = []
    if suffix == ".jsonl":
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                out.append(_obj_to_text(obj))
            except json.JSONDecodeError:
                out.append(line)
    else:
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return [text]
        if isinstance(data, list):
            for obj in data:
                out.append(_obj_to_text(obj))
        else:
            out.append(_obj_to_text(data))
    return [o for o in out if o]


def _obj_to_text(obj) -> str:
    if isinstance(obj, str):
        return obj
    if isinstance(obj, dict):
        # prefer common Q&A fields
        if "prompt" in obj and "response" in obj:
            return f"Q: {obj['prompt']}\nA: {obj['response']}"
        if "question" in obj and "answer" in obj:
            return f"Q: {obj['question']}\nA: {obj['answer']}"
        if "text" in obj:
            return str(obj["text"])
        parts = [f"{k}: {v}" for k, v in obj.items() if isinstance(v, (str, int, float))]
        return "\n".join(parts)
    return str(obj)


def _read_pdf(path: Path) -> list[str]:
    try:
        from pypdf import PdfReader
    except ImportError:
        raise DataLoadError(
            "读取 PDF 需要 pypdf：pip install pypdf（或改用 .txt/.md/.json 文件）"
        )
    reader = PdfReader(str(path))
    return [page.extract_text() or "" for page in reader.pages]


def _split_markdown(text: str) -> list[str]:
    """Split long text into paragraphs/heading blocks, dropping code fences markers."""
    lines = text.splitlines()
    blocks: list[str] = []
    current: list[str] = []
    in_code = False
    for line in lines:
        if line.strip().startswith("```"):
            in_code = not in_code
            continue
        if in_code:
            continue
        current.append(line)
        if line.strip() and (line.startswith("#") or line.strip().endswith(("。", "！", "？", "."))):
            block = "\n".join(current).strip()
            if block:
                blocks.append(block)
            current = []
    rest = "\n".join(current).strip()
    if rest:
        blocks.append(rest)
    return blocks


def _clean(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(line.rstrip() for line in text.splitlines())
    text = "\n\n".join(line for line in text.split("\n\n") if line.strip())
    return text.strip()
