"""Configuration: presets, environment, and paths."""

from __future__ import annotations

import os
from pathlib import Path

import yaml

PKG_DIR = Path(__file__).resolve().parent
PRESETS_PATH = PKG_DIR / "presets.yaml"
SAMPLE_DATA_PATH = PKG_DIR / "sample_data" / "distillation_qa.jsonl"
WEB_DIR = PKG_DIR / "web"
DEFAULT_TOOLS_DIR = PKG_DIR / "tools"

SUPPORTED_DATA_EXTENSIONS = {".txt", ".md", ".markdown", ".json", ".jsonl"}
PDF_EXTENSIONS = {".pdf"}


def _load_env_file(path: Path):
    """Minimal .env parser (no python-dotenv dependency)."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def load_env() -> None:
    _load_env_file(Path.cwd() / ".env")
    _load_env_file(PKG_DIR / ".env")


class Settings:
    def __init__(self) -> None:
        load_env()
        self._presets: dict = {}
        self.load_presets()

    # ---- presets -------------------------------------------------------
    def load_presets(self) -> None:
        with PRESETS_PATH.open(encoding="utf-8") as f:
            self._presets = yaml.safe_load(f) or {}

    @property
    def models(self) -> dict:
        return self._presets.get("models", {})

    @property
    def sizes(self) -> dict:
        return self._presets.get("sizes", {})

    @property
    def teachers(self) -> dict:
        return self._presets.get("teachers", {})

    @property
    def template(self) -> str:
        return self._presets.get("template", "{instruction}")

    def resolve_model(self, model: str, size: str) -> str:
        """Resolve a user-facing name ('ultra'/'balanced'/'smoke' or explicit id)."""
        if model:
            return model
        return self.models.get(size) or self.models.get("ultra")

    # ---- env helpers ----------------------------------------------------
    def env(self, key: str, default: str = "") -> str:
        return os.environ.get(key, default)

    @property
    def host(self) -> str:
        return self.env("HOST", "127.0.0.1")

    @property
    def port(self) -> int:
        try:
            return int(self.env("PORT", "8080"))
        except ValueError:
            return 8080

    @property
    def tools_dir(self) -> Path:
        return DEFAULT_TOOLS_DIR

    @property
    def llama_quantize(self) -> Path | None:
        p = Path(self.env("LLAMA_QUANTIZE")) if self.env("LLAMA_QUANTIZE") else None
        return p or (self.tools_dir / "llama-quantize.exe")

    @property
    def convert_hf_to_gguf(self) -> Path | None:
        p = Path(self.env("CONVERT_HF_TO_GGUF")) if self.env("CONVERT_HF_TO_GGUF") else None
        return p or (self.tools_dir / "convert_hf_to_gguf.py")

    def teacher_config(self, name: str = "") -> dict:
        name = name or self.env("TEACHER_NAME", "deepseek")
        preset = self.teachers.get(name, {})
        return {
            "name": name,
            "base_url": self.env("TEACHER_BASE_URL", preset.get("base_url", "")),
            "model": self.env("TEACHER_MODEL", preset.get("model", "")),
            "api_key": self.env("TEACHER_API_KEY", ""),
        }


settings = Settings()
