"""Training backends. Dispatches to Unsloth on GPU, transformers otherwise."""

from . import transformers_backend
from .unsloth_backend import available as _unsloth_available

__all__ = ["train", "pick_backend"]


def pick_backend(strategy) -> str:
    if getattr(strategy, "backend", "transformers") == "unsloth" and _unsloth_available():
        return "unsloth"
    return "transformers"


def train(spec, jsonl_path, out_dir, strategy, progress=None) -> dict:
    backend = pick_backend(strategy)
    if backend == "unsloth":
        from .unsloth_backend import train as _train

        return _train(spec, jsonl_path, out_dir, strategy, progress)
    return transformers_backend.train(spec, jsonl_path, out_dir, strategy, progress)
