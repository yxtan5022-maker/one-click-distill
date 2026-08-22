"""Training backends. Dispatches to Unsloth on GPU, transformers otherwise.

Backends are imported lazily so the REST/MCP server (and its tests) can run
without torch installed; the heavy imports only happen when a job actually
reaches the training stage.
"""

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
    from . import transformers_backend

    return transformers_backend.train(spec, jsonl_path, out_dir, strategy, progress)
