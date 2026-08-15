"""Unsloth GPU backend (QLoRA 4-bit, paged AdamW).

Only used when an NVIDIA GPU and the `unsloth` package are present. Exposes
the same train() signature as the transformers backend, so the runner can
pick the backend transparently.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..schema import JobSpec, ProgressCallback, Stage

INSTRUCTION_TEMPLATE = "问题：{instruction}\n回答："


class UnslothBackendError(Exception):
    pass


def available() -> bool:
    try:
        import torch

        if not torch.cuda.is_available():
            return False
        import unsloth  # noqa: F401

        return True
    except Exception:
        return False


def train(
    spec: JobSpec,
    jsonl_path: Path,
    out_dir: Path,
    strategy,
    progress: ProgressCallback | None = None,
) -> dict:
    if not available():
        raise UnslothBackendError("Unsloth 后端不可用（需要 NVIDIA GPU + pip install oneclick-distill[unsloth]）")

    from unsloth import FastLanguageModel  # type: ignore

    size_preset = _size_preset(spec.size)
    max_seq_len = int(strategy.max_seq_len)
    lr = float((size_preset or {}).get("lr", 1e-4))
    r = int((size_preset or {}).get("lora_r", 8))
    alpha = float((size_preset or {}).get("lora_alpha", 16))
    dropout = float((size_preset or {}).get("lora_dropout", 0.05))
    grad_accum = int(strategy.grad_accumulation_steps)
    batch_size = int(strategy.batch_size)

    if progress:
        progress(Stage.TRAIN, 0.0, f"Unsloth: 加载 {spec.model}（4-bit QLoRA）")

    model, tokenizer = FastLanguageModel.from_pretrained(
        spec.model,
        max_seq_length=max_seq_len,
        dtype=None,
        load_in_4bit=True,
        device_map="auto",
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=alpha,
        lora_dropout=dropout,
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    rows = []
    with jsonl_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict) and "prompt" in obj and "response" in obj:
                rows.append(obj)
    if not rows:
        raise UnslothBackendError("训练数据为空")

    data = [{"text": INSTRUCTION_TEMPLATE.format(instruction=r["prompt"]) + r["response"]} for r in rows]

    from unsloth import UnslothTrainer, UnslothTrainingArguments  # type: ignore

    args = UnslothTrainingArguments(
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        warmup_steps=5,
        max_steps=spec.max_steps or -1,
        num_train_epochs=(size_preset or {}).get("epochs", 3),
        learning_rate=lr,
        logging_steps=1,
        output_dir=str(out_dir),
        report_to="none",
    )
    trainer = UnslothTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=data,
        args=args,
        dataset_text_field="text",
        max_seq_length=max_seq_len,
    )
    trainer.train()

    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    if progress:
        progress(Stage.TRAIN, 1.0, "Unsloth 训练完成", {"backend": "unsloth"})
    return {"backend": "unsloth", "steps": spec.max_steps or "epochs"}


def _size_preset(size: str) -> dict | None:
    from ..config import settings

    return settings.sizes.get(size)
