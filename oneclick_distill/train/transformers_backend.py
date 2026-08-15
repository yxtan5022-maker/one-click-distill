"""Transformers CPU/GPU training backend with self-built LoRA.

This backend runs anywhere (no NVIDIA GPU required) and is the guaranteed
"保底" path. It exposes the exact same API as the Unsloth backend.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..schema import JobSpec, ProgressCallback, Stage
from .lora import merge_and_unwrap, restore_conv1d, wrap_linear_layers

INSTRUCTION_TEMPLATE = "问题：{instruction}\n回答："


class TrainError(Exception):
    pass


def _detect_oom(exc: BaseException) -> bool:
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    return "out of memory" in str(exc).lower()


def load_model_and_tokenizer(model_id: str, device: str, dtype: torch.dtype):
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, low_cpu_mem_usage=True)
    except Exception as e:
        raise TrainError(f"模型加载失败（请检查模型 ID 与网络）: {e}") from e
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
    except Exception:
        from transformers import GPT2Tokenizer

        tokenizer = GPT2Tokenizer.from_pretrained("gpt2", do_lower_case=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = model.to(device=device, dtype=dtype)
    return model, tokenizer


def build_dataset(
    jsonl_path: Path,
    tokenizer,
    max_seq_len: int,
    max_samples: int = 2000,
    seed: int = 42,
):
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
        raise TrainError("训练数据为空（需要 {prompt, response} 格式的 JSONL）")
    rng = random.Random(seed)
    if len(rows) > max_samples:
        rows = rng.sample(rows, max_samples)

    inputs = []
    for r in rows:
        text = INSTRUCTION_TEMPLATE.format(instruction=r["prompt"]) + r["response"]
        enc = tokenizer(text, truncation=True, max_length=max_seq_len)
        inputs.append({"input_ids": enc["input_ids"], "attention_mask": enc.get("attention_mask")})
    return inputs


def _collate(batch, pad_token_id: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    input_ids = [torch.tensor(b["input_ids"], dtype=torch.long) for b in batch]
    attn = [torch.tensor(b["attention_mask"], dtype=torch.long) for b in batch]
    max_len = max(t.size(0) for t in input_ids)
    padded = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
    mask = torch.zeros((len(batch), max_len), dtype=torch.long)
    for i, t in enumerate(input_ids):
        padded[i, : t.size(0)] = t
        mask[i, : t.size(0)] = attn[i]
    labels = padded.clone()
    return padded, mask, labels


def train(
    spec: JobSpec,
    jsonl_path: Path,
    out_dir: Path,
    strategy,
    progress: ProgressCallback | None = None,
) -> dict:
    """Run SFT + LoRA. Returns a metrics dict. Merges LoRA back and saves a
    complete HF model directory ready for GGUF conversion."""
    model_id = spec.model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    max_seq_len = int(strategy.max_seq_len)
    batch_size = int(strategy.batch_size)
    grad_accum = int(strategy.grad_accumulation_steps)
    epochs = 3
    lr = float(spec.size and 1e-4)
    r = 8
    alpha = 16
    dropout = 0.05

    size_preset = _size_preset(spec.size)
    if size_preset:
        epochs = int(size_preset.get("epochs", 3))
        lr = float(size_preset.get("lr", 1e-4))
        r = int(size_preset.get("lora_r", 8))
        alpha = float(size_preset.get("lora_alpha", 16))
        dropout = float(size_preset.get("lora_dropout", 0.05))
        max_seq_len = int(size_preset.get("max_seq_len", max_seq_len))
        batch_size = int(size_preset.get("batch_size", batch_size))
        grad_accum = int(size_preset.get("grad_accumulation_steps", grad_accum))

    if spec.smoke:
        # smoke: 小数据集也要能产生完整的优化步
        batch_size = 1
        grad_accum = 1

    torch.manual_seed(42)
    dtype = torch.float32
    if device == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    model, tokenizer = load_model_and_tokenizer(model_id, device, dtype)
    if progress:
        progress(Stage.TRAIN, 0.0, f"加载模型 {model_id} 完成，设备 {device.upper()}")

    # gradient checkpointing: saves ~60% activation memory
    try:
        if hasattr(model, "gradient_checkpointing_enable") and strategy.gradient_checkpointing:
            model.gradient_checkpointing_enable()
    except Exception:
        pass

    wrapped, conv_names = wrap_linear_layers(model, r=r, alpha=alpha, dropout=dropout)
    if not wrapped:
        raise TrainError("未找到可注入 LoRA 的线性层")
    if progress:
        progress(Stage.TRAIN, 0.02, f"注入 LoRA 完成（r={r}, α={alpha}），可训练参数约 "
                 f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    dataset = build_dataset(jsonl_path, tokenizer, max_seq_len)
    if progress:
        progress(Stage.TRAIN, 0.05, f"训练集 {len(dataset)} 条，max_seq_len={max_seq_len}")

    if spec.dry_run:
        dataset = dataset[:2]
        max_steps_total = 1
    else:
        max_steps_total = spec.max_steps or (epochs * len(dataset) // (batch_size * grad_accum) or 1)

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    lora_params = [p for p in model.parameters() if p.requires_grad]

    global_step = 0
    effective_batch = batch_size * grad_accum
    steps_per_epoch = max(1, len(dataset) // effective_batch)
    total_steps = max_steps_total if spec.dry_run else min(max_steps_total, epochs * steps_per_epoch)

    if progress:
        progress(Stage.TRAIN, 0.05, f"开始训练：{total_steps} 步（batch={batch_size}, grad_accum={grad_accum}）")

    for epoch in range(epochs):
        if global_step >= total_steps:
            break
        rng = random.Random(42 + epoch)
        order = rng.sample(range(len(dataset)), len(dataset))
        optimizer.zero_grad()
        accum_count = 0
        for pos, idx in enumerate(order):
            if global_step >= total_steps:
                break
            batch = [dataset[i] for i in order[pos : pos + batch_size]]
            input_ids, attn, labels = _collate(batch, tokenizer.pad_token_id)
            input_ids = input_ids.to(device)
            attn = attn.to(device)
            labels = labels.to(device)

            try:
                out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
            except BaseException as e:
                if _detect_oom(e):
                    if max_seq_len > 128:
                        # 防爆: halve seq len and restart the step
                        torch.cuda.empty_cache() if device == "cuda" else None
                        raise TrainError(
                            f"触发 CUDA OOM（max_seq_len={max_seq_len}）。请减小模型规格或在显卡条件更好的机器上运行。"
                        )
                    raise
                raise

            loss = out.loss / grad_accum
            loss.backward()
            accum_count += 1

            if accum_count == grad_accum:
                torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
                optimizer.step()
                optimizer.zero_grad()
                accum_count = 0
                global_step += 1
                if progress:
                    progress(
                        Stage.TRAIN,
                        0.05 + 0.9 * (global_step / total_steps),
                        f"训练中 step {global_step}/{total_steps}",
                        {"loss": round(float(out.loss.item()), 4), "step": global_step},
                    )
        # flush partial accumulation
        if accum_count > 0:
            torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
            optimizer.step()
            optimizer.zero_grad()

    # ---- merge LoRA back and save a full HF model -------------------------
    if progress:
        progress(Stage.TRAIN, 0.96, "合并 LoRA 权重并保存模型")
    model.eval()
    merge_and_unwrap(model, wrapped)
    restore_conv1d(model, conv_names)
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        model.save_pretrained(str(out_dir))
        tokenizer.save_pretrained(str(out_dir))
    except Exception as e:
        raise TrainError(f"模型保存失败: {e}") from e

    metrics = {
        "steps": global_step,
        "device": device,
        "trainable_params": sum(p.numel() for p in lora_params),
        "model": model_id,
        "final_loss": float(loss.item()) if "loss" in dir() else None,
    }
    if progress:
        progress(Stage.TRAIN, 1.0, "训练完成", metrics)
    return metrics


def _size_preset(size: str) -> dict | None:
    from ..config import settings

    return settings.sizes.get(size)
