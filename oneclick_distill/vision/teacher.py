"""Vision teacher model: train a strong backbone + generate soft labels.

The teacher is typically a large pretrained model (e.g. convnext_base, dinov2)
fine-tuned on the small labeled set with BCE loss, then used to predict
soft labels for the full unlabeled dataset.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from ..schema import JobSpec, LABEL_NAMES, ProgressCallback, Stage


class TeacherModel(nn.Module):
    """timm backbone + multi-label classification head."""

    def __init__(self, backbone_name: str, num_classes: int = 12, pretrained: bool = True):
        super().__init__()
        import timm

        self.backbone = timm.create_model(
            backbone_name, pretrained=pretrained, num_classes=0
        )
        feat_dim = self.backbone.num_features
        self.head = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H, W) → (B, 12) logits."""
        features = self.backbone(x)
        return self.head(features)


def train_teacher(
    model: TeacherModel,
    dataset: torch.utils.data.Dataset,
    *,
    epochs: int = 10,
    lr: float = 1e-4,
    batch_size: int = 8,
    device: str = "cpu",
    seed: int = 42,
    progress: Optional[ProgressCallback] = None,
    max_steps: Optional[int] = None,
) -> dict:
    """Fine-tune teacher model with BCEWithLogitsLoss. Returns metrics dict."""
    torch.manual_seed(seed)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.BCEWithLogitsLoss()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    global_step = 0
    total_steps = max_steps or (epochs * len(loader))
    t0 = time.time()

    for epoch in range(epochs):
        model.train()
        for images, labels, _ in loader:
            images = images.to(device)
            # handle multi-slice: pool over slices → mean before head
            if images.dim() == 5:
                B, N, C, H, W = images.shape
                images = images.view(B * N, C, H, W)
                if images.size(1) == 1:
                    images = images.repeat(1, 3, 1, 1)
                logits = model(images).view(B, N, -1).mean(dim=1)
            elif images.dim() == 4 and images.size(1) > 3:
                # (B, N, H, W) with N channels → treat as batch of N
                B, N, H, W = images.shape
                images = images.view(B * N, 1, H, W).repeat(1, 3, 1, 1)
                logits = model(images).view(B, N, -1).mean(dim=1)
            else:
                if images.size(1) == 1:
                    images = images.repeat(1, 3, 1, 1)
                logits = model(images)

            loss = criterion(logits, labels.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

            if progress and global_step % 5 == 0:
                progress(
                    Stage.TRAIN,
                    min(global_step / total_steps, 0.99),
                    f"Teacher training step {global_step}/{total_steps}",
                    {"loss": round(loss.item(), 4), "step": global_step},
                )

            if max_steps and global_step >= max_steps:
                break
        if max_steps and global_step >= max_steps:
            break

    scheduler.step()
    return {
        "steps": global_step,
        "device": device,
        "time_s": round(time.time() - t0, 1),
        "final_loss": round(loss.item(), 4) if "loss" in dir() else None,
    }


def predict_soft_labels(
    model: TeacherModel,
    dataset: torch.utils.data.Dataset,
    *,
    device: str = "cpu",
    batch_size: int = 16,
) -> tuple[list[str], np.ndarray]:
    """Run inference on a dataset → (study_uids, soft_probs[N,12])."""
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    all_probs = []
    all_uids = []
    sigmoid = nn.Sigmoid()

    with torch.no_grad():
        for images, _, study_uids in loader:
            images = images.to(device)
            if images.dim() == 5:
                B, N, C, H, W = images.shape
                images = images.view(B * N, C, H, W)
                if images.size(1) == 1:
                    images = images.repeat(1, 3, 1, 1)
                logits = model(images).view(B, N, -1).mean(dim=1)
            elif images.dim() == 4 and images.size(1) > 3:
                B, N, H, W = images.shape
                images = images.view(B * N, 1, H, W).repeat(1, 3, 1, 1)
                logits = model(images).view(B, N, -1).mean(dim=1)
            else:
                if images.size(1) == 1:
                    images = images.repeat(1, 3, 1, 1)
                logits = model(images)

            probs = sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_uids.extend(study_uids)

    return all_uids, np.concatenate(all_probs, axis=0)


def generate_soft_labels(
    model: TeacherModel,
    dataset: torch.utils.data.Dataset,
    output_path: Path,
    *,
    device: str = "cpu",
    batch_size: int = 16,
) -> dict:
    """Generate soft labels and save as JSONL. Returns stats."""
    study_uids, probs = predict_soft_labels(model, dataset, device=device, batch_size=batch_size)

    records = []
    for uid, prob in zip(study_uids, probs):
        records.append({"StudyInstanceUID": uid, "soft_labels": prob.tolist()})

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(__import__("json").dumps(r, ensure_ascii=False) + "\n")

    return {"studies": len(records), "output": str(output_path)}
