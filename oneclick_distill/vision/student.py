"""Student model: knowledge distillation from teacher soft labels + hard labels.

Loss = α · KL(σ(student/T) ∥ soft_teacher) + (1−α) · BCE(student, hard_labels)

where T is the distillation temperature.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..schema import JobSpec, LABEL_NAMES, ProgressCallback, Stage
from .teacher import TeacherModel


class StudentModel(nn.Module):
    """Lightweight vision backbone + classification head."""

    def __init__(self, backbone_name: str, num_classes: int = 12, pretrained: bool = True):
        super().__init__()
        import timm

        self.backbone = timm.create_model(
            backbone_name, pretrained=pretrained, num_classes=0
        )
        feat_dim = self.backbone.num_features
        self.head = nn.Linear(feat_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.head(features)


class DistillationDataset(torch.utils.data.Dataset):
    """Wraps an image dataset to also yield teacher soft-labels."""

    def __init__(self, image_dataset, soft_labels: dict[str, np.ndarray]):
        self.image_dataset = image_dataset
        self.soft_labels = soft_labels
        # align UIDs
        self.indices = [
            i for i in range(len(image_dataset))
            if image_dataset.study_uids[i] in soft_labels
        ]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        image, label, uid = self.image_dataset[real_idx]
        soft = torch.from_numpy(self.soft_labels[uid]).float()
        return image, label, soft, uid


def _flatten_multislice(x: torch.Tensor) -> torch.Tensor:
    """(B, N, C, H, W) or (B, N, H, W) → (B, C, H, W) via mean pooling."""
    if x.dim() == 5:
        out = x.mean(dim=1)  # (B, C, H, W)
        if out.size(1) == 1:
            out = out.repeat(1, 3, 1, 1)
        return out
    if x.dim() == 4 and x.size(1) > 3:
        B, N, H, W = x.shape
        return x.mean(dim=1).unsqueeze(1).repeat(1, 3, 1, 1)
    if x.size(1) == 1:
        return x.repeat(1, 3, 1, 1)
    return x


def distill_student(
    student: StudentModel,
    dataset,
    soft_labels: dict[str, np.ndarray],
    *,
    epochs: int = 10,
    lr: float = 1e-4,
    batch_size: int = 8,
    temperature: float = 3.0,
    alpha: float = 0.7,
    device: str = "cpu",
    seed: int = 42,
    progress: Optional[ProgressCallback] = None,
    max_steps: Optional[int] = None,
) -> dict:
    """Train student with distillation + hard-label BCE loss."""
    torch.manual_seed(seed)
    wrapped = DistillationDataset(dataset, soft_labels)
    loader = DataLoader(wrapped, batch_size=batch_size, shuffle=True, num_workers=0)

    student = student.to(device)
    optimizer = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=1e-4)
    bce_criterion = nn.BCEWithLogitsLoss()

    global_step = 0
    total_steps = max_steps or (epochs * len(loader))
    t0 = time.time()

    for epoch in range(epochs):
        student.train()
        for batch in loader:
            images, hard_labels, soft_labels_b, _ = batch
            images = _flatten_multislice(images.to(device))
            hard_labels = hard_labels.to(device)
            soft_labels_b = soft_labels_b.to(device)

            logits = student(images)

            # distillation loss (KL on soft targets)
            student_log_soft = F.log_softmax(logits / temperature, dim=1)
            teacher_soft = soft_labels_b
            # handle potential dimension mismatch
            if teacher_soft.dim() == 1:
                teacher_soft = teacher_soft.unsqueeze(0)
            teacher_soft = F.softmax(
                torch.log(teacher_soft + 1e-8) * temperature, dim=1
            )
            distill_loss = F.kl_div(student_log_soft, teacher_soft, reduction="batchmean")
            distill_loss = distill_loss * (temperature ** 2)

            # hard label loss
            hard_loss = bce_criterion(logits, hard_labels)

            loss = alpha * distill_loss + (1 - alpha) * hard_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

            if progress and global_step % 5 == 0:
                progress(
                    Stage.TRAIN,
                    min(global_step / total_steps, 0.99),
                    f"Student distillation step {global_step}/{total_steps}",
                    {"loss": round(loss.item(), 4), "step": global_step},
                )

            if max_steps and global_step >= max_steps:
                break
        if max_steps and global_step >= max_steps:
            break

    return {
        "steps": global_step,
        "device": device,
        "time_s": round(time.time() - t0, 1),
        "final_loss": round(loss.item(), 4) if "loss" in dir() else None,
    }
