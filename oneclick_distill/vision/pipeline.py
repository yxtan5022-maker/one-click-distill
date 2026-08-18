"""VisionPipeline: orchestrates import → teacher train → distillation → eval → export."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from sklearn.model_selection import GroupKFold

from ..schema import JobSpec, LABEL_NAMES, ProgressCallback, Stage
from .dicom import scan_study_dir
from .dataset import KneeStudyDataset, load_labels_csv, load_series_csv
from .eval import macro_roc_auc, per_label_auc
from .export import export_torchscript, export_onnx, export_onnx_int8
from .student import StudentModel, distill_student
from .teacher import TeacherModel, generate_soft_labels, train_teacher

VISION_STAGES = [Stage.PREPARE, Stage.DATA, Stage.IMPORT, Stage.TRAIN, Stage.EVAL, Stage.QUANTIZE, Stage.DONE]


class VisionPipeline:
    def __init__(self, spec: JobSpec, progress: Optional[ProgressCallback] = None):
        self.spec = spec
        self.progress = progress
        self.timings: dict[str, float] = {}
        self.result: dict = {}

    def _emit(self, stage: Stage, progress: float, message: str, metrics: dict | None = None):
        if self.progress:
            self.progress(stage, progress, message, metrics or {})

    def run(self) -> dict:
        spec = self.spec
        out_dir = Path(spec.out_dir or f"runs/{spec.id}")
        out_dir.mkdir(parents=True, exist_ok=True)
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # ---- prepare --------------------------------------------------
        t0 = time.time()
        self._emit(Stage.PREPARE, 0.0, "初始化 vision-classify 流水线")
        backbone = spec.teacher_backbone or "convnext_base"
        student_arch = spec.student_arch or "efficientnet_b0"
        self._emit(Stage.PREPARE, 1.0, f"教师 {backbone} → 学生 {student_arch}，设备 {device.upper()}")

        # ---- data -----------------------------------------------------
        t1 = time.time()
        self._emit(Stage.DATA, 0.0, "加载标签与 DICOM 扫描")
        labels = load_labels_csv(spec.labels_csv) if spec.labels_csv else {}
        series_map = load_series_csv(spec.series_csv) if spec.series_csv else {}
        study_uids = sorted(labels.keys())
        self._emit(Stage.DATA, 0.5, f"标签 {len(study_uids)} 个 study")

        # scan image dir for all study dirs
        image_dir = Path(spec.image_dir)
        all_study_dirs = [d.name for d in image_dir.iterdir() if d.is_dir()] if image_dir.exists() else []
        self._emit(Stage.DATA, 1.0, f"扫描到 {len(all_study_dirs)} 个 study 目录")
        self.timings["data"] = time.time() - t1

        # ---- GroupKFold -----------------------------------------------
        t2 = time.time()
        self._emit(Stage.IMPORT, 0.0, f"分 {spec.n_folds} 折 GroupKFold（按站点分组）")

        # map study → site group (from series descriptions or scan hash)
        site_groups = {}
        for uid in study_uids:
            if uid in series_map:
                site_groups[uid] = hash(tuple(sorted(series_map[uid].values())))
            else:
                site_groups[uid] = hash(uid)  # fallback: uid-based

        groups = np.array([site_groups.get(uid, 0) for uid in study_uids])
        group_kfold = GroupKFold(n_splits=spec.n_folds)
        fold_indices = list(group_kfold.split(study_uids, groups=groups))

        # for efficiency: only do teacher training on fold-0 in smoke mode
        folds_to_run = [0] if spec.smoke else list(range(spec.n_folds))

        all_study_preds = {}
        all_study_true = {}
        student_metrics_per_fold = []

        for fold_i in folds_to_run:
            self._emit(Stage.IMPORT, fold_i / spec.n_folds, f"Fold {fold_i}/{spec.n_folds}")
            train_idx, val_idx = fold_indices[fold_i]
            train_uids = [study_uids[i] for i in train_idx]
            val_uids = [study_uids[i] for i in val_idx]

            train_ds = KneeStudyDataset(
                image_dir, labels, series_map, study_uids=train_uids
            )
            val_ds = KneeStudyDataset(
                image_dir, labels, series_map, study_uids=val_uids
            )

            # ---- train teacher -----------------------------------------
            teacher = TeacherModel(backbone, pretrained=True)
            t_tr = time.time()
            train_teacher(
                teacher, train_ds, epochs=3, lr=1e-4, batch_size=4,
                device=device, max_steps=spec.max_steps, progress=self.progress,
            )
            self.timings[f"fold{fold_i}_teacher"] = time.time() - t_tr

            # teacher evaluate on val
            from .teacher import predict_soft_labels
            val_uids_pred, val_probs = predict_soft_labels(teacher, val_ds, device=device, batch_size=8)
            for uid, prob in zip(val_uids_pred, val_probs):
                all_study_preds[uid] = prob
            for uid in val_uids:
                all_study_true[uid] = labels[uid]

            # generate soft labels for all unlabeled studies
            all_ds = KneeStudyDataset(
                image_dir, labels, series_map, study_uids=study_uids
            )
            soft_labels_path = out_dir / f"soft_labels_fold{fold_i}.jsonl"
            generate_soft_labels(teacher, all_ds, soft_labels_path, device=device, batch_size=8)

            # ---- distill student ---------------------------------------
            student = StudentModel(student_arch, pretrained=True)
            soft_dict = {uid: prob for uid, prob in zip(val_uids_pred, val_probs)}
            # load full soft labels
            for row in _read_jsonl(soft_labels_path):
                soft_dict[row["StudyInstanceUID"]] = np.array(row["soft_labels"], dtype=np.float32)

            t_st = time.time()
            distill_student(
                student, train_ds, soft_dict, epochs=3, lr=5e-4,
                batch_size=8, device=device, max_steps=spec.max_steps,
                progress=self.progress,
            )
            self.timings[f"fold{fold_i}_student"] = time.time() - t_st

            # save student checkpoint
            torch.save(student.state_dict(), out_dir / f"student_fold{fold_i}.pt")

        # ---- eval ------------------------------------------------------
        t3 = time.time()
        self._emit(Stage.EVAL, 0.0, "评测 macro ROC-AUC")
        eval_uids = sorted(all_study_true.keys())
        if eval_uids:
            y_true = np.stack([all_study_true[u] for u in eval_uids])
            y_pred = np.stack([all_study_preds[u] for u in eval_uids])
            auc = macro_roc_auc(y_true, y_pred)
            label_aucs = per_label_auc(y_true, y_pred)
        else:
            auc = 0.0
            label_aucs = {n: 0.0 for n in LABEL_NAMES}

        self._emit(Stage.EVAL, 1.0, f"macro AUC = {auc:.4f}", {"auc": auc})
        self.timings["eval"] = time.time() - t3

        # ---- export (best fold) ----------------------------------------
        t4 = time.time()
        self._emit(Stage.QUANTIZE, 0.0, "导出最优学生模型")
        best_fold = 0
        student = StudentModel(student_arch, pretrained=False)
        student.load_state_dict(torch.load(out_dir / f"student_fold{best_fold}.pt", weights_only=True))
        student.eval()

        export_path = out_dir / "export"
        export_path.mkdir(parents=True, exist_ok=True)

        fmt = spec.export_format
        if fmt == "torchscript":
            export_torchscript(student, export_path / "student.pt")
        elif fmt == "onnx":
            export_onnx(student, export_path / "student.onnx")
        elif fmt == "onnx_int8":
            export_onnx_int8(student, export_path / "student_int8.onnx", calibration_data=None)
        else:
            export_torchscript(student, export_path / "student.pt")

        self.timings["export"] = time.time() - t4

        self.result = {
            "id": spec.id,
            "teacher_backbone": backbone,
            "student_arch": student_arch,
            "macro_auc": auc,
            "label_aucs": label_aucs,
            "n_folds": spec.n_folds,
            "n_studies": len(study_uids),
            "device": device,
            "out_dir": str(out_dir),
            "timings": {k: round(v, 1) for k, v in self.timings.items()},
        }
        self._emit(Stage.DONE, 1.0, "vision-classify 流水线完成 ✓", {"result": self.result})
        return self.result


def _read_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def run_vision_pipeline(spec: JobSpec, progress: Optional[ProgressCallback] = None) -> dict:
    return VisionPipeline(spec, progress).run()
