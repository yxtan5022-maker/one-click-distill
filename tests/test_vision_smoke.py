"""Smoke test for the vision-classify pipeline.

Creates synthetic DICOMs + labels CSV, runs the pipeline end-to-end
in smoke mode, and verifies the output structure.
"""

from __future__ import annotations

import csv
import tempfile
from pathlib import Path

import numpy as np


def _make_synthetic_dicom(path: Path, h: int = 64, w: int = 64):
    """Write a minimal synthetic DICOM file."""
    import pydicom
    from pydicom.dataset import Dataset, FileDataset
    from pydicom.uid import ExplicitVRLittleEndian

    file_meta = pydicom.Dataset()
    file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
    file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\x00" * 128)
    ds.Modality = "MR"
    ds.Rows = h
    ds.Columns = w
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.PixelRepresentation = 1
    ds.RescaleIntercept = 0
    ds.RescaleSlope = 1
    ds.WindowCenter = 400
    ds.WindowWidth = 600
    arr = np.random.randint(-200, 800, (h, w), dtype=np.int16)
    ds.PixelData = arr.tobytes()
    ds.save_as(str(path))


def _create_fake_dataset(base: Path, n_studies: int = 4, n_series: int = 2, n_slices: int = 2):
    """Create a minimal fake competition dataset."""
    base.mkdir(parents=True, exist_ok=True)
    labels = {}
    studies = []

    for i in range(n_studies):
        uid = f"study_{i:04d}"
        studies.append(uid)
        study_dir = base / uid
        for s in range(n_series):
            s_uid = f"series_{s:02d}"
            s_dir = study_dir / s_uid
            s_dir.mkdir(parents=True, exist_ok=True)
            for sl in range(n_slices):
                dcm_path = s_dir / f"slice_{sl:03d}.dcm"
                _make_synthetic_dicom(dcm_path)
        labels[uid] = {c: str(int(np.random.rand() > 0.5))
                       for c in ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
                                 "Medial OA", "Lateral OA", "PF OA", "Effusion",
                                 "Synovitis", "Baker's", "Contusion", "Fracture"]}
    return studies, labels


def test_vision_pipeline_smoke():
    """Full smoke test: synthetic DICOM → pipeline → export."""
    with tempfile.TemporaryDirectory(prefix="vision_smoke_") as tmpdir:
        tmpdir = Path(tmpdir)
        data_dir = tmpdir / "data"
        n_studies = 4
        studies, labels = _create_fake_dataset(data_dir, n_studies=n_studies)

        # write labels CSV
        label_names = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
                       "Medial OA", "Lateral OA", "PF OA", "Effusion",
                       "Synovitis", "Baker's", "Contusion", "Fracture"]
        csv_path = tmpdir / "train_labels.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["StudyInstanceUID"] + label_names)
            writer.writeheader()
            for uid, vals in labels.items():
                writer.writerow({"StudyInstanceUID": uid, **vals})

        out_dir = tmpdir / "runs"

        # run pipeline
        from oneclick_distill.schema import JobSpec
        from oneclick_distill.vision.pipeline import run_vision_pipeline

        spec = JobSpec(
            task="vision-classify",
            labels_csv=str(csv_path),
            image_dir=str(data_dir),
            teacher_backbone="resnet18",       # tiny for smoke
            student_arch="mobilenetv2_100",     # tiny for smoke
            n_folds=2,
            max_steps=2,                        # 2 training steps only
            smoke=True,
            out_dir=str(out_dir),
            export_format="torchscript",
        )

        result = run_vision_pipeline(spec)

        # assertions
        assert result["macro_auc"] >= 0.0, f"AUC negative: {result['macro_auc']}"
        assert result["n_studies"] == n_studies
        assert Path(result["out_dir"]).exists()
        assert "timings" in result
        assert "student_arch" in result

        # check export exists
        export_dir = out_dir / "export"
        assert export_dir.exists(), f"Export dir missing: {export_dir}"

        print(f"✓ Vision pipeline smoke passed: AUC={result['macro_auc']:.4f}, "
              f"studies={result['n_studies']}, timings={result['timings']}")
        return result


if __name__ == "__main__":
    test_vision_pipeline_smoke()
