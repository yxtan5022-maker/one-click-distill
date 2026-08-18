"""Export trained model to TorchScript / ONNX / ONNX-INT8 for Kaggle inference."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn


class _Wrapper(nn.Module):
    """Thin wrapper to normalize input before forwarding."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def _ensure_3ch_input(model: nn.Module) -> _Wrapper:
    """Wrap model so it handles (B,1,H,W) input by repeating to 3 channels."""
    class _Norm(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
        def forward(self, x):
            if x.size(1) == 1:
                x = x.repeat(1, 3, 1, 1)
            return self.m(x)
    return _Norm(model)


def export_torchscript(model: nn.Module, out_path: Path, img_size: int = 224) -> Path:
    """Export to TorchScript via tracing."""
    model.eval()
    wrapped = _ensure_3ch_input(model)
    dummy = torch.randn(1, 3, img_size, img_size)
    traced = torch.jit.trace(wrapped, dummy)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    traced.save(str(out_path))
    return out_path


def export_onnx(model: nn.Module, out_path: Path, img_size: int = 224) -> Path:
    """Export to ONNX."""
    import onnx

    model.eval()
    wrapped = _ensure_3ch_input(model)
    dummy = torch.randn(1, 3, img_size, img_size)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapped,
        dummy,
        str(out_path),
        opset_version=17,
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
    )
    onnx.checker.check_model(str(out_path))
    return out_path


def export_onnx_int8(model: nn.Module, out_path: Path, calibration_data, img_size: int = 224) -> Path:
    """Export to ONNX with static quantization (INT8)."""
    try:
        import onnxruntime.quantization as ort_quant
    except ImportError:
        raise ImportError("onnxruntime >= 1.17 required for INT8 quantization")

    # first export float32 ONNX
    float_path = out_path.with_suffix(".float.onnx")
    export_onnx(model, float_path, img_size)

    ort_quant.quantize_static(
        str(float_path),
        str(out_path),
        calibration_data=calibration_data,
    )
    return out_path
