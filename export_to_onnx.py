"""ONNX export utility — converts trained PyTorch models to ONNX for BATRunner.

BATRunner passes input as (N, 73, 1) — N frames, 73 features, 1 channel.
The ONNX model receives this and produces a single gesture-level vector.

Output shapes per model:
    floor: (1, 4)  — pair-group softmax probabilities for FT, HN, S, D
    spine: (1, 6)  — softmax probabilities for E, F, HG, LF, SR, U
    limb:  (1, 8)  — sigmoid probabilities for LB, SL, AS, A, G, UB, DL, SY
    space: (1, 5)  — sigmoid probabilities for RV, ST, SP, H, M

Usage:
    python export_to_onnx.py --model floor --checkpoint path/to/model.pt --output path/to/model.onnx
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


# ═══════════════════════════════════════════════════════════════════════
# BATRunner-compatible input wrapper
# ═══════════════════════════════════════════════════════════════════════

class BATRunnerInputWrapper(nn.Module):
    """Wraps a gesture model to accept BATRunner's (N, 73, 1) tensor format.

    BATRunner provides: (N_frames, 73, 1) = (time, features, channel)
    Training model expects: (B, features, time) = (1, 73, T)

    This wrapper reshapes the input and produces a single gesture-level output.
    """
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x):
        # x: (N, 73, 1) — BATRunner input
        x = x.squeeze(-1)          # (N, 73)
        x = x.unsqueeze(0)         # (1, N, 73)
        x = x.permute(0, 2, 1)     # (1, 73, N)
        return self.model(x)       # (1, num_codes)


# ═══════════════════════════════════════════════════════════════════════
# Output wrappers  (applied after model forward)
# ═══════════════════════════════════════════════════════════════════════

class LimbExpressionOutputWrapper(nn.Module):
    """Expands 4-head output to 8 Unity codes with sigmoid baked in."""

    def forward(self, logits_4):
        # logits_4: (B, 4) — raw logits from 4 heads
        probs = torch.sigmoid(logits_4)
        return torch.cat([
            1.0 - probs[:, 0:1],           # LB
            1.0 - probs[:, 1:2],           # SL
            probs[:, 2:3],                 # AS
            probs[:, 3:4],                 # A
            1.0 - probs[:, 3:4],           # G
            probs[:, 0:1],                 # UB
            probs[:, 1:2],                 # DL
            1.0 - probs[:, 2:3],           # SY
        ], dim=1)  # (B, 8)


class SpineOutputWrapper(nn.Module):
    """Applies softmax to 6-class logits."""

    def forward(self, logits_6):
        return torch.softmax(logits_6, dim=1)


class SigmoidOutputWrapper(nn.Module):
    """Applies sigmoid for multi-label output."""

    def forward(self, logits):
        return torch.sigmoid(logits)


class FloorOutputWrapper(nn.Module):
    """Applies per-group softmax for FloorSupport structured output.

    FloorSupport has two binary choices (FT↔HN, S↔D).  This wrapper
    normalises sigmoid probabilities within each pair so that:
        p(FT) + p(HN) = 1.0   and   p(S) + p(D) = 1.0
    guaranteeing the 4 valid combos: FT+D, FT+S, HN+D, HN+S.
    """

    def forward(self, logits):
        probs = torch.sigmoid(logits)
        # Group 0: FT↔HN (indices 0,1)
        pair_sum = probs[:, 0:1] + probs[:, 1:2] + 1e-8
        ft = probs[:, 0:1] / pair_sum
        hn = 1.0 - ft
        # Group 1: S↔D (indices 2,3)
        pair_sum = probs[:, 2:3] + probs[:, 3:4] + 1e-8
        s = probs[:, 2:3] / pair_sum
        d = 1.0 - s
        return torch.cat([ft, hn, s, d], dim=1)


# ═══════════════════════════════════════════════════════════════════════
# Combined wrappers for export
# ═══════════════════════════════════════════════════════════════════════

def make_onnx_model(inner_model: nn.Module, output_wrapper: nn.Module):
    """Chain: BATRunner input format → inner model → output activation → single output."""
    class ONNXModel(nn.Module):
        def __init__(self, inner, out_wrap):
            super().__init__()
            self.input_wrap = BATRunnerInputWrapper(inner)
            self.out_wrap = out_wrap

        def forward(self, x):
            y = self.input_wrap(x)      # (1, num_codes)
            return self.out_wrap(y)     # (1, num_codes)

    return ONNXModel(inner_model, output_wrapper)


# ═══════════════════════════════════════════════════════════════════════
# Export functions
# ═══════════════════════════════════════════════════════════════════════

def export_model(model: nn.Module, onnx_path: str, num_frames: int = 256,
                 batch_size: int = 1, opset_version: int = 11):
    """Export a PyTorch model to ONNX.

    Input shape:  (N, 73, 1)  — BATRunner format
    Output shape: (N, num_codes)  — where N=1 after temporal pooling wrapper
    """
    model.eval()
    dummy_input = torch.randn(num_frames, 73, 1, dtype=torch.float32)

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "frames"},       # variable number of frames
        },
        opset_version=opset_version,
        do_constant_folding=True,
    )
    print(f"  Exported to: {onnx_path}")


def verify_onnx(onnx_path: str, pytorch_model: nn.Module, num_frames: int = 256):
    """Verify ONNX model matches PyTorch output."""
    try:
        import onnxruntime as ort
    except ImportError:
        print("  onnxruntime not installed — skipping verification")
        return

    pytorch_model.eval()
    test_input = torch.randn(num_frames, 73, 1, dtype=torch.float32)

    with torch.no_grad():
        torch_out = pytorch_model(test_input).numpy()

    session = ort.InferenceSession(onnx_path)
    onnx_out = session.run(None, {"input": test_input.numpy()})[0]

    max_diff = np.abs(torch_out - onnx_out).max()
    print(f"  Max difference (PyTorch vs ONNX): {max_diff:.2e}")
    if max_diff < 1e-4:
        print("  ✓ ONNX output matches PyTorch")
    else:
        print(f"  ⚠ Difference is {max_diff:.2e} — may need investigation")
    return max_diff


# ═══════════════════════════════════════════════════════════════════════
# Per-model export helpers
# ═══════════════════════════════════════════════════════════════════════

def export_floor(checkpoint_path: str, onnx_path: str, target_frames: int = 256):
    """Export FloorSupport model (4 outputs with pair-group softmax constraint).

    Probabilities are normalised within each pair: p(FT)+p(HN)=1, p(S)+p(D)=1.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent / "FloorSupport"))
    from models.multilabel_floor import MultiLabelFloorModel

    m = MultiLabelFloorModel(device="cpu")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    m.model.load_state_dict(ckpt["model_state_dict"])
    model = make_onnx_model(m.model, FloorOutputWrapper())
    export_model(model, onnx_path)
    verify_onnx(onnx_path, model)


def export_spine(checkpoint_path: str, onnx_path: str, target_frames: int = 256):
    """Export Spine single-label model (6 softmax outputs)."""
    sys.path.insert(0, str(Path(__file__).resolve().parent / "Spine"))
    from models.multilabel_spine import MultiLabelSpineModel

    m = MultiLabelSpineModel(device="cpu")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    m.model.load_state_dict(ckpt["model_state_dict"])
    model = make_onnx_model(m.model, SpineOutputWrapper())
    export_model(model, onnx_path)
    verify_onnx(onnx_path, model)


def export_limb(checkpoint_path: str, onnx_path: str, target_frames: int = 128):
    """Export LimbExpression model (4→8 expanded sigmoid outputs)."""
    sys.path.insert(0, str(Path(__file__).resolve().parent / "LimbExpression"))
    from models.multilabel_cnn import MultiLabelCNN

    m = MultiLabelCNN(device="cpu")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    m.model.load_state_dict(ckpt["model_state_dict"])
    model = make_onnx_model(m.model, LimbExpressionOutputWrapper())
    export_model(model, onnx_path)
    verify_onnx(onnx_path, model)


def export_space(checkpoint_path: str, onnx_path: str, target_frames: int = 256):
    """Export Space multi-label model (5 sigmoid outputs)."""
    sys.path.insert(0, str(Path(__file__).resolve().parent / "Space"))
    from models.multilabel_space import MultiLabelSpaceModel

    m = MultiLabelSpaceModel(device="cpu")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    m.model.load_state_dict(ckpt["model_state_dict"])
    model = make_onnx_model(m.model, SigmoidOutputWrapper())
    export_model(model, onnx_path)
    verify_onnx(onnx_path, model)


EXPORTERS = {
    "floor": export_floor,
    "spine": export_spine,
    "limb": export_limb,
    "space": export_space,
}


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Export trained models to ONNX")
    parser.add_argument("--model", type=str, required=True,
                        choices=list(EXPORTERS),
                        help="Which model to export")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to .pt checkpoint")
    parser.add_argument("--output", type=str, default=None,
                        help="Output .onnx path (default: <model>_multilabel.onnx)")
    parser.add_argument("--target-frames", type=int, default=256,
                        help="Target frames for input (default: 256)")
    args = parser.parse_args()

    onnx_path = args.output or f"{args.model}_multilabel.onnx"
    export_fn = EXPORTERS[args.model]
    export_fn(args.checkpoint, onnx_path, args.target_frames)


if __name__ == "__main__":
    main()
