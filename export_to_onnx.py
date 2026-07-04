"""ONNX export utility — converts trained PyTorch models to ONNX for BATRunner.

BATRunner passes input as (N, 73, 1) — N frames, 73 features, 1 channel.
The ONNX model receives this and produces a single gesture-level vector.

Output shapes per model:
    floor: (1, 4)  — pair-group softmax probabilities for FT, HN, S, D
    spine: (1, 6)  — softmax probabilities for E, F, HG, LF, SR, U
    limb:  (1, 8)  — per-pair normalised probabilities for LB, SL, AS, A, G, UB, DL, SY
    space: (1, 7)  — pair-group softmax for ST, T, RV, SP | H, M, L

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
    """Applies per-pair normalisation for LimbExpression structured output.

    LimbExpression has four binary choices forming complementary pairs:
        LB↔UB, SL↔DL, AS↔SY, A↔G.
    This wrapper normalises probabilities within each pair so that:
        p(LB) + p(UB) = 1.0,  p(SL) + p(DL) = 1.0,
        p(AS) + p(SY) = 1.0,  p(A)  + p(G)  = 1.0
    Each label is predicted when its probability > 0.5.
    """

    def forward(self, logits_4):
        # logits_4: (B, 4) — raw logits from 4 heads [body, limb, symmetry, contact]
        probs = torch.sigmoid(logits_4)  # (B, 4) — [p_UB, p_DL, p_AS, p_A]

        # Pair 0: LB↔UB (body) — indices 0, 5 in 8-code output
        pair_sum = (1.0 - probs[:, 0:1]) + probs[:, 0:1] + 1e-8
        lb = (1.0 - probs[:, 0:1]) / pair_sum
        ub = 1.0 - lb
        # Pair 1: SL↔DL (limb) — indices 1, 6 in 8-code output
        pair_sum = (1.0 - probs[:, 1:2]) + probs[:, 1:2] + 1e-8
        sl = (1.0 - probs[:, 1:2]) / pair_sum
        dl = 1.0 - sl
        # Pair 2: AS↔SY (symmetry) — indices 2, 7 in 8-code output
        pair_sum = probs[:, 2:3] + (1.0 - probs[:, 2:3]) + 1e-8
        asym = probs[:, 2:3] / pair_sum
        sy = 1.0 - asym
        # Pair 3: A↔G (contact) — indices 3, 4 in 8-code output
        pair_sum = probs[:, 3:4] + (1.0 - probs[:, 3:4]) + 1e-8
        a = probs[:, 3:4] / pair_sum
        g = 1.0 - a

        return torch.cat([lb, sl, asym, a, g, ub, dl, sy], dim=1)  # (B, 8)


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


class SpaceOutputWrapper(nn.Module):
    """Applies per-group softmax for Space structured output.

    Space has two independent choices:
      Movement group (indices 0-3): ST, T, RV, SP  (4-way softmax)
      Energy group   (indices 4-6): H,  M, L       (3-way softmax)

    For groups with >2 codes, use argmax to select the winner.
    For binary groups (Floor, Limb), 0.5 threshold works.
    """

    def forward(self, logits):
        mv = torch.softmax(logits[:, 0:4], dim=1)
        en = torch.softmax(logits[:, 4:7], dim=1)
        return torch.cat([mv, en], dim=1)


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
    """Export LimbExpression model (4→8 per-pair normalised outputs)."""
    sys.path.insert(0, str(Path(__file__).resolve().parent / "LimbExpression"))
    from models.multilabel_cnn import MultiLabelCNN

    m = MultiLabelCNN(device="cpu")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    m.model.load_state_dict(ckpt["model_state_dict"])
    model = make_onnx_model(m.model, LimbExpressionOutputWrapper())
    export_model(model, onnx_path)
    verify_onnx(onnx_path, model)


def export_space(checkpoint_path: str, onnx_path: str, target_frames: int = 256):
    """Export Space model (7 outputs with pair-group softmax constraint).

    Movement group (indices 0-3): ST, T, RV, SP (4-way softmax)
    Energy group   (indices 4-6): H,  M, L      (3-way softmax)
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent / "Space"))
    from models.multilabel_space import MultiLabelSpaceModel

    m = MultiLabelSpaceModel(device="cpu")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    m.model.load_state_dict(ckpt["model_state_dict"])
    model = make_onnx_model(m.model, SpaceOutputWrapper())
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
