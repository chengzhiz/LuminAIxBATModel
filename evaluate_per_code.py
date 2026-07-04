#!/usr/bin/env python3
"""Per-code evaluation: precision, recall, F1, and accuracy for every BAT code.

Loads each region's best checkpoint, runs inference on the validation set,
and prints a per-code breakdown so you can answer questions like:
    "When the ground truth is dual feet, how often does the model get it right?"
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

# ── Path setup ──────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "FloorSupport"))

from plumbline_features import load_plumbline_features_from_json, sample_frames


# ═══════════════════════════════════════════════════════════════════════════
# Utility
# ═══════════════════════════════════════════════════════════════════════════

def _find_latest_checkpoint(ckpt_dir: Path, pattern: str = "*.pt") -> Path:
    """Return the .pt file with the highest acc in its name, falling back to newest."""
    pts = sorted(ckpt_dir.glob(pattern))
    if not pts:
        raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}")
    if len(pts) == 1:
        return pts[0]
    # Prefer highest accuracy in filename
    best, best_acc = pts[0], -1.0
    for p in pts:
        try:
            acc = float(p.stem.split("_acc")[-1])
        except (ValueError, IndexError):
            acc = -1.0
        if acc > best_acc:
            best, best_acc = p, acc
    return best


def load_val_data(
    val_dir: Path, label_parser, verbose: bool = True
) -> List[Tuple[List[List[float]], object]]:
    """Walk val_dir → (features_list, label) for every gesture JSON."""
    gestures: List[Tuple[List[List[float]], object]] = []
    if not val_dir.exists():
        print(f"  ⚠  val dir not found: {val_dir}")
        return gestures

    class_dirs = sorted(d for d in val_dir.iterdir() if d.is_dir())
    for class_dir in class_dirs:
        label = label_parser(class_dir.name)
        json_files = sorted(class_dir.glob("*.json"))
        for jp in json_files:
            feats = load_plumbline_features_from_json(jp)
            if feats:
                gestures.append((feats, label))
    if verbose:
        print(f"  Loaded {len(gestures)} gestures from {len(class_dirs)} classes")
    return gestures


def prepare_batch(
    gestures: List[Tuple[List[List[float]], object]],
    target_frames: int,
    device: str = "cpu",
) -> Tuple[torch.Tensor, List[object]]:
    """Sample frames and stack into (N, C, T) tensor."""
    X = np.stack(
        [sample_frames(feats, target_frames) for feats, _ in gestures], axis=0
    )
    labels = [label for _, label in gestures]
    X_t = torch.tensor(X, dtype=torch.float32, device=device).permute(0, 2, 1)
    return X_t, labels


# ═══════════════════════════════════════════════════════════════════════════
# Per-code metrics
# ═══════════════════════════════════════════════════════════════════════════

def multilabel_metrics(
    preds: np.ndarray, trues: np.ndarray, code_names: List[str]
) -> List[dict]:
    """Compute per-code precision, recall, F1, accuracy for multi-label preds.

    Args:
        preds: (N, C) binary predictions
        trues: (N, C) binary ground truth
        code_names: list of C code strings

    Returns list of dicts, one per code.
    """
    results = []
    for i, name in enumerate(code_names):
        tp = int(((preds[:, i] == 1) & (trues[:, i] == 1)).sum())
        fp = int(((preds[:, i] == 1) & (trues[:, i] == 0)).sum())
        fn = int(((preds[:, i] == 0) & (trues[:, i] == 1)).sum())
        tn = int(((preds[:, i] == 0) & (trues[:, i] == 0)).sum())

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0

        # Support: how many samples actually have this code
        support = int(trues[:, i].sum())

        results.append({
            "code": name,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "accuracy": acc,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "support": support,
        })
    return results


def singlelabel_metrics(
    preds: np.ndarray, trues: np.ndarray, code_names: List[str]
) -> List[dict]:
    """Compute per-class precision, recall, F1 for single-label (softmax) preds.

    Args:
        preds: (N,) int array of predicted class indices
        trues: (N,) int array of ground truth class indices
        code_names: list of C class names
    """
    n_classes = len(code_names)
    results = []
    for i, name in enumerate(code_names):
        tp = int(((preds == i) & (trues == i)).sum())
        fp = int(((preds == i) & (trues != i)).sum())
        fn = int(((preds != i) & (trues == i)).sum())
        tn = int(((preds != i) & (trues != i)).sum())

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
        support = int((trues == i).sum())

        results.append({
            "code": name,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "accuracy": acc,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "support": support,
        })
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Confusion / error analysis
# ═══════════════════════════════════════════════════════════════════════════

def print_confusion_matrix(
    preds: np.ndarray, trues: np.ndarray, code_names: List[str], title: str
):
    """Print a confusion matrix for single-label predictions.

    Rows = true labels, Columns = predicted labels.
    Also prints per-class error breakdowns.
    """
    n = len(code_names)
    matrix = np.zeros((n, n), dtype=int)
    for t, p in zip(trues, preds):
        matrix[t, p] += 1

    print(f"\n{'─'*100}")
    print(f"  {title} — Confusion Matrix")
    print(f"  Rows=True, Cols=Predicted")
    print(f"{'─'*100}")

    # Header
    header = "  " + "".join(f"{c:>7s}" for c in code_names) + f"  {'Total':>7s}"
    print(header)
    print(f"  {'─'*7 * (n+1)}")

    for i, name in enumerate(code_names):
        row = "".join(f"{matrix[i, j]:7d}" for j in range(n))
        total = matrix[i].sum()
        print(f"  {name:>4s} {row}  {total:7d}")

    print(f"  {'─'*7 * (n+1)}")
    total_row = "".join(f"{matrix[:, j].sum():7d}" for j in range(n))
    print(f"  {' ':>4s} {total_row}  {matrix.sum():7d}")
    print(f"{'─'*100}")

    # ── Per-class error breakdown ──
    print(f"\n  Per-class error breakdown:")
    print(f"  {'─'*80}")
    for i, name in enumerate(code_names):
        total_i = matrix[i].sum()
        correct = matrix[i, i]
        errors = []
        for j in range(n):
            if j != i and matrix[i, j] > 0:
                errors.append(f"{code_names[j]} ({matrix[i, j]})")
        if errors:
            err_str = ", ".join(errors)
            print(f"  True={name:>4s}: {correct}/{total_i} correct ({correct/total_i:.1%}) → misclassified as: {err_str}")
        else:
            print(f"  True={name:>4s}: {correct}/{total_i} correct ({correct/total_i:.1%}) → all correct!")
    print(f"{'─'*100}")


def print_multilabel_error_analysis(
    preds: np.ndarray, trues: np.ndarray, code_names: List[str], title: str
):
    """Print per-code error breakdown for multi-label predictions.

    For each code, shows:
    - False Negatives: how many times code was present but model said absent
    - False Positives: how many times code was absent but model said present
    - Which other codes were co-active when FN/FP occurred (top co-occurrences)
    """
    n_codes = len(code_names)

    print(f"\n{'─'*100}")
    print(f"  {title} — Multi-label Error Analysis")
    print(f"{'─'*100}")

    for i, name in enumerate(code_names):
        tp = int(((preds[:, i] == 1) & (trues[:, i] == 1)).sum())
        fp = int(((preds[:, i] == 1) & (trues[:, i] == 0)).sum())
        fn = int(((preds[:, i] == 0) & (trues[:, i] == 1)).sum())
        tn = int(((preds[:, i] == 0) & (trues[:, i] == 0)).sum())
        support = int(trues[:, i].sum())

        if support == 0 and fp == 0:
            print(f"\n  [{name}] Support=0, FP=0 — no errors to analyze")
            continue

        print(f"\n  ┌─ [{name}]  Support={support}  ──────────────────────────────")
        print(f"  │  TP={tp}  FP={fp}  FN={fn}  TN={tn}")

        # ── False Negatives: code should have been 1, but model said 0 ──
        if fn > 0:
            fn_mask = (preds[:, i] == 0) & (trues[:, i] == 1)
            fn_indices = np.where(fn_mask)[0]
            print(f"  │  ── False Negatives ({fn} cases): should be {name}=1, predicted {name}=0 ──")

            # What other codes were co-active (true=1) in these FN samples?
            co_active = {}
            for j in range(n_codes):
                if j == i:
                    continue
                co_count = int(trues[fn_indices, j].sum())
                if co_count > 0:
                    co_active[code_names[j]] = co_count

            if co_active:
                sorted_co = sorted(co_active.items(), key=lambda x: -x[1])
                co_str = ", ".join(f"{c}={n}" for c, n in sorted_co)
                print(f"  │    Co-active true labels in these FN cases: {co_str}")

            # What did the model predict instead for these FN samples?
            false_preds = {}
            for j in range(n_codes):
                if j == i:
                    continue
                fp_count = int(preds[fn_indices, j].sum())
                if fp_count > 0:
                    false_preds[code_names[j]] = fp_count

            if false_preds:
                sorted_fp = sorted(false_preds.items(), key=lambda x: -x[1])
                fp_str = ", ".join(f"{c}={n}" for c, n in sorted_fp)
                print(f"  │    Model falsely predicted: {fp_str}")
            else:
                # Model predicted all zeros for these samples
                all_zero_count = int((preds[fn_indices].sum(axis=1) == 0).sum())
                if all_zero_count > 0:
                    print(f"  │    Model predicted ALL ZEROS in {all_zero_count}/{fn} FN cases")

        # ── False Positives: code should have been 0, but model said 1 ──
        if fp > 0:
            fp_mask = (preds[:, i] == 1) & (trues[:, i] == 0)
            fp_indices = np.where(fp_mask)[0]
            print(f"  │  ── False Positives ({fp} cases): should be {name}=0, predicted {name}=1 ──")

            # What other codes were truly active in these FP samples?
            co_active = {}
            for j in range(n_codes):
                if j == i:
                    continue
                co_count = int(trues[fp_indices, j].sum())
                if co_count > 0:
                    co_active[code_names[j]] = co_count

            if co_active:
                sorted_co = sorted(co_active.items(), key=lambda x: -x[1])
                co_str = ", ".join(f"{c}={n}" for c, n in sorted_co)
                print(f"  │    True labels co-occurring in these FP cases: {co_str}")

            # What else did the model incorrectly predict in these FP samples?
            other_fp = {}
            for j in range(n_codes):
                if j == i:
                    continue
                other_fp_count = int(((preds[fp_indices, j] == 1) & (trues[fp_indices, j] == 0)).sum())
                if other_fp_count > 0:
                    other_fp[code_names[j]] = other_fp_count

            if other_fp:
                sorted_ofp = sorted(other_fp.items(), key=lambda x: -x[1])
                ofp_str = ", ".join(f"{c}={n}" for c, n in sorted_ofp)
                print(f"  │    Other codes also falsely predicted: {ofp_str}")

        if fn == 0 and fp == 0:
            print(f"  │  ✓ No errors for this code")

    print(f"{'─'*100}")


def print_metrics_table(results: List[dict], title: str):
    """Pretty-print per-code metrics."""
    print(f"\n{'─'*85}")
    print(f"  {title}")
    print(f"{'─'*85}")
    header = f"  {'Code':>6s}  {'Precision':>9s}  {'Recall':>7s}  {'F1':>7s}  {'Accuracy':>8s}  {'Support':>8s}"
    print(header)
    print(f"  {'─'*6}  {'─'*9}  {'─'*7}  {'─'*7}  {'─'*8}  {'─'*8}")
    for r in results:
        print(
            f"  {r['code']:>6s}  {r['precision']:8.1%}  {r['recall']:7.1%}  "
            f"{r['f1']:7.1%}  {r['accuracy']:8.1%}  {r['support']:8d}"
        )
    # Overall (macro average ignoring zero-support classes)
    valid = [r for r in results if r["support"] > 0]
    if valid:
        macro_prec = np.mean([r["precision"] for r in valid])
        macro_rec = np.mean([r["recall"] for r in valid])
        macro_f1 = np.mean([r["f1"] for r in valid])
        print(f"  {'─'*6}  {'─'*9}  {'─'*7}  {'─'*7}  {'─'*8}  {'─'*8}")
        print(
            f"  {'macro':>6s}  {macro_prec:8.1%}  {macro_rec:7.1%}  "
            f"{macro_f1:7.1%}  {'—':>8s}  {'—':>8s}"
        )
    print(f"{'─'*85}")


# ═══════════════════════════════════════════════════════════════════════════
# Region-specific evaluators
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_floorsupport(device: str = "cpu", plot: bool = False):
    """FloorSupport: multi-label, 4 codes [FT, HN, S, D], BiLSTM, target_frames=256."""
    from FloorSupport.models.multilabel_floor import MultiLabelFloorModel
    from FloorSupport.trainers import _parse_floor_label

    CODES = ["FT", "HN", "S", "D"]
    TARGET_FRAMES = 256
    CKPT = _find_latest_checkpoint(
        PROJECT_ROOT / "FloorSupport/checkpoints/multilabel_floor_lstm"
    )
    VAL_DIR = PROJECT_ROOT / "FloorSupport/dataset/floor_support_raw/val"

    print(f"\n{'█'*85}")
    print(f"  FloorSupport  —  4 codes: {', '.join(CODES)}")
    print(f"{'█'*85}")

    # Load model
    model = MultiLabelFloorModel(num_codes=4, target_frames=TARGET_FRAMES, device=device)
    ckpt = torch.load(str(CKPT), map_location=device, weights_only=False)
    # strict=False allows loading old (mean-pool) checkpoints into new (attn-pool) model
    missing, unexpected = model.model.load_state_dict(
        ckpt["model_state_dict"], strict=False
    )
    if missing:
        print(f"  ⚠  Missing keys (new layers — retrain to use): {missing}")
    if unexpected:
        print(f"  ⚠  Unexpected keys (old architecture): {unexpected}")
    model.model.eval()
    print(f"  Loaded: {CKPT.name}")

    # Load data
    gestures = load_val_data(VAL_DIR, _parse_floor_label)
    if not gestures:
        print("  No validation data found.")
        return

    X, labels_list = prepare_batch(gestures, TARGET_FRAMES, device)
    trues = np.array(labels_list, dtype=np.float32)  # (N, 4)

    # Inference — use pair-group argmax for structurally valid predictions
    with torch.no_grad():
        logits = model.model(X)
        if model._use_pair_groups:
            preds = model._group_logits_to_multi_hot(
                logits, model.pair_groups
            ).cpu().numpy()
        else:
            preds = (torch.sigmoid(logits) > 0.5).float().cpu().numpy()

    # Overall exact match
    exact_match = (preds == trues).all(axis=1).mean()
    print(f"  Overall exact-match accuracy: {exact_match:.1%}")

    results = multilabel_metrics(preds, trues, CODES)
    print_metrics_table(results, "FloorSupport per-code metrics")
    print_multilabel_error_analysis(preds, trues, CODES, "FloorSupport")
    if plot:
        plot_multilabel_dashboard(preds, trues, CODES, "FloorSupport")
        plot_truth_to_prediction(preds, trues, CODES, "FloorSupport")
    return results


def evaluate_spine(device: str = "cpu", plot: bool = False):
    """Spine: single-label softmax, 6 classes [E, F, HG, LF, SR, U], BiLSTM, target_frames=256."""
    from Spine.models.multilabel_spine import MultiLabelSpineModel
    from Spine.trainers import parse_spine_label

    CODES = ["E", "F", "HG", "LF", "SR", "U"]
    TARGET_FRAMES = 256
    CKPT = _find_latest_checkpoint(
        PROJECT_ROOT / "Spine/checkpoints/spine_classifier_lstm"
    )
    VAL_DIR = PROJECT_ROOT / "Spine/dataset/spine_raw/val"

    print(f"\n{'█'*85}")
    print(f"  Spine  —  6 classes: {', '.join(CODES)}")
    print(f"{'█'*85}")

    # Load model
    model = MultiLabelSpineModel(num_classes=6, target_frames=TARGET_FRAMES, device=device)
    ckpt = torch.load(str(CKPT), map_location=device, weights_only=False)
    model.model.load_state_dict(ckpt["model_state_dict"])
    model.model.eval()
    print(f"  Loaded: {CKPT.name}")

    # Load data
    gestures = load_val_data(VAL_DIR, parse_spine_label)
    if not gestures:
        print("  No validation data found.")
        return

    X, labels_list = prepare_batch(gestures, TARGET_FRAMES, device)
    trues = np.array(labels_list, dtype=np.int64)  # (N,) class indices

    # Inference
    with torch.no_grad():
        logits = model.model(X)  # (N, 6)
        preds = logits.argmax(dim=1).cpu().numpy()  # (N,) class indices

    # Overall accuracy
    overall = (preds == trues).mean()
    print(f"  Overall accuracy: {overall:.1%}")

    results = singlelabel_metrics(preds, trues, CODES)
    print_metrics_table(results, "Spine per-class metrics")
    print_confusion_matrix(preds, trues, CODES, "Spine")
    if plot:
        plot_singlelabel_dashboard(preds, trues, CODES, "Spine")
    return results


def evaluate_limbexpression(device: str = "cpu", plot: bool = False):
    """LimbExpression: multi-label, 4→8 codes, CNN, target_frames=128."""
    from LimbExpression.models.multilabel_cnn import MultiLabelCNN, parse_attributes

    CODES_8 = ["LB", "SL", "AS", "A", "G", "UB", "DL", "SY"]
    TARGET_FRAMES = 128
    CKPT = _find_latest_checkpoint(
        PROJECT_ROOT / "LimbExpression/checkpoints/multilabel_cnn"
    )
    VAL_DIR = PROJECT_ROOT / "LimbExpression/dataset/limb_expression_raw/val"

    print(f"\n{'█'*85}")
    print(f"  LimbExpression  —  8 codes: {', '.join(CODES_8)}")
    print(f"{'█'*85}")

    # Load model
    model = MultiLabelCNN(num_heads=4, target_frames=TARGET_FRAMES, device=device)
    ckpt = torch.load(str(CKPT), map_location=device, weights_only=False)
    model.model.load_state_dict(ckpt["model_state_dict"])
    model.model.eval()
    print(f"  Loaded: {CKPT.name}")

    # Load data — labels are 4-element [body, limb, symmetry, contact]
    gestures = load_val_data(VAL_DIR, parse_attributes)
    if not gestures:
        print("  No validation data found.")
        return

    X, labels_list = prepare_batch(gestures, TARGET_FRAMES, device)
    trues_4 = np.array(labels_list, dtype=np.float32)  # (N, 4)

    # Inference
    with torch.no_grad():
        logits_4 = model.model(X)  # (N, 4)
        probs_4 = torch.sigmoid(logits_4).cpu().numpy()
        preds_8 = _expand_preds_4to8(probs_4)

    # Expand ground truth 4→8
    trues_8 = _expand_labels_4to8(trues_4)  # (N, 8)

    # Overall exact match on 8 codes
    exact_match = (preds_8 == trues_8).all(axis=1).mean()
    print(f"  Overall exact-match accuracy (8-code): {exact_match:.1%}")

    results = multilabel_metrics(preds_8, trues_8, CODES_8)
    print_metrics_table(results, "LimbExpression per-code metrics")
    print_multilabel_error_analysis(preds_8, trues_8, CODES_8, "LimbExpression")
    if plot:
        plot_multilabel_dashboard(preds_8, trues_8, CODES_8, "LimbExpression")
    return results


def _expand_preds_4to8(probs_4: np.ndarray) -> np.ndarray:
    """Expand 4-head sigmoid probabilities to 8 binary predictions.

    probs_4: (N, 4) — [p_UB, p_DL, p_AS, p_A]
    Returns: (N, 8) binary — [LB, SL, AS, A, G, UB, DL, SY]

    Each complementary pair is normalised to sum to 1.0 before thresholding.
    Each label is predicted when its normalised probability > 0.5.
    """
    n = probs_4.shape[0]
    eps = 1e-8
    # Expand to 8 probabilities via complement
    lb = 1.0 - probs_4[:, 0]  # LB from body head
    ub = probs_4[:, 0]         # UB from body head
    sl = 1.0 - probs_4[:, 1]  # SL from limb head
    dl = probs_4[:, 1]         # DL from limb head
    asym = probs_4[:, 2]       # AS from symmetry head
    sy = 1.0 - probs_4[:, 2]  # SY from symmetry head
    a = probs_4[:, 3]          # A from contact head
    g = 1.0 - probs_4[:, 3]   # G from contact head

    # Normalise each pair so they sum to 1.0
    # Pair 0: LB↔UB
    pair_sum = lb + ub + eps
    lb, ub = lb / pair_sum, ub / pair_sum
    # Pair 1: SL↔DL
    pair_sum = sl + dl + eps
    sl, dl = sl / pair_sum, dl / pair_sum
    # Pair 2: AS↔SY
    pair_sum = asym + sy + eps
    asym, sy = asym / pair_sum, sy / pair_sum
    # Pair 3: A↔G
    pair_sum = a + g + eps
    a, g = a / pair_sum, g / pair_sum

    # Predict each label when normalised probability > 0.5
    preds_8 = np.zeros((n, 8), dtype=np.float32)
    preds_8[:, 0] = lb > 0.5
    preds_8[:, 1] = sl > 0.5
    preds_8[:, 2] = asym > 0.5
    preds_8[:, 3] = a > 0.5
    preds_8[:, 4] = g > 0.5
    preds_8[:, 5] = ub > 0.5
    preds_8[:, 6] = dl > 0.5
    preds_8[:, 7] = sy > 0.5
    return preds_8


def _expand_labels_4to8(labels_4: np.ndarray) -> np.ndarray:
    """Expand 4-element ground truth to 8-element.

    labels_4: (N, 4) — [body, limb, symmetry, contact]
        body:     0=LB, 1=UB
        limb:     0=SL, 1=DL
        symmetry: 0=SY, 1=AS
        contact:  0=G,  1=A
    Returns: (N, 8) — [LB, SL, AS, A, G, UB, DL, SY]
    """
    n = labels_4.shape[0]
    labels_8 = np.zeros((n, 8), dtype=np.float32)
    labels_8[:, 0] = 1.0 - labels_4[:, 0]  # LB
    labels_8[:, 1] = 1.0 - labels_4[:, 1]  # SL
    labels_8[:, 2] = labels_4[:, 2]         # AS
    labels_8[:, 3] = labels_4[:, 3]         # A
    labels_8[:, 4] = 1.0 - labels_4[:, 3]  # G
    labels_8[:, 5] = labels_4[:, 0]         # UB
    labels_8[:, 6] = labels_4[:, 1]         # DL
    labels_8[:, 7] = 1.0 - labels_4[:, 2]  # SY
    return labels_8


def _expand_8(logits_4):
    """Unused — expand_probs_4to8 equivalent."""
    from LimbExpression.models.multilabel_cnn import expand_probs_4to8
    return expand_probs_4to8(torch.sigmoid(logits_4))


def evaluate_space(device: str = "cpu", plot: bool = False):
    """Space: multi-label, 7 codes [ST, T, RV, SP, H, M, L] in 2 pair groups, BiLSTM."""
    from Space.models.multilabel_space import MultiLabelSpaceModel
    from Space.trainers import parse_space_label

    CODES = ["ST", "T", "RV", "SP", "H", "M", "L"]
    TARGET_FRAMES = 256
    CKPT = _find_latest_checkpoint(
        PROJECT_ROOT / "Space/checkpoints/multilabel_space_lstm"
    )
    VAL_DIR = PROJECT_ROOT / "Space/dataset/space_raw/val"

    print(f"\n{'█'*85}")
    print(f"  Space  —  {len(CODES)} codes: {', '.join(CODES)}")
    print(f"  pair_groups: movement [ST,T,RV,SP]  |  energy [H,M,L]")
    print(f"{'█'*85}")

    # Load model
    model = MultiLabelSpaceModel(num_codes=7, target_frames=TARGET_FRAMES, device=device)
    ckpt = torch.load(str(CKPT), map_location=device, weights_only=False)
    model.model.load_state_dict(ckpt["model_state_dict"])
    model.model.eval()
    print(f"  Loaded: {CKPT.name}")

    # Load data
    gestures = load_val_data(VAL_DIR, parse_space_label)
    if not gestures:
        print("  No validation data found.")
        return

    X, labels_list = prepare_batch(gestures, TARGET_FRAMES, device)
    trues = np.array(labels_list, dtype=np.float32)  # (N, 7)

    # Inference — use pair-group argmax for structurally valid predictions
    with torch.no_grad():
        logits = model.model(X)
        if model._use_pair_groups:
            preds = model._group_logits_to_multi_hot(
                logits, model.pair_groups
            ).cpu().numpy()
        else:
            preds = (torch.sigmoid(logits) > 0.5).float().cpu().numpy()

    # Overall exact match
    exact_match = (preds == trues).all(axis=1).mean()
    print(f"  Overall exact-match accuracy: {exact_match:.1%}")

    results = multilabel_metrics(preds, trues, CODES)
    print_metrics_table(results, "Space per-code metrics")
    print_multilabel_error_analysis(preds, trues, CODES, "Space")
    if plot:
        plot_multilabel_dashboard(preds, trues, CODES, "Space")
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Visualization
# ═══════════════════════════════════════════════════════════════════════════

def plot_multilabel_dashboard(
    preds: np.ndarray,
    trues: np.ndarray,
    code_names: List[str],
    title: str,
    save_path: str = None,
):
    """Generate a multi-panel dashboard for a multi-label classifier.

    Panels:
      1. Per-code metrics bar chart (precision, recall, F1)
      2. Sample-by-sample prediction-vs-truth heatmap (green=correct, red=error)
      3. Per-code error summary (FN, FP counts)
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    n_codes = len(code_names)
    n_samples = len(trues)

    # ── Compute per-code stats ────────────────────────────────────────
    per_code = []
    for i, name in enumerate(code_names):
        tp = int(((preds[:, i] == 1) & (trues[:, i] == 1)).sum())
        fp = int(((preds[:, i] == 1) & (trues[:, i] == 0)).sum())
        fn = int(((preds[:, i] == 0) & (trues[:, i] == 1)).sum())
        tn = int(((preds[:, i] == 0) & (trues[:, i] == 0)).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        support = int(trues[:, i].sum())
        per_code.append({
            "name": name, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": prec, "recall": rec, "f1": f1, "support": support,
        })

    exact_match = (preds == trues).all(axis=1).mean()
    per_bit_acc = (preds == trues).mean()

    # ── Create figure ─────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(f"{title} — Evaluation Dashboard", fontsize=14, fontweight="bold", y=0.98)

    # ── Panel 1: Per-code metrics bar chart ───────────────────────────
    ax1 = fig.add_subplot(2, 3, 1)
    x = np.arange(n_codes)
    width = 0.25
    prec_vals = [r["precision"] for r in per_code]
    rec_vals = [r["recall"] for r in per_code]
    f1_vals = [r["f1"] for r in per_code]

    bars1 = ax1.bar(x - width, prec_vals, width, label="Precision", color="#4CAF50", edgecolor="white")
    bars2 = ax1.bar(x, rec_vals, width, label="Recall", color="#2196F3", edgecolor="white")
    bars3 = ax1.bar(x + width, f1_vals, width, label="F1", color="#FF9800", edgecolor="white")

    # Annotate bars with values
    for bar, val in zip(bars1, prec_vals):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f"{val:.0%}" if val > 0 else "0", ha="center", va="bottom", fontsize=7)
    for bar, val in zip(bars2, rec_vals):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f"{val:.0%}" if val > 0 else "0", ha="center", va="bottom", fontsize=7)

    ax1.set_xticks(x)
    ax1.set_xticklabels(code_names, fontsize=9)
    ax1.set_ylim(0, 1.15)
    ax1.set_ylabel("Score", fontsize=9)
    ax1.set_title("Per-Code Metrics", fontsize=11, fontweight="bold")
    ax1.legend(loc="lower right", fontsize=7)
    ax1.grid(axis="y", alpha=0.3)

    # ── Panel 2: Per-code TP/FP/FN breakdown ──────────────────────────
    ax2 = fig.add_subplot(2, 3, 2)
    x = np.arange(n_codes)
    width = 0.3
    tp_vals = [r["tp"] for r in per_code]
    fp_vals = [r["fp"] for r in per_code]
    fn_vals = [r["fn"] for r in per_code]

    ax2.bar(x - width, tp_vals, width, label="TP (correct=1)", color="#4CAF50", edgecolor="white")
    ax2.bar(x, fp_vals, width, label="FP (false alarm)", color="#F44336", edgecolor="white")
    ax2.bar(x + width, fn_vals, width, label="FN (missed)", color="#FF9800", edgecolor="white")

    for i, (tp_v, fp_v, fn_v) in enumerate(zip(tp_vals, fp_vals, fn_vals)):
        if tp_v > 0:
            ax2.text(i - width, tp_v + max(0.5, tp_v * 0.02), str(tp_v), ha="center", fontsize=7, fontweight="bold")
        if fp_v > 0:
            ax2.text(i, fp_v + max(0.5, fp_v * 0.02), str(fp_v), ha="center", fontsize=7, color="#C62828")
        if fn_v > 0:
            ax2.text(i + width, fn_v + max(0.5, fn_v * 0.02), str(fn_v), ha="center", fontsize=7, color="#E65100")

    ax2.set_xticks(x)
    ax2.set_xticklabels(code_names, fontsize=9)
    ax2.set_ylabel("Count", fontsize=9)
    ax2.set_title("Per-Code Error Breakdown", fontsize=11, fontweight="bold")
    ax2.legend(loc="upper right", fontsize=7)
    ax2.grid(axis="y", alpha=0.3)

    # ── Panel 3: Sample-by-sample heatmap ─────────────────────────────
    ax3 = fig.add_subplot(2, 3, (3, 6))  # spans two columns on second row

    # Build a colour matrix: green=correct, red=wrong-0, orange=wrong-1
    # We'll encode: correct prediction = 0, FN = 1, FP = 2
    heatmap = np.zeros((n_samples, n_codes), dtype=int)
    for i in range(n_codes):
        for j in range(n_samples):
            if preds[j, i] == trues[j, i]:
                heatmap[j, i] = 0  # correct
            elif trues[j, i] == 1:
                heatmap[j, i] = 1  # FN: should be 1, predicted 0
            else:
                heatmap[j, i] = 2  # FP: should be 0, predicted 1

    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(["#4CAF50", "#F44336", "#FF9800"])  # green, red, orange

    im = ax3.imshow(heatmap.T, aspect="auto", cmap=cmap, interpolation="nearest", vmin=0, vmax=2)

    ax3.set_yticks(range(n_codes))
    ax3.set_yticklabels(code_names, fontsize=9)
    ax3.set_xlabel(f"Sample index (N={n_samples})", fontsize=9)
    ax3.set_title(f"Prediction vs Truth Heatmap  "
                  f"(exact match: {exact_match:.1%} | per-bit acc: {per_bit_acc:.1%})",
                  fontsize=11, fontweight="bold")

    # Colour legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#4CAF50", label="Correct"),
        Patch(facecolor="#F44336", label="FN (missed: true=1, pred=0)"),
        Patch(facecolor="#FF9800", label="FP (false alarm: true=0, pred=1)"),
    ]
    ax3.legend(handles=legend_elements, loc="upper right", fontsize=7,
               ncol=3, framealpha=0.9)

    # ── Panel 4: Summary text ─────────────────────────────────────────
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.axis("off")
    summary_lines = [
        f"Exact match: {exact_match:.1%}",
        f"Per-bit accuracy: {per_bit_acc:.1%}",
        f"Total samples: {n_samples}",
        "",
        "Per-code recall (when truth=1):",
    ]
    for r in per_code:
        marker = "⚠" if r["recall"] < 0.85 else "✓"
        summary_lines.append(f"  {marker} {r['name']}: {r['recall']:.1%}  "
                            f"(support={r['support']}, FN={r['fn']})")
    summary_lines.append("")
    summary_lines.append("Per-code precision (when pred=1):")
    for r in per_code:
        marker = "⚠" if r["precision"] < 0.85 else "✓"
        summary_lines.append(f"  {marker} {r['name']}: {r['precision']:.1%}  "
                            f"(FP={r['fp']})")

    ax4.text(0.05, 0.95, "\n".join(summary_lines), transform=ax4.transAxes,
             fontsize=9, fontfamily="monospace", verticalalignment="top",
             bbox=dict(boxstyle="round", facecolor="whitesmoke", alpha=0.8))

    # ── Panel 5: Per-code recall bar (highlight problem codes) ────────
    ax5 = fig.add_subplot(2, 3, 5)
    colors = ["#F44336" if r["recall"] < 0.7 else "#FF9800" if r["recall"] < 0.85 else "#4CAF50"
              for r in per_code]
    bars = ax5.barh(code_names, [r["recall"] for r in per_code], color=colors, edgecolor="white")
    for bar, r in zip(bars, per_code):
        ax5.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                 f"{r['recall']:.1%} ({r['tp']}/{r['support']})",
                 va="center", fontsize=8)
    ax5.set_xlim(0, 1.25)
    ax5.set_title("Recall by Code (green≥85%, yellow≥70%, red<70%)", fontsize=10, fontweight="bold")
    ax5.axvline(x=0.85, color="green", linestyle="--", alpha=0.5)
    ax5.grid(axis="x", alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path is None:
        save_path = f"{title.lower().replace(' ', '_')}_dashboard.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  📊 Dashboard saved: {save_path}")
    plt.close(fig)


def plot_singlelabel_dashboard(
    preds: np.ndarray,
    trues: np.ndarray,
    code_names: List[str],
    title: str,
    save_path: str = None,
):
    """Generate a dashboard for a single-label (softmax) classifier."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_classes = len(code_names)
    n_samples = len(trues)

    # Confusion matrix
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(trues, preds):
        cm[t, p] += 1

    overall = (preds == trues).mean()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"{title} — Evaluation Dashboard  (overall acc: {overall:.1%})",
                 fontsize=14, fontweight="bold")

    # ── Confusion matrix heatmap ──────────────────────────────────────
    ax1 = axes[0]
    im = ax1.imshow(cm, cmap="YlOrRd", aspect="auto")
    ax1.set_xticks(range(n_classes))
    ax1.set_xticklabels(code_names)
    ax1.set_yticks(range(n_classes))
    ax1.set_yticklabels(code_names)
    ax1.set_xlabel("Predicted", fontsize=10)
    ax1.set_ylabel("True", fontsize=10)
    ax1.set_title("Confusion Matrix", fontsize=11, fontweight="bold")

    for i in range(n_classes):
        for j in range(n_classes):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax1.text(j, i, str(cm[i, j]), ha="center", va="center",
                     fontsize=9, fontweight="bold", color=color)

    fig.colorbar(im, ax=ax1, shrink=0.8)

    # ── Per-class metrics ─────────────────────────────────────────────
    ax2 = axes[1]
    per_class = []
    for i, name in enumerate(code_names):
        total_i = cm[i].sum()
        correct = cm[i, i]
        rec = correct / total_i if total_i > 0 else 0.0
        # Precision: of all predicted as i, how many were truly i
        pred_total = cm[:, i].sum()
        prec = correct / pred_total if pred_total > 0 else 0.0
        per_class.append({"name": name, "recall": rec, "precision": prec,
                          "correct": correct, "total": total_i})

    x = np.arange(n_classes)
    width = 0.3
    ax2.bar(x - width / 2, [r["recall"] for r in per_class], width,
            label="Recall", color="#2196F3", edgecolor="white")
    ax2.bar(x + width / 2, [r["precision"] for r in per_class], width,
            label="Precision", color="#4CAF50", edgecolor="white")
    ax2.set_xticks(x)
    ax2.set_xticklabels(code_names)
    ax2.set_ylim(0, 1.15)
    ax2.set_ylabel("Score")
    ax2.set_title("Per-Class Precision & Recall", fontsize=11, fontweight="bold")
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)

    # Annotate
    for i, r in enumerate(per_class):
        ax2.text(i - width / 2, r["recall"] + 0.02, f"{r['recall']:.0%}", ha="center", fontsize=7)
        ax2.text(i + width / 2, r["precision"] + 0.02, f"{r['precision']:.0%}", ha="center", fontsize=7)

    plt.tight_layout()

    if save_path is None:
        save_path = f"{title.lower().replace(' ', '_')}_dashboard.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  📊 Dashboard saved: {save_path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# Truth → Prediction Visualisation  ("when true label is X, what did model say?")
# ═══════════════════════════════════════════════════════════════════════════

def _bits_to_label(bits: np.ndarray, code_names: List[str]) -> str:
    """Convert a 4-bit vector like [1,0,0,1] into a readable string "FT+D"."""
    active = [code_names[i] for i, v in enumerate(bits) if v == 1]
    return "+".join(active) if active else "(none)"


def _bits_to_dirname(bits: np.ndarray, code_names: List[str]) -> str:
    """Convert a 4-bit vector into the directory-name format e.g. 'd_ft_dual_feet'."""
    active = [code_names[i] for i, v in enumerate(bits) if v == 1]
    active_lower = [c.lower() for c in active]
    # Try to match known patterns: if D+FT → d_ft_dual_feet, S+FT → s_ft_single_feet
    has_d = "D" in code_names and bits[list(code_names).index("D")] == 1
    has_s = "S" in code_names and bits[list(code_names).index("S")] == 1
    has_ft = "FT" in code_names and bits[list(code_names).index("FT")] == 1
    has_hn = "HN" in code_names and bits[list(code_names).index("HN")] == 1
    if has_d and has_ft:
        return "d_ft_dual_feet"
    elif has_d and has_hn:
        return "d_hn_dual_hand"
    elif has_s and has_ft:
        return "s_ft_single_feet"
    return "+".join(active_lower)


def plot_truth_to_prediction(
    preds: np.ndarray,
    trues: np.ndarray,
    code_names: List[str],
    title: str,
    save_path: str = None,
):
    """For each true label (class), show a bar chart of what the model predicted.

    This is the intuitive "when truth is X, what happened?" view.
    Green bars = correct predictions.  Red bars = wrong predictions.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from collections import Counter

    # Group samples by their true label (4-bit vector → string)
    true_groups: dict = {}  # true_label_str → list of predicted label strings
    for i in range(len(trues)):
        true_key = _bits_to_label(trues[i], code_names)
        pred_key = _bits_to_label(preds[i], code_names)
        if true_key not in true_groups:
            true_groups[true_key] = []
        true_groups[true_key].append(pred_key)

    # Sort groups by size (largest first)
    sorted_groups = sorted(true_groups.items(), key=lambda x: -len(x[1]))
    n_groups = len(sorted_groups)

    # Gather all unique prediction labels across all groups
    all_pred_labels = set()
    for _, preds_list in sorted_groups:
        all_pred_labels.update(preds_list)
    all_pred_labels = sorted(all_pred_labels,
                             key=lambda x: (x != "", x))  # put empty last

    # Assign a consistent colour to each prediction label
    # correct pred = green shades, wrong = red/orange shades
    import matplotlib.cm as cm
    n_preds = len(all_pred_labels)

    fig, axes = plt.subplots(n_groups, 1, figsize=(14, 3 * n_groups),
                             squeeze=False)
    fig.suptitle(f"{title} — Truth → Prediction Flow\n"
                 f"(for each true label, what the model predicted)",
                 fontsize=14, fontweight="bold", y=0.99)

    for idx, (true_label, pred_list) in enumerate(sorted_groups):
        ax = axes[idx][0]
        total = len(pred_list)
        counts = Counter(pred_list)

        # Build bars in consistent order
        bar_labels = []
        bar_counts = []
        bar_colors = []
        for pl in all_pred_labels:
            c = counts.get(pl, 0)
            if c > 0:
                bar_labels.append(pl)
                bar_counts.append(c)
                if pl == true_label:
                    bar_colors.append("#2E7D32")   # dark green = correct
                elif pl == "(none)":
                    bar_colors.append("#B71C1C")    # dark red = all zeros
                else:
                    bar_colors.append("#E65100")    # orange = wrong combo

        y_pos = range(len(bar_labels))
        bars = ax.barh(y_pos, bar_counts, color=bar_colors, edgecolor="white", height=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(bar_labels, fontsize=10)

        # Annotate bars with count and percentage
        for j, (bar, count) in enumerate(zip(bars, bar_counts)):
            pct = count / total * 100
            ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                    f"{count}  ({pct:.0f}%)", va="center", fontsize=9,
                    fontweight="bold" if bar_labels[j] == true_label else "normal")

        # Correct count
        correct = counts.get(true_label, 0)
        acc = correct / total * 100

        ax.set_title(f"True: {true_label}   (n={total})   →   "
                     f"Correct: {correct}/{total} = {acc:.0f}%",
                     fontsize=12, fontweight="bold",
                     color="#2E7D32" if acc >= 80 else "#E65100" if acc >= 50 else "#B71C1C")
        ax.set_xlim(0, max(bar_counts) * 1.5 + 1)
        ax.set_xlabel("Number of samples", fontsize=9)
        ax.grid(axis="x", alpha=0.3)

        # Legend
        from matplotlib.patches import Patch
        legend_el = [
            Patch(facecolor="#2E7D32", label="Correct"),
            Patch(facecolor="#E65100", label="Wrong label combo"),
            Patch(facecolor="#B71C1C", label="Predicted nothing"),
        ]
        ax.legend(handles=legend_el, loc="lower right", fontsize=8, ncol=3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path is None:
        save_path = f"{title.lower().replace(' ', '_')}_truth_to_pred.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  📊 Truth→Prediction chart saved: {save_path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Per-code evaluation of all BAT models")
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    parser.add_argument("--plot", action="store_true",
                        help="Generate and save visualization dashboards")
    parser.add_argument(
        "--region", nargs="*", default=["all"],
        choices=["all", "FloorSupport", "Spine", "LimbExpression", "Space"],
        help="Which region(s) to evaluate (default: all)",
    )
    args = parser.parse_args()

    regions = args.region
    if "all" in regions:
        regions = ["FloorSupport", "Spine", "LimbExpression", "Space"]

    all_results: Dict[str, List[dict]] = {}

    for region in regions:
        try:
            if region == "FloorSupport":
                all_results[region] = evaluate_floorsupport(args.device, plot=args.plot)
            elif region == "Spine":
                all_results[region] = evaluate_spine(args.device, plot=args.plot)
            elif region == "LimbExpression":
                all_results[region] = evaluate_limbexpression(args.device, plot=args.plot)
            elif region == "Space":
                all_results[region] = evaluate_space(args.device, plot=args.plot)
        except Exception as e:
            print(f"  ✗ Error evaluating {region}: {e}")
            import traceback
            traceback.print_exc()

    # ── Summary ────────────────────────────────────────────────────────
    print(f"\n\n{'█'*85}")
    print(f"  SUMMARY: Per-code recall  (\"when ground truth is X, how often correct?\")")
    print(f"{'█'*85}")
    for region, results in all_results.items():
        if results:
            parts = [f"{r['code']}={r['recall']:.1%}" for r in results]
            print(f"  {region:>20s}:  {', '.join(parts)}")

    print()


if __name__ == "__main__":
    main()
