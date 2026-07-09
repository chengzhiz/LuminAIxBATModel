"""Confusion heatmaps for all BAT model categories.

Generates one heatmap per region:
  FloorSupport  — 4 valid combos (pair-group constrained)
  Spine         — 6-class single-label confusion matrix
  LimbExpression — 8-code multi-label, truth→pred by combo
  Space         — 5-code multi-label, truth→pred by combo
"""

import sys
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, ListedColormap

PROJECT_ROOT = Path(__file__).resolve().parent

# ═══════════════════════════════════════════════════════════════════════════
# Shared utilities
# ═══════════════════════════════════════════════════════════════════════════

def bits_to_label(bits, codes):
    active = [codes[i] for i, v in enumerate(bits) if v == 1]
    return "+".join(active) if active else "∅"


def _filter_nonempty(cm, labels):
    """Remove rows (and corresponding columns) with zero samples.

    Returns (filtered_cm, filtered_labels).  Labels whose row has at
    least one sample are kept; the rest are dropped.
    """
    keep = cm.sum(axis=1) > 0
    return cm[keep][:, keep], [l for l, k in zip(labels, keep) if k]


def _find_latest_checkpoint(ckpt_dir, pattern="*.pt"):
    pts = sorted(Path(ckpt_dir).glob(pattern))
    if not pts:
        raise FileNotFoundError(f"No checkpoints in {ckpt_dir}")
    best, best_acc = pts[0], -1.0
    for p in pts:
        try:
            acc = float(p.stem.split("_acc")[-1])
        except (ValueError, IndexError):
            acc = -1.0
        if acc > best_acc:
            best, best_acc = p, acc
    return best


def draw_heatmap(cm, row_labels, col_labels, title, out_path,
                 single_label=False):
    """Draw a confusion heatmap and save to out_path.

    cm: (R, C) numpy array of counts
    row_labels: list of R strings (true labels)
    col_labels: list of C strings (predicted labels)
    title: figure title
    single_label: if True, highlight diagonal; if False, highlight exact matches
    """
    n_rows, n_cols = cm.shape
    figsize_x = max(8, n_cols * 1.15)
    figsize_y = max(6, n_rows * 0.95)

    fig, ax = plt.subplots(figsize=(figsize_x, figsize_y))

    vmax = max(cm.max(), 1)
    im = ax.imshow(cm, cmap="YlOrRd", aspect="equal",
                   norm=LogNorm(vmax=vmax, vmin=0.5),
                   interpolation="nearest")

    row_totals = cm.sum(axis=1)

    for i in range(n_rows):
        for j in range(n_cols):
            count = cm[i, j]
            if count == 0:
                continue
            row_total = row_totals[i]
            pct = count / row_total * 100 if row_total > 0 else 0.0
            label = f"{count}\n({pct:.0f}%)"

            if i == j and row_total > 0:
                text_color = "#1B5E20"
                weight = "bold"
            else:
                frac = count / vmax
                text_color = "white" if frac > 0.35 else "black"
                weight = "normal"
            ax.text(j, i, label, ha="center", va="center",
                    fontsize=10, color=text_color, fontweight=weight)

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(col_labels, fontsize=10, rotation=45, ha="right")
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels, fontsize=10)
    ax.set_xlabel("Predicted", fontsize=12, fontweight="bold")
    ax.set_ylabel("True", fontsize=12, fontweight="bold")

    # Green border on diagonal
    for i in range(min(n_rows, n_cols)):
        if cm[i, i] > 0:
            rect = plt.Rectangle((i - 0.5, i - 0.5), 1, 1,
                                 linewidth=3.0, edgecolor="#2E7D32",
                                 facecolor="none", zorder=10)
            ax.add_patch(rect)

    exact = np.trace(cm) / cm.sum() * 100 if cm.sum() > 0 else 0
    ax.set_title(title, fontsize=13, fontweight="bold", pad=15)

    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label("Sample count (log scale)", fontsize=10)

    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"  Saved: {out_path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# FloorSupport
# ═══════════════════════════════════════════════════════════════════════════

def heatmap_floorsupport(split="val"):
    sys.path.insert(0, str(PROJECT_ROOT / "FloorSupport"))
    from FloorSupport.models.multilabel_floor import MultiLabelFloorModel
    from FloorSupport.trainers import FloorSupportMultiLabelTrainer

    CODES = ["FT", "HN", "S", "D"]
    CKPT = _find_latest_checkpoint(
        PROJECT_ROOT / "FloorSupport/checkpoints/multilabel_floor_lstm"
    )
    DATA_DIR = PROJECT_ROOT / "FloorSupport/dataset/floor_support_raw"

    split_label = "test" if split == "test" else "validation"

    model = MultiLabelFloorModel(num_codes=4, target_frames=256, device="cpu")
    ckpt = torch.load(str(CKPT), map_location="cpu", weights_only=False)
    model.model.load_state_dict(ckpt["model_state_dict"])
    model.model.eval()

    trainer = FloorSupportMultiLabelTrainer(
        model=model, data_dir=str(DATA_DIR), epochs=1
    )
    gestures = trainer._load_gestures(split)
    print(f"\n{'='*60}")
    print(f"  FloorSupport ({split_label})  —  {CKPT.name}")
    print(f"{'='*60}")
    print(f"  {len(gestures)} {split_label} gestures")

    all_probs, all_truths = [], []
    for features, label in gestures:
        probs = model.predict_proba([(features, label)])[0]
        all_probs.append(probs)
        all_truths.append(label)

    all_probs = np.array(all_probs)
    all_truths = np.array(all_truths)
    all_preds = (all_probs > 0.5).astype(int)

    # Only 4 valid combos
    valid_combos = [(1, 0, 0, 1), (1, 0, 1, 0), (0, 1, 0, 1), (0, 1, 1, 0)]
    valid_combos.sort(key=lambda b: bits_to_label(b, CODES))
    all_combos = [bits_to_label(b, CODES) for b in valid_combos]
    combo_to_idx = {c: i for i, c in enumerate(all_combos)}

    cm = np.zeros((len(all_combos), len(all_combos)), dtype=int)
    for t, p in zip(all_truths.astype(int), all_preds):
        cm[combo_to_idx[bits_to_label(t, CODES)],
           combo_to_idx[bits_to_label(p, CODES)]] += 1

    for i, c in enumerate(all_combos):
        total = cm[i].sum()
        correct = cm[i, i]
        if total > 0:
            print(f"  {c:>8s}: {correct}/{total} correct ({correct/total:.0%})")
        else:
            print(f"  {c:>8s}: (no samples)")

    # Only show categories that have data
    cm_filt, combos_filt = _filter_nonempty(cm, all_combos)

    exact = (all_preds == all_truths).all(axis=1).mean()
    N = len(all_probs)
    out_name = f"heatmap_floorsupport.png" if split == "val" else f"heatmap_floorsupport_test.png"
    draw_heatmap(
        cm_filt, combos_filt, combos_filt,
        f"FloorSupport ({split_label}) — True vs Predicted\n"
        f"{N} samples  •  {len(combos_filt)}/{len(all_combos)} combos  •  "
        f"exact match: {exact:.1%}",
        f"assets/{out_name}"
    )


# ═══════════════════════════════════════════════════════════════════════════
# Spine  (single-label 6-class softmax)
# ═══════════════════════════════════════════════════════════════════════════

def heatmap_spine(split="val"):
    sys.path.insert(0, str(PROJECT_ROOT / "Spine"))
    from Spine.models.multilabel_spine import MultiLabelSpineModel
    from Spine.trainers import parse_spine_label

    CODES = ["E", "F", "HG", "LF", "SR", "U"]
    CKPT = _find_latest_checkpoint(
        PROJECT_ROOT / "Spine/checkpoints/spine_classifier_lstm"
    )
    DATA_DIR = PROJECT_ROOT / "Spine/dataset/spine_raw"

    split_label = "test" if split == "test" else "validation"

    model = MultiLabelSpineModel(num_classes=6, target_frames=256, device="cpu")
    ckpt = torch.load(str(CKPT), map_location="cpu", weights_only=False)
    model.model.load_state_dict(ckpt["model_state_dict"])
    model.model.eval()

    # Load data
    from plumbline_features import load_plumbline_features_from_json
    from FloorSupport.models._gesture_base import sample_frames

    data_dir_split = DATA_DIR / split
    gestures = []
    if data_dir_split.exists():
        for class_dir in sorted(data_dir_split.iterdir()):
            if not class_dir.is_dir():
                continue
            label = parse_spine_label(class_dir.name)
            for jp in sorted(class_dir.glob("*.json")):
                feats = load_plumbline_features_from_json(jp)
                if feats:
                    gestures.append((feats, label))

    print(f"\n{'='*60}")
    print(f"  Spine ({split_label})  —  {CKPT.name}")
    print(f"{'='*60}")
    print(f"  {len(gestures)} {split_label} gestures")

    X = np.stack([sample_frames(feats, 256) for feats, _ in gestures], axis=0)
    trues = np.array([label for _, label in gestures], dtype=np.int64)
    X_t = torch.tensor(X, dtype=torch.float32).permute(0, 2, 1)

    with torch.no_grad():
        logits = model.model(X_t)
        preds = logits.argmax(dim=1).cpu().numpy()

    n_classes = len(CODES)
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(trues, preds):
        cm[t, p] += 1

    for i, c in enumerate(CODES):
        total = cm[i].sum()
        correct = cm[i, i]
        if total > 0:
            print(f"  {c:>6s}: {correct}/{total} correct ({correct/total:.0%})")
        else:
            print(f"  {c:>6s}: (no samples)")

    # Only show classes that have data
    cm_filt, codes_filt = _filter_nonempty(cm, CODES)

    out_name = f"heatmap_spine.png" if split == "val" else f"heatmap_spine_test.png"
    overall = (preds == trues).mean()
    draw_heatmap(
        cm_filt, codes_filt, codes_filt,
        f"Spine ({split_label}) — True vs Predicted\n"
        f"{len(gestures)} samples  •  {len(codes_filt)}/{n_classes} classes  •  "
        f"accuracy: {overall:.1%}",
        f"assets/{out_name}",
        single_label=True
    )


# ═══════════════════════════════════════════════════════════════════════════
# LimbExpression  (8-code multi-label via 4-head expand)
# ═══════════════════════════════════════════════════════════════════════════

def heatmap_limb(split="val"):
    sys.path.insert(0, str(PROJECT_ROOT / "LimbExpression"))
    from LimbExpression.models.multilabel_cnn import MultiLabelCNN, parse_attributes

    CODES = ["LB", "SL", "AS", "A", "G", "UB", "DL", "SY"]
    CKPT = _find_latest_checkpoint(
        PROJECT_ROOT / "LimbExpression/checkpoints/multilabel_cnn"
    )
    DATA_DIR = PROJECT_ROOT / "LimbExpression/dataset/limb_expression_raw"

    split_label = "test" if split == "test" else "validation"

    model = MultiLabelCNN(num_heads=4, target_frames=128, device="cpu")
    ckpt = torch.load(str(CKPT), map_location="cpu", weights_only=False)
    model.model.load_state_dict(ckpt["model_state_dict"])
    model.model.eval()

    from plumbline_features import load_plumbline_features_from_json
    from FloorSupport.models._gesture_base import sample_frames

    data_dir_split = DATA_DIR / split
    gestures = []
    if data_dir_split.exists():
        for class_dir in sorted(data_dir_split.iterdir()):
            if not class_dir.is_dir():
                continue
            label = parse_attributes(class_dir.name)
            for jp in sorted(class_dir.glob("*.json")):
                feats = load_plumbline_features_from_json(jp)
                if feats:
                    gestures.append((feats, label))

    print(f"\n{'='*60}")
    print(f"  LimbExpression ({split_label})  —  {CKPT.name}")
    print(f"{'='*60}")
    print(f"  {len(gestures)} {split_label} gestures")

    X = np.stack([sample_frames(feats, 128) for feats, _ in gestures], axis=0)
    trues_4 = np.array([label for _, label in gestures], dtype=np.float32)
    X_t = torch.tensor(X, dtype=torch.float32).permute(0, 2, 1)

    with torch.no_grad():
        logits_4 = model.model(X_t)
        probs_4 = torch.sigmoid(logits_4).cpu().numpy()

    # Expand 4→8 with per-pair normalisation
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

    # Normalise each pair to sum to 1.0
    pair_sum = lb + ub + eps; lb, ub = lb / pair_sum, ub / pair_sum
    pair_sum = sl + dl + eps; sl, dl = sl / pair_sum, dl / pair_sum
    pair_sum = asym + sy + eps; asym, sy = asym / pair_sum, sy / pair_sum
    pair_sum = a + g + eps; a, g = a / pair_sum, g / pair_sum

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

    trues_8 = np.zeros((n, 8), dtype=np.float32)
    trues_8[:, 0] = 1.0 - trues_4[:, 0]  # LB
    trues_8[:, 1] = 1.0 - trues_4[:, 1]  # SL
    trues_8[:, 2] = trues_4[:, 2]         # AS
    trues_8[:, 3] = trues_4[:, 3]         # A
    trues_8[:, 4] = 1.0 - trues_4[:, 3]  # G
    trues_8[:, 5] = trues_4[:, 0]         # UB
    trues_8[:, 6] = trues_4[:, 1]         # DL
    trues_8[:, 7] = 1.0 - trues_4[:, 2]  # SY

    true_labels = [bits_to_label(t.astype(int), CODES) for t in trues_8]
    pred_labels = [bits_to_label(p.astype(int), CODES) for p in preds_8]

    # Fixed grid of all 16 valid combos (4 complementary pairs → 2^4 = 16)
    all_4bit = [
        [0,0,0,0],[0,0,0,1],[0,0,1,0],[0,0,1,1],
        [0,1,0,0],[0,1,0,1],[0,1,1,0],[0,1,1,1],
        [1,0,0,0],[1,0,0,1],[1,0,1,0],[1,0,1,1],
        [1,1,0,0],[1,1,0,1],[1,1,1,0],[1,1,1,1],
    ]
    all_8bit = []
    for vec in all_4bit:
        bits = np.array([1-vec[0], 1-vec[1], vec[2], vec[3],
                         1-vec[3], vec[0], vec[1], 1-vec[2]], dtype=int)
        all_8bit.append(bits)
    all_combos = sorted(
        [bits_to_label(b, CODES) for b in all_8bit],
        key=lambda x: (x.count("+"), x)
    )
    combo_to_idx = {c: i for i, c in enumerate(all_combos)}
    n_combos = len(all_combos)

    cm = np.zeros((n_combos, n_combos), dtype=int)
    for t, p in zip(true_labels, pred_labels):
        cm[combo_to_idx[t], combo_to_idx[p]] += 1

    for c in all_combos:
        i = combo_to_idx[c]
        total = cm[i].sum()
        correct = cm[i, i]
        if total > 0:
            print(f"  {c:>12s}: {correct}/{total} correct ({correct/total:.0%})")
        else:
            print(f"  {c:>12s}: (no samples)")

    # Only show combos that have data
    cm_filt, combos_filt = _filter_nonempty(cm, all_combos)

    out_name = f"heatmap_limbexpression.png" if split == "val" else f"heatmap_limbexpression_test.png"
    exact = (preds_8 == trues_8).all(axis=1).mean()
    draw_heatmap(
        cm_filt, combos_filt, combos_filt,
        f"LimbExpression ({split_label}) — True vs Predicted\n"
        f"{len(gestures)} samples  •  {len(combos_filt)}/{n_combos} combos  •  "
        f"exact match: {exact:.1%}",
        f"assets/{out_name}"
    )


# ═══════════════════════════════════════════════════════════════════════════
# Space  (7-code multi-label with pair groups)
# ═══════════════════════════════════════════════════════════════════════════

def heatmap_space(split="val"):
    """Confusion heatmap for Space — 7 codes, 2 pair groups, 12 valid combos.

    Movement group (indices 0-3): ST, T, RV, SP  (4-way softmax)
    Energy group   (indices 4-6): H,  M,  L      (3-way softmax)
    3 energy × 4 movement = 12 valid combos.
    """
    sys.path.insert(0, str(PROJECT_ROOT / "Space"))
    from Space.models.multilabel_space import MultiLabelSpaceModel
    from Space.trainers import parse_space_label

    CODES = ["ST", "T", "RV", "SP", "H", "M", "L"]
    CKPT = _find_latest_checkpoint(
        PROJECT_ROOT / "Space/checkpoints/multilabel_space_lstm"
    )
    DATA_DIR = PROJECT_ROOT / "Space/dataset/space_raw"

    split_label = "test" if split == "test" else "validation"

    model = MultiLabelSpaceModel(num_codes=7, target_frames=256, device="cpu")
    ckpt = torch.load(str(CKPT), map_location="cpu", weights_only=False)
    model.model.load_state_dict(ckpt["model_state_dict"])
    model.model.eval()

    from plumbline_features import load_plumbline_features_from_json
    from FloorSupport.models._gesture_base import sample_frames

    data_dir_split = DATA_DIR / split
    gestures = []
    if data_dir_split.exists():
        for class_dir in sorted(data_dir_split.iterdir()):
            if not class_dir.is_dir():
                continue
            label = parse_space_label(class_dir.name)
            for jp in sorted(class_dir.glob("*.json")):
                feats = load_plumbline_features_from_json(jp)
                if feats:
                    gestures.append((feats, label))

    print(f"\n{'='*60}")
    print(f"  Space ({split_label})  —  {CKPT.name}")
    print(f"{'='*60}")
    print(f"  {len(gestures)} {split_label} gestures")

    X = np.stack([sample_frames(feats, 256) for feats, _ in gestures], axis=0)
    trues = np.array([label for _, label in gestures], dtype=np.float32)
    if trues.ndim == 1:
        trues = trues.reshape(-1, 1)
    if trues.shape[1] < 7:
        trues = np.pad(trues, ((0, 0), (0, 7 - trues.shape[1])), constant_values=0.0)
    X_t = torch.tensor(X, dtype=torch.float32).permute(0, 2, 1)

    with torch.no_grad():
        logits = model.model(X_t)
        if model._use_pair_groups:
            preds = model._group_logits_to_multi_hot(
                logits, model.pair_groups
            ).cpu().numpy()
        else:
            preds = (torch.sigmoid(logits) > 0.5).float().cpu().numpy()

    true_labels = [bits_to_label(t.astype(int), CODES) for t in trues]
    pred_labels = [bits_to_label(p.astype(int), CODES) for p in preds]

    # Show ALL 12 valid combos (4 movement x 3 energy), even empty ones
    all_combos = [f"{mv}+{en}" for mv in ["ST", "T", "RV", "SP"]
                               for en in ["H", "M", "L"]]
    combo_to_idx = {c: i for i, c in enumerate(all_combos)}
    n_combos = len(all_combos)  # 12

    cm = np.zeros((n_combos, n_combos), dtype=int)
    for t, p in zip(true_labels, pred_labels):
        cm[combo_to_idx[t], combo_to_idx[p]] += 1

    for c in all_combos:
        i = combo_to_idx[c]
        total = cm[i].sum()
        correct = cm[i, i]
        if total > 0:
            print(f"  {c:>8s}: {correct}/{total} correct ({correct/total:.0%})")
        else:
            print(f"  {c:>8s}: (no samples)")

    # Only show combos that have data
    cm_filt, combos_filt = _filter_nonempty(cm, all_combos)

    out_name = f"heatmap_space.png" if split == "val" else f"heatmap_space_test.png"
    exact = (preds == trues).all(axis=1).mean()
    draw_heatmap(
        cm_filt, combos_filt, combos_filt,
        f"Space ({split_label}) — True vs Predicted\n"
        f"{len(gestures)} samples  •  {len(combos_filt)}/{n_combos} combos  •  "
        f"exact match: {exact:.1%}",
        f"assets/{out_name}"
    )


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    for split in ["val", "test"]:
        for func in [heatmap_floorsupport, heatmap_spine, heatmap_limb, heatmap_space]:
            try:
                func(split)
            except FileNotFoundError as e:
                print(f"  ⚠  Skipping: {e}")
            except Exception as e:
                print(f"  ✗  Error: {e}")
                import traceback
                traceback.print_exc()

    print(f"\nAll heatmaps saved to assets/")
