"""Sweep D-weight values for FloorSupport S/D pair to find optimal balance."""
import sys, numpy as np, torch
from pathlib import Path
from collections import Counter

sys.path.insert(0, "FloorSupport")
from FloorSupport.models.multilabel_floor import MultiLabelFloorModel
from FloorSupport.trainers import FloorSupportMultiLabelTrainer

CODES = ["FT", "HN", "S", "D"]
DATA_DIR = "FloorSupport/dataset/floor_support_raw"
DEVICE = "cpu"

weights_to_try = [1.0, 1.5, 2.0, 2.5, 3.0]
results = {}

for dw in weights_to_try:
    print(f"\n{'='*50}")
    print(f"  D-weight = {dw}")
    print(f"{'='*50}")

    model = MultiLabelFloorModel(num_codes=4, target_frames=256, device=DEVICE)
    # Override pair_weights
    model._pair_weights = [None, torch.tensor([1.0, dw])]

    trainer = FloorSupportMultiLabelTrainer(model=model, data_dir=DATA_DIR, epochs=30)
    train_gestures = trainer._load_gestures("train")
    val_gestures = trainer._load_gestures("val")

    for epoch in range(30):
        result = model.train(train_gestures, val_gestures, epoch=epoch+1, total_epochs=30)

    # Evaluate on val
    all_probs, all_truths = [], []
    for features, label in val_gestures:
        probs = model.predict_proba([(features, label)])[0]
        all_probs.append(probs); all_truths.append(label)
    all_probs = np.array(all_probs); all_truths = np.array(all_truths)
    all_preds = (all_probs > 0.5).astype(int)

    def bits_to_label(bits):
        active = [CODES[i] for i, v in enumerate(bits) if v == 1]
        return "+".join(active) if active else "0"

    ft_d_acc = ft_s_acc = hn_d_acc = exact = 0
    for tc_name, tc_bits in [("FT+D", [1,0,0,1]), ("FT+S", [1,0,1,0]), ("HN+D", [0,1,0,1])]:
        mask = (all_truths == np.array(tc_bits)).all(axis=1)
        if mask.sum() == 0: continue
        correct = (all_preds[mask] == all_truths[mask]).all(axis=1).mean()
        if tc_name == "FT+D": ft_d_acc = correct
        elif tc_name == "FT+S": ft_s_acc = correct
        elif tc_name == "HN+D": hn_d_acc = correct

    exact = (all_preds == all_truths).all(axis=1).mean()
    balance = 2 * ft_d_acc * ft_s_acc / (ft_d_acc + ft_s_acc) if (ft_d_acc + ft_s_acc) > 0 else 0

    results[dw] = {"FT+D": ft_d_acc, "FT+S": ft_s_acc, "HN+D": hn_d_acc,
                    "exact": exact, "F1_balance": balance}
    print(f"  FT+D={ft_d_acc:.1%}  FT+S={ft_s_acc:.1%}  HN+D={hn_d_acc:.1%}  "
          f"exact={exact:.1%}  balance_F1={balance:.3f}")

print(f"\n{'='*70}")
print(f"  {'Weight':<8s} {'FT+D':>8s} {'FT+S':>8s} {'HN+D':>8s} {'Exact':>8s} {'F1_bal':>8s}")
print(f"  {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
for dw in weights_to_try:
    r = results[dw]
    print(f"  {dw:<8.1f} {r['FT+D']:7.1%} {r['FT+S']:7.1%} {r['HN+D']:7.1%} "
          f"{r['exact']:7.1%} {r['F1_balance']:7.3f}")

best = max(results, key=lambda w: results[w]['F1_balance'])
print(f"\n  Best F1 balance: D-weight = {best}")
