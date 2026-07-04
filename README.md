# LuminAIxBATModel

<img src="assets/BAT.png" width="600" alt="BAT codes">

This repository trains and deploys small neural networks that classify body movement into BAT (Body Articulation Type) codes across four body regions. All models use 73-dimensional plumb-line features per bodyframe and output per-pair normalised probabilities where a **0.5 threshold** selects the active label in each binary pair. For groups with >2 codes (Spine, Space), use **argmax**.

---

## Model Validation Accuracy

| Body Region | Architecture | Codes | Val Accuracy | Exact Match |
|:---|---:|---:|---:|:---|
| **FloorSupport** | BiLSTM + attention, pair groups | FT, HN, S, D (4 codes) | **86.3%** | 72.5% |
| **Spine** | BiLSTM + mean pool, class weights | E, F, HG, LF, SR, U (6 classes) | **69.5%** | 69.5% |
| **LimbExpression** | CNN + per-pair sigmoid heads | LB, SL, AS, A, G, UB, DL, SY (8 codes) | **95.2%** | 76.6% |
| **Space** | BiLSTM + mean pool, pair groups | ST, T, RV, SP, H, M, L (7 codes) | **90.3%** | 66.0% |

---

## FloorSupport — Foot/Hand/Ground Contact

**4 codes in 2 binary pairs:** `[FT, HN]` `[S, D]` → 4 valid combos

| Combo | Train | Val | Test |
|-------|------:|----:|-----:|
| FT+D (`d_ft_dual_feet`) | 77 | 16 | 17 |
| HN+D (`d_hn_dual_hand`) | 69 | 14 | 16 |
| FT+S (`s_ft_single_feet`) | 100 | 21 | 22 |
| **HN+S** (`s_hn_single_hand`) | **0** | **0** | **0** |

> **Gap:** HN+S has zero samples. The model has never seen hand-only contact with a single limb.

| True → Pred | FT+D | FT+S | HN+D |
|:---|---:|---:|---:|
| FT+D | **50%** | 50% | — |
| FT+S | 29% | **71%** | — |
| HN+D | — | — | **100%** |

---

## Spine — Spine Movement Patterns

**6-class single-label:** E (Extension), F (Flexion), HG (Hinge), LF (Lateral Flexion), SR (Spinal Rotation), U (Undefined)

| Class | Train | Val | Test |
|-------|------:|----:|-----:|
| E (`e_extension`) | 72 | 15 | 16 |
| F (`f_flexion`) | 87 | 18 | 20 |
| HG (`hg_hinge`) | 84 | 18 | 19 |
| LF (`lf_lateral_flexion`) | 40 | 8 | 10 |
| **SR** | **0** | **0** | **0** |
| **U** | **0** | **0** | **0** |

> **Gap:** SR and U have zero training data. LF has only 40 samples (half of other classes).

| True → Pred | E | F | HG | LF |
|:---|---:|---:|---:|---:|
| E | **73%** | 7% | 13% | 7% |
| F | — | **61%** | 39% | — |
| HG | 6% | 6% | **83%** | 6% |
| LF | 38% | 13% | — | **50%** |

---

## LimbExpression — Limb Expression Patterns

**4 binary pairs → 8 codes:** body (LB↔UB), limb (SL↔DL), symmetry (SY↔AS), contact (G↔A) → 16 valid combos

| Combo | Train | Val | Test |
|-------|------:|----:|-----:|
| LB+DL+SY+G | 79 | 16 | 18 |
| LB+SL+AS+A | 67 | 14 | 15 |
| LB+SL+AS+G | 81 | 17 | 19 |
| **All UB combos** | **0** | **0** | **0** |
| **All other 12 combos** | **0** | **0** | **0** |

> **Gap:** Only 3 of 16 valid combos have training data. All UpperBody (UB) combos are missing. No Air+Asymmetric combos outside the LB+SL+AS+A class.

| True → Pred | LB+DL+SY+G | LB+SL+AS+A | LB+SL+AS+G |
|:---|---:|---:|---:|
| LB+DL+SY+G | **100%** | — | — |
| LB+SL+AS+A | — | **93%** | 7% |
| LB+SL+AS+G | 6% | 24% | **71%** |

---

## Space — Spatial/Energy Patterns

**2 pair groups:** Movement `[ST, T, RV, SP]` × Energy `[H, M, L]` → 12 valid combos

| Combo | Train | Val | Test |
|-------|------:|----:|-----:|
| H+RV (`h_rv`) | 77 | 16 | 18 |
| H+SP (`h_sp`) | 95 | 20 | 22 |
| H+ST (`h_st`) | 65 | 14 | 15 |
| **All M combos** (M+RV, M+SP, M+ST) | **0** | **0** | **0** |
| **All L combos** | **0** | **0** | **0** |
| **All T combos** (T+H, T+M, T+L) | **0** | **0** | **0** |

> **Gap:** 9 of 12 combos have zero training data. No Medium energy, no Low energy, no Transition movement. Only High-energy gestures exist.

| True → Pred | ST+H | RV+H | SP+H |
|:---|---:|---:|---:|
| ST+H | **43%** | 7% | 50% |
| RV+H | 6% | **88%** | 6% |
| SP+H | — | 35% | **65%** |

---

## Confusion Matrices

### FloorSupport
<img src="assets/heatmap_floorsupport.png" width="700" alt="FloorSupport confusion matrix">

### Spine
<img src="assets/heatmap_spine.png" width="700" alt="Spine confusion matrix">

### LimbExpression
<img src="assets/heatmap_limbexpression.png" width="700" alt="LimbExpression confusion matrix">

### Space
<img src="assets/heatmap_space.png" width="700" alt="Space confusion matrix">
