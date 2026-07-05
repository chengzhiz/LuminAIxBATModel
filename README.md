# LuminAIxBATModel

<img src="assets/BAT.png" width="600" alt="BAT codes">

## Design Decisions

1. **COG-relative plumbline features.** All four models use 73-dimensional plumb-line features (21 distances + 52 angles between body keypoints) instead of raw 732-dim joint positions. These features are computed relative to the center of gravity, making them translation-invariant. The plumbline distances are ratios between body segments, so a tall adult and a short child produce the same feature values for the same pose.

2. **Height-invariant normalization.** For raw-feature models (gesture CNN variants), `HeightNormalize` divides all position channels by the per-sample skeleton height (max Y − min Y). For plumbline models, this is unnecessary — the distances are already scale-invariant since they measure proportions between keypoints, not absolute coordinates. Both approaches ensure the model generalizes across different body heights.

3. **D-weight penalty for FloorSupport false negatives.** The FloorSupport model has a built-in class weight: D (dual-foot) is weighted **4×** in the S↔D CrossEntropyLoss pair. This aggressively penalizes missing a dual-foot ("flamingo") prediction — it's better to falsely say "dual feet" for a single-foot gesture than to miss a genuine dual-foot. The weight is applied via `pair_weights = [None, tensor([1.0, 4.0])]` on the S/D group.

4. **Model architecture experiments per region.** Different architectures were tested for each body region before settling on the current ones:
   - **FloorSupport:** BiLSTM + attention pooling + pair-group CrossEntropyLoss (replaced BCE + soft mutex penalty which allowed invalid combos like "FT alone")
   - **Spine:** BiLSTM + mean pooling + class-weighted CrossEntropyLoss (attention pooling and deeper classifiers overfit on the small 283-sample dataset)
   - **LimbExpression:** 1D CNN trunk + 4 independent sigmoid heads with per-pair normalization (contrastive-pair architecture was the original design and works best)
   - **Space:** BiLSTM + mean pooling + pair-group CrossEntropyLoss (expanded from 5 to 7 codes to cover all 12 valid movement×energy combos)

5. **Random seeds for reproducibility.** All training scripts accept `--seed` (default 42) and set `random`, `numpy`, and `torch` seeds before training. Identical data shuffling, weight initialization, and dropout patterns across runs — critical for publishing and peer review.

## BAT Origins

BAT (Body Articulation Type) codes were developed at the **Georgia Tech Expressive Machinery Lab** as part of the **LuminAI** project — an interactive AI dance partner that improvises movement with a human participant. The taxonomy is grounded in **Laban Movement Analysis (LMA)**, specifically its **Body** component, which categorizes how body parts are coordinated during movement.

**Key contributors:** Milka Trajkova (research scientist, former professional ballet dancer), Brian Magerko (PI), Duri Long, Manoj Deshpande, and Andrea Knowlton (Kennesaw State University dance professor).

**Reference paper:** Trajkova, Long, Deshpande, Knowlton, & Magerko (2024). *"Exploring Collaborative Movement Improvisation Towards the Design of LuminAI — a Co-Creative AI Dance Partner."* CHI '24, ACM Conference on Human Factors in Computing Systems.

The four BAT regions map to LMA: floor contact patterns (FloorSupport), spine articulation (Spine), limb expression (LimbExpression), and spatial/energy levels (Space). Viewpoints movement theory (Overlie, Bogart & Landau) also influenced the taxonomy.

## Future Work

### Data Augmentation

Several augmentation strategies could expand the limited training set:

- **Time warping:** Stretch or compress gesture timelines (±20%) to simulate faster/slower execution.
- **Left-right mirroring:** Flip the skeleton's X-axis to double the dataset (especially helpful for LimbExpression SL↔DL and LB↔UB).
- **Gaussian noise injection:** Jitter (σ=0.02) on plumbline features to simulate Kinect sensor noise.
- **Frame dropout:** Randomly drop 10–20% of frames to simulate occluded keypoints.

### AI-Generated Moves for Missing Categories

Many valid BAT combinations have zero training data (Space: 9 of 12 combos, LimbExpression: 13 of 16, FloorSupport: HN+S, Spine: SR/U). A potential workflow:

1. **Motion generation:** Use a pretrained dance generation model (MDM, EDGE, or Motion Diffusion Model) conditioned on the missing BAT label combo.
2. **Human-in-the-loop labeling:** A dance expert reviews generated motions, keeping only those that correctly embody the target codes.
3. **Augment training:** Accepted motions join the training set; low-confidence ones could use soft labels during training.

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

---

## BAT Code Slogans

Each active code triggers a natural-language slogan displayed in the LuminAI UI. Codes marked `—` have no `batdesc_` GameObject in the current Unity scene and need one added.

### FloorSupport

| Code | Slogan |
|:---|:---|
| D-FT | I think you're balancing on two feet. |
| S-FT | I think you're balancing on one foot, like a flamingo. |
| D-HN | I think you're balancing on two hands, like an acrobat. |
| S-HN | — |

### Spine

| Code | Slogan |
|:---|:---|
| E | I think your spine is stretching back, like you're letting out a big... |
| F | I think your spine is bending forward, almost like you're bowing. |
| HG | I think you're moving your spine like an inflatable tube man. |
| LF | I think your spine is bending side to side like a fitness instructor. |
| SR | I think your spine is twisting at your waist like you're looking... |
| U | — |

### LimbExpression

| Code | Slogan |
|:---|:---|
| LB | — |
| SL | I believe you're using one limb. |
| AS | I think one or both limbs are moving differently from the other. |
| A | From what I can see, one or more limbs are moving in the air. |
| G | I think one or more limbs are moving, while still touching the ground. |
| UB | — |
| DL | I think you're using, let me think, two limbs. |
| SY | I think both limbs are mirroring each other. |

### Space

| Code | Slogan |
|:---|:---|
| ST | I think you're stationary or moving in place! |
| T | — |
| RV | I think you're turning or spinning like a top! |
| SP | I think you're jumping in the air! |
| H | I think you're in high space, meaning your feet—heels and toes—are... |
| M | I think you're in medium space, meaning your joints are vertically... |
| L | — |
