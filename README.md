# LuminAIxBATModel

<img src="assets/BAT.png" width="600" alt="BAT codes">

Classifies body movement into **BAT (Body Articulation Type)** codes across four regions: FloorSupport, Spine, LimbExpression, Space. All models use 73-dim plumbline features (COG-relative, scale-invariant) and output per-pair normalized probabilities. **0.5 threshold** for binary pairs, **argmax** for 3+ groups.

> **Two accuracy metrics:** The table below reports **per-bit accuracy** (each code checked independently — partial credit). The confusion heatmaps show **exact match** (all codes must be correct for a sample to count). Per-bit is always higher. For example, if only the D bit is wrong on an FT+D sample, it scores 3/4 = 75% per-bit but 0% exact match.

## Design

- **COG-relative features.** Plumbline distances/angles are invariant to body height and translation — no separate normalization needed.
- **Pair-group constraints.** Structurally enforced mutual exclusivity via per-group softmax + CrossEntropyLoss. No invalid combos possible.
- **Multi-head decoding.** Each mutually exclusive code group gets its own attention + classifier head on a shared backbone. The heads decide independently, so the frames that matter for one decision (e.g. the brief moment of dual-foot contact) don't have to compete with the frames that matter for another (foot vs hand). This also allows zero-shot compositional predictions — e.g. HN+S (single hand) is a reachable output even with no HN+S training data.
- **Attention pooling everywhere.** Attention beats mean pooling in every region — brief events (jumps, dual-foot contact) get washed out by averaging over 256 frames. Deep classifiers overfit on small datasets (<300 samples); shallow heads with LayerNorm work best.
- **Reproducible.** All `main.py` scripts accept `--seed` (default 42), setting `random`/`numpy`/`torch` seeds.

### Model Structures

| Region | Backbone | Pooling | Heads | Loss / Weights |
|:-------|:---------|:--------|:------|:---------------|
| **FloorSupport** | BiLSTM (73→256×2, 2 layers) | Per-head attention | **2 heads**: FT↔HN, S↔D (each: attn → Linear→LayerNorm→ReLU→Dropout→2) | CE per group; D weighted 2.0× (FT+D: 25%→69% val, 29%→82% test) |
| **Spine** | BiLSTM (73→256×2, 2 layers) | Single attention | **1 head**: 6-way softmax E/F/HG/LF/SR/U (single-label — one class per sample, nothing to split) | CE; class weights LF=3.0×, F=1.2× |
| **LimbExpression** | 1D CNN (73→128→256→512) | AdaptiveAvgPool | **4 heads**: LB↔UB, SL↔DL, AS↔SY, A↔G (each: Linear→ReLU→Dropout→1 sigmoid) | BCE per head |
| **Space** | BiLSTM (73→256×2, 2 layers) | Per-head attention | **2 heads**: movement ST/T/RV/SP (4-way), energy H/M/L (3-way) | CE per group; ST weighted 1.8× (ST+H: 43%→71% val, 20%→73% test) |

## BAT Origins

Developed at the **Georgia Tech Expressive Machinery Lab** (LuminAI project). Grounded in **Laban Movement Analysis** (Body component). Key paper: Trajkova et al., *"Exploring Collaborative Movement Improvisation Towards the Design of LuminAI"*, CHI '24.

---

## FloorSupport — Foot/Hand Contact (4 codes, 2 binary pairs)

| Combo | Train | Val | Test | Val Acc | Test Acc |
|-------|------:|----:|-----:|--------:|---------:|
| FT+D | 77 | 16 | 17 | **69%** | **82%** |
| FT+S | 100 | 21 | 22 | **57%** | **64%** |
| HN+D | 69 | 14 | 16 | **100%** | **94%** |
| HN+S | 0 | 0 | 0 | — | — |

> Two-head attention + D-weight 2.0× balances S↔D. HN+S has no data.

## Spine — Spine Movement (6-class single-label)

| Class | Train | Val | Test | Val Acc | Test Acc |
|-------|------:|----:|-----:|--------:|---------:|
| E (Extension) | 72 | 15 | 16 | **80%** | **81%** |
| F (Flexion) | 87 | 18 | 20 | **78%** | **90%** |
| HG (Hinge) | 84 | 18 | 19 | **78%** | **68%** |
| LF (Lat Flexion) | 40 | 8 | 10 | **50%** | **40%** |
| SR, U | 0 | 0 | 0 | — | — |

> Attention pooling + class weights (LF=3.0×, F=1.2×) + lr=5e-4, 80 epochs. Overall val acc 69.5% → 74.6%. SR and U have no data. LF has only 40 samples — more data needed.

## LimbExpression — Limb Patterns (8 codes, 4 binary pairs)

| Combo | Train | Val | Test | Val Acc | Test Acc |
|-------|------:|----:|-----:|--------:|---------:|
| LB+G+DL+SY | 79 | 16 | 18 | **100%** | **94%** |
| LB+SL+AS+A | 67 | 14 | 15 | **93%** | **80%** |
| LB+SL+AS+G | 81 | 17 | 19 | **71%** | **74%** |
| All UB combos (8) | 0 | 0 | 0 | — | — |
| Other 5 combos | 0 | 0 | 0 | — | — |

> Only 3/16 combos have data. All UpperBody combos missing.

## Space — Spatial/Energy (7 codes, 2 pair groups)

Movement `[ST, T, RV, SP]` × Energy `[H, M, L]` = 12 combos.

| Combo | Train | Val | Test | Val Acc | Test Acc |
|-------|------:|----:|-----:|--------:|---------:|
| RV+H | 77 | 16 | 18 | **88%** | **94%** |
| SP+H | 95 | 20 | 22 | **95%** | **95%** |
| ST+H | 65 | 14 | 15 | **71%** | **73%** |
| All M, L, T combos (9) | 0 | 0 | 0 | — | — |

> Two-head attention (movement/energy) + ST-weight 1.8×. Attention pooling fixed the ST↔SP confusion (jumps are brief events that mean pooling washed out): ST+H 43% → 71% (val), 20% → 73% (test); SP+H 65% → 95% (val), 50% → 95% (test). 9/12 combos missing — no Medium/Low energy, no Transition movement.

---

## Confusion Matrices

| Region | Validation | Test |
|:-------|:----------:|:----:|
| **FloorSupport** | <img src="assets/heatmap_floorsupport.png" width="340"> | <img src="assets/heatmap_floorsupport_test.png" width="340"> |
| **Spine** | <img src="assets/heatmap_spine.png" width="340"> | <img src="assets/heatmap_spine_test.png" width="340"> |
| **LimbExpression** | <img src="assets/heatmap_limbexpression.png" width="340"> | <img src="assets/heatmap_limbexpression_test.png" width="340"> |
| **Space** | <img src="assets/heatmap_space.png" width="340"> | <img src="assets/heatmap_space_test.png" width="340"> |

---

## Future Work

- **Data augmentation:** time warping, left-right mirroring, Gaussian noise, frame dropout.
- **AI-generated data:** motion diffusion models conditioned on missing combos → human expert labels → training set.
- **More data needed:** HN+S (FloorSupport), SR/U (Spine), all UB combos (LimbExpression), M/L/T combos (Space).

---

## BAT Code Slogans

Codes marked `—` need a `batdesc_` GameObject in the Unity scene.

| Region | Code | Slogan |
|:---|:---|:---|
| Floor | D-FT | I think you're balancing on two feet. |
| Floor | S-FT | I think you're balancing on one foot, like a flamingo. |
| Floor | D-HN | I think you're balancing on two hands, like an acrobat. |
| Floor | S-HN | — |
| Spine | E | I think your spine is stretching back... |
| Spine | F | I think your spine is bending forward, almost like you're bowing. |
| Spine | HG | I think you're moving your spine like an inflatable tube man. |
| Spine | LF | I think your spine is bending side to side like a fitness instructor. |
| Spine | SR | I think your spine is twisting at your waist... |
| Spine | U | — |
| Limb | LB | — |
| Limb | SL | I believe you're using one limb. |
| Limb | AS | I think one or both limbs are moving differently from the other. |
| Limb | A | From what I can see, one or more limbs are moving in the air. |
| Limb | G | I think one or more limbs are moving, while still touching the ground. |
| Limb | UB | — |
| Limb | DL | I think you're using, let me think, two limbs. |
| Limb | SY | I think both limbs are mirroring each other. |
| Space | ST | I think you're stationary or moving in place! |
| Space | T | — |
| Space | RV | I think you're turning or spinning like a top! |
| Space | SP | I think you're jumping in the air! |
| Space | H | I think you're in high space... |
| Space | M | I think you're in medium space... |
| Space | L | — |
