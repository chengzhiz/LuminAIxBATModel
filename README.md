# LuminAIxBATModel

This repository contains the LuminAIxBATModel project files.

Notes from Cat:

- the .csv file (BodyFrame) has a 4 dimensional vector for rotation (Quaternion), and 3 dimensional vectors for heading and position
- like the majority of things (by quantity) in a Gesture.json file would not be useful to train a model on

## Model Validation Accuracy

Each model predicts BAT (Body Articulation Type) codes for a different body region. Accuracy is measured as exact match across all code bits for each prediction.

| Body Region | Architecture | Codes | Val Accuracy |
|:---|:---|---:|---:|
| **FloorSupport** | Multi-label BiLSTM | FT, HN, S, D (4 codes) | **86.3%** |
| **Spine** | Spine Classifier LSTM | E, F, HG, LF, SR, U (6 codes) | **64.4%** |
| **LimbExpression** | Multi-label CNN | LB, SL, AS, A, G, UB, DL, SY (8 codes) | **95.2%** |
| **Space** | Multi-label Space LSTM | RV, ST, SP, H, M (5 codes) | **85.6%** |

### Per-region breakdown

**FloorSupport** — foot/hand/ground contact patterns
- Multi-label LSTM checkpoint: `FloorSupport/checkpoints/multilabel_floor_lstm/` — 86.3%
- Gesture CNN checkpoint: `FloorSupport/checkpoints/gesture_cnn/` — 80.4%

**Spine** — spine movement patterns
- Spine Classifier LSTM checkpoint: `Spine/checkpoints/spine_classifier_lstm/` — 61.0%
- Gesture CNN checkpoint: `Spine/checkpoints/gesture_cnn/` — 62.7%

**LimbExpression** — limb expression patterns
- Multi-label CNN checkpoint: `LimbExpression/checkpoints/multilabel_cnn/` — 95.2%

**Space** — spatial/energy patterns
- Multi-label Space LSTM checkpoint: `Space/checkpoints/multilabel_space_lstm/` — 85.6%
- Multi-task CNN checkpoint: `Space/checkpoints/multitask_cnn/` — 86.0%

### Deployed models (ONNX)

The ONNX exports used by the LuminAI Unity app:
- `best_model_FS_multilabel.onnx` — FloorSupport 86.3%
- `best_model_Spine_multilabel.onnx` — Spine 64.4%
- `best_model_LE_multilabel.onnx` — LimbExpression 95.2%
- `best_model_Space_multilabel.onnx` — Space 85.6%

## Confusion Matrices

### FloorSupport
<img src="assets/heatmap_floorsupport.png" width="700" alt="FloorSupport confusion matrix">

### Spine
<img src="assets/heatmap_spine.png" width="700" alt="Spine confusion matrix">

### LimbExpression
<img src="assets/heatmap_limbexpression.png" width="700" alt="LimbExpression confusion matrix">

### Space
<img src="assets/heatmap_space.png" width="700" alt="Space confusion matrix">

## Notes

- The FloorSupport directory is ignored for Git synchronization.
