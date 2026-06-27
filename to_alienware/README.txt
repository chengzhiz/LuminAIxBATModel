=== Files for the Alienware (LuminAI computer) ===

This folder contains everything needed. No building required.
No Unity rebuild required. No Visual Studio required.

Just copy this entire "to_alienware" folder to the LuminAI computer
and place the files as described below.

=== What BATRunner.exe does ===

BATRunner.exe is a standalone console app that:
1. Reads a gesture JSON file (bodyframe data from Kinect)
2. Extracts 73 plumb-line features per frame (COG + joint angles/distances)
3. Runs 4 ONNX neural network models (Space, Spine, Limb, Floor)
4. Writes predictions to bat_result.csv

The LuminAI Unity app launches BATRunner.exe whenever a new gesture
is recorded. Unity reads the CSV and applies the BAT codes to the gesture.

=== File placement ===

On the Alienware, in the LuminAI-master project folder:

1. BATRunner.exe
   → Replace: BATRunner/app/BATRunner.exe

2. onnxruntime.dll
   → Replace: BATRunner/app/onnxruntime.dll

3. onnxruntime_providers_shared.dll
   → Replace: BATRunner/app/onnxruntime_providers_shared.dll

4. Models/best_model_FS_multilabel.onnx   }
   Models/best_model_Spine_multilabel.onnx } → Place all 4 in:
   Models/best_model_LE_multilabel.onnx    }   BATRunner/app/Models/
   Models/best_model_Space_multilabel.onnx }

   (You can also place them in BATRunner/Models/ — the post-build
    script will copy them to app/Models/ if you ever rebuild)

=== What does NOT change ===
- BATPreprocess.cs       (not needed — logic is built into the new .exe)
- BATModelRunner.cs       (Unity) — backward compatible, no changes
- All Unity assets, scenes, prefabs — untouched

=== Model summary ===
FloorSupport:  4 codes (FT, HN, S, D)         - 86.3% accuracy
Spine:         6 codes (E, F, HG, LF, SR, U)   - 64.4% accuracy
LimbExpression: 8 codes (LB, SL, AS, A, G, UB, DL, SY) - 95.2% accuracy
Space:         5 codes (RV, ST, SP, H, M)       - 85.6% accuracy
