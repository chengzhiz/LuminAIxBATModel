"""Plumb-line feature extraction — replicates BATPreprocess.process_bodyframe() from C#.

Produces exactly 73 features per bodyframe:
    21 plumb-line distances  (alphabetical by joint name, fixed order)
  + 52 plumb-line angles      (sorted by bone key index)
  = 73

The BATPreprocess.cs reference is at:
    LuminAI-master/BATRunner/BATPreprocess.cs  →  process_bodyframe()
"""

import json
import math
import re
from pathlib import Path
from typing import List, Union

import numpy as np


# ═══════════════════════════════════════════════════════════════════════
# JSON loading (same preprocessing as existing load_feature_vectors_from_json)
# ═══════════════════════════════════════════════════════════════════════

def _preprocess_json(text: str) -> str:
    """Handle MongoDB extended JSON: ISODate(), NaN, ObjectId()."""
    text = re.sub(r'ISODate\("(.+?)"\)', r'"\1"', text)
    text = re.sub(r'ObjectId\("(.+?)"\)', r'"\1"', text)
    text = text.replace("NaN", "null")
    return text


def _parse_bodyframe_positions(bodyframe: dict) -> dict:
    """Extract bone positions as {bone_key: (x, y, z)} from one bodyframe."""
    positions = {}
    for entry in bodyframe.get("bodyFrameHumanPos", []):
        k = entry["k"]
        v = entry["v"]
        positions[k] = np.array([float(v[0]), float(v[1]), float(v[2])], dtype=np.float32)
    return positions


# ═══════════════════════════════════════════════════════════════════════
# COG weights  —  exact match of BATPreprocess.cs `weights` dictionary
# ═══════════════════════════════════════════════════════════════════════

_COG_WEIGHTS = {
    0:  0.07,                      # Head
    7:  0.25,  8:  0.25,          # Torso (Spine, Chest)
    1:  0.075, 2:  0.075,         # Upper legs
    3:  0.075, 4:  0.075,
    5:  0.035, 6:  0.035,         # Lower legs
    11: 0.03,  12: 0.03,           # Upper arms
    13: 0.03,  14: 0.03,
    15: 0.02,  16: 0.02,           # Lower arms
    17: 0.02,  18: 0.02,
}
_COG_DEFAULT_WEIGHT = 0.01


def _calculate_cog(positions: dict) -> np.ndarray:
    """Weighted center of gravity — matches BATPreprocess.cs calculate_cog()."""
    total_weight = 0.0
    weighted_sum = np.zeros(3, dtype=np.float32)

    for bone, pos in positions.items():
        w = _COG_WEIGHTS.get(bone, _COG_DEFAULT_WEIGHT)
        weighted_sum += pos * w
        total_weight += w

    if total_weight == 0:
        return np.zeros(3, dtype=np.float32)
    return weighted_sum / total_weight


# ═══════════════════════════════════════════════════════════════════════
# Distance order  —  exact match of BATPreprocess.cs `distance_order`
# ═══════════════════════════════════════════════════════════════════════

# Maps bone key → distance name  (BATPreprocess.cs `distance_mapping`)
_DISTANCE_MAPPING = {
    0:  "Hips_to_plumbline",
    1:  "LeftUpperLeg_to_plumbline",
    2:  "RightUpperLeg_to_plumbline",
    3:  "LeftLowerLeg_to_plumbline",
    4:  "RightLowerLeg_to_plumbline",
    5:  "LeftFoot_to_plumbline",
    6:  "RightFoot_to_plumbline",
    7:  "Spine_to_plumbline",
    8:  "Chest_to_plumbline",
    9:  "Neck_to_plumbline",
    10: "Head_to_plumbline",
    11: "LeftShoulder_to_plumbline",
    12: "RightShoulder_to_plumbline",
    13: "LeftUpperArm_to_plumbline",
    14: "RightUpperArm_to_plumbline",
    15: "LeftLowerArm_to_plumbline",
    16: "RightLowerArm_to_plumbline",
    17: "LeftHand_to_plumbline",
    18: "RightHand_to_plumbline",
    19: "LeftToes_to_plumbline",
    20: "RightToes_to_plumbline",
    21: "UpperChest_to_plumbline",   # not in distance_order
}

# Fixed order of 21 named distances  (BATPreprocess.cs `distance_order`)
_DISTANCE_ORDER = [
    "Chest_to_plumbline",          # k=8
    "Head_to_plumbline",           # k=10
    "Hips_to_plumbline",           # k=0
    "LeftFoot_to_plumbline",       # k=5
    "LeftHand_to_plumbline",       # k=17
    "LeftLowerArm_to_plumbline",   # k=15
    "LeftLowerLeg_to_plumbline",   # k=3
    "LeftShoulder_to_plumbline",   # k=11
    "LeftToes_to_plumbline",       # k=19
    "LeftUpperArm_to_plumbline",   # k=13
    "LeftUpperLeg_to_plumbline",   # k=1
    "Neck_to_plumbline",           # k=9
    "RightFoot_to_plumbline",      # k=6
    "RightHand_to_plumbline",      # k=18
    "RightLowerArm_to_plumbline",  # k=16
    "RightLowerLeg_to_plumbline",  # k=4
    "RightShoulder_to_plumbline",  # k=12
    "RightToes_to_plumbline",      # k=20
    "RightUpperArm_to_plumbline",  # k=14
    "RightUpperLeg_to_plumbline",  # k=2
    "Spine_to_plumbline",          # k=7
]

EXPECTED_MODEL_FEATURE_COUNT = 73


def process_bodyframe(bodyframe: dict) -> np.ndarray:
    """Compute 73 plumb-line features for a single bodyframe.

    Replicates BATPreprocess.cs process_bodyframe() exactly.
    """
    positions = _parse_bodyframe_positions(bodyframe)
    if not positions:
        raise ValueError("No positions found in bodyframe")

    cog = _calculate_cog(positions)
    up = np.array([0.0, 1.0, 0.0], dtype=np.float32)  # plumb_line_direction = Vector3.up

    # Compute per-joint plumb-line distance and angle
    plumb_distances = {}
    angles_to_plumb = {}

    for bone, joint_pos in positions.items():
        joint_to_cog = joint_pos - cog

        # Plumb distance = |joint_to_cog × up|  (perpendicular distance to vertical)
        plumb_dist = float(np.linalg.norm(np.cross(joint_to_cog, up)))

        # Angle between joint_to_cog and vertical
        jtc_norm = np.linalg.norm(joint_to_cog)
        if jtc_norm < 1e-10:
            angle = 0.0
        else:
            cos_angle = np.clip(np.dot(joint_to_cog, up) / jtc_norm, -1.0, 1.0)
            angle = float(math.degrees(math.acos(cos_angle)))

        plumb_distances[bone] = plumb_dist
        angles_to_plumb[bone] = angle

    # Build named distance map  (BATPreprocess.cs specific_distances)
    specific_distances = {}
    for bone, name in _DISTANCE_MAPPING.items():
        if bone in plumb_distances:
            specific_distances[name] = plumb_distances[bone]

    # Build tensor row: 21 distances in fixed order + 52 angles sorted by bone key
    tensor_row = []

    # 21 distances
    for dist_name in _DISTANCE_ORDER:
        tensor_row.append(specific_distances.get(dist_name, 0.0))

    # 52 angles sorted by bone key (ascending)
    for bone in sorted(angles_to_plumb.keys()):
        tensor_row.append(angles_to_plumb[bone])

    if len(tensor_row) != EXPECTED_MODEL_FEATURE_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_MODEL_FEATURE_COUNT} features, got {len(tensor_row)}"
        )

    return np.array(tensor_row, dtype=np.float32)


def load_plumbline_features_from_json(
    json_path: Union[str, Path]
) -> List[List[float]]:
    """Convert a single JSON recording into a list of 73-feature vectors,
    one per bodyframe.  This replaces load_feature_vectors_from_json()."""
    path = Path(json_path)
    text = path.read_text(encoding="utf-8")
    text = _preprocess_json(text)

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        print(f"Warning: could not parse JSON: {path}")
        return []

    bodyframes = data.get("bodyFrames", [])
    all_features = []

    for bf in bodyframes:
        try:
            features = process_bodyframe(bf)
            all_features.append(features.tolist())
        except (ValueError, KeyError) as e:
            # Skip frames with missing position data
            continue

    return all_features


# ═══════════════════════════════════════════════════════════════════════
# Frame sampling  —  same logic as sample_frames() in _gesture_base.py
# ═══════════════════════════════════════════════════════════════════════

def sample_frames(
    features_list: List[List[float]], target_len: int
) -> np.ndarray:
    """Uniformly sample *target_len* frames from a variable-length sequence."""
    T = len(features_list)
    arr = np.array(features_list, dtype=np.float32)
    if T == 0:
        return np.zeros((target_len, arr.shape[1] if arr.ndim == 2 else 0),
                        dtype=np.float32)
    if T < target_len:
        pad = np.tile(arr[-1:], (target_len - T, 1))
        arr = np.concatenate([arr, pad], axis=0)
        return arr
    indices = np.linspace(0, T - 1, target_len, dtype=int)
    return arr[indices]
