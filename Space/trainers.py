"""Multi-label trainer for Space — 5-code output: RV, ST, SP, H, M.

Matches Unity space_codes order in BATModelRunner.cs.
No training data for M or L — they train to near-0.
"""

import sys
from pathlib import Path
import importlib.util

_floor = Path(__file__).resolve().parents[1] / "FloorSupport"
sys.path.insert(0, str(_floor))

# Avoid circular import — load FloorSupport trainers explicitly
_fs_trainers_path = _floor / "trainers.py"
spec = importlib.util.spec_from_file_location("fs_trainers", _fs_trainers_path)
_fs_trainers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_fs_trainers)
MultiLabelGestureTrainer = _fs_trainers.MultiLabelGestureTrainer


# Code order matching Unity: RV, ST, SP, H, M
def parse_space_label(dir_name: str):
    """Parse Space directory names into 5-element multi-hot vectors.

    h_rv_high_revolution   → [1, 0, 0, 1, 0]  (RV, H)
    h_sp_high_spring       → [0, 0, 1, 1, 0]  (SP, H)
    h_st_high_stationary   → [0, 1, 0, 1, 0]  (ST, H)
    """
    parts = dir_name.split("_")

    # Parse energy: h → H=1, m → M=1, l → neither (no data)
    energy = parts[0]  # 'h', 'm', or 'l'

    # Parse movement
    movement = parts[1]  # 'rv', 'sp', or 'st'

    # Build label: [RV, ST, SP, H, M]
    label = [0.0, 0.0, 0.0, 0.0, 0.0]

    if movement == "rv":
        label[0] = 1.0  # RV
    elif movement == "st":
        label[1] = 1.0  # ST
    elif movement == "sp":
        label[2] = 1.0  # SP

    if energy == "h":
        label[3] = 1.0  # H
    elif energy == "m":
        label[4] = 1.0  # M

    return label


class SpaceMultiLabelTrainer(MultiLabelGestureTrainer):
    """Trainer that parses Space directory names into 5-element multi-hot labels."""

    def _parse_label(self, dir_name: str):
        return parse_space_label(dir_name)
