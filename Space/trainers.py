"""Multi-label trainer for Space — 7-code output in 2 pair groups.

Movement group: ST, T, RV, SP  (indices 0-3)
Energy group:   H,  M, L       (indices 4-6)

12 valid combos = 4 movement x 3 energy.
"""

import sys
from pathlib import Path
import importlib.util

_floor = Path(__file__).resolve().parents[1] / "FloorSupport"
sys.path.insert(0, str(_floor))

_fs_trainers_path = _floor / "trainers.py"
spec = importlib.util.spec_from_file_location("fs_trainers", _fs_trainers_path)
_fs_trainers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_fs_trainers)
MultiLabelGestureTrainer = _fs_trainers.MultiLabelGestureTrainer


# Code order: [ST, T, RV, SP, H, M, L]
#   indices 0-3: movement group (4-way softmax)
#   indices 4-6: energy group   (3-way softmax)
_SPACE_CODES = ["ST", "T", "RV", "SP", "H", "M", "L"]

# Mapping from directory-name tokens to group-local indices
_MOVEMENT_MAP = {"st": 0, "t": 1, "rv": 2, "sp": 3}
_ENERGY_MAP   = {"h": 4, "m": 5, "l": 6}


def parse_space_label(dir_name: str):
    """Parse Space directory names into 7-element multi-hot vectors.

    h_rv_high_revolution   -> [0, 0, 1, 0,  1, 0, 0]  (RV, H)
    h_sp_high_spring       -> [0, 0, 0, 1,  1, 0, 0]  (SP, H)
    h_st_high_stationary   -> [1, 0, 0, 0,  1, 0, 0]  (ST, H)
    """
    parts = dir_name.split("_")
    energy = parts[0]    # 'h', 'm', or 'l'
    movement = parts[1]  # 'rv', 'sp', 'st', or 't'

    label = [0.0] * 7

    if movement in _MOVEMENT_MAP:
        label[_MOVEMENT_MAP[movement]] = 1.0
    if energy in _ENERGY_MAP:
        label[_ENERGY_MAP[energy]] = 1.0

    return label


class SpaceMultiLabelTrainer(MultiLabelGestureTrainer):
    """Trainer that parses Space directory names into 7-element multi-hot labels."""

    def _parse_label(self, dir_name: str):
        return parse_space_label(dir_name)
