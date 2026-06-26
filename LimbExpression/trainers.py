"""Multi-label trainer for LimbExpression — 8-code output.

Matches Unity limb_codes order: LB, SL, AS, A, G, UB, DL, SY
"""

import sys
from pathlib import Path

_floor = Path(__file__).resolve().parents[1] / "FloorSupport"
sys.path.insert(0, str(_floor))
import importlib.util


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_fs_trainers = _load_module("fs_trainers", _floor / "trainers.py")
MultiLabelGestureTrainer = _fs_trainers.MultiLabelGestureTrainer


class LimbExpressionMultiLabelTrainer(MultiLabelGestureTrainer):
    """Trainer that parses LimbExpression directory names into 4-element labels."""

    def _parse_label(self, dir_name: str):
        from models.multilabel_cnn import parse_attributes
        return parse_attributes(dir_name)
