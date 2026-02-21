"""Snapshot dataclass for checkpoint serialisation."""

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, List

import torch


@dataclass
class Snapshot:
    model_state: 'OrderedDict[str, torch.Tensor]'
    optimizer_state: Dict[str, Any]
    finished_epoch: int
    epoch_test_accuracies: List[float] = None
