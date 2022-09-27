from dataclasses import dataclass
from functools import partial
import torch


@dataclass
class TrainConfig:
    iterations: int = 4
