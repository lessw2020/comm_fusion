from dataclasses import dataclass
from functools import partial
import torch


@dataclass
class TrainConfig:
    bucket_size: int = 100_000
    iterations: int = 4
    num_features: int = 20_000
