from functorch.compile import aot_function, aot_module, aot_drawgraph
from torch import nn
from torch.distributed import ProcessGroup

import torch
import torch.distributed as dist
import torch.fx as fx
import torch.multiprocessing as mp
import torch.utils._pytree as pytree

from dataclasses import dataclass
from enum import Enum, auto
from functools import partial, reduce

from typing import (
    Any, 
    Callable,
    List,
    Dict,
    Optional,
    Set,
)

import logging
import math
import os

class MyModel(nn.Module):
    def __init__(self, n_features, n_layers):
        super().__init__()
        self.seq = nn.Sequential(*[nn.Linear(n_features, n_features) for _ in range(n_layers)])
    
    def forward(self,x):
        return self.seq(x)

# type of dtensor
class DTensorType(Enum):
    Replicated = auto()
    Sharded = auto()
    Partial = auto()
    ONDEMAND = auto()

@dataclass
class DTensorTag:
    dttype: DTensorType = DTensorType.Replicated
    pg: ProcessGroup = None

def _tag_module(module: nn.Module, tag: DTensorTag):
    for p in module.parameters():
        if not hasattr(p,"_dtags"):
            p._dtags = []
        p._dtags.append(tag)
        