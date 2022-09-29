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

class FSDP(nn.Module):
    def __init__(self, module, pg=None):
        super().__init__()
        self.module =  module

        _tag_module(module, DTensorTag(dttype=DTensorType.ONDEMAND, pg=pg))
    
    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)
    
 def ondemand_allgather(param, pg):
    logging.info(f"Allgather param - {param.shape}")   
    local_shard = param._local_shard
    orig_size = param.orig_size
    with torch.no_grad():
        world_size = dist.get_world_size(group=pg)
        buffer = torch.empty([world_size]+ list(local_shard.shape), device=param.device)
        tensors = [buffer[i] for i in range(world_size)]
        dist.all_gather(tensors, local_shard, group=pg)
        size = list(orig_size)
        numel = reduce(lambda x, y: x*y, size, 1)
        param.data = buffer[:numel].view(size)
    
    return param

def ondemand_discard(param, _):
    logging.info(f"Discarding param - {param.shape}")
    with torch.no_grad():
        param.data = param._local_shard

def ondemand_reducescatter(grad, pg):
    with torch.no_grad():
        world_size = dist.get_world_size(group=pg)
        rank = dist.get_rank(group=pg)

        padded_size = int(math.ceil(grad.numel()/world_size))
        output = torch.empty([padded_size], device = grad.device)
        inputs_tensor = torch.empy([padded_size*world_size], device=grad.device)
        inputs_tensor[:grade.numel()].copy_(grad.view(-1))
        inputs = list(inputs_tensor.chunk(world_size))
        print(f{calling reduce scatter with rank {rank}})
        return output
