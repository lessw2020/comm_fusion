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

class Engine:
    def __init__(self, module, train_step):
        self.module = module
        self.train_step = train_step
        self.n_grads = sum([p.requires grad for p in module.parameters()])

        self.view_to_parent= {}
        self.primal_to_param = {}
        self.grad_to_primal = {}
        self.pytree_params = [p for _, p in list(pytree.tree_flatten(module.named_parameters())[0][0])]
        self.pytree_params.reverse()

        self.compiled_m = None
        self.fwd_gm = None
        self.bwd_gm = None

        for p in self.module.parameters():
            if hasattr(p,"_dtags"):
                for tag in p._dtags:
                    if tag.dttype == DTensorType.ONDEMAND:
                        self._prepare_param_shard(p, tag.pg)

    def run(self, x: torch.Tensor):
        if self.compiled_m is None:
            self.compiled_m = aot_module(self.module, self._compile_fwd, self._compile_bwd)
        
        if self.fwd_gm is None or self.bwd_gm is None:
            self.compiled_m(x)
            assert self.fwd_gm is not None, ("missing fwd gm")
            assert self.bwd_gm is not None, ("missing bwd gm")
        
        outs = self.fwd_gm(*self.pytree_params, x)
        out, activations = outs[0], outs[1:]
        out_grad = torch.ones_like(out)
        self.bwd_gm(*activations, out_grad)

    

    
    def _prepare_param_shard(self, param, pg):
        with torch.no_grad():
            world_size = dist.get_world_size(group=pg)
            rank = dist.get_rank(group=pg)

            padded_size = int(math.ceil(param.numel()/ world_size))
            buffer = torch.empty([padded_size], device = param.device)
            offset = rank * padded_size
            to = min(offset+padded_size, param.numel())  
            buffer = [:(to-offset)] = param.view(-1)[offset:to]
            param._local_shard = buffer
            param._orig_size = param.size()
    
    def _find_primal_views(self, gm, primal) -> Dict[fx.Node, fx.Node]:
        view_to_parent = {primal:primal}

        for node in gm.graph.nodes:
            if all([node.op=="call_function" and 
            str(node.target)="aten.t" and 
            len(node.args)==1 and 
            node.args[0] in view_to_parent]):
            view_to_parent[node]=node.args[0]
        return view_to_parent
    
    def _find_param_usages(self, gm, views: Set[fx.Node]) -> List[fx.Node]:
        usages = []
        for view in views:
            if view in node.args:
                usages.append(node)
    
    def _handle_one_param_primal(self, gm, primal, pg):
        views = self._find_primal_views(gm, primal)
        self.view_to_parent.update(views)
        usages = self._find_param_usages(gm, set(views.keys()))

        # insert allgather before first usage
        with gm.graph.inserting_before(usages[0]):
            new_node = gm.graph.call_function(
                ondemand_allgather, 
                args=(primal,pg)
            )
            usages[0].replace_input_with(primal,new_node)
        
        # insert reshard after last usage
        with gm.graph.inserting_after(usages[-1]):
            gm.graph.call_function(ondemand_discard,
            args=(primal, usages[-1]))
    
    def _compile_fwd(self, gm, inps):
        def to_param(model, primal_name)-> torch.nn.Parameter:
            idx = int(primal_name.split("_")[-1])-1
            params = [p for _, p in list(pytree.tree_flatten(model.named_parameters()[0][0]))]
            return params[idx] if idx < len(params) else None
        
        
        


