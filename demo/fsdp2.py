from functorch.compile import aot_function, aot_module, draw_graph
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
        #self.seq = nn.Sequential(*[nn.Linear(n_features, n_features) for _ in range(n_layers)])
        self.seq = nn.Linear(n_features, n_features)
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
        print(f"calling reduce scatter with rank {rank}")
        return output

class Engine:
    def __init__(self, module, train_step):
        self.module = module
        self.train_step = train_step
        self.n_grads = sum([p.requires_grad for p in module.parameters()])
        print(f"Num grads = {self.n_grads}")
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
            buffer[:(to-offset)] = param.view(-1)[offset:to]
            param._local_shard = buffer
            param._orig_size = param.size()
    
    def _find_primal_views(self, gm, primal) -> Dict[fx.Node, fx.Node]:

        view_to_parent = {primal:primal}

        for node in gm.graph.nodes:
            if all([
                node.op=="call_function" and 
                str(node.target)=="aten.t" and 
                len(node.args)==1 and 
                node.args[0] in view_to_parent
            ]):
                view_to_parent[node]=node.args[0]
        print(f"primal return {view_to_parent}")
        return view_to_parent
    
    def _find_param_usages(self, gm, views: Set[fx.Node]) -> List[fx.Node]:
        usages = []
        for node in gm.graph.nodes:
            for view in views:
                if view in node.args:
                    usages.append(node)
        return usages
    
    def _handle_one_param_primal(self, gm, primal, pg):
        views = self._find_primal_views(gm, primal)
        self.view_to_parent.update(views)
        usages = self._find_param_usages(gm, set(views.keys()))
        print(f"Alert: ")
        print(f"usages = {usages}")
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
            params = [p for _, p in list(pytree.tree_flatten(model.named_parameters())[0][0])]
            return params[idx] if idx < len(params) else None
        
        logging.info("Compiling Forward Pass")
        rank = dist.get_rank()
        if rank==0:
            gm.graph.print_tabular()
        
        # get tags on each node
        for node in gm.graph.nodes:
            if node.op == "placeholder" and node.target.startswith("primal"):
                p = to_param(self.module, node.name)
                if p is not None and hasattr(p,"_dtags"):
                    assert node.target not in self.primal_to_param, (f"inserting {node.target} twice")
                    self.primal_to_param[node.target] = p 
                #self.primal_to_param[node.target]
                    for tag in p._dtags:
                        if tag.dttype == DTensorType.ONDEMAND:
                            self._handle_one_param_primal(gm, node, tag.pg)
            if node.op == "output":
                new_args=([], )
                for i, arg in enumerate(node.args[0]):
                    if arg in self.view_to_parent:
                        view = arg
                        while view != self.view_to_parent[view]:
                            view = self.view_to_parent[view]
                        new_args[0].append(view)
                    else:
                        new_args[0].append(arg)
                node.args = new_args
        logging.info(
                "\n** Finished compiling forward, identified following Distributed Tensors\n" +
                "\n".join([f"{pl} : {pm._dtags}" for pl, pm in self.primal_to_param.items()])
            )
        
        gm.graph.lint()
        gm.recompile()
        logging.info("\n Modified Forward pass")
        if rank==0:
            gm.graph.print_tabular()
        self.fwd_gm = gm
        return gm
    
    def _compile_bwd(self, gm, inps):
        rank = dist.get_rank()
        
        logging.info("Compiling backward")
        logging.info("Original backward graph")
        if rank==0:
            gm.graph.print_tabular()
        
        # insert all gathers
        view_name_to_node = {v.name: v for v,p in self.view_to_parent.items()}

        for node in gm.graph.nodes:
            if node.op == "placeholder" and node.name in view_name_to_node:
                # found a view of param primal, retrieve all preceeding nodes that generate
                # this view
                view_node = view_name_to_node[node.name]
                node_to_insert=[]
                while view_node != self.view_to_parent[view_node]:
                    node_to_insert.append(view_node)
                    view_node = self.view_to_parent[view_node]
                
                node_to_insert.append(view_node)
                node_to_insert.reverse()
                new_nodes = {}
            
            def arg_transform(arg):
                if arg.name in new_nodes:
                    return new_nodes[arg.name]
                else:
                    raise RuntimeError(f"unrecognized arg {arg}")
                
                param_primal = None
                with gm.graph.inserting_before(node):
                    for to_insert in node_to_insert:
                        for arg in to_insert.args:
                            new_node = gm.graph.node_copy(arg, arg_transform=arg_transform )
                            new_nodes[arg.name] = new_node
                        new_node = gm.graph.node_copy(to_insert, arg_transform=arg_transform)
                        new_nodes[to_insert.name] = new_node
                        param_primal = new_node if new_node.op == "placeholder" else param_primal

                node.replace_all_uses_with(new_node)

                views = set(self._find_primal_views(gm, new_node).keys())
                usages = self._find_param_usages(gm, views)

                with gm.graph.inserting_after(usages[-1]):
                    gm.graph.call_function(
                        ondemand_discard,
                        args=(param_primal, usages[-1])

                    )  
                gm.graph_erase_node(node)
        logging.info("After recover param primals")
        if rank==0:
            gm.graph.print_tabular()
        
        logging.info("Insert Grad ReduceScatter")
        for node in gm.graph_nodes:
            if node.op == "output":
                new_output_args=[]
            
            i = 0
            for grad_node in node.args[0][:self.n_grads]:
                i+=1
                primal =f"primals_{i}"
                self.grad_to_primal[grad_node.name]=primal
                for dtag in self.primal_to_param[primal]._dtags:
                    if dtag.dttype == DTensorType.ONDEMAND:
                        with gm.graph.inserting_after(grad_node):
                            new_grad_node = gm.graph.call_function(
                                ondemand_reducescatter, 
                                args = (grad_node, dtag.pg)
                            
                            )
        gm.graph.lint()
        gm.recompile()
        logging.info(f"rank {rank} Modified backward graph")
        if rank==0:
            print(f"recompile backward results")
            gm.graph.print_tabular()
        
        logging.info(f"finished compiling backward, rank {rank}")
        self.bwd_gm = gm
        if self.bwd_gm is None and rank==0:
            print(f"failed backward compile")
        return gm

        

                






# ---- USER APIs --------------

def train_step(model, x):
    # autograd cannot train trace_step yet
    model(x).sum().backward()


def run_worker(rank, world_size):
    logging.getLogger().setLevel(logging.DEBUG if rank==0 else logging.CRITICAL)
    dist.init_process_group("nccl", rank = rank, world_size = world_size)

    n_features=30
    model=MyModel(n_features, 3)
    model.to(rank)
    model = FSDP(model)

    engine = Engine(model, train_step)
    for i in range(3):
        logging.info(f"----- iteration {i} ------")
        x = torch.randn(2, n_features).to(rank)
        engine.run(x)

if __name__=="__main__":
    os.environ["MASTER_ADDR"]="localhost"
    os.environ["MASTER_PORT"]="29500"
    world_size =2
    mp.spawn(run_worker, 
        args=(world_size,),
        nprocs = world_size, 
        join=True)

        


