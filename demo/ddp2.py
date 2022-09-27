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
from functools import partial

from typing import (
    Any,
    Callable,
    List,
    Optional,
)

import logging
import os


class MyModel(nn.Module):
    def __init__(self, num_features, num_layers):
        super().__init__()
        self.seq = nn.Sequential(
            *[nn.Linear(num_features, num_features) for _ in range(num_layers)]
        )

    def forward(self, x):
        return self.seq(x)


class DTensorType(Enum):
    Replicated = auto()
    Sharded = auto()
    Partial = auto()


@dataclass
class DTensorTag:
    dttype: DTensorType = DTensorType.Replicated
    pg: ProcessGroup = None


class DDP(nn.Module):
    """Tag each param as type Replicated"""

    def __init__(self, module, pg=None, tag_with=DTensorType.Replicated):
        super().__init__()
        self.module = module

        for p in module.parameters():
            if not hasattr(p, "_dtags"):
                p._dtags = []

            p._dtags.append(DTensorTag(dttype=tag_with, pg=pg))

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


def allreduce(tensor, pg):
    logging.info(f"AllReduce Tensor of shape {tensor.shape}")
    dist.all_reduce(tensor, group=pg)


def fused_allreduce(tensors, pg):
    logging.info(f"Fused AllReduce Tensor of shape {[t.shape for t in tensors]}")
    buffer = torch.empty(sum([t.numel() for t in tensors]))
    offset = 0
    for t in tensors:
        numel = t.numel()
        buffer[offset : offset + numel] = t.view(-1)
        offset += numel

    dist.all_reduce(buffer, group=pg)

    offset = 0
    for t in tensors:
        numel = t.numel()
        t = buffer[offset : offset + numel].view(t.shape)
        offset += numel


class Engine:
    """compile the train_step function.
    Then based on the tags in local module,
    insert comms ops and fuse them."""

    def __init__(self, module, train_step, bucket_mb: int = 5):
        self.module = module
        self.train_step = train_step
        self.bucket_mb = bucket_mb

        self.num_grads = sum([p.requires_grad for p in module.parameters()])

        self.primal_to_param = {}
        self.grad_to_primal = {}

        self.compiled = None

    def run(self, x: torch.Tensor):
        if self.compiled is None:
            self.compiled = aot_module(self.module, self.compile_fwd, self.compile_bwd)

        self.compiled(x).sum().backward()

    def compile_fwd(self, gm: fx.GraphModule, inputs):
        # use pytree order to map to primals, save for bwd
        def to_param(model: nn.Module, primal_name: str) -> torch.nn.Parameter:
            index = int(primal_name.split("_")[-1]) - 1
            params = [
                p for _, p in list(pytree.tree_flatten(model.named_parameters())[0][0])
            ]
            return params[index] if index < len(params) else None

        logging.info("Compiling forward")
        gm.graph.print_tabular()
        # get tags on each param
        for node in gm.graph.nodes:
            if node.op == "placeholder" and node.target.startswith("primal"):
                p = to_param(self.module, node.name)
                if p is not None:
                    assert (
                        node.target not in self.primal_to_param
                    ), f"inserting {node.target} twice"
                    self.primal_to_param[node.target] = p

        logging.info(
            "\nFinished fwd compile, identified following DTs\n"
            + "\n".join(
                [f"{pl}:{pm._dtags}" for pl, pm in self.primal_to_param.items()]
            )
        )

        return gm

    def compile_bwd(self, gm: fx.GraphModule, inputs):
        logging.info("compiling bwd")
        logging.info("Original bwd graph")
        gm.graph.print_tabular()

        # insert individual all reduce
        pgs = {}
        for node in gm.graph.nodes:
            if node.op == "output":
                # relying on assumption of primal and gradient ordering is the same
                for i, grad_node in enumerate(node.args[0][: self.num_grads]):
                    primal = f"primals_{i+1}"
                    self.grad_to_primal[grad_node.name] = primal
                    for dtag in self.primal_to_param[primal]._dtags:
                        if dtag.dttype == DTensorType.Replicated:
                            with gm.graph.inserting_after(grad_node):
                                gm.graph.call_function(
                                    allreduce, args=(grad_node, dtag.pg)
                                )
                                pgs[grad_node] = dtag.pg
                break

        # fuse all reduce ops based on bucket size
        comm_args, comm_nodes = [], []
        comm_size = 0
        pg = None

        for node in gm.graph.nodes:
            if node.name.startswith("allreduce"):
                comm_args.append(node.args[0])
                comm_size += self.primal_to_param[
                    self.grad_to_primal[node.args[0].name]
                ].numel()
                comm_nodes.append(node)
                assert (
                    pg is None or pg == pg[node.args[0]]
                ), "expecting same pg instance"
                pg = pgs[node.args[0]]
                last_node = node

                if comm_size >= self.bucket_mb * 1e6:
                    # accumulated comm size > bucket size, lets fuse them
                    with gm.graph.inserting_after(last_node):
                        gm.graph.call_function(fused_allreduce, args=(comm_args, pg))

                    comm_args = []
                    comm_size = 0

        if len(comm_args) > 0:
            with gm.graph.inserting_after(last_node):
                gm.graph.call_function(fused_allreduce, args=(comm_args, pg))

        for node in comm_nodes:
            gm.graph.erase_node(node)

        gm.graph.lint()
        gm.recompile()
        logging.info("Modified backward graph")
        print(f"----- Updated Graph ----")
        gm.graph.print_tabular()

        logging.info("finished compiling backward")
        return gm


# ---  user facing api ----
#
def train_step(model: nn.Module, x: torch.Tensor):
    model(x).sum().backward()


def run_worker(rank, world_size):
    logging.getLogger().setLevel(logging.DEBUG if rank == 0 else logging.CRITICAL)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    num_features = 10_000
    # create on cpu
    model = MyModel(num_features, 4)
    # tag all params as replicated
    model = DDP(model)

    # compile train_step, insert comm ops based on tags and fuse
    engine = Engine(model, train_step)
    for i in range(3):
        logging.info(f"--- iteration {i+1} ==========")
        x = torch.randn(2, num_features)
        engine.run(x)


if __name__ == "__main__":
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29501"
    world_size = 2
    mp.spawn(run_worker, args=(world_size,), nprocs=world_size, join=True)
