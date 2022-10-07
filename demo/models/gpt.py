# minGPT, based on Shen's benchmark GPT

import os
import math
import logging
from functools import partial

import torch
import torch.nn as nn
from torch.nn import functional as F

class GPTConfig:
    """base GPT config, params common to all GPT versions"""

    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    def __init__(self, vocab_size, block_size, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        for k, v in kwargs.items():
            setattr(self, k, v)

class GPTSmall(GPTConfig):
    """GPT3-small like network roughly 125M params"""

    n_layer = 12
    n_head = 12
    n_embd = 768

class GPTMedium(GPTConfig):
    """GPT3-large like network roughly 350M params"""

    n_layer = 24
    n_head = 16
    n_embd = 1024


class GPTLarge(GPTConfig):
    """GPT3-large like network roughly 760M params"""

    n_layer = 24
    n_head = 16
    n_embd = 1536


class GPTXL(GPTConfig):
    """GPT3-XL like network roughly 1.3B params"""

    n_layer = 24
    n_head = 24
    n_embd = 2064

class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config, device="cpu", dtype=torch.float32):
        super().__init__()
        assert (
            config.n_embd % config.n_head == 0
        ), f"n_embd={config.n_embd}, n_head={config.n_head} must be evenly divisible"

        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd, device=device, dtype=dtype)
        self.query = nn.Linear(config.n_embd, config.n_embd, device=device, dtype=dtype)
        self.value = nn.Linear(config.n_embd, config.n_embd, device=device, dtype=dtype)
        # regularization

        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)

        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd, device=device, dtype=dtype)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        # TODO: leave buffer on CPU for now, until we can do meta_tensor.to_empty()
        d = device if torch.device(device).type == "cuda" else "cpu"

        self.register_buffer(
            "mask",
            torch.tril(
                torch.ones(config.block_size, config.block_size, device=d, dtype=dtype)
            ).view(1, 1, config.block_size, config.block_size),
        )
        self.n_head = config.n_head

    def reset_parameters(self):
        for _, m in self.named_modules():
            if isinstance(m, nn.Linear):
                m.reset_parameters()

    def forward(self, x, layer_past=None):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = (
            self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        )  # (B, nh, T, hs)
        q = (
            self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        )  # (B, nh, T, hs)
        v = (
            self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        )  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y

    class EmbeddingStem(nn.Module):
        def __init__(self, config, device="cpu", dtype=torch.float32):
            super().__init__()

            # input embedding stem
            self.tok_emb = nn.Embedding(
                config.vocab_size, config.n_embd, device=device, dtype=dtype
            )
            self.pos_emb = nn.Parameter(
                torch.zeros(1, config.block_size, config.n_embd, device=device, dtype=dtype)
            )
            self.drop = nn.Dropout(config.embd_pdrop)
            self.block_size = config.block_size

        def reset_parameters(self):
            self.tok_emb.reset_parameters()

        def forward(self, idx):
            b, t = idx.size()
            assert t <= self.block_size, "Cannot forward, model block size is exhausted."

            token_embeddings = self.tok_emb(idx)  # each index maps to a (learnable) vector
            position_embeddings = self.pos_emb[
                :, :t, :
            ]  # each position maps to a (learnable) vector
            return self.drop(token_embeddings + position_embeddings)

    class Block(nn.Module):
        """an unassuming Transformer block"""

        def __init__(
            self,
            config,
            device=None,
            dtype=torch.float32,
            wrapper=lambda m: m,
            version="pytorch",
            cpu_offload=False,
        ):
            super().__init__()
            if version == "pytorch" or not cpu_offload:
                self.ln1 = nn.LayerNorm(config.n_embd, device=device, dtype=dtype)
                self.ln2 = nn.LayerNorm(config.n_embd, device=device, dtype=dtype)
                self.attn = CausalSelfAttention(config, device=device, dtype=dtype)
                self.mlp = nn.Sequential(
                    nn.Linear(
                            config.n_embd, 4 * config.n_embd, device=device, dtype=dtype
                        ),
                    nn.GELU(),
            
                    nn.Linear(
                            4 * config.n_embd, config.n_embd, device=device, dtype=dtype
                        ),
                    nn.Dropout(config.resid_pdrop),
                ) 
            else:
                print("cpu offload for block")
                self.ln1 = nn.LayerNorm(config.n_embd, device=device, dtype=dtype).cpu()
            
                self.ln2 = nn.LayerNorm(config.n_embd, device=device, dtype=dtype).cpu()
                
                self.attn = CausalSelfAttention(config, device=device, dtype=dtype).cpu()
            
                self.mlp = nn.Sequential(
                        nn.Linear(
                            config.n_embd, 4 * config.n_embd, device=device, dtype=dtype
                        ).cpu(),
                    nn.GELU(),
                        nn.Linear(
                            4 * config.n_embd, config.n_embd, device=device, dtype=dtype
                        ).cpu(),
                    nn.Dropout(config.resid_pdrop),
                ) 
        def reset_parameters(self):
            self.attn.reset_parameters()
            for _, m in self.named_modules():
                if isinstance(m, nn.LayerNorm) or isinstance(m, nn.Linear):
                    m.reset_parameters()

        def forward(self, x):
            x = x + self.attn(self.ln1(x))
            x = x + self.mlp(self.ln2(x))
            return x

class GPT(nn.Module):
    """the full GPT language model, with a context size of block_size"""

    def __init__(self, config, device="cpu", dtype=torch.float32):
        super().__init__()

        # input embedding stem
        self.emb_stem = EmbeddingStem(config, device=device, dtype=dtype)
        # transformer
        self.blocks = nn.Sequential(
            *[Block(config, device=device, dtype=dtype) for _ in range(config.n_layer)]
        )
        # decoder head
        self.ln_f = nn.LayerNorm(config.n_embd, device=device, dtype=dtype)
        self.head = nn.Linear(
            config.n_embd, config.vocab_size, bias=False, device=device, dtype=dtype
        )

        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )

    def forward(self, idx):
        x = self.emb_stem(idx)
        x = self.blocks(x)
        x = self.ln_f(x)
        return self.head(x)
  