"""The decoder: a 1D convolutional network that reads the raster directly.

Input for one example: (N neurons, 226 bins) spike counts in 10 ms bins, plus each
neuron's E/I tag. No weight is ever tied to a particular neuron, and the row order does
not matter, so the same trained network runs on a rat it has never seen, with any
number of cells.

Stages (10 weight layers in total):
  per-neuron   3 conv layers, kernel 3, dilation 1,2,4   -> reach 15 bins = 150 ms
               (shortest paper-defined OFF period is 75 ms; 2 layers reach only 70 ms)
  relational   1 attention layer across neurons at every 10 ms step
               (co-firing, E/I relationships; compares activity, never row numbers)
  pooling      1 attention-pooling layer with Q learned queries -> neuron count disappears
  population   3 conv layers, kernel 3, dilation 8,16,32 -> total reach 127 bins = 1.27 s
               (longest paper-defined OFF period is 1.25 s)
  readout      average over the 100 steps of the middle second -> 3 classes

All convolutions are "valid" (no padding), so 226 input bins give exactly 100 output
steps, one per 10 ms of the labeled second, each read with its full 1.27 s reach.

Input scale: each unit's counts are divided by that unit's mean count per bin over the
whole recording (label-free; see dataset.py). With raw 0/1 counts (about 1.5% of bins
occupied) the network stayed at chance on the training data itself; with the relative
input it learned within 100 steps. Normalization layers inside the network (ChannelNorm,
kept as an option) did not help and are off by default.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dataset import WINDOW, BINS_PER_S


class Conv1dK3(nn.Module):
    """Valid conv1d, kernel 3, dilation d. Same weights and result as nn.Conv1d, but computed
    as one matmul over [x_t, x_{t+d}, x_{t+2d}]: PyTorch's CPU path for dilated convolutions
    over tens of thousands of short rows (one per neuron per example) is several times slower."""

    def __init__(self, c_in: int, c_out: int, dilation: int):
        super().__init__()
        self.d = dilation
        ref = nn.Conv1d(c_in, c_out, kernel_size=3, dilation=dilation)   # standard initialization
        self.weight = nn.Parameter(ref.weight.detach().clone())          # (C_out, C_in, 3)
        self.bias = nn.Parameter(ref.bias.detach().clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d, L = self.d, x.shape[-1] - 2 * self.d
        cols = torch.cat([x[:, :, :L], x[:, :, d:d + L], x[:, :, 2 * d:2 * d + L]], dim=1)   # (R, 3*C_in, L)
        w = self.weight.permute(0, 2, 1).reshape(self.weight.shape[0], -1)                   # tap-major, matches cols
        return torch.einsum("rkl,ok->rol", cols, w) + self.bias.view(1, -1, 1)


class ChannelNorm(nn.Module):
    """LayerNorm over the channel dimension of a (rows, channels, time) tensor."""

    def __init__(self, c: int):
        super().__init__()
        self.ln = nn.LayerNorm(c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ln(x.transpose(1, 2)).transpose(1, 2)


def _conv_stack(c_in: int, c: int, dilations: tuple[int, ...], norm: bool) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i, d in enumerate(dilations):
        layers.append(Conv1dK3(c_in if i == 0 else c, c, d))
        if norm:
            layers.append(ChannelNorm(c))
        layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class SpikeNet(nn.Module):
    def __init__(self, filters: int = 16, queries: int = 2,
                 relational: bool = True, population: bool = True, norm: bool = False, n_classes: int = 3):
        super().__init__()
        C, Q = filters, queries
        self.relational, self.population = relational, population
        # input channels: spike count, and a constant 0/1 tag (E=0, I=1)
        self.neuron_conv = _conv_stack(2, C, (1, 2, 4), norm)
        if relational:
            self.norm = nn.LayerNorm(C)
            self.attn = nn.MultiheadAttention(C, num_heads=1, batch_first=True)
        self.queries = nn.Parameter(torch.randn(Q, C) / C ** 0.5)
        self.pool_key = nn.Linear(C, C)
        if population:
            self.pop_conv = _conv_stack(Q * C, C, (8, 16, 32), norm)
            self.readout = nn.Linear(C, n_classes)
        else:
            # without the population layers, the pooled signal goes straight to the readout
            self.readout = nn.Linear(Q * C, n_classes)
        self.reach_cut = 8 + 16 + 32   # bins lost on each side by the population stack

    def forward(self, x: torch.Tensor, types: torch.Tensor) -> torch.Tensor:
        """x: (B, N, 226) spike counts; types: (N,) 0/1. Returns (B, 3) logits."""
        B, N, T = x.shape
        tag = types.to(x.dtype).view(1, N, 1).expand(B, N, T)
        h = torch.stack([x, tag], dim=2).reshape(B * N, 2, T)       # (B*N, 2, T)
        h = self.neuron_conv(h)                                     # (B*N, C, T1)
        C, T1 = h.shape[1], h.shape[2]
        h = h.reshape(B, N, C, T1).permute(0, 3, 1, 2).reshape(B * T1, N, C)   # neurons as a set, per step
        if self.relational:
            z = self.norm(h)
            a, _ = self.attn(z, z, z, need_weights=False)
            h = h + a
        # attention pooling: Q queries score every neuron by its current features
        scores = torch.einsum("qc,bnc->bqn", self.queries, self.pool_key(h)) / C ** 0.5
        w = torch.softmax(scores, dim=-1)                           # (B*T1, Q, N), sums to 1 over neurons
        pooled = torch.einsum("bqn,bnc->bqc", w, h)                 # (B*T1, Q, C)
        pooled = pooled.reshape(B, T1, -1).permute(0, 2, 1)         # (B, Q*C, T1)
        if self.population:
            g = self.pop_conv(pooled)                               # (B, C, 100)
        else:
            g = pooled[:, :, self.reach_cut:T1 - self.reach_cut]    # same 100 middle steps
        assert g.shape[-1] == BINS_PER_S, g.shape
        return self.readout(g.mean(dim=-1))

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


if __name__ == "__main__":
    for C in (16, 32):
        for Q in (1, 2, 4):
            m = SpikeNet(C, Q)
            out = m(torch.zeros(4, 37, WINDOW), torch.zeros(37, dtype=torch.long))
            print(f"filters={C:2d} queries={Q}: {m.n_params():6d} weights, output {tuple(out.shape)}")
    for rel, pop in ((False, True), (True, False)):
        m = SpikeNet(16, 2, relational=rel, population=pop)
        print(f"relational={rel} population={pop}: {m.n_params()} weights, output {tuple(m(torch.zeros(2, 5, WINDOW), torch.zeros(5, dtype=torch.long)).shape)}")
