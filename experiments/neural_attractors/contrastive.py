# pyright: basic
"""Self-supervised contrastive encoder for steady-state trajectories (no attractor labels).

Each state channel is encoded independently by a dilated 1-D CNN, so one model applies to
systems with any number of states; a trajectory's embedding is the concatenation of its
per-state embeddings. Positive pairs are two random crops (random phase) of the same
trajectory with small additive noise; all other crops in the batch are negatives (InfoNCE).
Inputs are rescaled with pooled dataset statistics, not per series, so offset, sign and
amplitude remain visible to the encoder.
"""

import time

import numpy as np
import torch
from common import Dataset, dataset_scale
from torch import nn
from torch.nn import functional

CROP = 256
EMBED_DIM = 64
TEMPERATURE = 0.1
NOISE = 0.02


class DilatedEncoder(nn.Module):
    def __init__(self, hidden: int = 64, depth: int = 7, out: int = EMBED_DIM):
        super().__init__()
        self.inp = nn.Conv1d(1, hidden, kernel_size=1)
        self.blocks = nn.ModuleList(
            nn.Conv1d(hidden, hidden, kernel_size=3, dilation=2**i, padding=2**i)
            for i in range(depth)
        )
        self.out = nn.Conv1d(hidden, out, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x``: (B, T) -> (B, out) via global max pooling over time."""
        h = self.inp(x[:, None, :])
        for block in self.blocks:
            h = h + functional.gelu(block(h))
        return self.out(h).max(dim=-1).values


def _channels(ds: Dataset) -> np.ndarray:
    loc, scale = dataset_scale(ds.y)
    z = (ds.y - loc) / scale
    return np.arcsinh(z).reshape(-1, z.shape[-1]).astype(np.float32)


def _crop(x: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    starts = torch.randint(0, x.shape[1] - CROP + 1, (x.shape[0],), generator=gen)
    idx = starts[:, None] + torch.arange(CROP)[None, :]
    return torch.gather(x, 1, idx) + NOISE * torch.randn(x.shape[0], CROP, generator=gen)


def train(
    datasets: list[Dataset], steps: int = 600, batch: int = 256, seed: int = 0
) -> DilatedEncoder:
    """Train on every state channel of every trajectory in ``datasets``, sampled uniformly
    over datasets so a large system does not dominate."""
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    pools = [torch.from_numpy(_channels(ds)) for ds in datasets]
    min_len = min(p.shape[1] for p in pools)
    pools = [p[:, -min_len:] for p in pools]
    model = DilatedEncoder()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    start = time.perf_counter()
    for step in range(steps):
        which = torch.randint(0, len(pools), (batch,), generator=gen)
        rows = torch.stack([
            pools[w][torch.randint(0, len(pools[w]), (1,), generator=gen)[0]]
            for w in which.tolist()
        ])
        a = functional.normalize(model(_crop(rows, gen)), dim=1)
        b = functional.normalize(model(_crop(rows, gen)), dim=1)
        logits = a @ b.T / TEMPERATURE
        target = torch.arange(batch)
        loss = 0.5 * (
            functional.cross_entropy(logits, target) + functional.cross_entropy(logits.T, target)
        )
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 250 == 0 or step == steps - 1:
            print(f"  step {step:5d} loss {loss.item():.3f} ({time.perf_counter() - start:.0f}s)")
    return model.eval()


def embed(model: DilatedEncoder, ds: Dataset, batch: int = 512) -> np.ndarray:
    x = torch.from_numpy(_channels(ds))
    with torch.inference_mode():
        out = torch.cat([model(x[i : i + batch]) for i in range(0, len(x), batch)])
    return out.reshape(ds.y.shape[0], -1).numpy()
