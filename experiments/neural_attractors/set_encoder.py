# pyright: basic
"""Self-supervised set encoder over delay-embedded state samples (no attractor labels).

A trajectory window is treated as an unordered set of delay vectors
``(x(t), x(t + l_1), ..., x(t + l_k))`` per state channel. A shared point-wise MLP followed
by mean and max pooling (a DeepSets encoder) makes the embedding invariant to the order of
the samples, so a time shift of the window only changes which points of the same attractor
were sampled, not how they are arranged. This is the property positional sequence encoders
lack: they embed the orbit's phase relative to the window start.

Each channel is encoded independently, so one model applies to systems with any number of
states; a trajectory's embedding is the concatenation of its per-channel embeddings. Inputs
are rescaled with pooled dataset statistics, as in ``contrastive.py``.

Training (InfoNCE): the two views of a trajectory are random point subsets drawn from two
independent random sub-windows covering 50-100 % of the window, plus small noise. All other
trajectories in the batch are negatives.
"""

import time

import numpy as np
import torch
from common import Dataset, dataset_scale
from torch import nn
from torch.nn import functional

LAGS = (0, 4, 16, 64)
SET_SIZE = 256
EMBED_DIM = 64
TEMPERATURE = 0.1
NOISE = 0.02
MIN_VIEW_FRACTION = 0.5


class SetEncoder(nn.Module):
    def __init__(self, hidden: int = 128, out: int = EMBED_DIM):
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(len(LAGS), hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        self.rho = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.GELU(), nn.Linear(hidden, out))

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """``points``: (B, P, len(LAGS)) -> (B, out)."""
        h = self.phi(points)
        return self.rho(torch.cat([h.mean(dim=1), h.max(dim=1).values], dim=-1))


def _channels(ds: Dataset) -> torch.Tensor:
    loc, scale = dataset_scale(ds.y)
    z = np.arcsinh((ds.y - loc) / scale).reshape(-1, ds.y.shape[-1]).astype(np.float32)
    return torch.from_numpy(z)


def delay_points(x: torch.Tensor) -> torch.Tensor:
    """(B, T) -> (B, T - max(LAGS), len(LAGS)) delay vectors."""
    n = x.shape[1] - max(LAGS)
    return torch.stack([x[:, lag : lag + n] for lag in LAGS], dim=-1)


def _view(points: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    b, n, _ = points.shape
    length = torch.randint(int(MIN_VIEW_FRACTION * n), n + 1, (b,), generator=gen)
    start = (torch.rand(b, generator=gen) * (n - length + 1)).long()
    offsets = (torch.rand(b, SET_SIZE, generator=gen) * length[:, None]).long()
    idx = (start[:, None] + offsets)[..., None].expand(-1, -1, points.shape[-1])
    view = torch.gather(points, 1, idx)
    return view + NOISE * torch.randn(view.shape, generator=gen)


def train(datasets: list[Dataset], steps: int = 600, batch: int = 256, seed: int = 0) -> SetEncoder:
    """Train on every state channel of every trajectory in ``datasets``, sampled uniformly
    over datasets so a large system does not dominate."""
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    pools = [_channels(ds) for ds in datasets]
    min_len = min(p.shape[1] for p in pools)
    pools = [p[:, -min_len:] for p in pools]
    model = SetEncoder()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    start = time.perf_counter()
    for step in range(steps):
        which = torch.randint(0, len(pools), (batch,), generator=gen)
        rows = torch.stack([
            pools[w][torch.randint(0, len(pools[w]), (1,), generator=gen)[0]]
            for w in which.tolist()
        ])
        points = delay_points(rows)
        a = functional.normalize(model(_view(points, gen)), dim=1)
        b = functional.normalize(model(_view(points, gen)), dim=1)
        logits = a @ b.T / TEMPERATURE
        target = torch.arange(batch)
        loss = 0.5 * (
            functional.cross_entropy(logits, target) + functional.cross_entropy(logits.T, target)
        )
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 200 == 0 or step == steps - 1:
            print(f"  step {step:5d} loss {loss.item():.3f} ({time.perf_counter() - start:.0f}s)")
    return model.eval()


def untrained(seed: int = 0) -> SetEncoder:
    """Randomly initialised encoder: isolates what the architecture's invariance gives
    before any training."""
    torch.manual_seed(seed)
    return SetEncoder().eval()


def embed(model: SetEncoder, ds: Dataset, batch: int = 256) -> np.ndarray:
    """Embed every delay vector of the full window (no subsampling, deterministic)."""
    x = _channels(ds)
    with torch.inference_mode():
        out = torch.cat([model(delay_points(x[i : i + batch])) for i in range(0, len(x), batch)])
    return out.reshape(ds.y.shape[0], -1).numpy()
