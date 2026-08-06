"""Shared export/verification helper for the case-study graphs (see integrate_graph.py and
case_study_graph.py's module docstrings for why each pipeline stage is its own `.pte`)."""

import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from executorch.exir import to_edge

_t_last = [time.perf_counter()]

MAX_BATCH = 20_000


def mark(name: str) -> None:
    now = time.perf_counter()
    print(f"[{now - _t_last[0]:7.2f}s elapsed] {name}", flush=True)
    _t_last[0] = now


def export_and_save(
    module: nn.Module,
    example_inputs: tuple[torch.Tensor, ...],
    dynamic_shapes: dict[str, Any],
    pte_path: Path,
) -> tuple[torch.export.ExportedProgram, bytes]:
    """Export `module`, lower to ExecuTorch, and write the `.pte`. Reports stage timings.

    :param example_inputs: Positional example inputs, in `forward`'s argument order.
    :param dynamic_shapes: Per `torch.export.export`'s `dynamic_shapes` -- e.g.
        ``{"u01": {0: Dim("batch", min=1, max=MAX_BATCH)}, "min_limits": None, "max_limits": None}``
        for a graph with one batch-dynamic input and two fixed-shape ones.
    """
    module.eval()
    with torch.no_grad():
        exported = torch.export.export(module, example_inputs, dynamic_shapes=dynamic_shapes)
    mark("torch.export.export")

    with torch.no_grad():
        edge_program = to_edge(exported)
    mark("to_edge")

    with torch.no_grad():
        et_program = edge_program.to_executorch()
    mark("to_executorch")

    pte_path.write_bytes(et_program.buffer)
    mark(f"wrote {pte_path.name}, size={len(et_program.buffer)} bytes")
    return exported, et_program.buffer


def check_eager_vs_exported(
    module: nn.Module,
    exported: torch.export.ExportedProgram,
    test_inputs: tuple[torch.Tensor, ...],
) -> bool:
    with torch.no_grad():
        eager_out = module(*test_inputs)
        exported_out = exported.module()(*test_inputs)
    if isinstance(eager_out, tuple):
        match = all(torch.equal(a, b) for a, b in zip(eager_out, exported_out, strict=True))
    else:
        match = torch.equal(eager_out, exported_out)
    mark(f"eager-vs-exported check on different inputs: match={match}")
    return match
