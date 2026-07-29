"""Shared path constants for the Zig native solver package."""

from __future__ import annotations

from pathlib import Path

ZIG_ROOT: Path = Path(__file__).parent
SRC_DIR: Path = ZIG_ROOT / "src"
USER_ODE_DIR: Path = ZIG_ROOT / "user_odes"
LIB_DIR: Path = ZIG_ROOT / "zig-out" / "lib"
ODE_LIB_DIR: Path = ZIG_ROOT / "zig-out" / "ode_libs"
SOLVER_LIB_NAME: str = "libzig_ode_solver.so"
SOLVER_HASH_FILE: Path = ZIG_ROOT / ".solver_hash"
