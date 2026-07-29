"""Compilation and caching for the Zig solver library and ODE shared libraries.

Handles:
- Rebuilding the solver ``libzig_ode_solver.so`` when Zig sources change.
- Compiling individual ODE ``.zig`` or ``.c`` files to ``lib<name>.so``.
- Content-hash based caching so recompilation only happens on source changes.
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

from ._paths import (
    LIB_DIR,
    ODE_LIB_DIR,
    SOLVER_HASH_FILE,
    SOLVER_LIB_NAME,
    SRC_DIR,
    USER_ODE_DIR,
    ZIG_ROOT,
)

# ============================================================
# Solver library build
# ============================================================


def _compute_solver_hash() -> str:
    """Compute hash of solver source files."""
    hasher = hashlib.sha256()
    for zig_file in sorted(SRC_DIR.rglob("*.zig")):
        hasher.update(zig_file.name.encode())
        hasher.update(zig_file.read_bytes())
    build_zig = ZIG_ROOT / "build.zig"
    if build_zig.exists():
        hasher.update(build_zig.read_bytes())
    return hasher.hexdigest()


def solver_needs_rebuild() -> bool:
    """Check if solver library needs rebuilding."""
    lib_path = LIB_DIR / SOLVER_LIB_NAME
    if not lib_path.exists():
        return True
    if not SOLVER_HASH_FILE.exists():
        return True
    return _compute_solver_hash() != SOLVER_HASH_FILE.read_text().strip()


def rebuild_solver(optimize: bool = True) -> None:
    """Rebuild the solver library.

    :param optimize: If True, use ReleaseFast optimisation.
    """
    cmd: list[str] = ["zig", "build"]
    if optimize:
        cmd.extend(["-Doptimize=ReleaseFast"])

    print(f"Building solver library: {' '.join(cmd)}", file=sys.stderr)
    result = subprocess.run(cmd, cwd=ZIG_ROOT, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"Solver build failed:\n{result.stderr}")

    SOLVER_HASH_FILE.write_text(_compute_solver_hash())
    print("Solver build successful!", file=sys.stderr)


# ============================================================
# ODE source discovery & hashing
# ============================================================


def get_ode_source(ode_name: str) -> Path:
    """Return path to the ODE source file, preferring ``.zig`` over ``.c``.

    :param ode_name: ODE identifier (without extension).
    :raises FileNotFoundError: If neither a ``.zig`` nor a ``.c`` source exists.
    """
    for ext in (".zig", ".c"):
        p = USER_ODE_DIR / f"{ode_name}{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(
        f"No ODE source found for '{ode_name}' in {USER_ODE_DIR}\n"
        f"Create {ode_name}.zig (Zig) or {ode_name}.c (via SymPyODE)"
    )


def get_ode_lib(ode_name: str) -> Path:
    """Return path to compiled ODE shared library."""
    return ODE_LIB_DIR / f"lib{ode_name}.so"


def _get_ode_hash_file(ode_name: str) -> Path:
    return ODE_LIB_DIR / f".{ode_name}_hash"


def _compute_ode_hash(ode_name: str) -> str:
    source = get_ode_source(ode_name)
    return hashlib.sha256(source.read_bytes()).hexdigest()


def ode_needs_compile(ode_name: str) -> bool:
    """Check if an ODE needs to be compiled.

    :return: True if the library is missing or the source has changed.
    """
    lib_path = get_ode_lib(ode_name)
    hash_file = _get_ode_hash_file(ode_name)

    if not lib_path.exists():
        return True
    if not hash_file.exists():
        return True
    try:
        return _compute_ode_hash(ode_name) != hash_file.read_text().strip()
    except FileNotFoundError:
        return True


# ============================================================
# ODE compilation (Zig and C)
# ============================================================

_WRAPPER_TEMPLATE: Path = SRC_DIR / "ode_wrapper_template.zig"


def _compile_zig_ode(ode_name: str, source: Path, optimize: bool) -> None:
    ODE_LIB_DIR.mkdir(parents=True, exist_ok=True)

    wrapper_path = ZIG_ROOT / f"_wrapper_{ode_name}.zig"
    wrapper_code = _WRAPPER_TEMPLATE.read_text().format(user_ode_path=f"user_odes/{ode_name}.zig")
    wrapper_path.write_text(wrapper_code)

    lib_path = get_ode_lib(ode_name)
    opt_flag = "-OReleaseFast" if optimize else "-ODebug"
    wrapper_rel = wrapper_path.relative_to(ZIG_ROOT)
    lib_rel = lib_path.relative_to(ZIG_ROOT)

    cmd: list[str] = [
        "zig",
        "build-lib",
        "-dynamic",
        opt_flag,
        str(wrapper_rel),
        f"-femit-bin={lib_rel}",
    ]

    print(f"Compiling Zig ODE '{ode_name}': {' '.join(cmd)}", file=sys.stderr)
    result = subprocess.run(cmd, cwd=ZIG_ROOT, capture_output=True, text=True)

    if result.returncode != 0:
        wrapper_path.unlink(missing_ok=True)
        raise RuntimeError(f"ODE compile failed:\n{result.stderr}")


def _compile_c_ode(ode_name: str, source: Path, optimize: bool) -> None:
    ODE_LIB_DIR.mkdir(parents=True, exist_ok=True)

    lib_path = get_ode_lib(ode_name)
    opt_flag = "-OReleaseFast" if optimize else "-ODebug"
    source_rel = source.relative_to(ZIG_ROOT)
    lib_rel = lib_path.relative_to(ZIG_ROOT)

    cmd: list[str] = [
        "zig",
        "build-lib",
        "-dynamic",
        opt_flag,
        str(source_rel),
        f"-femit-bin={lib_rel}",
        "-lc",
    ]

    print(f"Compiling C ODE '{ode_name}': {' '.join(cmd)}", file=sys.stderr)
    result = subprocess.run(cmd, cwd=ZIG_ROOT, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"ODE compile failed:\n{result.stderr}")


def compile_ode(ode_name: str, optimize: bool = True) -> None:
    """Compile a single ODE to a shared library.

    Dispatches to ``zig build-lib`` for ``.zig`` sources or ``cc`` for ``.c``.

    :param ode_name: Name of the ODE (without extension).
    :param optimize: If True, compile with full optimisation.
    """
    source = get_ode_source(ode_name)

    if source.suffix == ".zig":
        _compile_zig_ode(ode_name, source, optimize)
    else:
        _compile_c_ode(ode_name, source, optimize)

    _get_ode_hash_file(ode_name).write_text(_compute_ode_hash(ode_name))
    print(f"ODE '{ode_name}' compiled successfully!", file=sys.stderr)


def compile_ode_if_needed(ode_name: str, optimize: bool = True) -> bool:
    """Compile ODE only if source has changed since last build.

    :return: True if compilation was performed.
    """
    if ode_needs_compile(ode_name):
        compile_ode(ode_name, optimize)
        return True
    return False


def list_available_odes() -> list[str]:
    """List all ODE source files in ``user_odes/`` (``.zig`` and ``.c``)."""
    if not USER_ODE_DIR.exists():
        return []
    return sorted({f.stem for f in USER_ODE_DIR.iterdir() if f.suffix in (".zig", ".c")})
