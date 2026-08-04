"""Core tstorch must stay importable and usable with only `torch` installed.

Runs the import + a minimal eager extraction in a subprocess so already-imported test
dependencies (dask, cupy, ...) in this test session can't mask a real leak. NumPy is
excluded from this runtime check: PyTorch itself imports NumPy internally, so its
presence in `sys.modules` after `import torch` is not something tstorch controls or can
avoid. `test_no_toplevel_numpy_or_dask_import_outside_optional_boundary` instead checks
statically that no tstorch source file *itself* imports numpy/dask at module level
outside the optional `tstorch.dask` subpackage and `tstorch.preprocessing`'s
function-local dispatch imports.
"""

import ast
import subprocess
import sys
from pathlib import Path

_CHECK_SCRIPT = """
import sys
import torch

from tstorch.torch_feature_extractor import TorchFeatureExtractor
from tstorch.preprocessing import ExtremeImputer, StandardScaler

leaked = [name for name in ("dask", "cupy", "dask_ml") if name in sys.modules]
assert not leaked, f"tstorch leaked optional imports at core-import time: {leaked}"

y = torch.randn(4, 1, 20)
extractor = TorchFeatureExtractor(features="minimal", use_compile=False)
features = extractor.extract_features(y)
imputed = ExtremeImputer().transform(features)
scaler = StandardScaler().fit(imputed)
scaler.transform(imputed)

leaked = [name for name in ("dask", "cupy", "dask_ml") if name in sys.modules]
assert not leaked, f"tstorch leaked optional imports during eager use: {leaked}"
print("OK")
"""


def test_core_import_and_eager_use_never_import_dask_or_cupy() -> None:
    result = subprocess.run(
        [sys.executable, "-c", _CHECK_SCRIPT],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK" in result.stdout


def test_no_toplevel_numpy_or_dask_import_outside_optional_boundary() -> None:
    src_root = Path(__file__).parents[2] / "src" / "tstorch"
    optional_dirs = {src_root / "dask"}
    forbidden = {"numpy", "dask", "dask_ml", "cupy"}
    violations: list[str] = []

    for path in src_root.rglob("*.py"):
        if any(path.is_relative_to(d) for d in optional_dirs):
            continue
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Import | ast.ImportFrom):
                continue
            if isinstance(node, ast.ImportFrom) and (node.col_offset > 0 or node.level > 0):
                continue  # function-local (e.g. preprocessing's optional Dask dispatch)
            module_names = (
                [alias.name for alias in node.names]
                if isinstance(node, ast.Import)
                else [node.module or ""]
            )
            for name in module_names:
                top_level = name.split(".")[0]
                if top_level in forbidden and node.col_offset == 0:
                    violations.append(f"{path.relative_to(src_root)}:{node.lineno} imports {name}")

    assert not violations, (
        "Top-level optional-dependency imports outside tstorch.dask:\n" + "\n".join(violations)
    )
