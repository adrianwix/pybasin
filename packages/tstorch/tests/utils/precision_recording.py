# pyright: basic
"""In-memory numerical-accuracy recording, dumped to JSON at the end of a pytest session.

Unit tests record comparisons through `assert_feature_matches_tsfresh` in
`tests/utils/unit_test_utils.py`. Records upsert by `(feature, params)`, mirroring the
performance store's upsert-by-key model.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

_RECORDS: dict[tuple[str, str, str], dict[str, Any]] = {}


def _current_test_hint() -> str:
    """Return the running test's `path::test_name[params]` id, stripped of pytest's phase
    suffix (` (call)`/` (setup)`) - this is the "where is this tested" pointer for the report.
    """
    raw = os.environ.get("PYTEST_CURRENT_TEST", "unknown")
    return raw.rsplit(" (", 1)[0]


def _param_key(params: dict[str, Any] | None) -> str:
    if not params:
        return "default"
    return repr(tuple(sorted((key, repr(value)) for key, value in params.items())))


def record_precision(
    feature: str,
    params: dict[str, Any] | None,
    *,
    dtype: str,
    rows_compared: int,
    total_abs_diff: float,
    max_abs_diff: float,
    nan_mismatches: int,
    passed: bool,
    atol: float,
    rtol: float,
) -> None:
    """Upsert comparison stats for one ``(feature, dtype, params)`` combination."""
    key = (feature, dtype, _param_key(params))
    existing = _RECORDS.get(key)
    hint = _current_test_hint()
    if existing is None:
        _RECORDS[key] = {
            "feature": feature,
            "dtype": dtype,
            "params": params or {},
            "rows_compared": rows_compared,
            "total_abs_diff": total_abs_diff,
            "max_abs_diff": max_abs_diff,
            "nan_mismatches": nan_mismatches,
            "passed": passed,
            "atol": atol,
            "rtol": rtol,
            "test": hint,
        }
        return
    existing["rows_compared"] += rows_compared
    existing["total_abs_diff"] += total_abs_diff
    existing["max_abs_diff"] = max(existing["max_abs_diff"], max_abs_diff)
    existing["nan_mismatches"] += nan_mismatches
    existing["passed"] = existing["passed"] and passed
    existing["atol"] = max(existing["atol"], atol)
    existing["rtol"] = max(existing["rtol"], rtol)
    if hint not in existing["test"]:
        existing["test"] += f", {hint}"


def dump_precision_records(path: Path) -> None:
    """Write every record accumulated this session to `path` as a JSON list."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(list(_RECORDS.values()), indent=2, sort_keys=True) + "\n")
