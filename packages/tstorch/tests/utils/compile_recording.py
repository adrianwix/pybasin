# pyright: basic
"""In-memory torch.compile(fullgraph=True) safety recording, dumped to JSON at session end.

Unit tests call `record_compile` (directly, or via `assert_feature_compiles_fullgraph` in
`tests/utils/benchmark_utils.py`) once per case they compile-check. Records upsert by
`(feature, params)` - calling it again for the same key merges into one aggregate entry (AND of
`compiled`, first non-empty `error`) instead of producing duplicates, mirroring
`precision_recording`'s upsert-by-key model.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

_RECORDS: dict[tuple[str, str], dict[str, Any]] = {}


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


def record_compile(
    feature: str,
    params: dict[str, Any] | None,
    *,
    compiled: bool,
    error: str | None,
) -> None:
    """Upsert one compile-check's result into the in-memory record for `(feature, params)`."""
    key = (feature, _param_key(params))
    existing = _RECORDS.get(key)
    hint = _current_test_hint()
    if existing is None:
        _RECORDS[key] = {
            "feature": feature,
            "params": params or {},
            "compiled": compiled,
            "error": error,
            "test": hint,
        }
        return
    existing["compiled"] = existing["compiled"] and compiled
    if error and not existing["error"]:
        existing["error"] = error
    if hint not in existing["test"]:
        existing["test"] += f", {hint}"


def dump_compile_records(path: Path) -> None:
    """Write every record accumulated this session to `path` as a JSON list."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(list(_RECORDS.values()), indent=2, sort_keys=True) + "\n")
