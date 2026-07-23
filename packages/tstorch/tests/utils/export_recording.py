# pyright: basic
"""In-memory `torch.export` safety recording, mirroring `utils/compile_recording.py` exactly -
same upsert-by-`(feature, params)` model, same JSON dump shape. Kept as a separate module
(not merged into `utils/compile_recording.py`) since `torch.export` is a genuinely independent
check from `torch.compile(fullgraph=True)` (different tracer, can pass one and fail the
other - see `references/api-design.md`), so its results are a distinct table, not more
rows in the compile-safety one.

`dump_export_records` mirrors `dump_compile_records`'s signature so the pytest session dump
keeps compile and export recording symmetric.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

_RECORDS: dict[tuple[str, str], dict[str, Any]] = {}


def _current_test_hint() -> str:
    raw = os.environ.get("PYTEST_CURRENT_TEST", "unknown")
    return raw.rsplit(" (", 1)[0]


def _param_key(params: dict[str, Any] | None) -> str:
    if not params:
        return "default"
    return repr(tuple(sorted((key, repr(value)) for key, value in params.items())))


def record_export(
    feature: str,
    params: dict[str, Any] | None,
    *,
    exported: bool,
    error: str | None,
) -> None:
    """Upsert one `torch.export` check's result into the in-memory record for
    `(feature, params)` - same merge semantics as `compile_recording.record_compile`."""
    key = (feature, _param_key(params))
    existing = _RECORDS.get(key)
    hint = _current_test_hint()
    if existing is None:
        _RECORDS[key] = {
            "feature": feature,
            "params": params or {},
            "exported": exported,
            "error": error,
            "test": hint,
        }
        return
    existing["exported"] = existing["exported"] and exported
    if error and not existing["error"]:
        existing["error"] = error
    if hint not in existing["test"]:
        existing["test"] += f", {hint}"


def dump_export_records(path: Path) -> None:
    """Write every record accumulated this session to `path` as a JSON list."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(list(_RECORDS.values()), indent=2, sort_keys=True) + "\n")
