import logging
import re
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


def _make_details() -> dict[str, str | int | float]:
    return {}


@dataclass
class StepRecord:
    label: str
    elapsed: float
    details: dict[str, str | int | float] = field(default_factory=_make_details)


_P_SUFFIX_RE = re.compile(r"^(.+?) \[p=(\d+)\]$")


class StepTimer:
    """Context-manager-based step timer that collects timing records for a summary log."""

    def __init__(self) -> None:
        self._records: list[StepRecord] = []
        self._total_start: float = 0.0

    def start(self) -> None:
        self._total_start = time.perf_counter()

    @contextmanager
    def step(self, label: str) -> Iterator[StepRecord]:
        record = StepRecord(label=label, elapsed=0.0)
        t0 = time.perf_counter()
        yield record
        record.elapsed = time.perf_counter() - t0
        self._records.append(record)
        # Per-parameter steps (contain " [p=N]") are noisy individually — the summary
        # aggregates them, so skip the immediate log for those.
        if not _P_SUFFIX_RE.match(label):
            details_str = "  ".join(f"{k}={v}" for k, v in record.details.items())
            if details_str:
                logger.info("  ✓ %-28s %.4fs  %s", label, record.elapsed, details_str)
            else:
                logger.info("  ✓ %-28s %.4fs", label, record.elapsed)

    @property
    def total_elapsed(self) -> float:
        return time.perf_counter() - self._total_start

    def summary(self) -> None:
        total = self.total_elapsed
        logger.info("BASIN STABILITY ESTIMATION COMPLETE")
        logger.info("Total time: %.4fs", total)
        logger.info("Timing Breakdown:")

        # Group records: collect per-parameter groups (label contains " [p=N]")
        # and keep singleton steps in insertion order.
        seen_groups: dict[str, list[StepRecord]] = {}
        order: list[str | StepRecord] = []  # str = group base name, StepRecord = singleton

        for rec in self._records:
            m = _P_SUFFIX_RE.match(rec.label)
            if m:
                base = m.group(1)
                if base not in seen_groups:
                    seen_groups[base] = []
                    order.append(base)
                seen_groups[base].append(rec)
            else:
                order.append(rec)

        for item in order:
            if isinstance(item, StepRecord):
                pct = (item.elapsed / total * 100) if total > 0 else 0.0
                logger.info("  %-30s %8.4fs  (%5.1f%%)", item.label, item.elapsed, pct)
                for key, val in item.details.items():
                    logger.info("    %s: %s", key, val)
            else:
                # Aggregated per-parameter group
                group = seen_groups[item]
                times = [r.elapsed for r in group]
                n = len(times)
                mean_t = sum(times) / n
                min_t = min(times)
                max_t = max(times)
                total_t = sum(times)
                pct = (total_t / total * 100) if total > 0 else 0.0

                logger.info(
                    "  %-30s %8.4fs  (%5.1f%%)  [×%d  mean=%.4fs  min=%.4fs  max=%.4fs]",
                    item,
                    total_t,
                    pct,
                    n,
                    mean_t,
                    min_t,
                    max_t,
                )

                # Outliers: any run whose time deviates more than 2× the mean
                if n > 2:
                    threshold = 2.0 * mean_t
                    outliers = [(i, r) for i, r in enumerate(group) if r.elapsed > threshold]
                    for p_idx, rec in outliers:
                        m = _P_SUFFIX_RE.match(rec.label)
                        p_label = m.group(2) if m else str(p_idx)
                        logger.info(
                            "    ⚠ outlier  p=%-4s  %.4fs  (%.1f× mean)",
                            p_label,
                            rec.elapsed,
                            rec.elapsed / mean_t,
                        )
                        for key, val in rec.details.items():
                            logger.info("      %s: %s", key, val)
