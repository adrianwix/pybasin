import logging
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

    @property
    def total_elapsed(self) -> float:
        return time.perf_counter() - self._total_start

    def summary(self) -> None:
        total = self.total_elapsed
        logger.info("BASIN STABILITY ESTIMATION COMPLETE")
        logger.info("Total time: %.4fs", total)
        logger.info("Timing Breakdown:")
        for rec in self._records:
            pct = (rec.elapsed / total * 100) if total > 0 else 0.0
            logger.info("  %-24s %8.4fs  (%5.1f%%)", rec.label, rec.elapsed, pct)
            for key, val in rec.details.items():
                logger.info("    %s: %s", key, val)
