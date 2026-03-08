# Plan: Simplify Logging in `BasinStabilityEstimator`

## Problem

`estimate_bs()` is 335 lines long. **94 of those lines are logging/timing boilerplate** — that's ~28% of the method dedicated purely to instrumentation, not business logic.

Three concrete issues:

1. **Duplicate timing**: Every step logs its time inline (`logger.info("... in %.4fs", elapsed)`) AND again in the summary table via `_log_timing(...)`. The same information is emitted twice.

2. **Manual timing boilerplate everywhere**: Every step repeats this pattern:
   ```python
   logger.info("STEP N: ...")
   tN = time.perf_counter()
   # ... actual work ...
   tN_elapsed = time.perf_counter() - tN
   logger.info("  ... in %.4fs", tN_elapsed)
   ```
   There are 12+ timing variables (`t1`, `t2_start`, `t2a_elapsed`, `t2b_elapsed`, `t3b_elapsed`, `t_orbit_elapsed`, `t5b_elapsed`, etc.) scattered throughout the method.

3. **Business logic is buried**: The actual algorithm (sample → integrate → detect unbounded → extract features → filter → classify → compute BS) is hard to follow because every 2-3 lines of logic are interrupted by logging calls.

## Requirements

- Know how long every step takes (for optimization work)
- Know what happened in each step (shapes, counts, modes chosen)
- Keep the business logic readable

## Proposed Solution: `StepTimer` Context Manager

Use a context manager that handles timing automatically and collects results for a single summary at the end. This is a lightweight, stdlib-only pattern (no new dependencies).

### The `StepTimer` class

```python
import time
import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator

logger = logging.getLogger(__name__)

@dataclass
class StepRecord:
    label: str
    elapsed: float
    details: dict[str, str | int | float] = field(default_factory=dict)

class StepTimer:
    def __init__(self) -> None:
        self._records: list[StepRecord] = []
        self._total_start: float = 0.0
        self._current: StepRecord | None = None

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
```

### How `estimate_bs()` would look after

```python
def estimate_bs(self, parallel_integration: bool = True) -> StudyResult:
    timer = StepTimer()
    timer.start()

    # Step 1: Sampling
    with timer.step("Sampling") as step:
        self.y0 = self.sampler.sample(self.n)
        step.details["n_samples"] = len(self.y0)

    # Step 2: Integration
    with timer.step("Integration") as step:
        t, y = self._integrate(parallel_integration)
        step.details["trajectory_shape"] = str(y.shape)
        step.details["mode"] = "parallel" if (parallel_integration and self.template_integrator) else "sequential"

    # Step 3: Solution
    with timer.step("Solution") as step:
        self.solution = Solution(initial_condition=self.y0, time=t, y=y)

    # Step 3b: Unbounded detection
    unbounded_mask, n_unbounded = None, 0
    total_samples = len(self.y0)
    if self.detect_unbounded:
        with timer.step("Unbounded Detection") as step:
            unbounded_mask = self._detect_unbounded_trajectories(y)
            n_unbounded = int(unbounded_mask.sum().item())
            step.details["n_unbounded"] = n_unbounded
            step.details["pct"] = f"{(n_unbounded / total_samples) * 100:.1f}%"

        if n_unbounded == total_samples:
            # ... early return for all-unbounded case ...

        if n_unbounded > 0:
            # ... filter to bounded only ...

    # Step 4: Feature extraction
    with timer.step("Feature Extraction") as step:
        features = self.feature_extractor.extract_features(self.solution)
        feature_names = self._get_feature_names()
        self.solution.set_extracted_features(features, feature_names)
        step.details["shape"] = str(features.shape)

    # Step 5: Feature filtering
    with timer.step("Feature Filtering") as step:
        features, feature_names = self._filter_features(features, feature_names)
        step.details["n_features"] = features.shape[1]

    # Step 6: Classification
    with timer.step("Classification") as step:
        labels = self._classify(features, unbounded_mask, n_unbounded, total_samples)

    # Step 7: Basin stability
    with timer.step("BS Computation") as step:
        self.bs_vals = self._compute_bs(labels)
        for label, val in self.bs_vals.items():
            step.details[label] = f"{val * 100:.2f}%"

    timer.summary()
    return StudyResult(...)
```

### What this achieves

| Aspect | Before | After |
|--------|--------|-------|
| `estimate_bs()` length | ~335 lines | ~80-90 lines |
| Timing variables | 12+ manual variables | 0 (managed by `StepTimer`) |
| Duplicate logs | Every step logged twice | Single summary table |
| Business logic visibility | Buried in logging | Clear linear flow |
| Timing info preserved | Yes (inline + summary) | Yes (summary table + `details`) |
| Step details preserved | Yes (scattered `logger.info`) | Yes (via `step.details` dict) |

### Extract methods for complex steps

Some steps have branching logic that further clutters `estimate_bs()`. Extract these into private methods:

- `_integrate(parallel: bool) -> tuple[Tensor, Tensor]` — handles parallel vs sequential, template vs main
- `_filter_features(features, names) -> tuple[Tensor, list[str]]` — wraps `_apply_feature_filtering` + solution bookkeeping
- `_classify(features, unbounded_mask, ...) -> ndarray` — handles bounded/unbounded label reconstruction + solution restoration
- `_compute_bs(labels) -> dict[str, float]` — string conversion, unique counting, fraction math

These methods contain the logic that's currently inline. They don't take timing parameters — `StepTimer` handles that externally.

## File placement

Create `src/pybasin/step_timer.py` for the `StepTimer` and `StepRecord` classes.

## Migration steps

1. Create `StepTimer` in `src/pybasin/step_timer.py`
2. Extract `_integrate`, `_filter_features`, `_classify`, `_compute_bs` private methods
3. Rewrite `estimate_bs()` using `StepTimer` context managers
4. Remove `_log_timing()` module-level function
5. Remove all inline timing variables and duplicate `logger.info` calls
6. Keep `logger.debug` calls for verbose diagnostic info (feature sample values, etc.)
7. Run `bash scripts/ci.sh` and existing tests to verify nothing breaks

## Alternatives Considered

**Decorators on extracted methods**: Clean for simple cases but doesn't work well with branching logic (parallel vs sequential integration) or steps that need to record runtime metadata (shapes, counts). Context managers are more flexible.

**structlog / structured logging**: Good library but adds a dependency for something solvable with stdlib. Could be adopted later if JSON-structured logs become needed.

**Pipeline pattern (list of step callables)**: Too rigid — `estimate_bs` has conditional steps (unbounded detection, template integration) and early returns that don't fit a linear pipeline.
