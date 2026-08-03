# Design Studies

These studies preserve the experiments behind solver and batching decisions. They are narrower than case studies: each asks one implementation question and records enough context to interpret the timing or numerical difference.

- [Diffrax batching, precision, and devices](diffrax-batching-precision.ipynb) crosses CPU and GPU execution with shared and independent adaptive stepping, float32 and float64, tolerance choices, and cold versus warmed execution.
- [Event stopping](lorenz-event-stopping.ipynb) checks how chunked batches interact with a Lorenz unboundedness event.

The notebooks contain committed outputs and are not re-executed when the documentation is built. Reproduction commands are included in each notebook.
