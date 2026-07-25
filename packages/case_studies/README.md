# Case Studies

This directory contains case studies from the original bSTAB MATLAB paper, ported to Python using the pybasin library.

## Purpose

1. **Validation**: Verify that pybasin produces the same basin stability values as the MATLAB implementation
2. **Thesis Artifacts**: Generate figures, results, and reports for the bachelor thesis
3. **Examples**: Serve as usage examples for the library

## Case Studies

### Duffing Oscillator (`duffing_oscillator/`)
- Forced Duffing oscillator with two attractors
- Supervised and unsupervised learning approaches
- **Reference**: Original bSTAB paper, Section X TODO: Update

### Lorenz System (`lorenz/`)
- Classic chaotic system with parameter variations
- Studies on sigma parameter and hyperparameter N
- **Reference**: Original bSTAB paper, Section Y TODO: Update

### Pendulum (`pendulum/`)
- Forced pendulum with different parameter cases
- Bifurcation analysis
- Grid-based sampling studies
- **Reference**: Original bSTAB paper, Section Z TODO: Update

### Friction System (`friction/`)
- System with friction effects
- Parameter variation studies
- **Reference**: Original bSTAB paper, Section W TODO: Update

## Running Case Studies

Each case study can be run independently:

```bash
# From the project root
uv run python case_studies/duffing_oscillator/main_supervised.py
uv run python case_studies/lorenz/main_lorenz.py
# etc.
```

## Generated Artifacts

All outputs (figures, results, reports) are saved to the `artifacts/` directory at the project root.

## Converting to Tests

The case studies are also converted to integration tests in `tests/integration/` to automatically validate correctness against the MATLAB implementation.
