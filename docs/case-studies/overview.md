# Case Studies Overview

Documented here are the case studies that validate pybasin against its original MATLAB counterpart, bSTAB. Each study targets a specific dynamical system and compares the two implementations on identical initial conditions.

## Classification Quality Metrics

To assess whether pybasin reproduces the MATLAB bSTAB results, we compare predicted attractor labels from pybasin with ground truth labels produced by bSTAB. Both tools classify trajectories into discrete attractor categories (e.g., "FP", "LC", "chaos"), which makes it possible to apply standard classification metrics directly.

### Methodology

Given that attractor labels from two independent implementations are compared, we treat the problem as a classification task and evaluate agreement using established metrics.

Concretely, each test case proceeds as follows:

1. Load the exact initial conditions exported from MATLAB ground truth CSV files
2. Classify those initial conditions with pybasin
3. Match the resulting labels against the MATLAB ground truth
4. Evaluate the classification metrics described below

### Metrics Used

#### 1. F1-Score (Per Class)

Per-class classification quality is measured by the [F1-score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score):

$$F1 = \frac{2 \cdot TP}{2 \cdot TP + FP + FN}$$

where TP = true positives, FP = false positives, FN = false negatives for that class.

**Range:** [0, 1], where 1.0 = perfect classification for that class

#### 2. Macro F1-Score (Overall)

Averaging the per-class F1-scores yields the macro F1-score, which captures overall classification quality without weighting by class frequency:

$$\text{Macro F1} = \frac{1}{K} \sum_{k=1}^{K} F1_k$$

where $K$ is the number of classes (attractor types).

**Range:** [0, 1], where 1.0 = perfect classification across all classes

#### 3. Matthews Correlation Coefficient (MCC)

As a global measure of prediction–ground truth correlation, we also report the [Matthews correlation coefficient](https://scikit-learn.org/stable/modules/model_evaluation.html#matthews-corrcoef).

For **binary classification**:

$$\text{MCC} = \frac{TP \cdot TN - FP \cdot FN}{\sqrt{(TP + FP)(TP + FN)(TN + FP)(TN + FN)}}$$

For **multiclass classification**, scikit-learn implements a generalization based on the confusion matrix $C$:

$$\text{MCC} = \frac{c \cdot s - \sum_k p_k \cdot t_k}{\sqrt{(s^2 - \sum_k p_k^2)(s^2 - \sum_k t_k^2)}}$$

where:

- $t_k = \sum_i^K C_{ik}$ — the number of times class $k$ truly occurred
- $p_k = \sum_i^K C_{ki}$ — the number of times class $k$ was predicted
- $c = \sum_k^K C_{kk}$ — the total number of samples correctly predicted
- $s = \sum_i^K \sum_j^K C_{ij}$ — the total number of samples

**Range:** [-1, 1], where:

- +1 = perfect prediction
- 0 = random prediction
- -1 = complete disagreement

Because basin stability problems often feature one dominant attractor, MCC is well-suited here; it remains informative even under class imbalance.

### Quality Thresholds

| Metric       | Excellent | Good   | Acceptable | Poor   |
| ------------ | --------- | ------ | ---------- | ------ |
| **F1**       | ≥ 0.95    | ≥ 0.90 | ≥ 0.80     | < 0.80 |
| **Macro F1** | ≥ 0.95    | ≥ 0.90 | ≥ 0.80     | < 0.80 |
| **MCC**      | ≥ 0.90    | ≥ 0.80 | ≥ 0.70     | < 0.70 |

Scores in the Excellent or Good range confirm that pybasin faithfully reproduces the MATLAB behaviour. Acceptable scores point to minor discrepancies—often numerical precision or boundary-case effects. Poor scores warrant further investigation.

### Reading Comparison Tables

Each case study contains a comparison table with the following columns:

- **Attractor**: The attractor type (e.g., "FP", "LC", "chaos")
- **pybasin BS +/- SE**: Basin stability and standard error from the Python implementation
- **bSTAB BS +/- SE**: Corresponding values from the MATLAB reference
- **F1**: Per-class F1-score

Above each table, a summary line reports the macro F1-score and the MCC for that comparison.

## Purpose

These case studies fulfil several roles at once. First, they provide a systematic validation path: comparing pybasin results against MATLAB bSTAB on the same initial conditions establishes correctness. Beyond validation, they double as usage examples for different classes of dynamical systems. They also produce the figures and numerical tables needed for documentation. Finally, they serve as performance benchmarks.

## Available Case Studies

### [Duffing Oscillator](duffing.md)

A periodically forced Duffing oscillator that exhibits multistability with five coexisting limit cycle attractors.

**Key Features:**

- Five coexisting limit cycle attractors
- Supervised vs. unsupervised classification comparison
- Feature extraction based on maximum amplitude and standard deviation

**Reference:** Thomson, J. M. T., & Stewart, H. B. (2002). _Nonlinear dynamics and chaos_ (2nd ed.). Wiley. (See p. 9, Fig. 1.9)

**Files:** `case_studies/duffing_oscillator/`

---

### [Lorenz System](lorenz.md)

A modified Lorenz system in which two chaotic attractors coexist alongside unbounded trajectories.

**Key Features:**

- Two coexisting chaotic attractors and unbounded solutions
- Parameter sweep over σ
- Sample size (N) convergence study
- Sensitivity analysis of integrator tolerances (rtol/atol)

**Reference:** Li, C., & Sprott, J. C. (2014). Multistability in the Lorenz system: A broken butterfly. _International Journal of Bifurcation and Chaos_, _24_(10), Article 1450131. https://doi.org/10.1142/S0218127414501314

**Files:** `case_studies/lorenz/`

---

### [Pendulum](pendulum.md)

A periodically forced pendulum, studied under several different forcing parameter configurations.

**Key Features:**

- Multiple parameter cases
- Fixed point and limit cycle attractors
- Supervised classification

**Reference:** Menck, P., Heitzig, J., Marwan, N., & Kurths, J. (2013). How basin stability complements the linear-stability paradigm. _Nature Physics_, _9_, 89–92. https://doi.org/10.1038/nphys2516

**Files:** `case_studies/pendulum/`

---

### [Friction System](friction.md)

A mechanical oscillator subject to dry friction, resulting in non-smooth dynamics and coexisting fixed point and limit cycle attractors.

**Key Features:**

- Fixed point and limit cycle attractors
- Non-smooth dynamics due to friction
- Parameter sweep over the driving velocity $v_d$

**Reference:** Stender, M., Hoffmann, N., & Papangelo, A. (2020). The basin stability of bi-stable friction-excited oscillators. _Lubricants_, _8_(12), Article 105. https://doi.org/10.3390/lubricants8120105

**Files:** `case_studies/friction/`

---

### [Rössler Network](rossler-network.md)

A network of coupled Rössler oscillators, used to study synchronization and its basin stability.

**Key Features:**

- Coupled oscillator dynamics
- Synchronization analysis
- Network-level basin stability estimation

**Reference:** Menck, P., Heitzig, J., Marwan, N., & Kurths, J. (2013). How basin stability complements the linear-stability paradigm. _Nature Physics_, _9_, 89–92. https://doi.org/10.1038/nphys2516

**Files:** `case_studies/rossler_network/`

## Running Case Studies

From the project root, individual case studies can be executed as follows:

```bash
# Navigate to project root
cd /path/to/pybasinWorkspace

# Run a specific case study
uv run python -m case_studies.duffing_oscillator.main_duffing_oscillator_supervised
uv run python -m case_studies.lorenz.main_lorenz
uv run python -m case_studies.pendulum.main_pendulum_case1
```

## Integration Tests

Each case study has a corresponding integration test that automatically checks correctness:

```bash
# Run all integration tests
uv run pytest tests/integration/

# Run specific case study test
uv run pytest tests/integration/test_duffing.py
```

## Generated Artifacts

Outputs are written to two locations:

- **Figures**: `docs/assets/` — plots and visualisations
- **Results**: `artifacts/results/` — numerical data (JSON, CSV)

Passing the `--generate-artifacts` flag to the test runner regenerates these outputs:

```bash
# Generate artifacts for all case studies
uv run pytest tests/integration/ --generate-artifacts

# Generate artifacts for a specific case study
uv run pytest tests/integration/test_duffing.py --generate-artifacts
```

## Contributing New Case Studies

Adding a new case study involves the following steps:

1. Create a directory under `case_studies/`
2. Implement the ODE system and feature extractor
3. Write a main script that runs the analysis
4. Add a matching integration test
5. Document the study in this section

See the [Contributing Guide](../development/contributing.md) for further details.
