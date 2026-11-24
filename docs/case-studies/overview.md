# Case Studies Overview

The case studies exercise pybasin on continuous systems, a discrete map, smooth and
non-smooth dynamics, supervised classification, and unsupervised grouping. Their
reference claims are deliberately separated because the available evidence is not the
same for every system.

## Origin of the Ground Truth Data

The Duffing, Lorenz, pendulum, and friction datasets preserve accepted outputs from the
original bSTAB validation work. Their current automated tests are frozen regression
checks because the stored CSVs were subsequently regenerated with pybasin. Repeating the
bSTAB runs would restore an independently reproducible comparison, but the present docs
do not describe those snapshots as a new cross-implementation result.

The [Hénon map](henon-map.md) uses a current, reproducible Attractors.jl generator for
an independent implementation comparison.
The [Rossler network](rossler-network.md) checks aggregate basin-stability values reported
by Menck et al. (2013); it does not currently contain per-trajectory Attractors.jl labels.

## Classification Quality Metrics

Where per-trajectory reference labels exist, pybasin classifies the same initial
conditions and compares the paired labels. The source of those labels is stated on each
case-study page.

### Methodology

Each test follows the same measurement structure:

1. Load the exact initial conditions from the stored ground truth CSV files
2. Classify those initial conditions with pybasin
3. Match the resulting labels against the reference
4. Compute MCC, per-class and macro F1, a confusion matrix, and the mismatch count
5. Compare basin fractions using their absolute and relative sampling errors

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

Acceptance thresholds belong to the scientific claim made by each case, rather than to
generic labels such as "good" or "excellent." The Hénon map requires MCC and every
per-class F1 score to reach 0.99 because no ODE integration separates the implementations.
The historical regression cases normally
require MCC of at least 0.95, with explicitly documented exceptions for sensitivity
studies.

Alongside label agreement, each basin fraction reports
$e_{abs}=\sqrt{S_B(1-S_B)/N}$ and $e_{rel}=1/\sqrt{NS_B}$. For paired-label studies, the
aggregate difference must lie within 1.96 combined absolute standard errors. This is a
supplementary sampling-uncertainty check; MCC, F1, and the mismatch count remain the
direct evidence for identical initial conditions.

### Reading Comparison Tables

Each case study contains a comparison table with the following columns:

- **Attractor**: The attractor type (e.g., "FP", "LC", "chaos")
- **pybasin BS**: Basin stability with absolute and relative sampling errors
- **Reference BS**: Corresponding reference fraction and sampling errors
- **F1**: Per-class F1-score
- **z**: Difference in basin fractions divided by their combined absolute error

Above each table, the summary reports macro F1, MCC, and the paired mismatch count.

## Purpose

These studies provide regression coverage, independent implementation checks where a
current generator exists, and reproducible examples for several classes of dynamical
systems. Performance measurements live in the benchmark suites and are not inferred from
the accuracy tests.

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

---

### [Hénon Map](henon-map.md)

This discrete map validates `HistogramClusterer` and `PairwiseComparisonClusterer`
against labels generated by Attractors.jl from the same 1,000 initial conditions.

**Reference:** Hénon, M. (1976). A two-dimensional mapping with a strange attractor. _Communications in Mathematical Physics_, _50_(1), 69–77. https://doi.org/10.1007/BF01608556

**Files:** `case_studies/henon_map/`

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
uv run pytest tests/integration/duffing/test_duffing.py
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
uv run pytest tests/integration/duffing/test_duffing.py --generate-artifacts
```

## Contributing New Case Studies

Adding a new case study involves the following steps:

1. Create a directory under `case_studies/`
2. Implement the ODE system and feature extractor
3. Write a main script that runs the analysis
4. Add a matching integration test
5. Document the study in this section

See the [Contributing Guide](../development/contributing.md) for further details.
