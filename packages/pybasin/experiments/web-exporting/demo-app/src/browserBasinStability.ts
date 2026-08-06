/**
 * Orchestrates the three independent pipeline stages for one case study: sample+integrate
 * (zigode WASM), extract features (ExecuTorch `.pte`), classify (ExecuTorch `.pte`,
 * supervised or unsupervised). Each stage is a separate call so a UI control that only
 * affects one stage doesn't need to redo the others -- see the experiment README's
 * "independent per-stage" requirement.
 *
 * Feature extraction always computes the same fixed `FEATURE_NAMES` set for every state (see
 * feature_extract_graph.py's module docstring), so a UI "choose features" control is a pure
 * client-side column slice of that fixed output (`classify()`'s `selection` argument) -- it
 * doesn't even need to re-run extraction, only re-run classification on the already-cached
 * full feature matrix.
 */

import { ExecuTorchRunner, type EtModuleInstance } from "./wasm/executorchRunner";
import { ZigOdeModule } from "./wasm/zigOde";

/** Must match `_FEATURE_NAMES` in
 * packages/pybasin/src/pybasin/feature_extractors/default_feature_extractor.py exactly --
 * that ordering is what `FeatureExtractGraph`'s `(n, state_dim * 10)` output columns are
 * grouped by (state-major, then this order within each state). */
export const FEATURE_NAMES = [
  "median",
  "mean",
  "standard_deviation",
  "variance",
  "root_mean_square",
  "maximum",
  "absolute_maximum",
  "minimum",
  "delta",
  "log_delta",
] as const;
export type FeatureName = (typeof FEATURE_NAMES)[number];

export interface FeatureSelectionEntry {
  state: number;
  name: FeatureName;
}
export type FeatureSelection = FeatureSelectionEntry[];

export interface PendulumConfig {
  alpha: number;
  torque: number;
  stiffness: number;
  /** Integration start (anchor); zigode's adaptive Dopri5 integrates continuously from here
   * to `windowEnd` and only *records* the window's save points -- unlike the fixed-step
   * solver Phase 1 tuned this window for, an adaptive solver pays no efficiency cost for a
   * long smooth transient, so `t0=0` here is the true reference start, not a tuning choice. */
  t0: number;
  /** Steady-state observation window. (150, 200) matches exactly what
   * extract_features_dim2.pte/classify_supervised.pte were exported for (Phase 1/2's tuned
   * config, already validated against the true `t_eval=(950,1000)` diffrax reference) --
   * changing this requires re-exporting those graphs with a matching `t_steps`. */
  windowStart: number;
  windowEnd: number;
  /** Must equal the exported graphs' `t_steps` (50). */
  nSave: number;
  rtol: number;
  atol: number;
  maxSteps: number;
}

export const PENDULUM_DEFAULT_CONFIG: PendulumConfig = {
  alpha: 0.1,
  torque: 0.5,
  stiffness: 1.0,
  t0: 0,
  windowStart: 150,
  windowEnd: 200,
  nSave: 50,
  rtol: 1e-8,
  atol: 1e-6,
  maxSteps: 100_000,
};

export const PENDULUM_STATE_DIM = 2;
/** Real physical trajectories with known labels (matches production `TemplateIntegrator`
 * usage) -- integrated once at load time via `integrateAndExtractTemplates`, not baked into
 * `classify_supervised.pte` (see classify_graph.py's module docstring). */
export const PENDULUM_TEMPLATE_Y0: readonly [number, number][] = [
  [0.4, 0.0],
  [2.7, 0.0],
];
export const PENDULUM_LABELS = ["FP", "LC"] as const;
/** Matches the feature the real production classifier uses (`setup_pendulum_system.py`'s
 * `{1: {"log_delta": None}}`) -- the demo UI's feature-selection control can add or swap
 * this freely (see `FeatureSelection`). */
export const PENDULUM_DEFAULT_SELECTION: FeatureSelection = [{ state: 1, name: "log_delta" }];

export interface SampleResult {
  y0: Float64Array; // (n * 2,)
  trajectories: Float64Array; // (n, nSave, stateDim) row-major, zigode's native layout
  fullFeatures: Float32Array; // (n, stateDim * FEATURE_NAMES.length)
}

export interface ClassifyResult {
  labelIdx: Int32Array; // (n,)
  basinStability: Map<number, number>; // label index -> fraction of samples
}

function columnIndices(stateDim: number, selection: FeatureSelection): number[] {
  return selection.map(({ state, name }) => state * FEATURE_NAMES.length + FEATURE_NAMES.indexOf(name));
}

function sliceFeatures(full: Float32Array, n: number, fullF: number, cols: number[]): Float32Array {
  const out = new Float32Array(n * cols.length);
  for (let i = 0; i < n; i++) {
    for (let c = 0; c < cols.length; c++) {
      out[i * cols.length + c] = full[i * fullF + (cols[c] ?? 0)] ?? 0;
    }
  }
  return out;
}

export class BrowserBasinStability {
  constructor(
    private readonly zig: ZigOdeModule,
    private readonly et: ExecuTorchRunner,
    private readonly extractFeaturesModel: EtModuleInstance,
    private readonly stateDim: number,
  ) {}

  private get fullFeatureCount(): number {
    return this.stateDim * FEATURE_NAMES.length;
  }

  /** Stages 1+2: sample `n` initial conditions uniformly in `[minLimits, maxLimits]`,
   * integrate (zigode WASM), and extract the full fixed feature set (ExecuTorch WASM).
   * Doesn't classify -- call `classify()` separately (and cache this result) so a
   * feature-selection or classifier-mode change can reuse it without resampling. */
  sampleIntegrateExtract(
    n: number,
    minLimits: [number, number],
    maxLimits: [number, number],
    config: PendulumConfig,
  ): SampleResult {
    const y0 = new Float64Array(n * 2);
    for (let i = 0; i < n; i++) {
      y0[i * 2 + 0] = minLimits[0] + Math.random() * (maxLimits[0] - minLimits[0]);
      y0[i * 2 + 1] = minLimits[1] + Math.random() * (maxLimits[1] - minLimits[1]);
    }
    const trajectories = this.integrate(y0, n, config);
    const fullFeatures = this.extractFullFeatures(trajectories, n, config.nSave);
    return { y0, trajectories, fullFeatures };
  }

  /** Integrates a fixed set of initial conditions (e.g. labeled template trajectories) and
   * extracts the same full fixed feature set -- used once at load time to build supervised
   * mode's reference data, kept in sync with the live batch's feature computation by
   * construction (same graphs, same config). */
  integrateAndExtractTemplates(
    templateY0: readonly [number, number][],
    config: PendulumConfig,
  ): Float32Array {
    const n = templateY0.length;
    const y0 = new Float64Array(n * 2);
    templateY0.forEach(([theta, omega], i) => {
      y0[i * 2] = theta;
      y0[i * 2 + 1] = omega;
    });
    const trajectories = this.integrate(y0, n, config);
    return this.extractFullFeatures(trajectories, n, config.nSave);
  }

  /** Stage 3: slice the selected feature columns out of already-computed full feature
   * matrices and classify -- no re-integration, no re-extraction. Reused for both a
   * feature-selection change and a classifier-mode change, since both only affect this
   * stage. */
  classify(
    fullFeatures: Float32Array,
    n: number,
    selection: FeatureSelection,
    classifyModel: EtModuleInstance,
    referenceFullFeatures?: Float32Array,
    nTemplates?: number,
  ): ClassifyResult {
    const cols = columnIndices(this.stateDim, selection);
    const features = sliceFeatures(fullFeatures, n, this.fullFeatureCount, cols);
    const reference =
      referenceFullFeatures !== undefined && nTemplates !== undefined
        ? {
            features: sliceFeatures(referenceFullFeatures, nTemplates, this.fullFeatureCount, cols),
            nTemplates,
          }
        : undefined;
    const labelIdxBig = this.et.runClassify(classifyModel, features, n, cols.length, reference);
    const labelIdx = Int32Array.from(labelIdxBig, v => Number(v));
    return { labelIdx, basinStability: fractionsOf(labelIdx) };
  }

  private integrate(y0: Float64Array, n: number, config: PendulumConfig): Float64Array {
    const saveAt = new Float64Array(config.nSave);
    for (let i = 0; i < config.nSave; i++) {
      saveAt[i] = config.windowStart + ((config.windowEnd - config.windowStart) * i) / (config.nSave - 1);
    }
    return this.zig.solveBatchPendulum({
      y0s: y0,
      t0: config.t0,
      t1: config.windowEnd,
      saveAt,
      rtol: config.rtol,
      atol: config.atol,
      maxSteps: config.maxSteps,
      alpha: config.alpha,
      torque: config.torque,
      stiffness: config.stiffness,
    });
  }

  private extractFullFeatures(trajectories: Float64Array, n: number, nSave: number): Float32Array {
    // ExecuTorch's feature extractor expects (n, stateDim, tSteps) float32 with the time axis
    // last and contiguous; zigode's output is (n, tSteps, stateDim) float64 row-major (state
    // pairs per step). Transpose + downcast here rather than in Zig, since this is a one-time
    // small reshape and keeps the WASM module's output layout simple.
    const y = new Float32Array(n * this.stateDim * nSave);
    for (let i = 0; i < n; i++) {
      for (let t = 0; t < nSave; t++) {
        for (let d = 0; d < this.stateDim; d++) {
          y[i * this.stateDim * nSave + d * nSave + t] =
            trajectories[i * nSave * this.stateDim + t * this.stateDim + d] ?? 0;
        }
      }
    }
    return this.et.runExtractFeatures(this.extractFeaturesModel, y, n, this.stateDim, nSave);
  }
}

function fractionsOf(labelIdx: Int32Array): Map<number, number> {
  const counts = new Map<number, number>();
  for (const label of labelIdx) counts.set(label, (counts.get(label) ?? 0) + 1);
  const fractions = new Map<number, number>();
  for (const [label, count] of counts) fractions.set(label, count / labelIdx.length);
  return fractions;
}
