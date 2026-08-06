import { useCallback, useEffect, useRef, useState } from "react";
import "./index.css";
import {
  BrowserBasinStability,
  FEATURE_NAMES,
  PENDULUM_DEFAULT_CONFIG,
  PENDULUM_DEFAULT_SELECTION,
  PENDULUM_LABELS,
  PENDULUM_STATE_DIM,
  PENDULUM_TEMPLATE_Y0,
  type FeatureName,
  type FeatureSelection,
  type SampleResult,
} from "./browserBasinStability";
import { ExecuTorchRunner, loadExecuTorchScript, type EtModuleInstance } from "./wasm/executorchRunner";
import { ZigOdeModule } from "./wasm/zigOde";

// Matches setup_pendulum_system.py's sampling range exactly.
const DEFAULT_THETA_MIN = -Math.PI + Math.asin(0.5);
const DEFAULT_THETA_MAX = Math.PI + Math.asin(0.5);
const DEFAULT_OMEGA_MIN = -10;
const DEFAULT_OMEGA_MAX = 10;

// Must not exceed the exported graphs' dynamic batch-dim upper bound (export_utils.py's
// MAX_BATCH).
const MAX_TRAJECTORIES = 20_000;

const STATE_NAMES = ["theta", "omega"];

type Mode = "supervised" | "unsupervised";

interface Loaded {
  bbs: BrowserBasinStability;
  supervisedModel: EtModuleInstance;
  unsupervisedModel: EtModuleInstance;
  /** Template trajectories' full feature matrix, integrated+extracted once at load time --
   * supervised mode slices this the same way as the live batch (see classify_graph.py's
   * module docstring on why `reference_features` is a runtime input, not baked in). */
  templateFullFeatures: Float32Array;
}

interface DisplayResult {
  y0: Float64Array;
  labelIdx: Int32Array;
  basinStability: Map<number, number>;
}

export function App() {
  const [loaded, setLoaded] = useState<Loaded | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [n, setN] = useState(300);
  const [thetaRange, setThetaRange] = useState<[number, number]>([DEFAULT_THETA_MIN, DEFAULT_THETA_MAX]);
  const [omegaRange, setOmegaRange] = useState<[number, number]>([DEFAULT_OMEGA_MIN, DEFAULT_OMEGA_MAX]);
  const [mode, setMode] = useState<Mode>("supervised");
  const [selection, setSelection] = useState<FeatureSelection>(PENDULUM_DEFAULT_SELECTION);
  const [result, setResult] = useState<DisplayResult | null>(null);
  const [running, setRunning] = useState(false);
  const [runError, setRunError] = useState<string | null>(null);
  const lastSample = useRef<SampleResult | null>(null);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const zig = await ZigOdeModule.load("/wasm/zigode/pendulum.wasm");
        await loadExecuTorchScript("/wasm/executorch/pybasin_wasm_lib.js");
        const et = await ExecuTorchRunner.load(window.createExecuTorchModule);

        const extractFeaturesModel = await et.loadModel("/pte/extract_features_dim2.pte");
        const supervisedModel = await et.loadModel("/pte/classify_supervised.pte");
        const unsupervisedModel = await et.loadModel("/pte/classify_unsupervised_k2.pte");

        const bbs = new BrowserBasinStability(zig, et, extractFeaturesModel, PENDULUM_STATE_DIM);
        const templateFullFeatures = bbs.integrateAndExtractTemplates(
          PENDULUM_TEMPLATE_Y0,
          PENDULUM_DEFAULT_CONFIG,
        );
        if (!cancelled) {
          setLoaded({ bbs, supervisedModel, unsupervisedModel, templateFullFeatures });
        }
      } catch (err) {
        if (!cancelled) setLoadError(String(err));
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  const config = PENDULUM_DEFAULT_CONFIG;

  const modelFor = (loadedState: Loaded, m: Mode): EtModuleInstance =>
    m === "supervised" ? loadedState.supervisedModel : loadedState.unsupervisedModel;

  const classifyAndSetResult = useCallback(
    (loadedState: Loaded, sample: SampleResult, sel: FeatureSelection, m: Mode) => {
      const { labelIdx, basinStability } = loadedState.bbs.classify(
        sample.fullFeatures,
        sample.y0.length / 2,
        sel,
        modelFor(loadedState, m),
        m === "supervised" ? loadedState.templateFullFeatures : undefined,
        m === "supervised" ? PENDULUM_TEMPLATE_Y0.length : undefined,
      );
      setResult({ y0: sample.y0, labelIdx, basinStability });
    },
    [],
  );

  const run = useCallback(() => {
    if (!loaded) return;
    setRunning(true);
    setRunError(null);
    // Yield to the browser so the "running" state actually paints before the (synchronous,
    // WASM-bound) solve blocks the main thread.
    setTimeout(() => {
      try {
        const sample = loaded.bbs.sampleIntegrateExtract(
          n,
          [thetaRange[0], omegaRange[0]],
          [thetaRange[1], omegaRange[1]],
          config,
        );
        lastSample.current = sample;
        classifyAndSetResult(loaded, sample, selection, mode);
      } catch (err) {
        setRunError(String(err));
      } finally {
        setRunning(false);
      }
    }, 0);
  }, [loaded, n, thetaRange, omegaRange, mode, selection, config, classifyAndSetResult]);

  // Re-classify without re-sampling/re-integrating/re-extracting when only the classifier
  // mode or feature selection changes and we already have a sample -- the whole point of
  // splitting the pipeline into independent stages.
  const reclassifyWithCurrentSample = useCallback(
    (m: Mode, sel: FeatureSelection) => {
      if (loaded && lastSample.current) {
        try {
          classifyAndSetResult(loaded, lastSample.current, sel, m);
          setRunError(null);
        } catch (err) {
          setRunError(String(err));
        }
      }
    },
    [loaded, classifyAndSetResult],
  );

  const onModeChange = useCallback(
    (next: Mode) => {
      setMode(next);
      reclassifyWithCurrentSample(next, selection);
    },
    [selection, reclassifyWithCurrentSample],
  );

  const onSelectionChange = useCallback(
    (next: FeatureSelection) => {
      setSelection(next);
      reclassifyWithCurrentSample(mode, next);
    },
    [mode, reclassifyWithCurrentSample],
  );

  return (
    <div className="max-w-5xl mx-auto p-8">
      <h1 className="text-3xl font-bold mb-2">pybasin browser demo -- Pendulum</h1>
      <p className="text-sm text-gray-400 mb-6">
        Sampling, integration (zigode/Zig, adaptive Dopri5, WASM) and classification
        (ExecuTorch, WASM) all run client-side.
      </p>

      {loadError && <div className="text-red-500 mb-4">Failed to load: {loadError}</div>}
      {!loaded && !loadError && <div className="mb-4">Loading WASM modules...</div>}
      {runError && <div className="text-red-500 mb-4">Run failed: {runError}</div>}

      <div className="grid grid-cols-2 gap-6 mb-6">
        <RangeControl label="theta range" value={thetaRange} onChange={setThetaRange} min={-6} max={6} step={0.01} />
        <RangeControl label="omega range" value={omegaRange} onChange={setOmegaRange} min={-15} max={15} step={0.1} />
      </div>

      <div className="mb-6">
        <FeatureSelector selection={selection} onChange={onSelectionChange} />
      </div>

      <div className="flex items-center gap-6 mb-6">
        <label className="flex items-center gap-2">
          n trajectories
          <input
            type="number"
            min={1}
            max={MAX_TRAJECTORIES}
            value={n}
            onChange={e => setN(Math.max(1, Math.min(MAX_TRAJECTORIES, Number(e.target.value))))}
            className="w-24 bg-transparent border rounded px-2 py-1"
          />
        </label>

        <div className="flex items-center gap-2">
          <button
            className={`px-3 py-1 rounded border ${mode === "supervised" ? "bg-blue-600" : ""}`}
            onClick={() => onModeChange("supervised")}
          >
            Supervised
          </button>
          <button
            className={`px-3 py-1 rounded border ${mode === "unsupervised" ? "bg-blue-600" : ""}`}
            onClick={() => onModeChange("unsupervised")}
          >
            Unsupervised
          </button>
        </div>

        <button
          disabled={!loaded || running}
          onClick={run}
          className="px-4 py-2 rounded bg-green-600 disabled:opacity-50"
        >
          {running ? "Running..." : "Run"}
        </button>
      </div>

      {result && (
        <div className="grid grid-cols-2 gap-6">
          <PhasePortrait result={result} mode={mode} />
          <BasinStabilityBars result={result} mode={mode} />
        </div>
      )}
    </div>
  );
}

/** Checkbox grid over `state x FEATURE_NAMES` -- the demo UI's "choose features" control.
 * Since `FeatureExtractGraph` always computes every one of these for every state (see
 * browserBasinStability.ts), toggling a checkbox never re-runs integration or extraction,
 * only classification (see `onSelectionChange` in `App`). At least one feature must stay
 * selected -- an empty selection has no valid feature-count to classify on. */
function FeatureSelector({
  selection,
  onChange,
}: {
  selection: FeatureSelection;
  onChange: (next: FeatureSelection) => void;
}) {
  const isSelected = (state: number, name: FeatureName) =>
    selection.some(e => e.state === state && e.name === name);

  const toggle = (state: number, name: FeatureName) => {
    if (isSelected(state, name)) {
      if (selection.length <= 1) return; // keep at least one feature selected
      onChange(selection.filter(e => !(e.state === state && e.name === name)));
    } else {
      onChange([...selection, { state, name }]);
    }
  };

  return (
    <div>
      <div className="text-sm mb-1">features</div>
      <div className="grid grid-cols-2 gap-4">
        {Array.from({ length: PENDULUM_STATE_DIM }, (_, state) => (
          <div key={state} className="border rounded p-2">
            <div className="text-xs text-gray-400 mb-1">{STATE_NAMES[state] ?? `state ${state}`}</div>
            <div className="flex flex-wrap gap-x-3 gap-y-1">
              {FEATURE_NAMES.map(name => (
                <label key={name} className="flex items-center gap-1 text-xs">
                  <input
                    type="checkbox"
                    checked={isSelected(state, name)}
                    onChange={() => toggle(state, name)}
                  />
                  {name}
                </label>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function RangeControl({
  label,
  value,
  onChange,
  min,
  max,
  step,
}: {
  label: string;
  value: [number, number];
  onChange: (v: [number, number]) => void;
  min: number;
  max: number;
  step: number;
}) {
  return (
    <div>
      <div className="text-sm mb-1">
        {label}: [{value[0].toFixed(2)}, {value[1].toFixed(2)}]
      </div>
      <div className="flex gap-2">
        <input
          type="range"
          min={min}
          max={max}
          step={step}
          value={value[0]}
          onChange={e => onChange([Number(e.target.value), value[1]])}
          className="flex-1"
        />
        <input
          type="range"
          min={min}
          max={max}
          step={step}
          value={value[1]}
          onChange={e => onChange([value[0], Number(e.target.value)])}
          className="flex-1"
        />
      </div>
    </div>
  );
}

function PhasePortrait({ result, mode }: { result: DisplayResult; mode: Mode }) {
  const width = 320;
  const height = 320;
  const n = result.y0.length / 2;
  const points = Array.from({ length: n }, (_, i) => ({
    theta: result.y0[i * 2] ?? 0,
    omega: result.y0[i * 2 + 1] ?? 0,
    label: result.labelIdx[i] ?? 0,
  }));
  const thetaMin = Math.min(...points.map(p => p.theta));
  const thetaMax = Math.max(...points.map(p => p.theta));
  const omegaMin = Math.min(...points.map(p => p.omega));
  const omegaMax = Math.max(...points.map(p => p.omega));

  const colors = mode === "supervised" ? ["#60a5fa", "#f97316"] : ["#34d399", "#f472b6", "#a78bfa", "#fbbf24"];

  return (
    <div>
      <h2 className="text-lg font-semibold mb-2">Initial conditions (colored by label)</h2>
      <svg width={width} height={height} className="border rounded bg-black/20">
        {points.map((p, i) => {
          const x = ((p.theta - thetaMin) / (thetaMax - thetaMin || 1)) * (width - 20) + 10;
          const y = height - (((p.omega - omegaMin) / (omegaMax - omegaMin || 1)) * (height - 20) + 10);
          return <circle key={i} cx={x} cy={y} r={2.5} fill={colors[p.label % colors.length]} opacity={0.8} />;
        })}
      </svg>
    </div>
  );
}

function BasinStabilityBars({ result, mode }: { result: DisplayResult; mode: Mode }) {
  const entries = [...result.basinStability.entries()].sort((a, b) => a[0] - b[0]);
  const names = mode === "supervised" ? PENDULUM_LABELS : entries.map(([label]) => `cluster ${label}`);

  return (
    <div>
      <h2 className="text-lg font-semibold mb-2">Basin stability</h2>
      <div className="space-y-2">
        {entries.map(([label, fraction], i) => (
          <div key={label} className="flex items-center gap-2">
            <div className="w-24 text-sm">{names[i] ?? `cluster ${label}`}</div>
            <div className="flex-1 bg-black/20 rounded h-4 overflow-hidden">
              <div className="h-full bg-blue-500" style={{ width: `${fraction * 100}%` }} />
            </div>
            <div className="w-12 text-sm text-right">{(fraction * 100).toFixed(1)}%</div>
          </div>
        ))}
      </div>
    </div>
  );
}

export default App;
