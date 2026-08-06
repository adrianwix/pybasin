/**
 * Thin wrapper around ExecuTorch's official Emscripten/embind JS API
 * (executorch/extension/wasm/wasm_bindings.cpp -- `Module`/`Tensor` classes), exposing the
 * two graph shapes every case study's pipeline reduces to downstream of integration (see
 * browserBasinStability.ts's module docstring): extract_features and classify. Integration
 * is NOT part of this module -- it runs through zigOde.ts instead (see the experiment
 * README on why an ExecuTorch-only pipeline doesn't scale for the integration step).
 *
 * Using the official binding instead of a hand-rolled C ABI + manual heap marshaling gets a
 * cleaner JS API for free (real typed arrays via `Tensor.fromArray`, multi-tensor
 * input/output via `execute()`) and lets `.pte` files be fetched over HTTP at runtime
 * instead of needing to be baked into the WASM binary at build time.
 */

export interface EtTensor {
  readonly data: Float32Array | BigInt64Array;
  readonly sizes: number[];
  delete(): void;
}

export interface EtModuleInstance {
  execute(method: string, inputs: EtTensor[]): EtTensor[];
  forward(inputs: EtTensor[]): EtTensor[];
  delete(): void;
}

interface EmscriptenExecuTorchModule {
  Module: { load(data: Uint8Array): EtModuleInstance };
  Tensor: {
    fromArray(sizes: number[], data: number[] | Float32Array): EtTensor;
  };
}

type ModuleFactory = () => Promise<EmscriptenExecuTorchModule>;

declare global {
  interface Window {
    createExecuTorchModule: ModuleFactory;
  }
}

/** Loads the Emscripten UMD glue via a runtime-inserted `<script>` tag rather than a static
 * `<script src>` in index.html -- Bun's HTML bundler treats the latter as a build-time module
 * reference and tries (and fails) to resolve/bundle it instead of leaving it as a runtime URL. */
export function loadExecuTorchScript(url: string): Promise<void> {
  return new Promise((resolve, reject) => {
    const script = document.createElement("script");
    script.src = url;
    script.onload = () => resolve();
    script.onerror = () => reject(new Error(`Failed to load ${url}`));
    document.head.appendChild(script);
  });
}

export class ExecuTorchRunner {
  private constructor(private readonly module: EmscriptenExecuTorchModule) {}

  static async load(moduleFactory: ModuleFactory): Promise<ExecuTorchRunner> {
    const module = await moduleFactory();
    return new ExecuTorchRunner(module);
  }

  /** Fetches a `.pte` over HTTP and loads it. The returned instance stays loaded until
   * `.delete()` is called -- kept alive for repeated calls, not reloaded per run. */
  async loadModel(url: string): Promise<EtModuleInstance> {
    const buffer = new Uint8Array(await (await fetch(url)).arrayBuffer());
    return this.module.Module.load(buffer);
  }

  /** trajectory (n, stateDim, tSteps) -> features (n, nFeatures) */
  runExtractFeatures(model: EtModuleInstance, y: Float32Array, n: number, stateDim: number, tSteps: number): Float32Array {
    const input = this.module.Tensor.fromArray([n, stateDim, tSteps], y);
    try {
      const [output] = model.forward([input]);
      if (!output) throw new Error("extract_features: no output tensor");
      const result = Float32Array.from(output.data as Float32Array);
      output.delete();
      return result;
    } finally {
      input.delete();
    }
  }

  /** features (n, nFeatures) -> label/cluster index (n,). `reference` (supervised mode only)
   * is passed as a second runtime input tensor rather than baked into the `.pte` -- see
   * classify_graph.py's module docstring for why, and browserBasinStability.ts for how the
   * feature columns it contains are kept in sync with `features`'s. */
  runClassify(
    model: EtModuleInstance,
    features: Float32Array,
    n: number,
    nFeatures: number,
    reference?: { features: Float32Array; nTemplates: number },
  ): BigInt64Array {
    const input = this.module.Tensor.fromArray([n, nFeatures], features);
    const referenceInput = reference
      ? this.module.Tensor.fromArray([reference.nTemplates, nFeatures], reference.features)
      : undefined;
    try {
      const inputs = referenceInput ? [input, referenceInput] : [input];
      const [output] = model.forward(inputs);
      if (!output) throw new Error("classify: no output tensor");
      const result = BigInt64Array.from(output.data as BigInt64Array);
      output.delete();
      return result;
    } finally {
      input.delete();
      referenceInput?.delete();
    }
  }
}
