/**
 * Thin wrapper around the zigode-compiled WASM module (packages/zigode/src/zigode/wasm/).
 * A raw WASM module with no Emscripten glue -- Zig's `export fn`s become WASM exports
 * directly, so this is just `WebAssembly.instantiate` plus manual linear-memory access.
 */

export interface SolvePendulumOptions {
  y0s: Float64Array; // [nIcs * 2], row-major (theta, theta_dot)
  t0: number;
  t1: number;
  saveAt: Float64Array; // [nSave], strictly increasing
  rtol: number;
  atol: number;
  maxSteps: number;
  alpha: number;
  torque: number;
  stiffness: number;
}

type ZigExports = {
  memory: WebAssembly.Memory;
  wasm_alloc(n: number): number;
  wasm_free(ptr: number, n: number): void;
  solve_batch_pendulum(
    y0sPtr: number,
    nIcs: number,
    t0: number,
    t1: number,
    saveAtPtr: number,
    nSave: number,
    rtol: number,
    atol: number,
    maxSteps: number,
    alpha: number,
    torque: number,
    stiffness: number,
    resultsPtr: number,
  ): number;
};

export class ZigOdeModule {
  private constructor(private readonly exports: ZigExports) {}

  static async load(wasmUrl: string): Promise<ZigOdeModule> {
    const bytes = await (await fetch(wasmUrl)).arrayBuffer();
    const { instance } = await WebAssembly.instantiate(bytes, { env: {} });
    return new ZigOdeModule(instance.exports as unknown as ZigExports);
  }

  /**
   * Solves the pendulum ODE for a batch of initial conditions.
   *
   * @returns `[nIcs * nSave * 2]` row-major trajectory tensor.
   */
  solveBatchPendulum(opts: SolvePendulumOptions): Float64Array {
    const { exports } = this;
    const nIcs = opts.y0s.length / 2;
    const nSave = opts.saveAt.length;

    const y0Ptr = exports.wasm_alloc(opts.y0s.length * 8);
    const saveAtPtr = exports.wasm_alloc(opts.saveAt.length * 8);
    const resultsPtr = exports.wasm_alloc(nIcs * nSave * 2 * 8);
    if (!y0Ptr || !saveAtPtr || !resultsPtr) {
      throw new Error("zigode: wasm_alloc failed");
    }

    try {
      // Re-read `memory.buffer` at every access: std.heap.wasm_allocator can grow linear
      // memory (via a new allocation), which detaches any previously-captured ArrayBuffer.
      new Float64Array(exports.memory.buffer, y0Ptr, opts.y0s.length).set(opts.y0s);
      new Float64Array(exports.memory.buffer, saveAtPtr, opts.saveAt.length).set(opts.saveAt);

      const rc = exports.solve_batch_pendulum(
        y0Ptr,
        nIcs,
        opts.t0,
        opts.t1,
        saveAtPtr,
        nSave,
        opts.rtol,
        opts.atol,
        opts.maxSteps,
        opts.alpha,
        opts.torque,
        opts.stiffness,
        resultsPtr,
      );
      if (rc !== 0) {
        throw new Error(`zigode: solve_batch_pendulum failed with code ${rc}`);
      }

      return new Float64Array(exports.memory.buffer, resultsPtr, nIcs * nSave * 2).slice();
    } finally {
      exports.wasm_free(y0Ptr, opts.y0s.length * 8);
      exports.wasm_free(saveAtPtr, opts.saveAt.length * 8);
      exports.wasm_free(resultsPtr, nIcs * nSave * 2 * 8);
    }
  }
}
