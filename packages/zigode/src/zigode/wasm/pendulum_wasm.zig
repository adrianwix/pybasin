//! WASM entry point for the pendulum ODE, for the pybasin browser demo.
//!
//! Unlike the native path (build.zig's shared library, dynamically loaded by Python via
//! ctypes with a separately-compiled ODE .so passed in as a function pointer), a browser
//! demo needs one self-contained WASM module per case study: WASM has no equivalent of
//! dlopen-ing a second module and passing function pointers across the boundary, so the
//! solver core and the ODE right-hand side are compiled together here. Single-threaded (no
//! pthreads -- WASM threads need SharedArrayBuffer/Atomics, an extra deployment requirement
//! this demo doesn't need): zigode's own benchmarks show ~50-100us/IC, so even 20,000
//! trajectories run in roughly 1-2s single-threaded.
const std = @import("std");
const dopri5_generic = @import("dopri5_generic");
const pendulum = @import("pendulum");

const Dopri5 = dopri5_generic.Dopri5Generic;

fn ode_wrapper(
    t: f64,
    y: [*]const f64,
    dydt: [*]f64,
    params_ptr: ?*const anyopaque,
    dim: usize,
) callconv(.c) void {
    _ = dim;
    const params: *const pendulum.Params = @ptrCast(@alignCast(params_ptr.?));
    pendulum.ode(t, y[0..pendulum.DIM], dydt[0..pendulum.DIM], params);
}

/// Allocates `n_bytes` in the module's linear memory; JS writes/reads through this pointer
/// via a typed array view over `memory.buffer`. Returns null on allocation failure.
export fn wasm_alloc(n_bytes: usize) ?[*]u8 {
    const mem = std.heap.wasm_allocator.alloc(u8, n_bytes) catch return null;
    return mem.ptr;
}

export fn wasm_free(ptr: [*]u8, n_bytes: usize) void {
    std.heap.wasm_allocator.free(ptr[0..n_bytes]);
}

/// Solves the pendulum ODE for `n_ics` initial conditions.
///
/// :param y0s: `[n_ics * 2]` row-major initial conditions (theta, theta_dot).
/// :param save_at: `[n_save]` time points to save, strictly increasing.
/// :param results: `[n_ics * n_save * 2]` row-major output, preallocated by the caller
///     (via `wasm_alloc`).
/// :return: 0 on success, negative on error.
export fn solve_batch_pendulum(
    y0s: [*]const f64,
    n_ics: usize,
    t0: f64,
    t1: f64,
    save_at: [*]const f64,
    n_save: usize,
    rtol: f64,
    atol: f64,
    max_steps: usize,
    alpha: f64,
    torque: f64,
    stiffness: f64,
    results: [*]f64,
) i32 {
    const params = pendulum.Params{ .alpha = alpha, .T = torque, .K = stiffness };
    const solver = Dopri5.init(rtol, atol, max_steps);
    const allocator = std.heap.wasm_allocator;
    const save_at_slice = save_at[0..n_save];
    const traj_size = n_save * pendulum.DIM;

    var i: usize = 0;
    while (i < n_ics) : (i += 1) {
        const y0_slice = y0s[i * pendulum.DIM .. (i + 1) * pendulum.DIM];
        const out_ptr = results + i * traj_size;
        solver.solve_into(
            ode_wrapper,
            &params,
            y0_slice,
            pendulum.DIM,
            t0,
            t1,
            save_at_slice,
            out_ptr,
            allocator,
        ) catch return -1;
    }
    return 0;
}
