const std = @import("std");
const c = @cImport({
    @cInclude("pthread.h");
    @cInclude("unistd.h");
});
const dopri5_generic = @import("solver/dopri5_generic.zig");

const Dopri5 = dopri5_generic.Dopri5Generic;

// ============================================================
// C ABI Exports
// ============================================================
// This is the SOLVER library - it takes a function pointer to an ODE.
// ODEs are compiled separately and loaded dynamically by Python.

/// Error codes returned to Python
pub const ErrorCode = enum(i32) {
    success = 0,
    invalid_ode_fn = -1,
    allocation_failed = -2,
    solve_failed = -3,
    thread_spawn_failed = -4,
};

/// ODE function signature (C ABI compatible).
/// This is what user-defined ODE libraries must export.
///
/// Arguments:
///   t: Current time
///   y: Current state vector (length = dim)
///   dydt: Output buffer for derivatives (length = dim)
///   params: Opaque pointer to parameter struct
///   dim: State dimension
pub const OdeFnC = *const fn (
    t: f64,
    y: [*]const f64,
    dydt: [*]f64,
    params: ?*const anyopaque,
    dim: usize,
) callconv(.c) void;

/// Solve an ODE system using Dopri5.
///
/// This is the main entry point. Python loads an ODE library separately,
/// gets the function pointer, and passes it here.
///
/// Arguments:
///   ode_fn: Function pointer to the ODE right-hand side (from user's ODE library)
///   y0: Initial conditions array of length `dim`
///   dim: Dimension of the state space
///   t0, t1: Integration time span
///   save_at: Array of time points to save solution at
///   n_save: Length of save_at array
///   rtol, atol: Tolerances
///   max_steps: Maximum integration steps
///   params: Pointer to ODE-specific parameters struct
///   result: Pre-allocated output buffer of shape [n_save * dim], row-major
///
/// Returns: ErrorCode (0 = success, negative = error)
export fn solve_ode(
    ode_fn: ?OdeFnC,
    y0: [*]const f64,
    dim: usize,
    t0: f64,
    t1: f64,
    save_at: [*]const f64,
    n_save: usize,
    rtol: f64,
    atol: f64,
    max_steps: usize,
    params: ?*const anyopaque,
    result: [*]f64,
) callconv(.c) i32 {
    // Validate function pointer
    const fn_ptr = ode_fn orelse {
        return @intFromEnum(ErrorCode.invalid_ode_fn);
    };

    // Wrap C function to match Zig's internal signature
    const zig_ode_fn = struct {
        fn call(t: f64, y: [*]const f64, dydt: [*]f64, p: ?*const anyopaque, d: usize) void {
            // Call the C function pointer stored in thread-local or closure
            // We use a simple wrapper approach here
            @call(.auto, fn_ptr, .{ t, y, dydt, p, d });
        }
    }.call;
    _ = zig_ode_fn;

    // Set up allocator
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    // Create solver
    const solver = Dopri5.init(rtol, atol, max_steps);

    // Wrap arrays as slices
    const y0_slice = y0[0..dim];
    const save_at_slice = save_at[0..n_save];

    // Solve directly into the output buffer (zero-copy)
    solver.solve_into(
        fn_ptr,
        params,
        y0_slice,
        dim,
        t0,
        t1,
        save_at_slice,
        result,
        allocator,
    ) catch {
        return @intFromEnum(ErrorCode.solve_failed);
    };

    return @intFromEnum(ErrorCode.success);
}

/// Per-worker data - contains everything needed to solve a chunk of ICs
const WorkerData = struct {
    ode_fn: OdeFnC,
    y0s: [*]const f64,
    results: [*]f64,
    save_at: [*]const f64,
    dim: usize,
    n_save: usize,
    t0: f64,
    t1: f64,
    rtol: f64,
    atol: f64,
    max_steps: usize,
    params: ?*const anyopaque,
    start_ic: usize,
    end_ic: usize, // exclusive
};

/// Worker function for pthreads (C ABI)
fn workerFnC(arg: ?*anyopaque) callconv(.c) ?*anyopaque {
    const data: *const WorkerData = @ptrCast(@alignCast(arg.?));
    const traj_size = data.n_save * data.dim;

    // Each thread gets its own arena allocator
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const solver_instance = Dopri5.init(data.rtol, data.atol, data.max_steps);
    const save_at_slice = data.save_at[0..data.n_save];

    var i = data.start_ic;
    while (i < data.end_ic) : (i += 1) {
        const y0_slice = data.y0s[i * data.dim .. (i + 1) * data.dim];
        const output_ptr = data.results + i * traj_size;

        solver_instance.solve_into(
            data.ode_fn,
            data.params,
            y0_slice,
            data.dim,
            data.t0,
            data.t1,
            save_at_slice,
            output_ptr,
            allocator,
        ) catch {
            // On error, leave zeros in output
            continue;
        };

        // Reset arena to avoid memory growth
        _ = arena.reset(.retain_capacity);
    }
    return null;
}

/// Solve ODE for multiple initial conditions in parallel.
///
/// Arguments:
///   ode_fn: Function pointer to the ODE right-hand side
///   y0s: Initial conditions array of shape [n_ics * dim], row-major
///   n_ics: Number of initial conditions
///   dim: Dimension of the state space
///   t0, t1: Integration time span
///   save_at: Array of time points to save solution at
///   n_save: Length of save_at array
///   rtol, atol: Tolerances
///   max_steps: Maximum integration steps
///   params: Pointer to ODE-specific parameters struct
///   results: Pre-allocated output buffer of shape [n_ics * n_save * dim], row-major
///   n_threads: Number of threads (0 = auto-detect CPU count)
///
/// Returns: ErrorCode (0 = success, negative = error)
pub export fn solve_batch(
    ode_fn: ?OdeFnC,
    y0s: [*]const f64,
    n_ics: usize,
    dim: usize,
    t0: f64,
    t1: f64,
    save_at: [*]const f64,
    n_save: usize,
    rtol: f64,
    atol: f64,
    max_steps: usize,
    params: ?*const anyopaque,
    results: [*]f64,
    n_threads: usize,
) callconv(.c) i32 {
    // Validate function pointer
    const fn_ptr = ode_fn orelse {
        return @intFromEnum(ErrorCode.invalid_ode_fn);
    };

    const traj_size = n_save * dim;
    _ = traj_size;

    // Use specified thread count (0 = auto)
    const cpu_count: usize = blk: {
        const result = c.sysconf(c._SC_NPROCESSORS_ONLN);
        break :blk if (result > 0) @intCast(result) else 4;
    };
    const actual_threads: usize = if (n_threads == 0) cpu_count else n_threads;
    const num_workers = @min(actual_threads, n_ics);

    // For single thread or single IC, run sequentially
    if (num_workers <= 1) {
        var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
        defer arena.deinit();
        const allocator = arena.allocator();

        const solver_instance = Dopri5.init(rtol, atol, max_steps);
        const save_at_slice = save_at[0..n_save];
        const out_traj_size = n_save * dim;

        for (0..n_ics) |i| {
            const y0_slice = y0s[i * dim .. (i + 1) * dim];
            const output_ptr = results + i * out_traj_size;

            solver_instance.solve_into(
                fn_ptr,
                params,
                y0_slice,
                dim,
                t0,
                t1,
                save_at_slice,
                output_ptr,
                allocator,
            ) catch {
                return @intFromEnum(ErrorCode.solve_failed);
            };

            // Reset arena to avoid memory growth
            _ = arena.reset(.retain_capacity);
        }
        return @intFromEnum(ErrorCode.success);
    }

    // Parallel: chunk ICs across workers
    // Calculate chunk sizes
    const chunk_size = (n_ics + num_workers - 1) / num_workers;

    // Allocate threads and worker data arrays dynamically
    var thread_arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer thread_arena.deinit();
    const thread_alloc = thread_arena.allocator();

    const threads = thread_alloc.alloc(c.pthread_t, num_workers) catch {
        return @intFromEnum(ErrorCode.allocation_failed);
    };

    const worker_data = thread_alloc.alloc(WorkerData, num_workers) catch {
        return @intFromEnum(ErrorCode.allocation_failed);
    };

    var spawned: usize = 0;
    for (0..num_workers) |i| {
        const start = i * chunk_size;
        const end = @min(start + chunk_size, n_ics);
        if (start >= end) break;

        worker_data[i] = WorkerData{
            .ode_fn = fn_ptr,
            .y0s = y0s,
            .results = results,
            .save_at = save_at,
            .dim = dim,
            .n_save = n_save,
            .t0 = t0,
            .t1 = t1,
            .rtol = rtol,
            .atol = atol,
            .max_steps = max_steps,
            .params = params,
            .start_ic = start,
            .end_ic = end,
        };

        const rc = c.pthread_create(&threads[i], null, workerFnC, @ptrCast(&worker_data[i]));
        if (rc != 0) {
            for (0..spawned) |j| {
                _ = c.pthread_join(threads[j], null);
            }
            return @intFromEnum(ErrorCode.thread_spawn_failed);
        }
        spawned += 1;
    }

    for (0..spawned) |i| {
        _ = c.pthread_join(threads[i], null);
    }

    return @intFromEnum(ErrorCode.success);
}
