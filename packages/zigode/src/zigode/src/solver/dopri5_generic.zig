const std = @import("std");
const math = std.math;
const Allocator = std.mem.Allocator;

/// Generic ODE function type for C ABI compatibility.
/// This allows loading ODE functions from external shared libraries.
/// Uses opaque params pointer instead of compile-time generic.
pub const GenericOdeFn = *const fn (t: f64, y: [*]const f64, dydt: [*]f64, params: ?*const anyopaque, dim: usize) callconv(.c) void;

/// Dormand-Prince 5(4) adaptive step-size Runge-Kutta integrator.
/// This version uses runtime dispatch for ODEs (opaque params pointer).
pub const Dopri5Generic = struct {
    const Self = @This();

    // Butcher tableau coefficients (same as typed version)
    const a21: f64 = 1.0 / 5.0;
    const a31: f64 = 3.0 / 40.0;
    const a32: f64 = 9.0 / 40.0;
    const a41: f64 = 44.0 / 45.0;
    const a42: f64 = -56.0 / 15.0;
    const a43: f64 = 32.0 / 9.0;
    const a51: f64 = 19372.0 / 6561.0;
    const a52: f64 = -25360.0 / 2187.0;
    const a53: f64 = 64448.0 / 6561.0;
    const a54: f64 = -212.0 / 729.0;
    const a61: f64 = 9017.0 / 3168.0;
    const a62: f64 = -355.0 / 33.0;
    const a63: f64 = 46732.0 / 5247.0;
    const a64: f64 = 49.0 / 176.0;
    const a65: f64 = -5103.0 / 18656.0;
    const a71: f64 = 35.0 / 384.0;
    const a73: f64 = 500.0 / 1113.0;
    const a74: f64 = 125.0 / 192.0;
    const a75: f64 = -2187.0 / 6784.0;
    const a76: f64 = 11.0 / 84.0;

    const c2: f64 = 1.0 / 5.0;
    const c3: f64 = 3.0 / 10.0;
    const c4: f64 = 4.0 / 5.0;
    const c5: f64 = 8.0 / 9.0;

    // Shampine error coefficients (c_sol - c_hat, matching torchdiffeq)
    const e1: f64 = 35.0 / 384.0 - 1951.0 / 21600.0;
    const e3: f64 = 500.0 / 1113.0 - 22642.0 / 50085.0;
    const e4: f64 = 125.0 / 192.0 - 451.0 / 720.0;
    const e5: f64 = -2187.0 / 6784.0 - -12231.0 / 42400.0;
    const e6: f64 = 11.0 / 84.0 - 649.0 / 6300.0;
    const e7: f64 = -1.0 / 60.0;

    // DPS_C_MID coefficients for Hermite interpolation (Shampine midpoint)
    const mid1: f64 = 6025192743.0 / 30085553152.0 / 2.0;
    const mid3: f64 = 51252292925.0 / 65400821598.0 / 2.0;
    const mid4: f64 = -2691868925.0 / 45128329728.0 / 2.0;
    const mid5: f64 = 187940372067.0 / 1594534317056.0 / 2.0;
    const mid6: f64 = -1776094331.0 / 19743644256.0 / 2.0;
    const mid7: f64 = 11237099.0 / 235043384.0 / 2.0;

    rtol: f64,
    atol: f64,
    max_steps: usize,
    safety: f64 = 0.9,
    min_factor: f64 = 0.2,
    max_factor: f64 = 10.0,

    pub fn init(rtol: f64, atol: f64, max_steps: usize) Self {
        return .{
            .rtol = rtol,
            .atol = atol,
            .max_steps = max_steps,
        };
    }

    /// Solve an ODE and write results directly into a pre-allocated contiguous buffer.
    ///
    /// The output buffer layout is row-major: output[t_idx * dim + d] = y[d] at time save_at[t_idx].
    /// This avoids all intermediate allocations for the result — only working buffers (k-stages) are allocated.
    pub fn solve_into(
        self: *const Self,
        ode_fn: GenericOdeFn,
        params: ?*const anyopaque,
        y0: []const f64,
        dim: usize,
        t0: f64,
        t1: f64,
        save_at: []const f64,
        output: [*]f64,
        allocator: Allocator,
    ) !void {
        // Working buffers
        const y = try allocator.alloc(f64, dim);
        defer allocator.free(y);
        const y_new = try allocator.alloc(f64, dim);
        defer allocator.free(y_new);
        const y_err = try allocator.alloc(f64, dim);
        defer allocator.free(y_err);

        // FSAL buffers
        var k_bufs: [7][]f64 = undefined;
        for (&k_bufs) |*kb| {
            kb.* = try allocator.alloc(f64, dim);
        }
        defer for (&k_bufs) |kb| {
            allocator.free(kb);
        };
        var k1_idx: usize = 0;
        var k7_idx: usize = 6;

        const y_stage = try allocator.alloc(f64, dim);
        defer allocator.free(y_stage);

        // Hermite interpolation coefficient buffers
        const interp_a = try allocator.alloc(f64, dim);
        defer allocator.free(interp_a);
        const interp_b = try allocator.alloc(f64, dim);
        defer allocator.free(interp_b);
        const interp_c = try allocator.alloc(f64, dim);
        defer allocator.free(interp_c);

        // Initialize
        @memcpy(y, y0);
        var t = t0;
        var save_idx: usize = 0;

        // Initial k1 (needed for Hairer initial step size)
        ode_fn(t, y.ptr, k_bufs[k1_idx].ptr, params, dim);
        var dt = try self.initialStepSize(ode_fn, params, y0, dim, t0, k_bufs[k1_idx], allocator);

        // Main loop
        var step_count: usize = 0;
        while (t < t1 and step_count < self.max_steps) : (step_count += 1) {
            if (t + dt > t1) dt = t1 - t;
            if (dt <= 0) break;

            const k1 = k_bufs[k1_idx];
            const k2 = k_bufs[1];
            const k3 = k_bufs[2];
            const k4 = k_bufs[3];
            const k5 = k_bufs[4];
            const k6 = k_bufs[5];
            const k7 = k_bufs[k7_idx];

            // Stage 2
            for (0..dim) |i| {
                y_stage[i] = y[i] + dt * a21 * k1[i];
            }
            ode_fn(t + c2 * dt, y_stage.ptr, k2.ptr, params, dim);

            // Stage 3
            for (0..dim) |i| {
                y_stage[i] = y[i] + dt * (a31 * k1[i] + a32 * k2[i]);
            }
            ode_fn(t + c3 * dt, y_stage.ptr, k3.ptr, params, dim);

            // Stage 4
            for (0..dim) |i| {
                y_stage[i] = y[i] + dt * (a41 * k1[i] + a42 * k2[i] + a43 * k3[i]);
            }
            ode_fn(t + c4 * dt, y_stage.ptr, k4.ptr, params, dim);

            // Stage 5
            for (0..dim) |i| {
                y_stage[i] = y[i] + dt * (a51 * k1[i] + a52 * k2[i] + a53 * k3[i] + a54 * k4[i]);
            }
            ode_fn(t + c5 * dt, y_stage.ptr, k5.ptr, params, dim);

            // Stage 6
            for (0..dim) |i| {
                y_stage[i] = y[i] + dt * (a61 * k1[i] + a62 * k2[i] + a63 * k3[i] + a64 * k4[i] + a65 * k5[i]);
            }
            ode_fn(t + dt, y_stage.ptr, k6.ptr, params, dim);

            // 5th-order solution
            for (0..dim) |i| {
                y_new[i] = y[i] + dt * (a71 * k1[i] + a73 * k3[i] + a74 * k4[i] + a75 * k5[i] + a76 * k6[i]);
            }

            // Stage 7
            ode_fn(t + dt, y_new.ptr, k7.ptr, params, dim);

            // Error estimate
            for (0..dim) |i| {
                y_err[i] = dt * (e1 * k1[i] + e3 * k3[i] + e4 * k4[i] + e5 * k5[i] + e6 * k6[i] + e7 * k7[i]);
            }

            // Error norm
            var err_norm: f64 = 0.0;
            for (0..dim) |i| {
                const sc = self.atol + self.rtol * @max(@abs(y[i]), @abs(y_new[i]));
                const ratio = y_err[i] / sc;
                err_norm += ratio * ratio;
            }
            err_norm = @sqrt(err_norm / @as(f64, @floatFromInt(dim)));

            if (err_norm <= 1.0) {
                const t_new = t + dt;

                // Precompute Hermite interpolation coefficients (torchdiffeq-style)
                if (save_idx < save_at.len and save_at[save_idx] <= t_new + 1e-12) {
                    for (0..dim) |i| {
                        const ymid = y[i] + dt * (mid1 * k1[i] + mid3 * k3[i] + mid4 * k4[i] + mid5 * k5[i] + mid6 * k6[i] + mid7 * k7[i]);
                        interp_a[i] = 2.0 * dt * (k7[i] - k1[i]) - 8.0 * (y_new[i] + y[i]) + 16.0 * ymid;
                        interp_b[i] = dt * (5.0 * k1[i] - 3.0 * k7[i]) + 18.0 * y[i] + 14.0 * y_new[i] - 32.0 * ymid;
                        interp_c[i] = dt * (k7[i] - 4.0 * k1[i]) - 11.0 * y[i] - 5.0 * y_new[i] + 16.0 * ymid;
                    }

                    while (save_idx < save_at.len and save_at[save_idx] <= t_new + 1e-12) {
                        const out_row = output[save_idx * dim .. (save_idx + 1) * dim];
                        if (save_at[save_idx] <= t + 1e-12) {
                            @memcpy(out_row, y);
                        } else {
                            const x = (save_at[save_idx] - t) / dt;
                            for (0..dim) |i| {
                                out_row[i] = y[i] + x * (dt * k1[i] + x * (interp_c[i] + x * (interp_b[i] + x * interp_a[i])));
                            }
                        }
                        save_idx += 1;
                    }
                }

                @memcpy(y, y_new);
                t = t_new;

                const tmp = k1_idx;
                k1_idx = k7_idx;
                k7_idx = tmp;
            }

            // Step size adaptation (torchdiffeq-style: don't shrink on accepted steps)
            const accepted = err_norm <= 1.0;
            const dfactor = if (accepted) 1.0 else self.min_factor;
            const factor = if (err_norm == 0.0)
                self.max_factor
            else
                @min(self.max_factor, @max(dfactor, self.safety * math.pow(f64, err_norm, -0.2)));
            dt *= factor;
        }

        while (save_idx < save_at.len) : (save_idx += 1) {
            const out_row = output[save_idx * dim .. (save_idx + 1) * dim];
            @memcpy(out_row, y);
        }
    }

    /// Solve an ODE and return results as allocated slices (convenience wrapper).
    /// Prefer solve_into() for batch solving to avoid intermediate allocations.
    pub fn solve(
        self: *const Self,
        ode_fn: GenericOdeFn,
        params: ?*const anyopaque,
        y0: []const f64,
        dim: usize,
        t0: f64,
        t1: f64,
        save_at: []const f64,
        allocator: Allocator,
    ) ![][]f64 {
        // Allocate contiguous buffer
        const flat = try allocator.alloc(f64, save_at.len * dim);

        try self.solve_into(ode_fn, params, y0, dim, t0, t1, save_at, flat.ptr, allocator);

        // Wrap flat buffer as [][]f64 slices (no copy, same memory)
        const result = try allocator.alloc([]f64, save_at.len);
        for (result, 0..) |*row, i| {
            row.* = flat[i * dim .. (i + 1) * dim];
        }
        return result;
    }

    fn initialStepSize(
        self: *const Self,
        ode_fn: GenericOdeFn,
        params: ?*const anyopaque,
        y0: []const f64,
        dim: usize,
        t0: f64,
        f0: []const f64,
        allocator: Allocator,
    ) !f64 {
        // Hairer, Nørsett & Wanner algorithm (Solving ODEs I, Sec. II.4)
        const fdim: f64 = @floatFromInt(dim);
        const scale = try allocator.alloc(f64, dim);
        defer allocator.free(scale);
        const y_tmp = try allocator.alloc(f64, dim);
        defer allocator.free(y_tmp);
        const f_tmp = try allocator.alloc(f64, dim);
        defer allocator.free(f_tmp);

        var d0: f64 = 0.0;
        var d1: f64 = 0.0;
        for (0..dim) |i| {
            scale[i] = self.atol + @abs(y0[i]) * self.rtol;
            const r0 = y0[i] / scale[i];
            d0 += r0 * r0;
            const r1 = f0[i] / scale[i];
            d1 += r1 * r1;
        }
        d0 = @sqrt(d0 / fdim);
        d1 = @sqrt(d1 / fdim);

        const h0: f64 = if (d0 < 1e-5 or d1 < 1e-5) 1e-6 else 0.01 * d0 / d1;

        for (0..dim) |i| {
            y_tmp[i] = y0[i] + h0 * f0[i];
        }
        ode_fn(t0 + h0, y_tmp.ptr, f_tmp.ptr, params, dim);

        var d2: f64 = 0.0;
        for (0..dim) |i| {
            const r = (f_tmp[i] - f0[i]) / scale[i];
            d2 += r * r;
        }
        d2 = @sqrt(d2 / fdim) / h0;

        const h1: f64 = if (d1 <= 1e-15 and d2 <= 1e-15)
            @max(1e-6, h0 * 1e-3)
        else
            math.pow(f64, 0.01 / @max(d1, d2), 0.2);

        return @min(100.0 * h0, h1);
    }

    pub fn freeResult(result: [][]f64, allocator: Allocator) void {
        for (result) |row| {
            allocator.free(row);
        }
        allocator.free(result);
    }
};
