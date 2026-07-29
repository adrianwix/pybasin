//! Pendulum ODE - Example of a user-defined ODE for the Zig solver.
//!
//! Users only need to define:
//!   - DIM: State space dimension
//!   - Params: Parameter struct (extern struct for C ABI compatibility)
//!   - ode: The ODE function with a clean Zig signature
//!
//! The Python wrapper auto-generates the C ABI exports.

const std = @import("std");

// ============================================================
// Required: Dimension
// ============================================================

pub const DIM = 2;

// ============================================================
// Required: Parameters
// ============================================================

/// Parameters for the pendulum ODE.
/// Must be `extern struct` for C ABI compatibility.
pub const Params = extern struct {
    alpha: f64, // Damping coefficient
    T: f64, // External torque
    K: f64, // Stiffness coefficient
};

// ============================================================
// Required: ODE Function
// ============================================================

/// Pendulum ODE: dθ/dt = θ̇, dθ̇/dt = -α·θ̇ + T - K·sin(θ)
///
/// Clean Zig signature - no raw pointers, no C ABI concerns.
/// The wrapper handles conversion to/from C types.
pub fn ode(t: f64, y: []const f64, dydt: []f64, params: *const Params) void {
    _ = t; // Autonomous system

    const theta = y[0];
    const theta_dot = y[1];

    dydt[0] = theta_dot;
    dydt[1] = -params.alpha * theta_dot + params.T - params.K * @sin(theta);
}
