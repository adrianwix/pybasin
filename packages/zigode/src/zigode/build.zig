const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // ============================================================
    // Shared Library (for Python integration)
    // ============================================================
    const lib = b.addLibrary(.{
        .linkage = .dynamic,
        .name = "zig_ode_solver",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/c_api.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });

    lib.root_module.link_libc = true;

    b.installArtifact(lib);

    // ============================================================
    // WASM (for the pybasin browser demo)
    // ============================================================
    // Always wasm32-freestanding regardless of -Dtarget: this artifact runs in a browser,
    // not on whatever platform `zig build` itself is invoked from. One self-contained module
    // per case study ODE -- see wasm/pendulum_wasm.zig's module docstring for why.
    const wasm_target = b.resolveTargetQuery(.{
        .cpu_arch = .wasm32,
        .os_tag = .freestanding,
    });

    const wasm_dopri5_mod = b.createModule(.{
        .root_source_file = b.path("src/solver/dopri5_generic.zig"),
        .target = wasm_target,
        .optimize = .ReleaseFast,
    });
    const wasm_pendulum_ode_mod = b.createModule(.{
        .root_source_file = b.path("user_odes/pendulum.zig"),
        .target = wasm_target,
        .optimize = .ReleaseFast,
    });
    const wasm_pendulum_mod = b.createModule(.{
        .root_source_file = b.path("wasm/pendulum_wasm.zig"),
        .target = wasm_target,
        .optimize = .ReleaseFast,
    });
    wasm_pendulum_mod.addImport("dopri5_generic", wasm_dopri5_mod);
    wasm_pendulum_mod.addImport("pendulum", wasm_pendulum_ode_mod);

    const wasm_pendulum = b.addExecutable(.{
        .name = "pendulum",
        .root_module = wasm_pendulum_mod,
    });
    wasm_pendulum.entry = .disabled;
    wasm_pendulum.rdynamic = true;

    const wasm_step = b.step("wasm", "Build case-study WASM modules for the browser demo");
    wasm_step.dependOn(&b.addInstallArtifact(wasm_pendulum, .{
        .dest_dir = .{ .override = .{ .custom = "wasm" } },
    }).step);
}
