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

    lib.linkLibC();

    b.installArtifact(lib);
}
