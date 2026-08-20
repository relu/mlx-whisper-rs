// mlx-whisper-rs has no native build steps of its own.
//
// All MLX linking is owned by `mlx-sys`, which this crate reaches through
// `mlx-rs`. `mlx-sys`'s own build script cmake-builds the vendored `mlx-c`
// sources with `MLX_C_USE_SYSTEM_MLX=OFF`, so MLX itself is fetched from git
// (pinned to v0.25.1) and compiled from source, then linked statically as
// `mlx` + `mlxc` alongside the Foundation/Metal/Accelerate frameworks.
//
// There is deliberately no attempt here to link a system MLX (e.g. from
// `brew install mlx`) instead:
//
//   * `mlx-sys` 0.2.0 exposes no environment variable or cargo feature to
//     skip its cmake build, so a system MLX would be linked *in addition to*
//     the static one rather than in place of it.
//   * That puts two copies of MLX in one binary, with duplicate global state
//     (Metal device singleton, default stream, allocator).
//   * Homebrew ships a much newer MLX than the v0.25.1 that `mlx-c` 0.2.0 is
//     written against, so the two are not ABI-compatible even in isolation.
//
// Using a system MLX would require patching `mlx-sys` to pass
// `MLX_C_USE_SYSTEM_MLX=ON` through to cmake and to link dynamically. That
// belongs upstream in oxideai/mlx-rs, not in a downstream build script.

fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    if !cfg!(target_os = "macos") {
        println!(
            "cargo:warning=mlx-whisper-rs requires macOS on Apple Silicon; \
             mlx-sys links Foundation/objc/Metal/Accelerate unconditionally."
        );
    }
}
