# Verifying on Apple Silicon

Everything in this repository is written on a Linux machine, and the crate
cannot be built there: `mlx-sys` links Foundation, objc, Metal and Accelerate
unconditionally, so `cargo build`, `cargo check` and `cargo test` all fail
before reaching any of this crate's own code. There is no local compile signal
at all — not even a type check.

So the four changes below have been reasoned about carefully against the
upstream Python and against the unpacked `mlx-rs` source, but **none of them
has ever been compiled or run**. This file is the checklist that closes that
gap. Work through it in order; step 1 gates everything else.

## 1. Does it compile, and do the existing tests still pass?

```sh
cargo build --release
cargo test
```

Four things changed shape and are the likely sites of a compile error:

- `load_model` gained a `dtype` argument.
- `Whisper::new`, `AudioEncoder::new` and `TextDecoder::new` gained a `dtype`
  argument.
- `AudioEncoder::forward` gained an `audio_ctx: Option<usize>` argument.
- `MultiHeadAttention::{query,key,value,out}`,
  `ResidualAttentionBlock::{mlp1,mlp2}` and `TextDecoder::token_embedding`
  changed type from `Linear`/`Embedding` to `MaybeQuantized<…>`.

The test suites pass `Dtype::Float32` explicitly, so their existing numeric
tolerances are unaffected by the dtype work.

## 2. Is the fp16 knob actually delivering?

This is the one to run first after the build, because it is the only claim
that was established by *reading* mlx-rs source rather than by measuring
anything.

```sh
cargo run --release --example transcribe -- sample.wav                 # f16, the new default
cargo run --release --example transcribe -- sample.wav --fp32          # the old behaviour
```

Compare wall-clock and peak memory. The default should be meaningfully faster
and use roughly half the memory.

**If the two are near-identical, the change did not work.** Two dtype
promotion sites were found and fixed — the `q * scale` multiply in
`qkv_attention`, and `mlx_rs::nn::gelu`, which returns f32 regardless of input
dtype because it divides by literal `Dtype::Float32` scalars. A third site
would produce exactly this symptom. Dump the encoder activations layer by
layer and find where `dtype` stops being `Float16`.

Also confirm the transcript is still correct: fp16 changes the numerics, and
`compression_ratio` and `no_speech_prob` feed hard thresholds that decide
temperature fallback.

## 3. Does `audio_ctx` cut latency on short audio?

```sh
cargo run --release --example transcribe -- short-clip.wav                    # full 1500-position context
cargo run --release --example transcribe -- short-clip.wav --audio-ctx 250    # ~5 seconds' worth
```

`n` encoder positions correspond to `2*n` mel frames, so 250 covers about five
seconds of a 30-second window. Encoder time should drop substantially.

Check accuracy as well as speed. The model was trained to attend over the full
window, so a truncated one is off-distribution and the transcript is expected
to degrade as `n` shrinks — the question is where the useful trade-off sits
for your workload, which is exactly what makes this knob worth having.

## 4. Do quantized checkpoints load and decode?

```sh
cargo run --release --example transcribe -- sample.wav --model mlx-community/whisper-tiny-mlx-4bit
```

The mlx-community naming convention is a `-4bit` / `-8bit` suffix; substitute
whichever quantized repo you can confirm exists, since these come and go.

Previously this failed with a clear "not supported yet" error. It should now
load and produce a sensible transcript.

Two specific things to watch:

- Garbled output rather than an error means the packed weights and their
  scales are being paired up wrongly. Layers are quantized per-module, only
  when that module's `.scales` key is present in the checkpoint, mirroring
  upstream's `class_predicate`.
- `quantized_matmul` documents that it only supports 2-D inputs whose
  dimensions are multiples of 32. Whisper's published dimensions satisfy this,
  but that is not validated at load time, so a custom checkpoint would fail
  inside mlx rather than with a clear message here.

Compare resident memory against the f16 model to confirm the win is real.

Combining `--fp32` with a quantized checkpoint may fail inside
`quantized_matmul`, since the activations would be f32 while the checkpoint's
scales are fp16. Upstream has the same exposure, so that would be parity
rather than a bug in this port — don't chase it unless the default f16 path
fails too.

## 5. Is the release binary self-contained?

```sh
./scripts/check-self-contained.sh --audio sample.wav
```

This is the check that cannot be answered by reading source at all. `mlx-sys`
links `mlx` and `mlxc` statically and otherwise only against system
frameworks, which is encouraging — but the Metal shader packaging is decided
inside MLX's own C++ tree, which `mlx-sys` does not vendor. It fetches it from
GitHub at tag `v0.25.1` during the macOS configure step, so whether the
shaders end up embedded in the binary or as a loose `.metallib` is only
observable on a machine that has actually run that build.

The script builds the example, classifies every `otool -L` entry as system or
non-system, prints `LC_RPATH` entries, looks for a `.metallib`, and then runs
the binary alone from an empty directory with the `DYLD_*` variables scrubbed
and Homebrew off `PATH`. It needs `--audio` (the repo ships only numeric
fixtures, no playable audio) and defaults to `mlx-community/whisper-tiny-mlx`
with assets from `./assets`.

Note that it moves any `.metallib` under `target/` aside for the isolated run
and restores it on exit.

## Reporting back

Paste the output of whichever step fails first. Steps 2, 3 and 4 each have a
plausible failure mode that looks like success — near-identical timings, an
unchanged transcript, or plausible-looking garbage — so it is worth recording
the actual numbers even when nothing errors.
