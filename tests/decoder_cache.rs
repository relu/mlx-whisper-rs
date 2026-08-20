//! Equivalence between the decoder's cached and uncached paths.
//!
//! Decoding runs one token at a time, feeding each block's previous keys and
//! values back in through `kv_cache`. `TextDecoder::forward` derives the
//! current position from the *shape* of that cache and uses it twice: to slice
//! the positional embedding, and to slice the causal mask. Both are silent
//! failures — a wrong offset produces perfectly well-formed logits that are
//! simply worse, so transcription quality degrades and nothing ever raises.
//!
//! The invariant that catches it needs no reference implementation and no
//! downloaded weights: feeding a whole token sequence at once must produce the
//! same logits, position by position, as feeding it one token at a time with
//! the cache. That holds for *any* weights, so random initialisation is fine.
//!
//! Two things make the test non-vacuous:
//!
//! * `TextDecoder::new` initialises `positional_embedding` to **zeros**, and
//!   `zeros[a..b] == zeros[c..d]` for every `a`, `b`, `c`, `d` — a wrong offset
//!   would be invisible. Each test overwrites it with `sinusoids`, which is
//!   what `load_models` puts there for a real model anyway.
//! * The token sequence repeats a token at several positions, and
//!   `logits_depend_on_position` asserts those positions produce *different*
//!   logits. Without that, a degenerate model would satisfy the equivalence
//!   trivially.

mod common;

use std::sync::{Mutex, MutexGuard};

use mlx_rs::{Array, Dtype, ops::indexing::IndexOp, transforms::eval};
use mlx_whisper_rs::whisper::{BlockCache, ModelDimensions, Whisper, sinusoids};

/// A model small enough to build in milliseconds. None of the dimensions are
/// Whisper's real ones — the cache invariant does not depend on them — but the
/// shapes relate the same way: `n_text_state` divides evenly by `n_text_head`,
/// and `n_text_ctx` is longer than the token sequence under test.
fn tiny_dims() -> ModelDimensions {
    ModelDimensions {
        n_mels: 80,
        n_audio_ctx: 8,
        n_audio_state: 32,
        n_audio_head: 4,
        n_audio_layer: 2,
        n_vocab: 64,
        n_text_ctx: 16,
        n_text_state: 32,
        n_text_head: 4,
        n_text_layer: 2,
    }
}

/// Token ids to decode. `3` appears at positions 0, 2 and 4 so that
/// `logits_depend_on_position` has something to compare.
const IDS: [i32; 6] = [3, 17, 3, 5, 3, 0];

/// Deterministic pseudo-random floats in `[-1, 1)`.
///
/// The same 64-bit LCG `tests/common/mod.rs` uses for synthetic logits. Model
/// weights still come from MLX's own initialisers, which are not seeded here —
/// the invariant holds for any weights — but the encoder output is pinned so a
/// failure reproduces with the same numbers every run.
fn lcg(n: usize, seed: u64) -> Vec<f32> {
    let mut out = Vec::with_capacity(n);
    let mut state: u64 = seed;
    for _ in 0..n {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let u = (state >> 40) as f32 / (1u32 << 24) as f32;
        out.push(u * 2.0 - 1.0);
    }
    out
}

/// Serialises the tests in this file. **Required, not defensive.**
///
/// libtest runs a file's tests on separate threads, and these three are the
/// only ones in the suite that build a model and run a forward pass. Doing
/// that from three threads at once segfaults inside MLX — reproducibly, on
/// CI, before any test could report, which is what a SIGSEGV in a test binary
/// looks like: no assertion, no backtrace, all three tests simply gone. The
/// arrays are never shared between the threads; concurrent graph construction
/// on MLX's global stream is enough on its own.
///
/// A `Mutex` rather than `--test-threads=1` so the constraint travels with the
/// code instead of living in a CI invocation someone can copy without.
///
/// Anything added here that touches MLX must take this lock too. If a future
/// crash ever looks like this one again, print a marker before each MLX call
/// and read the last line under `--nocapture`; that is how this one was found.
static SERIAL: Mutex<()> = Mutex::new(());

/// Lock `SERIAL`, ignoring poisoning — a panic in one test should fail that
/// test, not cascade into "the mutex is poisoned" for the other two.
fn serial() -> MutexGuard<'static, ()> {
    SERIAL.lock().unwrap_or_else(|e| e.into_inner())
}

/// A model whose positional embedding is actually position-dependent, plus a
/// stand-in for the encoder output to cross-attend to.
fn model_and_audio_features() -> (Whisper, Array) {
    common::init_device();
    let dims = tiny_dims();
    // f32, not the crate's fp16 default: this suite checks kv-cache offset
    // arithmetic against random weights, unrelated to PARITY.md MODEL-3's
    // dtype knob, and f32 keeps every intermediate at full precision.
    let mut model = Whisper::new(dims.clone(), Dtype::Float32).expect("build the model");

    // Overwrite the zeros the constructor leaves behind — see the module docs.
    let pe = sinusoids(dims.n_text_ctx, dims.n_text_state).expect("sinusoids");
    eval([&pe]).expect("evaluate the positional embedding");
    model.decoder.positional_embedding = pe;

    let n = dims.n_audio_ctx * dims.n_audio_state;
    let xa = Array::from_slice(
        &lcg(n, 0x5EED),
        &[1, dims.n_audio_ctx as i32, dims.n_audio_state as i32],
    );
    eval([&xa]).expect("evaluate the audio features");

    (model, xa)
}

/// One `[n_vocab]` row of a `[1, seq_len, n_vocab]` logits array.
///
/// `reshape` forces a contiguous copy and `eval` materialises it before
/// `as_slice` reads the buffer, matching how `src/transcribe.rs` evaluates
/// before `as_slice`. Elsewhere in this suite a bare `as_slice` is fine,
/// because those arrays are whole contiguous results — a `sinusoids` table, a
/// log-mel spectrogram — rather than one row sliced out of a larger array.
fn logits_row(logits: &Array, pos: i32) -> Vec<f32> {
    let row = logits
        .index((0i32,))
        .index((pos,))
        .reshape(&[-1])
        .expect("reshape a logits row to a contiguous 1-D array");
    eval([&row]).expect("evaluate a logits row");
    let data: &[f32] = row.as_slice();
    data.to_vec()
}

/// Cached and uncached decoding differ in how many keys each `matmul` reduces
/// over, so the two are not bit-identical in `f32`. A wrong cache offset moves
/// logits by far more than this: the positional embeddings it would mix up are
/// O(1) values added before a layer norm.
const TOL: f32 = 1e-3;

#[test]
fn cached_decoding_matches_a_single_full_pass() {
    let _serial = serial();
    let (mut model, xa) = model_and_audio_features();

    let all = Array::from_slice(&IDS[..], &[1, IDS.len() as i32]);
    let (full, _, _) = model
        .decoder
        .forward(&all, &xa, None)
        .expect("full-sequence forward");
    assert_eq!(
        full.shape(),
        &[1, IDS.len() as i32, tiny_dims().n_vocab as i32],
        "full-pass logits shape"
    );

    let mut cache: Option<Vec<Option<BlockCache>>> = None;
    for (i, &id) in IDS.iter().enumerate() {
        let one = Array::from_slice(&[id], &[1, 1]);
        let (step, new_cache, _) = model
            .decoder
            .forward(&one, &xa, cache.take())
            .expect("cached forward");
        cache = Some(new_cache);

        common::assert_close(
            &logits_row(&step, 0),
            &logits_row(&full, i as i32),
            TOL,
            &format!("cached step {i} (token {id}) vs the full pass"),
        );
    }
}

/// The cache has to grow by exactly one key per step. If `TextDecoder` ever
/// stopped concatenating — or concatenated on the wrong axis — the equivalence
/// test above could still pass for the first token and then drift, so pin the
/// shape directly.
#[test]
fn the_self_attention_cache_grows_one_position_per_step() {
    let _serial = serial();
    let (mut model, xa) = model_and_audio_features();
    let dims = tiny_dims();

    let mut cache: Option<Vec<Option<BlockCache>>> = None;
    for (i, &id) in IDS.iter().enumerate() {
        let one = Array::from_slice(&[id], &[1, 1]);
        let (_, new_cache, _) = model
            .decoder
            .forward(&one, &xa, cache.take())
            .expect("cached forward");

        assert_eq!(
            new_cache.len(),
            dims.n_text_layer,
            "one cache entry per decoder block"
        );
        for (layer, block) in new_cache.iter().enumerate() {
            let (self_kv, cross_kv) = block.as_ref().expect("block cache");
            let (k, v) = self_kv.as_ref().expect("self-attention cache");
            assert_eq!(
                k.shape(),
                &[1, i as i32 + 1, dims.n_text_state as i32],
                "step {i}, layer {layer}: cached keys"
            );
            assert_eq!(v.shape(), k.shape(), "step {i}, layer {layer}: cached values");

            // Cross-attention keys come from the encoder output and never grow;
            // recomputing them every step would be silent, just slower.
            let (ck, _) = cross_kv.as_ref().expect("cross-attention cache");
            assert_eq!(
                ck.shape(),
                &[1, dims.n_audio_ctx as i32, dims.n_text_state as i32],
                "step {i}, layer {layer}: cached cross-attention keys"
            );
        }

        cache = Some(new_cache);
    }
}

/// Guards the test above from passing vacuously: if the decoder's output did
/// not depend on position at all, cached and uncached decoding would agree for
/// trivial reasons and the offset arithmetic would be untested.
#[test]
fn logits_depend_on_position() {
    let _serial = serial();
    let (mut model, xa) = model_and_audio_features();

    let all = Array::from_slice(&IDS[..], &[1, IDS.len() as i32]);
    let (full, _, _) = model
        .decoder
        .forward(&all, &xa, None)
        .expect("full-sequence forward");

    // Token `3` sits at positions 0, 2 and 4. Same token, different position
    // and different preceding context, so the logits must differ.
    let first = logits_row(&full, 0);
    for pos in [2i32, 4] {
        let later = logits_row(&full, pos);
        let max_diff = first
            .iter()
            .zip(&later)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff > TOL,
            "token 3 produced the same logits at positions 0 and {pos} \
             (max diff {max_diff:.3e}) — the decoder is position-blind, which \
             would make the cache-equivalence test vacuous"
        );
    }
}
