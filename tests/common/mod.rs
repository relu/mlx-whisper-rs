//! Shared helpers for the golden-fixture tests.
//!
//! Fixtures live in `tests/fixtures/` and are generated from the upstream
//! Python `mlx_whisper` package by `tools/gen_fixtures.py`. See
//! `tests/fixtures/README.md` for the layout and for the pinned versions.

#![allow(dead_code)]

use std::path::{Path, PathBuf};

use serde_json::Value;

/// Absolute path to `tests/fixtures/`.
pub fn fixtures_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

/// Load a fixture JSON document by stem, e.g. `json("audio")`.
pub fn json(stem: &str) -> Value {
    let path = fixtures_dir().join(format!("{stem}.json"));
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("missing fixture {}: {e}\nRegenerate with tools/gen_fixtures.py", path.display()));
    serde_json::from_str(&text)
        .unwrap_or_else(|e| panic!("malformed fixture {}: {e}", path.display()))
}

/// Load a raw little-endian `f32` fixture by stem, e.g. `f32s("mel_80")`.
pub fn f32s(stem: &str) -> Vec<f32> {
    let path = fixtures_dir().join(format!("{stem}.f32"));
    let bytes = std::fs::read(&path)
        .unwrap_or_else(|e| panic!("missing fixture {}: {e}\nRegenerate with tools/gen_fixtures.py", path.display()));
    assert_eq!(
        bytes.len() % 4,
        0,
        "{} is not a whole number of f32 values",
        path.display()
    );
    bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

/// The synthetic audio signal the fixtures were generated from: one second of
/// a 440 Hz sine at 0.5 amplitude, then one second of silence, at 16 kHz.
///
/// Regenerated here rather than committed — `tests/fixtures/audio.json` records
/// the formula and a checksum so the two cannot silently drift apart.
pub fn synth_audio() -> Vec<f32> {
    let sr = 16_000usize;
    let mut out = Vec::with_capacity(2 * sr);
    for i in 0..sr {
        let t = i as f32 / sr as f32;
        out.push(0.5 * (2.0 * std::f32::consts::PI * 440.0 * t).sin());
    }
    out.extend(std::iter::repeat(0.0).take(sr));
    out
}

/// The synthetic logits the logit-filter fixtures were generated from.
///
/// A plain 64-bit LCG, chosen so both Python and Rust can produce the exact
/// same stream without agreeing on an RNG implementation. Row 0 has its
/// timestamp region pushed down and row 1 pushed up, so the "sum of timestamp
/// probability beats every text token" rule is exercised in both directions.
pub fn synth_logits(rows: usize, n_vocab: usize, seed: u64, timestamp_begin: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(rows * n_vocab);
    let mut state: u64 = seed;
    for _ in 0..rows * n_vocab {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let u = (state >> 40) as f32 / (1u32 << 24) as f32;
        out.push(u * 16.0 - 8.0);
    }
    for (row, shift) in [(0usize, -12.0f32), (1usize, 6.0f32)] {
        if row >= rows {
            continue;
        }
        for i in timestamp_begin..n_vocab {
            out[row * n_vocab + i] += shift;
        }
    }
    out
}

/// Assert two float slices agree elementwise within `tol`.
///
/// Reports the worst offender rather than the first, so a systematic shift is
/// obvious from a single failure message.
pub fn assert_close(actual: &[f32], expected: &[f32], tol: f32, what: &str) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{what}: length mismatch (got {}, want {})",
        actual.len(),
        expected.len()
    );
    let mut worst = (0usize, 0.0f32);
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        let d = (a - e).abs();
        if d > worst.1 {
            worst = (i, d);
        }
    }
    if worst.1 > tol {
        let i = worst.0;
        panic!(
            "{what}: max abs diff {:.3e} > tol {:.3e} at index {i} (got {}, want {})",
            worst.1, tol, actual[i], expected[i]
        );
    }
}

/// Read a `[usize]` list out of a fixture JSON array.
pub fn as_u32s(v: &Value) -> Vec<u32> {
    v.as_array()
        .expect("expected a JSON array")
        .iter()
        .map(|x| x.as_u64().expect("expected an integer") as u32)
        .collect()
}

/// Read `[[start, end_inclusive], ...]` suppressed-index ranges from a fixture.
pub fn as_ranges(v: &Value) -> Vec<(usize, usize)> {
    v.as_array()
        .expect("expected a JSON array of ranges")
        .iter()
        .map(|p| {
            let p = p.as_array().expect("expected a [start, end] pair");
            (p[0].as_u64().unwrap() as usize, p[1].as_u64().unwrap() as usize)
        })
        .collect()
}

/// Expand `[[start, end_inclusive], ...]` into a sorted index list.
pub fn expand_ranges(ranges: &[(usize, usize)]) -> Vec<usize> {
    let mut out = Vec::new();
    for &(a, b) in ranges {
        out.extend(a..=b);
    }
    out
}

/// Collect the indices of `-inf` entries in a row, compressed to ranges, so
/// failures print `[[0, 50363]]` rather than fifty thousand numbers.
pub fn neg_inf_ranges(row: &[f32]) -> Vec<(usize, usize)> {
    let idx: Vec<usize> = row
        .iter()
        .enumerate()
        .filter(|(_, v)| v.is_infinite() && v.is_sign_negative())
        .map(|(i, _)| i)
        .collect();
    let mut out = Vec::new();
    let mut it = idx.into_iter();
    let Some(first) = it.next() else { return out };
    let (mut start, mut prev) = (first, first);
    for v in it {
        if v == prev + 1 {
            prev = v;
            continue;
        }
        out.push((start, prev));
        start = v;
        prev = v;
    }
    out.push((start, prev));
    out
}

/// Whether the tokenizer assets are present. The `.tiktoken` vocabularies are
/// ~1.6 MB and are not committed; `tools/extract_assets.py` writes them into
/// `assets/`. Tests that need real BPE skip themselves when they are absent
/// rather than failing, so `cargo test` is useful on a fresh clone.
pub fn assets_dir() -> Option<PathBuf> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("assets");
    let needed = ["multilingual.tiktoken", "gpt2.tiktoken"];
    if needed.iter().all(|f| dir.join(f).exists()) {
        Some(dir)
    } else {
        None
    }
}

/// `assets_dir()` with a skip notice printed when it is absent.
///
/// Call as `let Some(assets) = common::require_assets() else { return };` —
/// a plain function rather than a macro, so it composes with `let ... else`
/// and needs no `#[macro_export]` gymnastics across test binaries.
pub fn require_assets() -> Option<PathBuf> {
    let dir = assets_dir();
    if dir.is_none() {
        eprintln!(
            "SKIP: tokenizer assets missing; run `python3 tools/extract_assets.py` \
             to populate assets/"
        );
    }
    dir
}
