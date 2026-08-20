//! Parity tests for the closed-form model math in `whisper`.
//!
//! `sinusoids` is the encoder's positional embedding. It is added to every
//! audio frame, so an off-by-one in the timescale denominator or a swapped
//! sin/cos half degrades transcription quality everywhere without ever
//! raising an error — exactly the class of bug a golden fixture catches.

mod common;

use mlx_whisper_rs::whisper::{ModelDimensions, Whisper, sinusoids};

#[test]
fn sinusoids_small_case_matches_upstream() {
    common::init_device();
    let want = common::f32s("sinusoids_10x8");
    let s = sinusoids(10, 8).expect("sinusoids(10, 8)");

    assert_eq!(s.shape(), &[10, 8], "sinusoids(10, 8) shape");
    let got: &[f32] = s.as_slice();
    common::assert_close(got, &want, 1e-6, "sinusoids(10, 8)");
}

#[test]
fn sinusoids_encoder_shape_matches_upstream() {
    common::init_device();
    // The real shape used by every Whisper encoder: n_audio_ctx x n_audio_state
    // for the `base` model. Committed as checksums plus probe points rather
    // than 2.3 MB of floats.
    let doc = common::json("model_math");
    let fx = &doc["sinusoids_1500x384"];
    let s = sinusoids(1500, 384).expect("sinusoids(1500, 384)");
    assert_eq!(s.shape(), &[1500, 384]);

    let got: &[f32] = s.as_slice();
    let sum: f64 = got.iter().map(|&x| x as f64).sum();
    let abs_sum: f64 = got.iter().map(|&x| (x as f64).abs()).sum();

    let want_sum = fx["sum"].as_f64().unwrap();
    let want_abs = fx["abs_sum"].as_f64().unwrap();
    assert!(
        (sum - want_sum).abs() < 1e-1,
        "sinusoids(1500, 384) sum {sum} != {want_sum}"
    );
    assert!(
        (abs_sum - want_abs).abs() < 1e-1,
        "sinusoids(1500, 384) abs-sum {abs_sum} != {want_abs}"
    );

    for probe in fx["probes"].as_array().unwrap() {
        let r = probe["row"].as_u64().unwrap() as usize;
        let c = probe["col"].as_u64().unwrap() as usize;
        let want = probe["value"].as_f64().unwrap() as f32;
        let got_v = got[r * 384 + c];
        assert!(
            (got_v - want).abs() < 1e-5,
            "sinusoids(1500, 384)[{r}, {c}] = {got_v}, want {want}"
        );
    }
}

#[test]
fn sinusoids_first_row_is_sin0_then_cos0() {
    common::init_device();
    // Row 0 has scaled_time == 0 for every channel, so the concatenation order
    // is directly observable: the sin half must be all 0.0 and the cos half
    // all 1.0. A swapped concat shows up here immediately.
    let s = sinusoids(4, 8).expect("sinusoids(4, 8)");
    let got: &[f32] = s.as_slice();

    for c in 0..4 {
        assert!(
            got[c].abs() < 1e-6,
            "row 0 channel {c} should be sin(0) == 0.0, got {}",
            got[c]
        );
    }
    for c in 4..8 {
        assert!(
            (got[c] - 1.0).abs() < 1e-6,
            "row 0 channel {c} should be cos(0) == 1.0, got {}",
            got[c]
        );
    }
}

// ── Vocabulary-derived model configuration ───────────────────────────────────

/// A model whose only interesting dimension is `n_vocab`. Everything else is
/// as small as the constructors allow — `n_*_state` has to divide evenly by
/// `n_*_head`, and `sinusoids` needs an even channel count.
fn dims_with_vocab(n_vocab: usize) -> ModelDimensions {
    ModelDimensions {
        n_mels: 80,
        n_audio_ctx: 2,
        n_audio_state: 8,
        n_audio_head: 2,
        n_audio_layer: 1,
        n_vocab,
        n_text_ctx: 4,
        n_text_state: 8,
        n_text_head: 2,
        n_text_layer: 2,
    }
}

/// `is_multilingual` and `num_languages` are read off `n_vocab` alone, and both
/// feed the tokenizer: the language block starts at `sot + 1` and runs for
/// `num_languages` entries, so an off-by-one in either shifts every language
/// token and every SOT sequence built from them. Both are one-line integer
/// expressions with no test, and the boundary is the whole risk.
///
/// The three vocabulary sizes are the ones that actually ship: 51864 for the
/// English-only models, 51865 for multilingual up to `large-v2`, and 51866 for
/// the `large-v3` family, which added Cantonese as a 100th language.
#[test]
fn multilinguality_is_derived_from_the_vocabulary_size() {
    common::init_device();

    let english_only = Whisper::new(dims_with_vocab(51864)).expect("english-only model");
    assert!(!english_only.is_multilingual(), "51864 is English-only");
    assert_eq!(english_only.num_languages(), 51864 - 51765);

    let multilingual = Whisper::new(dims_with_vocab(51865)).expect("multilingual model");
    assert!(multilingual.is_multilingual(), "51865 is multilingual");
    assert_eq!(multilingual.num_languages(), 99);

    let large_v3 = Whisper::new(dims_with_vocab(51866)).expect("large-v3 model");
    assert!(large_v3.is_multilingual(), "51866 is multilingual");
    assert_eq!(
        large_v3.num_languages(),
        100,
        "large-v3 has 100 languages: the 99 of large-v2 plus `yue`"
    );
}
