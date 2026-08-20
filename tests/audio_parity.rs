//! Parity tests for `audio`: constants, Hann window, mel filterbanks, the
//! STFT/log-Mel pipeline, and `pad_or_trim`.
//!
//! Expectations come from `tests/fixtures/`, generated from upstream Python
//! `mlx_whisper` — see `tests/fixtures/README.md`.

mod common;

use mlx_rs::Array;
use mlx_whisper_rs::audio::{
    self, log_mel_spectrogram, mel_filters, pad_or_trim, CHUNK_LENGTH, FRAMES_PER_SECOND,
    HOP_LENGTH, N_FFT, N_FRAMES, N_SAMPLES, N_SAMPLES_PER_TOKEN, SAMPLE_RATE, TOKENS_PER_SECOND,
};

#[test]
fn constants_match_upstream() {
    common::init_device();
    let doc = common::json("audio");
    let c = &doc["constants"];
    let got: [(&str, usize); 9] = [
        ("SAMPLE_RATE", SAMPLE_RATE),
        ("N_FFT", N_FFT),
        ("HOP_LENGTH", HOP_LENGTH),
        ("CHUNK_LENGTH", CHUNK_LENGTH),
        ("N_SAMPLES", N_SAMPLES),
        ("N_FRAMES", N_FRAMES),
        ("N_SAMPLES_PER_TOKEN", N_SAMPLES_PER_TOKEN),
        ("FRAMES_PER_SECOND", FRAMES_PER_SECOND),
        ("TOKENS_PER_SECOND", TOKENS_PER_SECOND),
    ];
    for (name, value) in got {
        assert_eq!(
            c[name].as_u64().unwrap() as usize,
            value,
            "constant {name} diverges from upstream"
        );
    }
}

#[test]
fn synth_audio_matches_the_fixture_generator() {
    common::init_device();
    // Guards the uncommitted-input trick: if the Rust and Python signal
    // generators ever drift, every mel comparison below would silently be
    // comparing different things.
    let sig = common::synth_audio();
    let doc = common::json("audio");
    let want = &doc["input"];
    assert_eq!(sig.len(), want["samples"].as_u64().unwrap() as usize);

    let sum: f64 = sig.iter().map(|&x| x as f64).sum();
    let abs_sum: f64 = sig.iter().map(|&x| (x as f64).abs()).sum();
    let (want_sum, want_abs) = (
        want["checksum_sum"].as_f64().unwrap(),
        want["checksum_abs_sum"].as_f64().unwrap(),
    );
    assert!(
        (sum - want_sum).abs() < 1e-2,
        "signal sum {sum} != {want_sum}"
    );
    assert!(
        (abs_sum - want_abs).abs() < 1e-1,
        "signal abs-sum {abs_sum} != {want_abs}"
    );
}

#[test]
fn hann_window_is_periodic_not_symmetric() {
    common::init_device();
    // np.hanning(N+1)[:-1]. The symmetric np.hanning(N) is the classic wrong
    // answer here: it differs from the periodic form in every interior sample
    // and puts an exact 0.0 at the final index.
    let want = common::f32s("hanning_400");
    assert_eq!(want.len(), N_FFT);

    let w = audio::hanning(N_FFT);
    let got: &[f32] = w.as_slice();
    common::assert_close(got, &want, 1e-6, "hanning(400)");

    // The distinguishing property, asserted directly so a failure says why.
    assert!(
        got[N_FFT - 1] > 1e-5,
        "hanning(400) looks symmetric: last sample is {}, expected the periodic \
         window's non-zero tail",
        got[N_FFT - 1]
    );
}

#[test]
fn mel_filterbanks_match_upstream() {
    common::init_device();
    let dir = common::fixtures_dir();
    let audio_fx = common::json("audio");

    for n_mels in [80usize, 128] {
        let want = &audio_fx["filters"][format!("mel_filters_{n_mels}")];
        let f = mel_filters(n_mels, &dir).expect("mel_filters should load the fixture .npy");

        let shape: Vec<i64> = want["shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_i64().unwrap())
            .collect();
        assert_eq!(
            f.shape(),
            &[shape[0] as i32, shape[1] as i32],
            "mel_filters({n_mels}) shape"
        );

        let got: &[f32] = f.as_slice();
        let sum: f64 = got.iter().map(|&x| x as f64).sum();
        let want_sum = want["sum"].as_f64().unwrap();
        assert!(
            (sum - want_sum).abs() < 1e-3,
            "mel_filters({n_mels}) sum {sum} != {want_sum}"
        );
    }
}

/// The STFT reflect-pad must actually reflect.
///
/// `mlx-rs` resolves `(..n).stride_by(-1)` as `start = size - 1, end = n`,
/// i.e. Python `x[:n:-1]` — an *empty* slice when `n == size`. The reversal
/// idiom that means `x[::-1]` is `(..).stride_by(-1)`, with the range left
/// unbounded. Getting this wrong drops the padding entirely, shifting every
/// frame by `N_FFT / 2` samples and losing three frames per 30 s chunk.
#[test]
fn log_mel_produces_the_upstream_frame_count() {
    common::init_device();
    let dir = common::fixtures_dir();
    let sig = common::synth_audio();
    let a = Array::from_slice(&sig, &[sig.len() as i32]);

    let mel = log_mel_spectrogram(a, 80, &dir, 0).expect("log_mel_spectrogram");
    let doc = common::json("audio");
    let want_frames = doc["mel_80"]["shape"][0].as_i64().unwrap() as i32;

    assert_eq!(
        mel.shape()[0],
        want_frames,
        "frame count diverges from upstream — reflect padding is very likely \
         being dropped (see the doc comment on this test)"
    );
    assert_eq!(mel.shape()[1], 80, "mel bin count");
}

#[test]
fn log_mel_full_chunk_yields_exactly_n_frames() {
    common::init_device();
    // The property the whole sliding-window pipeline is built on: one 30 s
    // chunk must produce exactly N_FRAMES (3000) frames.
    let dir = common::fixtures_dir();
    let sig = vec![0.0f32; N_SAMPLES];
    let a = Array::from_slice(&sig, &[sig.len() as i32]);

    let mel = log_mel_spectrogram(a, 80, &dir, 0).expect("log_mel_spectrogram");
    assert_eq!(
        mel.shape()[0],
        N_FRAMES as i32,
        "a {CHUNK_LENGTH}s chunk must yield exactly N_FRAMES frames"
    );
}

#[test]
fn log_mel_matches_upstream_values() {
    common::init_device();
    let dir = common::fixtures_dir();
    let sig = common::synth_audio();

    for n_mels in [80usize, 128] {
        let want = common::f32s(&format!("mel_{n_mels}"));
        let a = Array::from_slice(&sig, &[sig.len() as i32]);
        let mel = log_mel_spectrogram(a, n_mels, &dir, 0).expect("log_mel_spectrogram");
        let got: &[f32] = mel.as_slice();

        // 1e-4 absolute: the pipeline runs log10 over a float32 FFT, so the
        // last couple of mantissa bits legitimately differ between backends.
        common::assert_close(got, &want, 1e-4, &format!("log_mel_spectrogram(n_mels={n_mels})"));
    }
}

#[test]
fn log_mel_normalisation_floor_is_global() {
    common::init_device();
    // log_spec = maximum(log_spec, log_spec.max() - 8.0); (x + 4) / 4
    // so the minimum of the result is exactly (max*4 - 8 - ... ) => the span
    // between min and max is capped at 8/4 == 2.0, globally across the whole
    // spectrogram rather than per-frame or per-bin.
    let dir = common::fixtures_dir();
    let sig = common::synth_audio();
    let a = Array::from_slice(&sig, &[sig.len() as i32]);
    let mel = log_mel_spectrogram(a, 80, &dir, 0).expect("log_mel_spectrogram");
    let got: &[f32] = mel.as_slice();

    let max = got.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let min = got.iter().cloned().fold(f32::INFINITY, f32::min);
    assert!(
        (max - min - 2.0).abs() < 1e-4,
        "dynamic range should be clamped to exactly 2.0 after normalisation, got {}",
        max - min
    );
}

#[test]
fn pad_or_trim_cases() {
    common::init_device();
    for case in common::json("audio")["pad_or_trim"].as_array().unwrap() {
        let in_len = case["in_len"].as_u64().unwrap() as usize;
        let target = case["target"].as_u64().unwrap() as usize;
        let out_len = case["out_len"].as_u64().unwrap() as usize;

        // Ramp input, so a wrong-end trim is visible in the values.
        let data: Vec<f32> = (0..in_len).map(|i| i as f32).collect();
        let a = Array::from_slice(&data, &[in_len as i32]);
        let out = pad_or_trim(a, target).expect("pad_or_trim");

        assert_eq!(out.shape(), &[out_len as i32], "pad_or_trim({in_len} -> {target}) shape");
        let got: &[f32] = out.as_slice();

        if in_len >= target {
            // Keeps the prefix, not the suffix.
            for i in 0..target {
                assert_eq!(got[i], i as f32, "pad_or_trim should keep the prefix");
            }
        } else {
            for i in 0..in_len {
                assert_eq!(got[i], i as f32, "pad_or_trim must not disturb the head");
            }
            for i in in_len..target {
                assert_eq!(got[i], 0.0, "pad_or_trim must zero-fill the tail");
            }
        }
    }
}

#[test]
fn pad_or_trim_is_identity_at_exact_length() {
    common::init_device();
    let data: Vec<f32> = (0..N_FRAMES).map(|i| i as f32).collect();
    let a = Array::from_slice(&data, &[N_FRAMES as i32]);
    let out = pad_or_trim(a, N_FRAMES).expect("pad_or_trim");
    let got: &[f32] = out.as_slice();
    assert_eq!(got.len(), N_FRAMES);
    assert_eq!(got[0], 0.0);
    assert_eq!(got[N_FRAMES - 1], (N_FRAMES - 1) as f32);
}

/// `padding` appends zero **samples** before the STFT, the way upstream's
/// `log_mel_spectrogram(audio, n_mels, padding=N_SAMPLES)` does.
///
/// `transcribe` relies on this to compute `content_frames = frames - N_FRAMES`
/// and to run language detection over the padded mel, so the frame arithmetic
/// has to come out exactly.
#[test]
fn log_mel_padding_adds_exactly_n_frames() {
    common::init_device();
    let dir = common::fixtures_dir();
    let sig = common::synth_audio();

    let unpadded = {
        let a = Array::from_slice(&sig, &[sig.len() as i32]);
        log_mel_spectrogram(a, 80, &dir, 0).expect("log_mel_spectrogram")
    };
    let padded = {
        let a = Array::from_slice(&sig, &[sig.len() as i32]);
        log_mel_spectrogram(a, 80, &dir, N_SAMPLES).expect("log_mel_spectrogram")
    };

    assert_eq!(
        padded.shape()[0] - unpadded.shape()[0],
        N_FRAMES as i32,
        "padding with N_SAMPLES must add exactly N_FRAMES mel frames"
    );
}
