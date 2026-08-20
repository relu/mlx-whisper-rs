//! Tests for the in-memory audio entry points.
//!
//! These have no Python counterpart — upstream `mlx_whisper` always shells out
//! to ffmpeg — so they are correctness tests on their own terms rather than
//! parity tests. `audio_from_wav_bytes` returns a `Result`, so malformed input
//! must produce `Err`, never a panic: it is a parser fed untrusted bytes.

mod common;

use mlx_whisper_rs::audio::{audio_from_pcm_s16le, audio_from_wav_bytes, SAMPLE_RATE};

/// Build a minimal RIFF/WAVE buffer.
fn wav(
    audio_format: u16,
    channels: u16,
    sample_rate: u32,
    bits: u16,
    data: &[u8],
) -> Vec<u8> {
    let block_align = channels * bits / 8;
    let byte_rate = sample_rate * block_align as u32;
    let mut out = Vec::new();
    out.extend(b"RIFF");
    out.extend(((36 + data.len()) as u32).to_le_bytes());
    out.extend(b"WAVE");
    out.extend(b"fmt ");
    out.extend(16u32.to_le_bytes());
    out.extend(audio_format.to_le_bytes());
    out.extend(channels.to_le_bytes());
    out.extend(sample_rate.to_le_bytes());
    out.extend(byte_rate.to_le_bytes());
    out.extend(block_align.to_le_bytes());
    out.extend(bits.to_le_bytes());
    out.extend(b"data");
    out.extend((data.len() as u32).to_le_bytes());
    out.extend(data);
    out
}

fn s16(samples: &[i16]) -> Vec<u8> {
    samples.iter().flat_map(|s| s.to_le_bytes()).collect()
}

#[test]
fn parses_mono_s16le() {
    common::init_device();
    let bytes = wav(1, 1, SAMPLE_RATE as u32, 16, &s16(&[0, 16384, -16384, 32767]));
    let (a, sr) = audio_from_wav_bytes(&bytes).expect("valid mono s16 WAV");

    assert_eq!(sr, SAMPLE_RATE as u32);
    assert_eq!(a.shape(), &[4]);
    let got: &[f32] = a.as_slice();

    assert_eq!(got[0], 0.0);
    assert!((got[1] - 0.5).abs() < 1e-6, "16384/32768 == 0.5, got {}", got[1]);
    assert!((got[2] + 0.5).abs() < 1e-6, "-16384/32768 == -0.5, got {}", got[2]);
    assert!(got[3] <= 1.0, "full-scale positive must not exceed 1.0");
}

#[test]
fn averages_stereo_to_mono() {
    common::init_device();
    // L = +0.5, R = -0.5 must average to exactly 0.0.
    let bytes = wav(1, 2, SAMPLE_RATE as u32, 16, &s16(&[16384, -16384, 8192, 8192]));
    let (a, _) = audio_from_wav_bytes(&bytes).expect("valid stereo WAV");

    assert_eq!(a.shape(), &[2], "stereo must collapse to one sample per frame");
    let got: &[f32] = a.as_slice();
    assert!(got[0].abs() < 1e-6, "L+R downmix should cancel, got {}", got[0]);
    assert!((got[1] - 0.25).abs() < 1e-6, "equal channels should average, got {}", got[1]);
}

#[test]
fn skips_odd_sized_chunks_before_data() {
    common::init_device();
    // RIFF chunks are word-aligned: a chunk with an odd declared size is
    // followed by a pad byte. A parser that advances by the raw size lands
    // one byte off and never finds `data`.
    let payload = s16(&[16384, -16384]);
    let mut bytes = Vec::new();
    bytes.extend(b"RIFF");
    bytes.extend(0u32.to_le_bytes()); // patched below
    bytes.extend(b"WAVE");
    bytes.extend(b"fmt ");
    bytes.extend(16u32.to_le_bytes());
    bytes.extend(1u16.to_le_bytes());
    bytes.extend(1u16.to_le_bytes());
    bytes.extend((SAMPLE_RATE as u32).to_le_bytes());
    bytes.extend((SAMPLE_RATE as u32 * 2).to_le_bytes());
    bytes.extend(2u16.to_le_bytes());
    bytes.extend(16u16.to_le_bytes());
    // An odd-sized LIST chunk plus its pad byte.
    bytes.extend(b"LIST");
    bytes.extend(5u32.to_le_bytes());
    bytes.extend(b"INFOx");
    bytes.push(0);
    bytes.extend(b"data");
    bytes.extend((payload.len() as u32).to_le_bytes());
    bytes.extend(&payload);
    let n = (bytes.len() - 8) as u32;
    bytes[4..8].copy_from_slice(&n.to_le_bytes());

    let (a, _) = audio_from_wav_bytes(&bytes).expect("data chunk after an odd LIST chunk");
    assert_eq!(a.shape(), &[2]);
}

#[test]
fn rejects_non_riff_input() {
    common::init_device();
    assert!(audio_from_wav_bytes(b"not a wav file at all").is_err());
    assert!(audio_from_wav_bytes(&[]).is_err());
}

/// A `fmt ` chunk may declare `chunk_size >= 16` while the buffer is truncated
/// before those 16 bytes exist. Reading the format fields without a bounds
/// check indexes past the end and panics, in a `Result`-returning parser.
#[test]
fn rejects_truncated_fmt_chunk_without_panicking() {
    common::init_device();
    let mut bytes = Vec::new();
    bytes.extend(b"RIFF");
    bytes.extend(20u32.to_le_bytes());
    bytes.extend(b"WAVE");
    bytes.extend(b"fmt ");
    bytes.extend(16u32.to_le_bytes()); // claims 16 bytes of format data...
    bytes.extend(1u16.to_le_bytes()); // ...but only 2 are present
    assert!(
        audio_from_wav_bytes(&bytes).is_err(),
        "a truncated fmt chunk must return Err, not panic"
    );
}

#[test]
fn rejects_truncated_header_prefixes() {
    common::init_device();
    // Every prefix of a valid file must fail cleanly rather than panic — and
    // *fail*, not succeed with a garbage sample rate. The canonical header this
    // helper writes is exactly 44 bytes, so every prefix below that is short of
    // a complete `fmt `/`data` pair and has no valid parse.
    let full = wav(1, 1, SAMPLE_RATE as u32, 16, &s16(&[1, 2, 3, 4]));
    for len in 0..full.len().min(44) {
        assert!(
            audio_from_wav_bytes(&full[..len]).is_err(),
            "a {len}-byte prefix of a 44-byte header must return Err"
        );
    }
}

#[test]
fn rejects_zero_channels() {
    common::init_device();
    let bytes = wav(1, 0, SAMPLE_RATE as u32, 16, &s16(&[1, 2]));
    assert!(
        audio_from_wav_bytes(&bytes).is_err(),
        "channels == 0 would divide by zero in the downmix"
    );
}

#[test]
fn rejects_zero_sample_rate() {
    common::init_device();
    let bytes = wav(1, 1, 0, 16, &s16(&[1, 2]));
    match audio_from_wav_bytes(&bytes) {
        Err(_) => {}
        Ok((_, sr)) => assert_ne!(
            sr, 0,
            "a zero sample rate must be rejected, not reported back to the caller"
        ),
    }
}

#[test]
fn reports_non_16k_sample_rate_to_the_caller() {
    common::init_device();
    // The function does not resample; the contract is that it hands the rate
    // back so the caller can reject it. Pin that, so the contract cannot
    // silently change to "assumes 16 kHz".
    let bytes = wav(1, 1, 44_100, 16, &s16(&[1, 2, 3, 4]));
    let (_, sr) = audio_from_wav_bytes(&bytes).expect("44.1 kHz WAV should parse");
    assert_eq!(sr, 44_100, "the true sample rate must reach the caller");
    assert_ne!(sr, SAMPLE_RATE as u32);
}

#[test]
fn parses_float32_payload() {
    common::init_device();
    let data: Vec<u8> = [0.0f32, 0.5, -0.5, 1.0]
        .iter()
        .flat_map(|f| f.to_le_bytes())
        .collect();
    let bytes = wav(3, 1, SAMPLE_RATE as u32, 32, &data);
    let (a, _) = audio_from_wav_bytes(&bytes).expect("IEEE float32 WAV");
    let got: &[f32] = a.as_slice();
    assert_eq!(got.len(), 4);
    assert!((got[1] - 0.5).abs() < 1e-6);
    assert!((got[2] + 0.5).abs() < 1e-6);
}

#[test]
fn pcm_s16le_matches_wav_payload_scaling() {
    common::init_device();
    // The headerless path must agree with the WAV path on identical samples.
    let samples = [0i16, 16384, -16384, 32767];
    let raw = audio_from_pcm_s16le(&s16(&samples));
    let (from_wav, _) = audio_from_wav_bytes(&wav(1, 1, SAMPLE_RATE as u32, 16, &s16(&samples)))
        .expect("valid WAV");

    let a: &[f32] = raw.as_slice();
    let b: &[f32] = from_wav.as_slice();
    common::assert_close(a, b, 0.0, "pcm_s16le vs wav payload");
}

#[test]
fn pcm_s16le_ignores_a_trailing_odd_byte() {
    common::init_device();
    // A ring buffer can hand over a partial frame; it must be dropped, not
    // read past the end.
    let mut bytes = s16(&[1000, -1000]);
    bytes.push(0x7f);
    let a = audio_from_pcm_s16le(&bytes);
    assert_eq!(a.shape(), &[2], "a trailing half-sample must be discarded");
}

#[test]
fn pcm_s16le_handles_empty_input() {
    common::init_device();
    let a = audio_from_pcm_s16le(&[]);
    assert_eq!(a.size(), 0);
}
