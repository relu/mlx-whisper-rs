// Translated from mlx-whisper/transcribe.py (Apple Inc.)

use std::collections::HashMap;
use std::path::Path;

use anyhow::Result;
use mlx_rs::{Array, ops::{indexing::IndexOp, softmax_axis}, transforms::eval};

use crate::audio::{
    log_mel_spectrogram, pad_or_trim, HOP_LENGTH, N_FRAMES, N_SAMPLES, SAMPLE_RATE,
};
use crate::decoding::{decode, DecodingOptions, DecodingResult};
use crate::tokenizer::{get_tokenizer, LANGUAGES};
use crate::whisper::Whisper;

// ── Types ─────────────────────────────────────────────────────────────────────

pub struct Segment {
    pub id: usize,
    /// Frame offset in the original mel spectrogram
    pub seek: usize,
    pub start: f32,
    pub end: f32,
    pub text: String,
    pub tokens: Vec<u32>,
    pub temperature: f32,
    pub avg_logprob: f32,
    pub compression_ratio: f32,
    pub no_speech_prob: f32,
}

pub struct TranscribeResult {
    pub text: String,
    pub segments: Vec<Segment>,
    pub language: String,
}

pub struct TranscribeOptions {
    /// Language code (e.g. "en", "zh"); None → auto-detect for multilingual models
    pub language: Option<String>,
    pub task: String,
    /// Temperature schedule: tries each value in order if quality check fails
    pub temperatures: Vec<f32>,
    /// Compression ratio above this → retry with next temperature (too repetitive)
    pub compression_ratio_threshold: Option<f32>,
    /// Avg log-probability below this → retry with next temperature
    pub logprob_threshold: Option<f32>,
    /// No-speech probability above this → skip the segment (silence detection)
    pub no_speech_threshold: Option<f32>,
    /// Feed the previous window's output as a prompt for the next window
    pub condition_on_previous_text: bool,
    /// Text to prepend as context for the first window
    pub initial_prompt: Option<String>,
    /// Suppress blank tokens at the first generated position
    pub suppress_blank: bool,
    /// Token suppression: "-1" = non-speech symbols, or comma-separated IDs
    pub suppress_tokens: Option<String>,
    pub without_timestamps: bool,
    /// Print decoded text to stderr as it is produced
    pub verbose: bool,
    /// Override the encoder context length, mirroring whisper.cpp's `-ac` /
    /// `audio_ctx` flag. Whisper always encodes a full padded 30-second window
    /// regardless of how much of it is real audio, so a short utterance pays
    /// the same encoder cost as a full one; this truncates the encoder to `n`
    /// positions, cutting that cost at some accuracy price (the model was
    /// trained to see the full window, so attending over a truncated one is
    /// off-distribution).
    ///
    /// `None` (default) preserves today's behaviour exactly. `Some(n)` must
    /// satisfy `1 <= n <= dims.n_audio_ctx`; anything else is a clear error
    /// from `AudioEncoder::forward` rather than a panic. This does not change
    /// the 30-second sliding-window/seek chunking or the timestamp arithmetic
    /// — only how much of each window's mel the encoder attends to.
    pub audio_ctx: Option<usize>,
}

impl Default for TranscribeOptions {
    fn default() -> Self {
        Self {
            language: None,
            task: "transcribe".to_string(),
            temperatures: vec![0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            compression_ratio_threshold: Some(2.4),
            logprob_threshold: Some(-1.0),
            no_speech_threshold: Some(0.6),
            condition_on_previous_text: true,
            initial_prompt: None,
            suppress_blank: true,
            suppress_tokens: Some("-1".to_string()),
            without_timestamps: false,
            verbose: false,
            audio_ctx: None,
        }
    }
}

// ── Language detection ────────────────────────────────────────────────────────

/// Detect the spoken language in a mel spectrogram.
///
/// Returns `(best_language_code, probabilities)` where `probabilities` maps
/// each supported language code to its probability.
///
/// `mel` can be:
/// - 2-D `[n_frames, n_mels]` — will be padded/trimmed to 30 s and batched
/// - 3-D `[1, n_frames, n_mels]` — used as-is (must already be 30-s window)
///
/// Only meaningful for multilingual models; returns `"en"` with p=1.0 for
/// English-only models.
pub fn detect_language(
    model: &mut Whisper,
    mel: &Array,
    assets_dir: &Path,
) -> Result<(String, HashMap<String, f32>)> {
    let n_mels = model.dims.n_mels;

    if !model.is_multilingual() {
        let mut probs = HashMap::new();
        probs.insert("en".to_string(), 1.0f32);
        return Ok(("en".to_string(), probs));
    }

    // Normalise to [1, N_FRAMES, n_mels]
    let mel_batch = match mel.ndim() {
        2 => pad_or_trim(mel.clone(), N_FRAMES)?
            .reshape(&[1, N_FRAMES as i32, n_mels as i32])?,
        3 => mel.clone(),
        d => anyhow::bail!("mel must be 2-D or 3-D, got {d}-D"),
    };

    // Python: `mel_segment = pad_or_trim(mel, N_FRAMES, axis=-2).astype(dtype)`
    // (`transcribe.py`, right before `model.detect_language(mel_segment)`). An
    // f32 mel would promote the encoder's fp16 conv weights back to f32 at the
    // first layer (PARITY.md MODEL-3).
    let mel_batch = mel_batch.as_dtype(model.dtype)?;
    // Always encode the full context here: language ID benefits from as much
    // audio as is available, and this is unrelated to the caller-settable
    // `TranscribeOptions::audio_ctx` override used by the main decode loop.
    let audio_features = model.encoder.forward(&mel_batch, None)?;
    let num_langs = model.num_languages();
    let bare_tok = get_tokenizer(model.is_multilingual(), num_langs, None, None, assets_dir)?;

    // Feed the SOT token, look at position 0 logits
    let sot_arr = Array::from_slice(&[bare_tok.sot() as i32], &[1, 1]);
    let (logits, _, _) = model.decoder.forward(&sot_arr, &audio_features, None)?;
    let logits_1d = logits.index((0i32,)).index((0i32,)); // [n_vocab]

    // Mask out all non-language tokens
    let n_vocab = model.dims.n_vocab;
    let lang_tokens = bare_tok.all_language_tokens();
    let lang_codes = bare_tok.all_language_codes();
    let mut mask = vec![f32::NEG_INFINITY; n_vocab];
    for &lt in &lang_tokens {
        if (lt as usize) < n_vocab {
            mask[lt as usize] = 0.0;
        }
    }
    let mask_arr = Array::from_slice(&mask, &[n_vocab as i32]);
    let probs_arr = softmax_axis(&logits_1d + &mask_arr, 0, false)?;
    eval([&probs_arr])?;
    let probs_data: &[f32] = probs_arr.as_slice();

    // Build probability map and find best language
    let mut lang_probs: HashMap<String, f32> = HashMap::new();
    let mut best_lang = "en".to_string();
    let mut best_prob = 0.0f32;
    for (&tok, code) in lang_tokens.iter().zip(lang_codes.iter()) {
        let p = probs_data.get(tok as usize).copied().unwrap_or(0.0);
        if p > best_prob {
            best_prob = p;
            best_lang = code.clone();
        }
        lang_probs.insert(code.clone(), p);
    }

    Ok((best_lang, lang_probs))
}

/// Internal helper used by `transcribe()`: accepts a pre-batched mel and returns
/// only the best language string.
fn detect_language_best(
    model: &mut Whisper,
    mel_batch: &Array,
    assets_dir: &Path,
) -> Result<String> {
    let (lang, _) = detect_language(model, mel_batch, assets_dir)?;
    Ok(lang)
}

// ── Window arithmetic ─────────────────────────────────────────────────────────
//
// The seek loop's decision points, lifted out of `transcribe()` as free
// functions over plain numbers. They used to be inline, which made them
// unreachable without real model weights — the loop has to actually decode to
// get there. Out here they are pinned against Python by the fixture that
// `tools/reference_segmentation.py` emits, the same way `logit_filters.json`
// pins the logit filters.

/// Whether a decode result fails the quality bar, so the next temperature in
/// the ladder should be tried.
///
/// Clause order matters and is upstream's: the no-speech check runs last and
/// *clears* the flag, so a window that looks like silence is never retried
/// however badly it scored on the other two.
fn needs_fallback(
    compression_ratio: f32,
    avg_logprob: f32,
    no_speech_prob: f32,
    compression_ratio_threshold: Option<f32>,
    logprob_threshold: Option<f32>,
    no_speech_threshold: Option<f32>,
) -> bool {
    let mut needs = false;
    if let Some(crt) = compression_ratio_threshold {
        if compression_ratio > crt {
            needs = true; // too repetitive
        }
    }
    if let Some(lpt) = logprob_threshold {
        if avg_logprob < lpt {
            needs = true; // average log probability is too low
        }
    }
    if let Some(nst) = no_speech_threshold {
        if no_speech_prob > nst {
            needs = false; // silence
        }
    }
    needs
}

/// Whether a window is silence and should be skipped without emitting anything.
///
/// Upstream computes `no_speech_prob > threshold` and then *clears* it when
/// `avg_logprob > logprob_threshold`; this folds both into one expression by
/// testing `avg_logprob < logprob_threshold` instead. The two agree everywhere
/// except at exact float equality, where upstream skips and this does not —
/// recorded in `PARITY.md` under the unnumbered `transcribe.rs` risks and
/// pinned by `silence_skip_matches_the_port_semantics` below, which asserts the
/// divergence is exactly one case wide.
fn should_skip_window(
    no_speech_prob: f32,
    avg_logprob: f32,
    no_speech_threshold: Option<f32>,
    logprob_threshold: Option<f32>,
) -> bool {
    let Some(nst) = no_speech_threshold else {
        return false;
    };
    no_speech_prob > nst && logprob_threshold.is_none_or(|lpt| avg_logprob < lpt)
}

/// The tokens that become a segment's text.
///
/// `< eot`, not `< timestamp_begin`: upstream's `new_segment` keeps only tokens
/// below EOT, so every special token — language ids and `<|notimestamps|>`
/// included, which sit between EOT and `timestamp_begin` — is dropped rather
/// than decoded into the segment text as a literal `<|xx|>`.
fn text_tokens(tokens: &[u32], eot: u32) -> Vec<u32> {
    tokens.iter().copied().filter(|&t| t < eot).collect()
}

/// One segment carved out of a decoded window, before its text is decoded.
///
/// Decoding text needs the tokenizer; everything here is integer and float
/// arithmetic, which is what makes the split testable without a model.
#[derive(Debug, Clone, PartialEq)]
struct SegmentSpan {
    /// Seconds from the start of the file.
    start: f32,
    end: f32,
    /// The timestamp-token offset behind `start`, in `time_precision` units.
    /// Signed, because upstream's is: see `split_window`.
    ///
    /// Only the tests read this and `end_pos`. They are carried on the struct
    /// rather than recomputed there so the fixture can assert exact integers,
    /// instead of comparing seconds through a float tolerance and missing an
    /// off-by-one that lands inside it.
    #[cfg_attr(not(test), allow(dead_code))]
    start_pos: i64,
    /// The offset behind `end`, or `None` when the segment runs to the end of
    /// the window and `end` came from the window duration instead. That is a
    /// distinct case from `Some(0)`.
    #[cfg_attr(not(test), allow(dead_code))]
    end_pos: Option<i64>,
    /// The window tokens this segment covers, timestamp tokens included.
    tokens: Vec<u32>,
}

#[derive(Debug, Clone, PartialEq)]
struct WindowSplit {
    spans: Vec<SegmentSpan>,
    /// Mel frames to advance `seek` by, exactly as upstream computes it.
    ///
    /// Can be `0` — when the last consumed timestamp is `<|0.00|>`, upstream
    /// does `seek += 0` and spins on the same window forever. The anti-stall
    /// guard lives in the caller rather than here, so this stays a faithful
    /// reference and the divergence has exactly one home.
    seek_delta: usize,
}

/// Carve one window's decoded tokens into segments and compute the seek advance.
///
/// Transcribed from the `while seek < seek_clip_end` body of
/// `mlx_whisper/transcribe.py`, reduced to the default `clip_timestamps` (the
/// only case this port implements), which is why `segment_size` is a parameter
/// rather than derived from a clip end.
fn split_window(
    tokens: &[u32],
    ts_begin: u32,
    seek: usize,
    segment_size: usize,
    input_stride: usize,
) -> WindowSplit {
    let time_offset = seek as f32 * HOP_LENGTH as f32 / SAMPLE_RATE as f32;
    let time_precision = input_stride as f32 * HOP_LENGTH as f32 / SAMPLE_RATE as f32;
    let segment_duration = segment_size as f32 * HOP_LENGTH as f32 / SAMPLE_RATE as f32;

    let ts_flags: Vec<bool> = tokens.iter().map(|&t| t >= ts_begin).collect();

    // `timestamp_tokens[-2:] == [False, True]`: a trailing timestamp with a
    // text token in front of it, i.e. an unpaired one. False for a window of
    // fewer than two tokens, where upstream's slice is too short to compare
    // equal.
    let single_ts_ending = ts_flags.len() >= 2
        && !ts_flags[ts_flags.len() - 2]
        && ts_flags[ts_flags.len() - 1];

    // `np.where(timestamp_tokens[:-1] & timestamp_tokens[1:])[0] + 1`
    let consecutive: Vec<usize> = (0..ts_flags.len().saturating_sub(1))
        .filter(|&i| ts_flags[i] && ts_flags[i + 1])
        .map(|i| i + 1)
        .collect();

    let mut spans: Vec<SegmentSpan> = Vec::new();

    if consecutive.is_empty() {
        // No consecutive pair: one segment covering the whole window, ending at
        // the last timestamp if there is one that isn't `<|0.00|>`.
        let mut end_pos: Option<i64> = None;
        let mut duration = segment_duration;
        if let Some(&last_ts) = tokens.iter().filter(|&&t| t >= ts_begin).last() {
            if last_ts != ts_begin {
                let pos = (last_ts - ts_begin) as i64;
                end_pos = Some(pos);
                duration = pos as f32 * time_precision;
            }
        }
        spans.push(SegmentSpan {
            start: time_offset,
            end: time_offset + duration,
            // The window opens at its own origin, so this offset is 0 by
            // construction rather than read off a token.
            start_pos: 0,
            end_pos,
            tokens: tokens.to_vec(),
        });
        return WindowSplit {
            spans,
            seek_delta: segment_size,
        };
    }

    let mut slices = consecutive;
    if single_ts_ending {
        slices.push(tokens.len());
    }

    let mut last_slice = 0usize;
    for &current_slice in &slices {
        let sliced = &tokens[last_slice..current_slice];
        // Upstream has no such guard and would raise `IndexError`. It is
        // unreachable — `consecutive` is strictly increasing and every entry is
        // at least 1 — but a panic here would take down a transcription.
        if sliced.is_empty() {
            last_slice = current_slice;
            continue;
        }
        // Signed, matching upstream's plain integer subtraction: a slice that
        // does not open on a timestamp yields a *negative* offset and a segment
        // that starts before the file does. Every reachable slice opens on one,
        // but clamping to zero here would disagree with the reference the
        // moment that stopped being true.
        let start_pos = sliced[0] as i64 - ts_begin as i64;
        let end_pos = sliced[sliced.len() - 1] as i64 - ts_begin as i64;
        spans.push(SegmentSpan {
            start: time_offset + start_pos as f32 * time_precision,
            end: time_offset + end_pos as f32 * time_precision,
            start_pos,
            end_pos: Some(end_pos),
            tokens: sliced.to_vec(),
        });
        last_slice = current_slice;
    }

    let seek_delta = if single_ts_ending {
        // A single timestamp at the end means no speech after it.
        segment_size
    } else {
        // Otherwise drop the unfinished tail and seek to the last consumed
        // timestamp. `last_slice` is at least 1 here — every entry in
        // `consecutive` is an index plus one — so the token it names is a real
        // timestamp and this subtraction cannot go negative.
        let last_ts = tokens
            .get(last_slice.saturating_sub(1))
            .copied()
            .unwrap_or(ts_begin)
            .saturating_sub(ts_begin) as usize;
        last_ts * input_stride
    };

    WindowSplit { spans, seek_delta }
}

// ── Fallback decode ───────────────────────────────────────────────────────────

fn decode_with_fallback(
    model: &mut Whisper,
    mel_batch: &Array,
    tokenizer: &crate::tokenizer::Tokenizer,
    base_options: &DecodingOptions,
    temperatures: &[f32],
    compression_ratio_threshold: Option<f32>,
    logprob_threshold: Option<f32>,
    no_speech_threshold: Option<f32>,
) -> Result<DecodingResult> {
    let mut result: Option<DecodingResult> = None;

    for &temp in temperatures {
        let mut opts = base_options.clone();
        opts.temperature = temp;
        let r = decode(model, mel_batch, tokenizer, &opts)?;

        let retry = needs_fallback(
            r.compression_ratio,
            r.avg_logprob,
            r.no_speech_prob,
            compression_ratio_threshold,
            logprob_threshold,
            no_speech_threshold,
        );
        // Kept even when it fails the bar: if the ladder runs out, the last
        // result is the answer, exactly as upstream returns `decode_result`
        // after the loop rather than raising.
        result = Some(r);
        if !retry {
            break;
        }
    }

    result.ok_or_else(|| anyhow::anyhow!("No decode result produced"))
}

// ── Timestamp formatting ──────────────────────────────────────────────────────

fn format_timestamp(seconds: f32) -> String {
    let ms = (seconds * 1000.0).round() as u64;
    let h = ms / 3_600_000;
    let ms = ms % 3_600_000;
    let m = ms / 60_000;
    let ms = ms % 60_000;
    let s = ms / 1_000;
    let ms = ms % 1_000;
    if h > 0 {
        format!("{h:02}:{m:02}:{s:02}.{ms:03}")
    } else {
        format!("{m:02}:{s:02}.{ms:03}")
    }
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Transcribe a 16 kHz mono audio array using a sliding 30-second window.
///
/// # Parameters
/// - `audio`: 1-D float32 array of 16 kHz mono samples
/// - `model`: pre-loaded Whisper model
/// - `assets_dir`: directory containing `*.tiktoken` vocab files and `mel_filters_*.npy`
/// - `options`: transcription settings
pub fn transcribe(
    audio: Array,
    model: &mut Whisper,
    assets_dir: &Path,
    options: &TranscribeOptions,
) -> Result<TranscribeResult> {
    let n_mels = model.dims.n_mels;
    let n_audio_ctx = model.dims.n_audio_ctx;

    // ── Compute mel spectrogram ─────────────────────────────────────────────
    // Upstream pads the audio with a full 30s of silence before the STFT, so
    // the tail of the final segment is real mel-of-silence rather than literal
    // 0.0 in normalised log space. `content_frames` then excludes that padding.
    let mel = log_mel_spectrogram(audio, n_mels, assets_dir, N_SAMPLES)?; // [n_frames, n_mels]
    let content_frames = (mel.shape()[0] as usize).saturating_sub(N_FRAMES); // real frames

    // ── Determine language ──────────────────────────────────────────────────
    let language: String = if let Some(ref lang) = options.language {
        lang.clone()
    } else if !model.is_multilingual() {
        "en".to_string()
    } else {
        if options.verbose {
            eprintln!("Detecting language from first 30 seconds…");
        }
        // Upstream takes the first N_FRAMES of the *padded* mel
        // (`pad_or_trim(mel, N_FRAMES)`), so for audio shorter than 30s the
        // tail is real mel-of-silence rather than zeros in log space. Slicing
        // to `content_frames` first and zero-padding would feed the encoder a
        // different input for language ID.
        let mel_30s = pad_or_trim(mel.clone(), N_FRAMES)?
            .reshape(&[1, N_FRAMES as i32, n_mels as i32])?;
        let detected = detect_language_best(model, &mel_30s, assets_dir)?;
        if options.verbose {
            let lang_name = LANGUAGES
                .iter()
                .find(|(c, _)| *c == detected)
                .map(|(_, n)| *n)
                .unwrap_or(&detected);
            let mut name = lang_name.to_string();
            if let Some(c) = name.get_mut(0..1) {
                c.make_ascii_uppercase();
            }
            eprintln!("Detected language: {name}");
        }
        detected
    };

    // ── Build tokenizer ─────────────────────────────────────────────────────
    let tokenizer = get_tokenizer(
        model.is_multilingual(),
        model.num_languages(),
        Some(&language),
        Some(&options.task),
        assets_dir,
    )?;

    // ── Timing constants ────────────────────────────────────────────────────
    // input_stride: mel frames per decoder output token (typically 2). The
    // seconds-per-token precision derives from it inside `split_window`.
    let input_stride = N_FRAMES / n_audio_ctx;

    // ── Build base DecodingOptions (shared across all windows) ─────────────
    let base_decode_opts = DecodingOptions {
        task: options.task.clone(),
        language: Some(language.clone()),
        temperature: options.temperatures.first().copied().unwrap_or(0.0),
        suppress_blank: options.suppress_blank,
        suppress_tokens: options.suppress_tokens.clone(),
        without_timestamps: options.without_timestamps,
        max_initial_timestamp: Some(1.0),
        prompt: None, // filled per-window below
        audio_ctx: options.audio_ctx,
        ..Default::default()
    };

    // ── Prepare initial prompt tokens ───────────────────────────────────────
    let mut all_tokens: Vec<u32> = Vec::new();
    let mut prompt_reset_since = 0usize;
    // Remembered here rather than re-encoded after the loop: the prompt is fed
    // through BPE exactly once, and the two encodes cannot drift apart.
    let mut initial_prompt_len = 0usize;

    if let Some(ref init_prompt) = options.initial_prompt {
        let pt = tokenizer.encode(&format!(" {}", init_prompt.trim()));
        initial_prompt_len = pt.len();
        all_tokens.extend_from_slice(&pt);
        prompt_reset_since = 0; // include initial prompt always
    }

    // ── Sliding window decode loop ──────────────────────────────────────────
    let mut all_segments: Vec<Segment> = Vec::new();
    let mut seek = 0usize;

    while seek < content_frames {
        let segment_size = N_FRAMES.min(content_frames - seek);

        // Slice mel and pad/trim to exactly N_FRAMES
        let mel_slice = mel.index((seek as i32..(seek + segment_size) as i32,));
        let mel_segment = pad_or_trim(mel_slice, N_FRAMES)?
            .reshape(&[1, N_FRAMES as i32, n_mels as i32])?;

        // Build per-window prompt from accumulated tokens.
        //
        // Upstream passes `all_tokens[prompt_reset_since:]` unconditionally and
        // moves `prompt_reset_since` forward *after* the window. So with
        // condition_on_previous_text = false the first window still receives
        // the initial prompt (the cursor is still 0) and later windows receive
        // nothing. Gating on the flag here instead dropped `initial_prompt`
        // entirely in that mode.
        let prompt: Option<Vec<u32>> = {
            let p = &all_tokens[prompt_reset_since..];
            if p.is_empty() { None } else { Some(p.to_vec()) }
        };
        let mut window_opts = base_decode_opts.clone();
        window_opts.prompt = prompt;

        // Decode with temperature fallback
        let result = decode_with_fallback(
            model,
            &mel_segment,
            &tokenizer,
            &window_opts,
            &options.temperatures,
            options.compression_ratio_threshold,
            options.logprob_threshold,
            options.no_speech_threshold,
        )?;

        // ADDED FOR DEBUGGING
        if options.verbose {
            eprintln!("DEBUG: window decode result tokens = {:?}", result.tokens);
            eprintln!("DEBUG: window decode result text = {:?}", result.text);
        }

        let previous_seek = seek;

        // ── Silence check ───────────────────────────────────────────────────
        if should_skip_window(
            result.no_speech_prob,
            result.avg_logprob,
            options.no_speech_threshold,
            options.logprob_threshold,
        ) {
            seek += segment_size; // fast-forward to the next segment boundary
            continue;
        }

        // ── Parse timestamp tokens into segments ────────────────────────────
        let split = split_window(
            &result.tokens,
            tokenizer.timestamp_begin(),
            previous_seek,
            segment_size,
            input_stride,
        );

        let mut current_segments: Vec<Segment> = split
            .spans
            .into_iter()
            .enumerate()
            .map(|(i, span)| Segment {
                id: all_segments.len() + i,
                seek: previous_seek,
                start: span.start,
                end: span.end,
                // Not trimmed: upstream returns `tokenizer.decode(...)`
                // verbatim, which keeps Whisper's conventional leading space.
                // Trimming here made every segment byte-different from the
                // Python reference and lost the inter-segment spacing when
                // segments are concatenated.
                text: tokenizer.decode(&text_tokens(&span.tokens, tokenizer.eot())),
                tokens: span.tokens,
                temperature: result.temperature,
                avg_logprob: result.avg_logprob,
                compression_ratio: result.compression_ratio,
                no_speech_prob: result.no_speech_prob,
            })
            .collect();

        // Anti-stall guard. A zero advance means the last consumed timestamp
        // was `<|0.00|>`; upstream then re-decodes the same window forever.
        // Jumping a whole window drops that window's audio, which is the lesser
        // of the two failures. Documented in `PARITY.md`.
        seek += if split.seek_delta == 0 {
            segment_size
        } else {
            split.seek_delta
        };

        // ── Verbose output ──────────────────────────────────────────────────
        if options.verbose {
            for seg in &current_segments {
                eprintln!(
                    "[{} --> {}] text={:?} tokens={:?}",
                    format_timestamp(seg.start),
                    format_timestamp(seg.end),
                    seg.text,
                    seg.tokens
                );
            }
        }

        // Clear blank/instantaneous segments *before* accumulating.
        //
        // Upstream empties a blank segment's `text` and `tokens` in place and
        // keeps the segment itself, so the id sequence stays stable across a
        // blank window. Dropping the entry instead diverged from the reference
        // in both segment count and ids.
        //
        // The ordering matters independently of that: clearing has to happen
        // before `all_tokens` is extended, or blank segments contaminate
        // `all_tokens[prompt_reset_since..]`, which is fed as <|startofprev|>
        // context to every later window — an error that compounds across a
        // long file.
        for seg in &mut current_segments {
            if seg.start == seg.end || seg.text.trim().is_empty() {
                seg.text.clear();
                seg.tokens.clear();
            }
        }

        // ── Accumulate tokens for condition_on_previous_text ────────────────
        for seg in &current_segments {
            all_tokens.extend_from_slice(&seg.tokens);
        }
        // If temperature was high or condition disabled, reset the prompt cursor
        if !options.condition_on_previous_text || result.temperature > 0.5 {
            prompt_reset_since = all_tokens.len();
        }

        all_segments.extend(current_segments);
    }

    // ── Assemble full text ──────────────────────────────────────────────────
    // Segment ids are assigned at construction from `all_segments.len()`, and
    // nothing is dropped any more, so they are already 0..n in order — the old
    // re-indexing pass only existed to close the gaps left by the drop filter.
    //
    // Not trimmed, matching upstream's
    // `tokenizer.decode(all_tokens[len(initial_prompt_tokens):])`.
    let full_text = tokenizer.decode(&all_tokens[initial_prompt_len..]);

    Ok(TranscribeResult {
        text: full_text,
        segments: all_segments,
        language,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_common as common;
    use serde_json::Value;

    /// Everything here is integer and float arithmetic — no MLX, no tokenizer
    /// assets, no model — so these run on a fresh clone with nothing fetched.
    ///
    /// Golden values come from `tests/fixtures/segmentation.json`, emitted by
    /// `tools/reference_segmentation.py`, which transcribes upstream's seek
    /// loop and imports nothing outside the standard library.
    fn fixture() -> Value {
        common::json("segmentation")
    }

    /// Objects in the fixture, in the order the file stores them.
    fn cases(section: &Value) -> Vec<(String, Value)> {
        section
            .as_object()
            .expect("fixture section should be an object")
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect()
    }

    fn f32_of(v: &Value) -> f32 {
        v.as_f64().expect("expected a number") as f32
    }

    fn opt_f32(v: &Value) -> Option<f32> {
        if v.is_null() { None } else { Some(f32_of(v)) }
    }

    /// `time_offset + pos * time_precision` is computed in `f32` here and in
    /// `f64` upstream, and 0.02 is not exactly representable in either. The
    /// gap is ~2e-5 even for the pathological negative-offset case, which sits
    /// three orders of magnitude away from every other value in the table.
    const TIME_TOL: f32 = 1e-3;

    fn assert_time(actual: f32, expected: f64, what: &str) {
        let d = (actual as f64 - expected).abs();
        assert!(
            d <= TIME_TOL as f64,
            "{}: {} != {} (diff {:.3e}, tol {:.3e})",
            what,
            actual,
            expected,
            d,
            TIME_TOL
        );
    }

    // ── Constants ────────────────────────────────────────────────────────

    #[test]
    fn fixture_constants_match_this_crate() {
        let f = fixture();
        let c = &f["constants"];
        assert_eq!(c["sample_rate"].as_u64().unwrap() as usize, SAMPLE_RATE);
        assert_eq!(c["hop_length"].as_u64().unwrap() as usize, HOP_LENGTH);
        assert_eq!(c["n_frames"].as_u64().unwrap() as usize, N_FRAMES);
        // The fixture's `input_stride` is `N_FRAMES / n_audio_ctx` for the
        // 1500-frame audio context every Whisper model uses.
        let input_stride = c["input_stride"].as_u64().unwrap() as usize;
        assert_eq!(input_stride, N_FRAMES / 1500);
        let time_precision = input_stride as f32 * HOP_LENGTH as f32 / SAMPLE_RATE as f32;
        assert_time(
            time_precision,
            c["time_precision"].as_f64().unwrap(),
            "time_precision",
        );
    }

    // ── split_window ─────────────────────────────────────────────────────

    #[test]
    fn split_window_matches_upstream() {
        let f = fixture();
        let ts_begin = f["constants"]["timestamp_begin"].as_u64().unwrap() as u32;
        let input_stride = f["constants"]["input_stride"].as_u64().unwrap() as usize;

        let all = cases(&f["split_window"]);
        assert!(!all.is_empty(), "fixture has no split_window cases");

        for (name, case) in &all {
            let tokens = common::as_u32s(&case["tokens"]);
            let seek = case["seek"].as_u64().unwrap() as usize;
            let segment_size = case["segment_size"].as_u64().unwrap() as usize;

            let got = split_window(&tokens, ts_begin, seek, segment_size, input_stride);

            let want = case["segments"].as_array().unwrap();
            assert_eq!(
                got.spans.len(),
                want.len(),
                "{name}: segment count (got {:?})",
                got.spans
            );

            for (i, (span, expected)) in got.spans.iter().zip(want).enumerate() {
                let at = format!("{name}[{i}]");
                // Integer offsets are compared exactly; only the seconds they
                // are multiplied into need a tolerance.
                assert_eq!(
                    span.start_pos,
                    expected["start_pos"].as_i64().unwrap(),
                    "{at}: start_pos"
                );
                assert_eq!(
                    span.end_pos,
                    expected["end_pos"].as_i64(),
                    "{at}: end_pos (None means the window duration was used)"
                );
                assert_eq!(
                    span.tokens,
                    common::as_u32s(&expected["tokens"]),
                    "{at}: tokens"
                );
                assert_time(span.start, expected["start"].as_f64().unwrap(), &format!("{at}: start"));
                assert_time(span.end, expected["end"].as_f64().unwrap(), &format!("{at}: end"));
            }

            assert_eq!(
                got.seek_delta,
                case["seek_delta"].as_u64().unwrap() as usize,
                "{name}: seek_delta"
            );
        }
    }

    /// The whole point of splitting a window is that `seek` lands somewhere
    /// specific afterwards; an off-by-one here silently re-reads or skips
    /// audio, and nothing downstream ever complains.
    #[test]
    fn split_window_advance_is_pinned_including_the_stall_case() {
        let f = fixture();
        let ts_begin = f["constants"]["timestamp_begin"].as_u64().unwrap() as u32;
        let input_stride = f["constants"]["input_stride"].as_u64().unwrap() as usize;

        let mut saw_stall = false;
        for (name, case) in cases(&f["split_window"]) {
            let tokens = common::as_u32s(&case["tokens"]);
            let seek = case["seek"].as_u64().unwrap() as usize;
            let segment_size = case["segment_size"].as_u64().unwrap() as usize;

            let split = split_window(&tokens, ts_begin, seek, segment_size, input_stride);
            // The guard as `transcribe()` applies it.
            let advance = if split.seek_delta == 0 {
                segment_size
            } else {
                split.seek_delta
            };
            assert_eq!(
                advance,
                case["seek_delta_port"].as_u64().unwrap() as usize,
                "{name}: advance after the anti-stall guard"
            );
            if split.seek_delta == 0 {
                saw_stall = true;
                assert_eq!(advance, segment_size, "{name}: the guard should fire");
            }
        }
        assert!(
            saw_stall,
            "no fixture case produces a zero advance, so the anti-stall guard \
             is untested — the `stall_zero_advance` case should"
        );
    }

    /// Upstream subtracts plain Python integers, so a slice that does not open
    /// on a timestamp yields a negative offset. Clamping it to zero, as this
    /// port used to, would move the segment boundary without any error.
    #[test]
    fn split_window_keeps_a_negative_start_offset() {
        let f = fixture();
        let ts_begin = f["constants"]["timestamp_begin"].as_u64().unwrap() as u32;
        let input_stride = f["constants"]["input_stride"].as_u64().unwrap() as usize;

        let case = &f["split_window"]["negative_start_offset"];
        let tokens = common::as_u32s(&case["tokens"]);
        let split = split_window(
            &tokens,
            ts_begin,
            case["seek"].as_u64().unwrap() as usize,
            case["segment_size"].as_u64().unwrap() as usize,
            input_stride,
        );

        let start_pos = split.spans[0].start_pos;
        assert!(
            start_pos < 0,
            "expected a negative offset, got {start_pos} — a saturating \
             subtraction would report 0 here"
        );
        assert_eq!(
            start_pos,
            case["segments"][0]["start_pos"].as_i64().unwrap()
        );
        assert!(split.spans[0].start < 0.0);
    }

    // ── Silence skip ─────────────────────────────────────────────────────

    #[test]
    fn silence_skip_matches_the_port_semantics() {
        let f = fixture();
        let mut divergent: Vec<String> = Vec::new();

        for (name, case) in cases(&f["silence_skip"]) {
            let got = should_skip_window(
                f32_of(&case["no_speech_prob"]),
                f32_of(&case["avg_logprob"]),
                opt_f32(&case["no_speech_threshold"]),
                opt_f32(&case["logprob_threshold"]),
            );
            assert_eq!(got, case["port"].as_bool().unwrap(), "{name}");
            if case["port"] != case["upstream"] {
                divergent.push(name);
            }
        }

        // The port folds upstream's "clear the skip" clause into a single `<`
        // comparison, which disagrees with upstream only when `avg_logprob`
        // lands exactly on the threshold. Pinned rather than fixed — see
        // `PARITY.md`. If this list ever grows, the fold has drifted into
        // something wider than a float-equality edge.
        assert_eq!(
            divergent,
            vec!["logprob_exactly_at_threshold".to_string()],
            "unexpected set of cases where the port and upstream disagree"
        );
    }

    // ── Temperature fallback ─────────────────────────────────────────────

    #[test]
    fn needs_fallback_matches_upstream() {
        let f = fixture();
        for (name, case) in cases(&f["needs_fallback"]) {
            let got = needs_fallback(
                f32_of(&case["compression_ratio"]),
                f32_of(&case["avg_logprob"]),
                f32_of(&case["no_speech_prob"]),
                opt_f32(&case["compression_ratio_threshold"]),
                opt_f32(&case["logprob_threshold"]),
                opt_f32(&case["no_speech_threshold"]),
            );
            assert_eq!(got, case["needs_fallback"].as_bool().unwrap(), "{name}");
        }
    }

    /// The clause order is the part that is easy to get wrong: a silent window
    /// must not climb the temperature ladder even when both quality gates fail,
    /// which only holds because the no-speech clause runs last.
    #[test]
    fn no_speech_clears_a_failing_quality_check() {
        assert!(needs_fallback(3.0, -2.0, 0.1, Some(2.4), Some(-1.0), Some(0.6)));
        assert!(!needs_fallback(3.0, -2.0, 0.9, Some(2.4), Some(-1.0), Some(0.6)));
    }

    // ── Segment text tokens ──────────────────────────────────────────────

    #[test]
    fn text_tokens_match_upstream() {
        let f = fixture();
        let eot = f["constants"]["eot"].as_u64().unwrap() as u32;
        for (name, case) in cases(&f["text_tokens"]) {
            let got = text_tokens(&common::as_u32s(&case["tokens"]), eot);
            assert_eq!(got, common::as_u32s(&case["text_tokens"]), "{name}");
        }
    }
}
