// Translated from mlx-whisper/decoding.py (Apple Inc.)

use anyhow::Result;
use mlx_rs::{
    Array,
    ops::{indexing::argmax_axis, logsumexp_axis, max_axis, softmax_axis},
    ops::indexing::IndexOp,
    transforms::eval,
};

use crate::audio::CHUNK_LENGTH;
use crate::tokenizer::Tokenizer;
use crate::whisper::Whisper;

// ── Options & Result ─────────────────────────────────────────────────────────

#[derive(Clone)]
pub struct DecodingOptions {
    pub task: String,
    pub language: Option<String>,
    pub temperature: f32,
    /// Maximum number of tokens to sample; defaults to n_text_ctx / 2
    pub sample_len: Option<usize>,
    /// Suppress blank outputs at the first sampled position
    pub suppress_blank: bool,
    /// "-1" → use non_speech_tokens; comma-separated IDs; None → no suppression
    pub suppress_tokens: Option<String>,
    pub without_timestamps: bool,
    /// Maximum initial timestamp in seconds (default 1.0)
    pub max_initial_timestamp: Option<f32>,
    /// Previous context token IDs; prepended as prompt before the SOT sequence
    pub prompt: Option<Vec<u32>>,
}

impl Default for DecodingOptions {
    fn default() -> Self {
        Self {
            task: "transcribe".to_string(),
            language: None,
            temperature: 0.0,
            sample_len: None,
            suppress_blank: true,
            suppress_tokens: Some("-1".to_string()),
            without_timestamps: false,
            max_initial_timestamp: Some(1.0),
            prompt: None,
        }
    }
}

pub struct DecodingResult {
    pub language: String,
    pub tokens: Vec<u32>,
    pub text: String,
    pub avg_logprob: f32,
    pub no_speech_prob: f32,
    pub temperature: f32,
    pub compression_ratio: f32,
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// zlib compression ratio (raw bytes / compressed bytes), for hallucination detection.
fn compression_ratio(text: &str) -> f32 {
    use std::io::Write;
    let bytes = text.as_bytes();
    if bytes.is_empty() {
        return 0.0;
    }
    let mut enc = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::default());
    let _ = enc.write_all(bytes);
    match enc.finish() {
        Ok(compressed) if !compressed.is_empty() => bytes.len() as f32 / compressed.len() as f32,
        _ => 1.0,
    }
}

/// Build a [n_vocab] float32 mask: 0.0 for allowed tokens, NEG_INFINITY for suppressed.
fn build_mask(suppress_ids: &[u32], n_vocab: usize) -> Vec<f32> {
    let mut mask = vec![0.0f32; n_vocab];
    for &id in suppress_ids {
        if (id as usize) < n_vocab {
            mask[id as usize] = f32::NEG_INFINITY;
        }
    }
    mask
}

// ── Logit filters ─────────────────────────────────────────────────────────────

/// SuppressBlank: at the very first generated position, suppress blank/EOT tokens.
fn apply_suppress_blank(logits: &Array, tokens_len: usize, sample_begin: usize, mask: &Array) -> Array {
    if tokens_len == sample_begin {
        logits + mask
    } else {
        logits.clone()
    }
}

/// SuppressTokens: unconditionally suppress a set of token IDs.
fn apply_suppress_tokens(logits: &Array, mask: &Array) -> Array {
    logits + mask
}

/// ApplyTimestampRules: enforce timestamp token pairing and ordering constraints.
fn apply_timestamp_rules(
    logits: &Array,
    tokens: &[u32],
    sample_begin: usize,
    tokenizer: &Tokenizer,
    max_initial_timestamp_index: Option<usize>,
) -> Result<Array> {
    let n_vocab = logits.shape()[0] as usize;
    let ts_begin = tokenizer.timestamp_begin() as usize;
    let eot = tokenizer.eot() as usize;
    let no_ts = tokenizer.no_timestamps() as usize;

    let mut mask = vec![0.0f32; n_vocab];

    // Suppress <|notimestamps|> (handled by without_timestamps flag)
    if no_ts < n_vocab {
        mask[no_ts] = f32::NEG_INFINITY;
    }

    // Inspect the generated part of the sequence so far
    let seq: &[u32] = if tokens.len() > sample_begin { &tokens[sample_begin..] } else { &[] };
    let last_was_timestamp = seq.last().map_or(false, |&t| t as usize >= ts_begin);
    let penultimate_was_timestamp = seq.len() < 2 || seq[seq.len() - 2] as usize >= ts_begin;

    if last_was_timestamp {
        if penultimate_was_timestamp {
            // Two consecutive timestamps → next must be a non-timestamp
            for i in ts_begin..n_vocab { mask[i] = f32::NEG_INFINITY; }
        } else {
            // Last was timestamp after text → next cannot be a regular text token
            for i in 0..eot { mask[i] = f32::NEG_INFINITY; }
        }
    }

    // Timestamps must not decrease; also force non-zero segment length
    let timestamps: Vec<usize> = seq.iter()
        .filter(|&&t| t as usize > ts_begin)
        .map(|&t| t as usize)
        .collect();
    if !timestamps.is_empty() {
        let mut last_ts = *timestamps.last().unwrap();
        if !last_was_timestamp || penultimate_was_timestamp {
            last_ts += 1;
        }
        for i in ts_begin..last_ts.min(n_vocab) {
            mask[i] = f32::NEG_INFINITY;
        }
    }

    // At the very beginning of generation, force a timestamp token
    if tokens.len() == sample_begin {
        for i in 0..ts_begin { mask[i] = f32::NEG_INFINITY; }
        if let Some(max_idx) = max_initial_timestamp_index {
            let last_allowed = ts_begin + max_idx;
            for i in (last_allowed + 1)..n_vocab { mask[i] = f32::NEG_INFINITY; }
        }
    }

    let mask_arr = Array::from_slice(&mask, &[n_vocab as i32]);
    let logits_masked = logits + &mask_arr;

    // If total timestamp logprob > max text token logprob → suppress text tokens
    if ts_begin > 0 && ts_begin < n_vocab {
        let logprobs = &logits_masked - &logsumexp_axis(&logits_masked, 0, true)?;
        // Sum prob of all timestamps
        let ts_lp = logsumexp_axis(logprobs.index((ts_begin as i32..,)), 0, false)?;
        // Max prob of any text token
        let max_text_lp = max_axis(logprobs.index((..ts_begin as i32,)), 0, false)?;
        eval([&ts_lp, &max_text_lp])?;
        if ts_lp.item::<f32>() > max_text_lp.item::<f32>() {
            for i in 0..ts_begin { mask[i] = f32::NEG_INFINITY; }
            let final_mask = Array::from_slice(&mask, &[n_vocab as i32]);
            return Ok(logits + &final_mask);
        }
    }

    Ok(logits_masked)
}

// ── Token selection ───────────────────────────────────────────────────────────

/// Greedy (temp=0) or sampled (temp>0) token selection; accumulates log-probability.
fn select_next_token(
    logits: &Array,
    temperature: f32,
    sum_logprobs: &mut f32,
    tokens: &[u32],
) -> Result<u32> {
    let eot = tokens.first().copied().unwrap_or(0); // placeholder — not used here

    let next_arr = if temperature == 0.0 {
        argmax_axis(logits, 0, None)?
    } else {
        mlx_rs::random::categorical(logits * (1.0 / temperature), None, None, None)?
    };
    eval([&next_arr])?;
    let next_token = next_arr.item::<u32>();

    // Accumulate log-probability (skip if last real token was EOT)
    let last = tokens.last().copied().unwrap_or(0);
    let _ = eot; // unused above
    if last != tokens[0] || tokens.len() == 1 {
        // Always accumulate — the Python check is: multiply by (last != eot)
        // We do it unconditionally since our loop stops at EOT anyway.
    }
    let logprobs = logits - &logsumexp_axis(logits, 0, true)?;
    let lp = logprobs.index((next_token as i32,));
    eval([&lp])?;
    *sum_logprobs += lp.item::<f32>();

    Ok(next_token)
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Decode a 30-second mel spectrogram chunk.
///
/// # Parameters
/// - `mel`: 2D `[n_frames, n_mels]` or 3D `[1, n_frames, n_mels]` array
/// - `tokenizer`: must be configured with the correct language and task
/// - `options`: decoding hyperparameters
pub fn decode(
    model: &mut Whisper,
    mel: &Array,
    tokenizer: &Tokenizer,
    options: &DecodingOptions,
) -> Result<DecodingResult> {
    let n_vocab = model.dims.n_vocab;
    let n_audio_ctx = model.dims.n_audio_ctx;
    let n_text_ctx = model.dims.n_text_ctx;

    // ── Encoder forward pass ────────────────────────────────────────────────
    // mel expected as [n_frames, n_mels]; reshape to [1, n_frames, n_mels] for batch
    let mel_batch = if mel.ndim() == 2 {
        mel.reshape(&[1, mel.shape()[0], mel.shape()[1]])?
    } else {
        mel.clone()
    };
    let audio_features = model.encoder.forward(&mel_batch)?; // [1, n_audio_ctx, n_audio_state]

    // ── Initial token sequence ──────────────────────────────────────────────
    let mut sot_seq = tokenizer.sot_sequence.clone();
    if options.without_timestamps {
        sot_seq.push(tokenizer.no_timestamps());
    }

    // Prepend previous-context prompt: [sot_prev, ...prompt..., sot, lang, task]
    if let Some(ref prompt) = options.prompt {
        if !prompt.is_empty() {
            if let Some(&sot_prev) = tokenizer.special_tokens.get("<|startofprev|>") {
                let max_prompt = n_text_ctx / 2 - 1;
                let start = prompt.len().saturating_sub(max_prompt);
                let mut full = vec![sot_prev];
                full.extend_from_slice(&prompt[start..]);
                full.extend(sot_seq);
                sot_seq = full;
            }
        }
    }

    let sample_begin = sot_seq.len();
    let sot_index = sot_seq.iter().position(|&t| t == tokenizer.sot()).unwrap_or(0);
    let sample_len = options.sample_len.unwrap_or(n_text_ctx / 2);

    // ── Build logit filter masks ────────────────────────────────────────────

    // SuppressBlank: suppress space tokens and EOT at the first generated position
    let blank_mask_arr: Option<Array> = if options.suppress_blank {
        let space_ids = tokenizer.encode(" ");
        let eot_id = tokenizer.eot();
        let mut ids: Vec<u32> = space_ids;
        ids.push(eot_id);
        Some(Array::from_slice(&build_mask(&ids, n_vocab), &[n_vocab as i32]))
    } else {
        None
    };

    // SuppressTokens: always-on token suppression
    let suppress_mask_arr: Array = {
        let mut suppress_ids: Vec<u32> = Vec::new();

        if let Some(ref s) = options.suppress_tokens {
            let parsed: Vec<i64> = s.split(',')
                .filter_map(|x| x.trim().parse().ok())
                .collect();
            if parsed.contains(&-1) {
                suppress_ids.extend(tokenizer.non_speech_tokens());
            } else {
                suppress_ids.extend(parsed.iter().filter(|&&t| t >= 0).map(|&t| t as u32));
            }
        }

        // Always suppress these control tokens
        suppress_ids.push(tokenizer.transcribe_token());
        suppress_ids.push(tokenizer.translate_token());
        suppress_ids.push(tokenizer.sot());
        suppress_ids.push(tokenizer.no_speech());
        if let Some(&id) = tokenizer.special_tokens.get("<|startofprev|>") {
            suppress_ids.push(id);
        }
        if let Some(&id) = tokenizer.special_tokens.get("<|startoflm|>") {
            suppress_ids.push(id);
        }

        Array::from_slice(&build_mask(&suppress_ids, n_vocab), &[n_vocab as i32])
    };

    // max_initial_timestamp_index: how many 0.02s steps the initial timestamp may be
    let max_ts_index: Option<usize> = if options.without_timestamps {
        None
    } else {
        options.max_initial_timestamp.map(|t| {
            let precision = CHUNK_LENGTH as f32 / n_audio_ctx as f32; // typically 0.02s
            (t / precision).round() as usize
        })
    };

    // ── Decode loop ─────────────────────────────────────────────────────────
    let mut tokens: Vec<u32> = sot_seq.clone();
    let mut sum_logprobs = 0.0f32;

    // Helper: apply all logit filters
    macro_rules! apply_filters {
        ($logits:expr) => {{
            let logits = $logits;
            let logits = match &blank_mask_arr {
                Some(mask) => apply_suppress_blank(&logits, tokens.len(), sample_begin, mask),
                None => logits,
            };
            let logits = apply_suppress_tokens(&logits, &suppress_mask_arr);
            let logits = if !options.without_timestamps {
                apply_timestamp_rules(&logits, &tokens, sample_begin, tokenizer, max_ts_index)?
            } else {
                logits
            };
            logits
        }};
    }

    // ── Step 1: feed all initial tokens, extract no_speech_prob ────────────
    let (no_speech_prob, mut kv_cache) = {
        let tokens_i32: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
        let tokens_arr = Array::from_slice(&tokens_i32, &[1, tokens.len() as i32]);
        let (pre_logits, new_cache, _) = model.decoder.forward(&tokens_arr, &audio_features, None)?;

        // [1, seq_len, n_vocab] → [seq_len, n_vocab]
        let l2d = pre_logits.index((0i32,));

        // no_speech_prob from the SOT position
        let probs_sot = softmax_axis(l2d.index((sot_index as i32,)), 0, false)?;
        let ns_arr = probs_sot.index((tokenizer.no_speech() as i32,));
        eval([&ns_arr])?;
        let ns_prob = ns_arr.item::<f32>();

        // Logits at last position → apply filters → select token
        let logits = apply_filters!(l2d.index((-1i32,)));
        let next = select_next_token(&logits, options.temperature, &mut sum_logprobs, &tokens)?;
        tokens.push(next);

        (ns_prob, Some(new_cache))
    };

    // ── Steps 2+: feed only the last token with KV cache ───────────────────
    for _ in 1..sample_len {
        if tokens.len() > n_text_ctx {
            break;
        }
        if *tokens.last().unwrap() == tokenizer.eot() {
            break;
        }

        let last_i32 = *tokens.last().unwrap() as i32;
        let last_arr = Array::from_slice(&[last_i32], &[1, 1]);
        let (pre_logits, new_cache, _) =
            model.decoder.forward(&last_arr, &audio_features, kv_cache)?;
        kv_cache = Some(new_cache);

        // [1, 1, n_vocab] → [n_vocab]
        let logits = pre_logits.index((0i32,)).index((0i32,));
        let logits = apply_filters!(logits);
        let next = select_next_token(&logits, options.temperature, &mut sum_logprobs, &tokens)?;
        tokens.push(next);
    }

    // ── Post-process ────────────────────────────────────────────────────────
    let eot = tokenizer.eot();
    let generated = &tokens[sample_begin..];
    let end = generated.iter().position(|&t| t == eot).unwrap_or(generated.len());
    let result_tokens: Vec<u32> = generated[..end].to_vec();

    let text = tokenizer.decode(&result_tokens).trim().to_string();
    let avg_logprob = if result_tokens.is_empty() {
        f32::NEG_INFINITY
    } else {
        sum_logprobs / (result_tokens.len() + 1) as f32
    };

    Ok(DecodingResult {
        language: tokenizer.language.clone().unwrap_or_else(|| "en".to_string()),
        tokens: result_tokens,
        text: text.clone(),
        avg_logprob,
        no_speech_prob,
        temperature: options.temperature,
        compression_ratio: compression_ratio(&text),
    })
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer::get_tokenizer;
    use std::path::{Path, PathBuf};

    /// Real Whisper multilingual vocabulary size (large-v3 family).
    const N_VOCAB: usize = 51866;

    fn fixtures() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
    }

    /// Tokenizer assets are ~1.6 MB and are not committed; tests that need real
    /// BPE skip themselves rather than fail on a fresh clone.
    /// Run `python3 tools/extract_assets.py` to populate `assets/`.
    fn tokenizer() -> Option<Tokenizer> {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("assets");
        if !dir.join("multilingual.tiktoken").exists() {
            eprintln!("SKIP: run `python3 tools/extract_assets.py` to populate assets/");
            return None;
        }
        get_tokenizer(true, 99, Some("en"), Some("transcribe"), &dir).ok()
    }

    /// The same LCG stream `tools/gen_fixtures.py` uses, so Rust and Python
    /// filter the identical logits without committing a 400 KiB blob.
    fn synth_logits_row0(n_vocab: usize, ts_begin: usize) -> Vec<f32> {
        let mut all = Vec::with_capacity(n_vocab);
        let mut state: u64 = 12345;
        for _ in 0..n_vocab {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let u = (state >> 40) as f32 / (1u32 << 24) as f32;
            all.push(u * 16.0 - 8.0);
        }
        // Row 0 is the "text dominant" variant: pushing the timestamp region
        // down keeps the final logsumexp rule from firing, so the earlier
        // clauses stay observable instead of being buried under an all-`-inf`
        // row.
        for v in all.iter_mut().skip(ts_begin) {
            *v -= 12.0;
        }
        all
    }

    fn neg_inf_indices(row: &[f32]) -> Vec<usize> {
        row.iter()
            .enumerate()
            .filter(|(_, v)| v.is_infinite() && v.is_sign_negative())
            .map(|(i, _)| i)
            .collect()
    }

    // ── compression_ratio ────────────────────────────────────────────────

    #[test]
    fn compression_ratio_empty_is_zero() {
        // Upstream computes len(text_bytes) / len(zlib.compress(...)); for the
        // empty string that is 0 / 8 == 0.0. It must not be 1.0 or NaN, since
        // `transcribe` compares it against compression_ratio_threshold.
        assert_eq!(compression_ratio(""), 0.0);
    }

    #[test]
    fn compression_ratio_grows_with_redundancy() {
        let varied = compression_ratio("the quick brown fox jumps over the lazy dog");
        let repetitive = compression_ratio(&"a".repeat(200));
        assert!(
            repetitive > varied,
            "a highly compressible string must score higher ({repetitive} vs {varied})"
        );
        assert!(varied > 0.0);
    }

    #[test]
    fn compression_ratio_matches_cpython_zlib() {
        // Pinned from CPython's zlib at the default level, via
        // tools/gen_fixtures.py -> tests/fixtures/text_metrics.json.
        //
        // NOTE: `flate2`'s default backend is miniz_oxide, which is *not*
        // byte-identical to C zlib at the same level. If this test fails by a
        // few bytes of compressed length, the fix is to build flate2 against
        // zlib (`features = ["zlib"]`) rather than to loosen the tolerance —
        // the 2.4 hallucination threshold is a cliff, and a few bytes either
        // way flips real segments.
        let text = std::fs::read_to_string(fixtures().join("text_metrics.json"))
            .expect("fixture text_metrics.json");
        let doc: serde_json::Value = serde_json::from_str(&text).unwrap();

        for case in doc["compression_ratio"].as_array().unwrap() {
            let s = case["text"].as_str().unwrap();
            let Some(want) = case["ratio"].as_f64() else {
                continue; // the empty-string case, covered above
            };
            let got = compression_ratio(s) as f64;
            assert!(
                (got - want).abs() < 1e-6,
                "compression_ratio({s:?}) = {got}, upstream zlib gives {want}"
            );
        }
    }

    // ── SuppressBlank ────────────────────────────────────────────────────

    #[test]
    fn suppress_blank_applies_only_at_sample_begin() {
        let logits = Array::from_slice(&vec![0.0f32; 16], &[16]);
        let mask = Array::from_slice(&build_mask(&[3, 7], 16), &[16]);

        let at = apply_suppress_blank(&logits, 5, 5, &mask);
        let at_slice: &[f32] = at.as_slice();
        assert_eq!(neg_inf_indices(at_slice), vec![3, 7]);

        for tokens_len in [4usize, 6] {
            let off = apply_suppress_blank(&logits, tokens_len, 5, &mask);
            let off_slice: &[f32] = off.as_slice();
            assert!(
                neg_inf_indices(off_slice).is_empty(),
                "SuppressBlank must be inert at tokens_len {tokens_len} (sample_begin 5)"
            );
        }
    }

    #[test]
    fn suppress_blank_mask_is_space_plus_eot() {
        let Some(tk) = tokenizer() else { return };
        // Upstream: mask[tokenizer.encode(" ") + [tokenizer.eot]] = -inf
        let mut want = tk.encode(" ");
        want.push(tk.eot());
        want.sort_unstable();

        let mask = build_mask(&want, N_VOCAB);
        let got: Vec<usize> = mask
            .iter()
            .enumerate()
            .filter(|(_, v)| v.is_infinite())
            .map(|(i, _)| i)
            .collect();
        assert_eq!(got, want.iter().map(|&t| t as usize).collect::<Vec<_>>());
    }

    // ── ApplyTimestampRules ──────────────────────────────────────────────

    #[test]
    fn timestamp_rules_always_suppress_no_timestamps() {
        let Some(tk) = tokenizer() else { return };
        let ts_begin = tk.timestamp_begin() as usize;
        let logits = Array::from_slice(&synth_logits_row0(N_VOCAB, ts_begin), &[N_VOCAB as i32]);
        let sot = tk.sot_sequence.clone();
        let sample_begin = sot.len();

        // At the initial position, after a text token, and after a timestamp.
        let text_tok = tk.encode(" hello")[0];
        let seqs: Vec<Vec<u32>> = vec![
            sot.clone(),
            {
                let mut s = sot.clone();
                s.push(ts_begin as u32 + 10);
                s.push(text_tok);
                s
            },
            {
                let mut s = sot.clone();
                s.push(ts_begin as u32 + 10);
                s
            },
        ];

        for seq in seqs {
            let out = apply_timestamp_rules(&logits, &seq, sample_begin, &tk, None).unwrap();
            let row: &[f32] = out.as_slice();
            assert!(
                row[tk.no_timestamps() as usize].is_infinite(),
                "<|notimestamps|> must be suppressed at every step (seq len {})",
                seq.len()
            );
        }
    }

    #[test]
    fn timestamp_rules_force_a_timestamp_at_the_initial_position() {
        let Some(tk) = tokenizer() else { return };
        let ts_begin = tk.timestamp_begin() as usize;
        let logits = Array::from_slice(&synth_logits_row0(N_VOCAB, ts_begin), &[N_VOCAB as i32]);
        let sot = tk.sot_sequence.clone();

        let out = apply_timestamp_rules(&logits, &sot, sot.len(), &tk, None).unwrap();
        let row: &[f32] = out.as_slice();

        for i in 0..ts_begin {
            assert!(
                row[i].is_infinite(),
                "every non-timestamp token must be suppressed at the initial position (index {i})"
            );
        }
        assert!(
            row[ts_begin].is_finite(),
            "the timestamp region must stay open at the initial position"
        );
    }

    #[test]
    fn timestamp_rules_max_initial_timestamp_boundary_is_inclusive() {
        let Some(tk) = tokenizer() else { return };
        let ts_begin = tk.timestamp_begin() as usize;
        let logits = Array::from_slice(&synth_logits_row0(N_VOCAB, ts_begin), &[N_VOCAB as i32]);
        let sot = tk.sot_sequence.clone();
        let max_idx = 50usize; // round(1.0 / (30 / 1500))

        let out = apply_timestamp_rules(&logits, &sot, sot.len(), &tk, Some(max_idx)).unwrap();
        let row: &[f32] = out.as_slice();

        assert!(
            row[ts_begin + max_idx].is_finite(),
            "ts_begin + max_initial_timestamp_index must remain ALLOWED (inclusive bound)"
        );
        assert!(
            row[ts_begin + max_idx + 1].is_infinite(),
            "one past the bound must be suppressed"
        );
    }

    #[test]
    fn timestamp_rules_pair_constraint_after_two_timestamps() {
        let Some(tk) = tokenizer() else { return };
        let ts_begin = tk.timestamp_begin() as usize;
        let logits = Array::from_slice(&synth_logits_row0(N_VOCAB, ts_begin), &[N_VOCAB as i32]);

        // ... <ts> <ts>  => the next token must NOT be a timestamp.
        let mut seq = tk.sot_sequence.clone();
        let sample_begin = seq.len();
        seq.push(ts_begin as u32 + 10);
        seq.push(ts_begin as u32 + 20);

        let out = apply_timestamp_rules(&logits, &seq, sample_begin, &tk, None).unwrap();
        let row: &[f32] = out.as_slice();
        for i in ts_begin..N_VOCAB {
            assert!(
                row[i].is_infinite(),
                "after two consecutive timestamps the whole timestamp region must be closed (index {i})"
            );
        }
    }

    #[test]
    fn timestamp_rules_text_is_closed_after_a_lone_timestamp() {
        let Some(tk) = tokenizer() else { return };
        let ts_begin = tk.timestamp_begin() as usize;
        let eot = tk.eot() as usize;
        let logits = Array::from_slice(&synth_logits_row0(N_VOCAB, ts_begin), &[N_VOCAB as i32]);

        // ... <text> <ts>  => the next token must NOT be ordinary text, but EOT
        // stays reachable (upstream masks `[..eot]`, not `[..=eot]`).
        let mut seq = tk.sot_sequence.clone();
        let sample_begin = seq.len();
        seq.push(tk.encode(" hello")[0]);
        seq.push(ts_begin as u32 + 10);

        let out = apply_timestamp_rules(&logits, &seq, sample_begin, &tk, None).unwrap();
        let row: &[f32] = out.as_slice();

        for i in 0..eot {
            assert!(row[i].is_infinite(), "text token {i} must be closed");
        }
        assert!(
            row[eot].is_finite(),
            "EOT must remain reachable directly after a timestamp"
        );
    }

    /// Timestamps must not decrease.
    ///
    /// This clause is where `mlx_whisper` 0.4.3 and `openai/whisper` disagree,
    /// and where this port follows **openai/whisper**. Upstream mlx collects
    /// list *indices* (`[i for i, v in enumerate(seq) if v > ts_begin]`) where
    /// openai collects token *values*, so upstream's mask range
    /// `[ts_begin : last_timestamp]` is empty and the constraint never fires
    /// at all. See `tests/fixtures/README.md`.
    ///
    /// If bug-for-bug parity with `mlx_whisper` is ever chosen over
    /// correctness, this is the test to invert.
    #[test]
    fn timestamp_rules_enforce_monotonic_timestamps() {
        let Some(tk) = tokenizer() else { return };
        let ts_begin = tk.timestamp_begin() as usize;
        let logits = Array::from_slice(&synth_logits_row0(N_VOCAB, ts_begin), &[N_VOCAB as i32]);
        let text_tok = tk.encode(" hello")[0];

        // ... <ts+50> <text>  => a timestamp below ts+50 must be closed.
        let mut seq = tk.sot_sequence.clone();
        let sample_begin = seq.len();
        seq.push(ts_begin as u32 + 50);
        seq.push(text_tok);

        let out = apply_timestamp_rules(&logits, &seq, sample_begin, &tk, None).unwrap();
        let row: &[f32] = out.as_slice();

        assert!(
            row[ts_begin + 10].is_infinite(),
            "a timestamp earlier than the last one must be suppressed"
        );
        assert!(
            row[ts_begin + 51].is_finite(),
            "a later timestamp must stay allowed"
        );
        // The `+1` bump forces a non-zero segment length: repeating the last
        // timestamp is not allowed when the previous token was text.
        assert!(
            row[ts_begin + 50].is_infinite(),
            "repeating the last timestamp would produce a zero-length segment"
        );
    }

    /// The final "sample a timestamp if their total probability beats every
    /// text token" rule must normalise the **unmasked** logits.
    ///
    /// Python (`decoding.py:383-394`) computes `logprobs` from the logits as
    /// they arrived — carrying only SuppressBlank/SuppressTokens — not from
    /// this filter's own mask. Normalising the masked logits instead changes
    /// both the `logsumexp` denominator and which entries are still finite,
    /// so the `ts_lp > max_text_lp` comparison can flip.
    ///
    /// This case is built to flip it deterministically. After `[text, ts]` the
    /// pairing rule masks `[0, eot)`, so:
    ///
    ///   * unmasked — one very strong text token at index 100 dominates, so
    ///     `max_text_lp ~= 0` and `ts_lp ~= -12.7`: the rule does NOT fire and
    ///     EOT stays reachable;
    ///   * masked — index 100 is now `-inf`, `max_text_lp` drops to ~-17.3
    ///     while `ts_lp ~= 0`: the rule fires and closes EOT too.
    ///
    /// So `logits[eot]` being finite is exactly the discriminator.
    #[test]
    fn timestamp_rules_normalise_unmasked_logits() {
        let Some(tk) = tokenizer() else { return };
        let ts_begin = tk.timestamp_begin() as usize;
        let eot = tk.eot() as usize;

        let mut v = vec![-10.0f32; N_VOCAB];
        v[100] = 20.0; // a text token below eot, dominant before masking
        for x in v.iter_mut().skip(ts_begin) {
            *x = 0.0; // ~1502 timestamps, individually weak but numerous
        }
        let logits = Array::from_slice(&v, &[N_VOCAB as i32]);

        let mut seq = tk.sot_sequence.clone();
        let sample_begin = seq.len();
        seq.push(tk.encode(" hello")[0]);
        seq.push(ts_begin as u32 + 10);

        let out = apply_timestamp_rules(&logits, &seq, sample_begin, &tk, None).unwrap();
        let row: &[f32] = out.as_slice();

        assert!(
            row[eot].is_finite(),
            "the timestamp-vs-text rule was evaluated on the masked logits: with the \
             dominant text token already suppressed by the pairing rule, the rule fires \
             and closes EOT, which upstream never does here"
        );
    }

    // ── select_next_token ────────────────────────────────────────────────

    #[test]
    fn greedy_selection_never_picks_a_suppressed_token() {
        let mut v = vec![0.0f32; 32];
        v[7] = 10.0; // would win outright...
        let mut logits = v.clone();
        logits[7] = f32::NEG_INFINITY; // ...but is suppressed
        logits[3] = 5.0;

        let arr = Array::from_slice(&logits, &[32]);
        let mut sum_logprobs = 0.0f32;
        // `tokens` is the sequence so far; non-empty because select_next_token
        // indexes tokens[0] unconditionally.
        let tokens = [1u32, 2, 3];
        let tok = select_next_token(&arr, 0.0, &mut sum_logprobs, &tokens).unwrap();

        assert_eq!(tok, 3, "greedy selection must skip -inf entries");
        assert!(
            sum_logprobs < 0.0 && sum_logprobs.is_finite(),
            "accumulated logprob should be finite and negative, got {sum_logprobs}"
        );
    }

    #[test]
    fn greedy_logprob_is_log_softmax_of_the_filtered_logits() {
        // Two live tokens with equal logits => each has probability 0.5, so the
        // accumulated logprob must be ln(0.5).
        let mut logits = vec![f32::NEG_INFINITY; 8];
        logits[2] = 1.0;
        logits[5] = 1.0;

        let arr = Array::from_slice(&logits, &[8]);
        let mut sum_logprobs = 0.0f32;
        let tokens = [1u32, 2, 3];
        let _ = select_next_token(&arr, 0.0, &mut sum_logprobs, &tokens).unwrap();

        let want = 0.5f32.ln();
        assert!(
            (sum_logprobs - want).abs() < 1e-5,
            "expected ln(0.5) = {want}, got {sum_logprobs}"
        );
    }
}
