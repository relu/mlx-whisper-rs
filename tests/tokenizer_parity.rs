//! Parity tests for `tokenizer`.
//!
//! Special-token ids are **positional**: they are assigned by their order in
//! the special-token list appended after `<|endoftext|>`, and that list's
//! length depends on `num_languages`. A single misordered or missing entry
//! shifts every later id, which silently corrupts the sot sequence, the
//! timestamp base, and the suppression sets. These tests pin the whole layout.
//!
//! Tests needing real BPE skip themselves when `assets/` is unpopulated; run
//! `python3 tools/extract_assets.py` first.

mod common;

use mlx_whisper_rs::tokenizer::{get_tokenizer, LANGUAGES};

#[test]
fn language_table_matches_upstream() {
    common::init_device();
    let fx = common::json("tokenizer");
    let want = fx["languages"].as_array().unwrap();

    assert_eq!(
        LANGUAGES.len(),
        fx["n_languages_in_table"].as_u64().unwrap() as usize,
        "LANGUAGES length drives every language token id"
    );
    assert_eq!(LANGUAGES.len(), want.len());

    // Order matters as much as membership: the language token for entry `i`
    // is `sot + 1 + i`.
    for (i, (code, name)) in LANGUAGES.iter().enumerate() {
        let w_code = want[i][0].as_str().unwrap();
        let w_name = want[i][1].as_str().unwrap();
        assert_eq!(*code, w_code, "LANGUAGES[{i}] code");
        assert_eq!(*name, w_name, "LANGUAGES[{i}] name");
    }
}

/// Table-driven over every tokenizer variant captured in the fixtures.
#[test]
fn special_token_layout_matches_upstream() {
    common::init_device();
    let Some(assets) = common::require_assets() else {
        return;
    };
    let fx = common::json("tokenizer");

    for (label, v) in fx["variants"].as_object().unwrap() {
        let kw = &v["kwargs"];
        let multilingual = kw["multilingual"].as_bool().unwrap();
        let num_languages = kw["num_languages"].as_u64().unwrap() as usize;
        let language = kw["language"].as_str();
        let task = kw["task"].as_str();

        let tk = match get_tokenizer(multilingual, num_languages, language, task, &assets) {
            Ok(t) => t,
            Err(e) => panic!("get_tokenizer({label}) failed: {e}"),
        };

        let checks: [(&str, u32); 6] = [
            ("eot", tk.eot()),
            ("sot", tk.sot()),
            ("no_speech", tk.no_speech()),
            ("no_timestamps", tk.no_timestamps()),
            ("timestamp_begin", tk.timestamp_begin()),
            ("transcribe", tk.transcribe_token()),
        ];
        for (field, got) in checks {
            let want = v[field].as_u64().unwrap() as u32;
            assert_eq!(got, want, "{label}: {field}");
        }

        assert_eq!(
            tk.sot_sequence,
            common::as_u32s(&v["sot_sequence"]),
            "{label}: sot_sequence"
        );
        assert_eq!(
            tk.sot_sequence_including_notimestamps(),
            common::as_u32s(&v["sot_sequence_including_notimestamps"]),
            "{label}: sot_sequence_including_notimestamps"
        );
    }
}

#[test]
fn non_speech_tokens_match_upstream() {
    common::init_device();
    let Some(assets) = common::require_assets() else {
        return;
    };
    let fx = common::json("tokenizer");
    let v = &fx["variants"]["multilingual_99_en_transcribe"];

    let tk = get_tokenizer(true, 99, Some("en"), Some("transcribe"), &assets)
        .expect("get_tokenizer");

    let mut got = tk.non_speech_tokens();
    got.sort_unstable();
    got.dedup();

    let want = common::as_u32s(&v["non_speech_tokens"]);
    assert_eq!(
        got.len(),
        want.len(),
        "non_speech_tokens count: got {}, want {}",
        got.len(),
        want.len()
    );
    assert_eq!(got, want, "non_speech_tokens set");
}

#[test]
fn all_language_tokens_are_contiguous_from_sot() {
    common::init_device();
    let Some(assets) = common::require_assets() else {
        return;
    };
    let fx = common::json("tokenizer");
    let v = &fx["variants"]["multilingual_99_en_transcribe"];

    let tk = get_tokenizer(true, 99, Some("en"), Some("transcribe"), &assets)
        .expect("get_tokenizer");

    let toks = tk.all_language_tokens();
    assert_eq!(
        toks.len(),
        v["all_language_tokens_len"].as_u64().unwrap() as usize,
        "all_language_tokens length must equal num_languages"
    );
    assert_eq!(
        toks.iter().take(8).cloned().collect::<Vec<_>>(),
        common::as_u32s(&v["all_language_tokens_first8"]),
        "all_language_tokens prefix"
    );

    let codes = tk.all_language_codes();
    let want_codes: Vec<String> = v["all_language_codes_first8"]
        .as_array()
        .unwrap()
        .iter()
        .map(|s| s.as_str().unwrap().to_string())
        .collect();
    assert_eq!(codes.iter().take(8).cloned().collect::<Vec<_>>(), want_codes);
}

#[test]
fn encode_decode_round_trips_match_upstream() {
    common::init_device();
    let Some(assets) = common::require_assets() else {
        return;
    };
    let tk = get_tokenizer(true, 99, Some("en"), Some("transcribe"), &assets)
        .expect("get_tokenizer");

    for case in common::json("tokenizer")["encode_decode"].as_array().unwrap() {
        let text = case["text"].as_str().unwrap();
        let want_tokens = common::as_u32s(&case["tokens"]);
        let want_decoded = case["decoded"].as_str().unwrap();

        let got = tk.encode(text);
        assert_eq!(got, want_tokens, "encode({text:?})");
        assert_eq!(tk.decode(&got), want_decoded, "decode(encode({text:?}))");
    }
}

#[test]
fn decode_with_timestamps_matches_upstream() {
    common::init_device();
    let Some(assets) = common::require_assets() else {
        return;
    };
    let tk = get_tokenizer(true, 99, Some("en"), Some("transcribe"), &assets)
        .expect("get_tokenizer");

    let doc = common::json("tokenizer");
    let case = &doc["decode_with_timestamps"];
    let tokens = common::as_u32s(&case["tokens"]);

    assert_eq!(
        tk.decode_with_timestamps(&tokens),
        case["text"].as_str().unwrap(),
        "decode_with_timestamps"
    );
    assert_eq!(
        tk.decode(&tokens),
        case["plain"].as_str().unwrap(),
        "decode must strip timestamp tokens"
    );
}

#[test]
fn split_to_word_tokens_matches_upstream() {
    common::init_device();
    let Some(assets) = common::require_assets() else {
        return;
    };
    let fx = common::json("tokenizer");

    // "en" goes through the space-splitting path, "zh" through the
    // unicode-splitting path — both are exercised.
    for (lang, case) in fx["split_to_word_tokens"].as_object().unwrap() {
        let tk = get_tokenizer(true, 99, Some(lang), Some("transcribe"), &assets)
            .expect("get_tokenizer");
        let tokens = common::as_u32s(&case["tokens"]);

        let (words, word_tokens) = tk.split_to_word_tokens(&tokens);
        let want_words: Vec<String> = case["words"]
            .as_array()
            .unwrap()
            .iter()
            .map(|s| s.as_str().unwrap().to_string())
            .collect();
        assert_eq!(words, want_words, "split_to_word_tokens({lang}) words");

        let want_tokens: Vec<Vec<u32>> = case["word_tokens"]
            .as_array()
            .unwrap()
            .iter()
            .map(common::as_u32s)
            .collect();
        assert_eq!(
            word_tokens, want_tokens,
            "split_to_word_tokens({lang}) token groups"
        );
    }
}

#[test]
fn language_lookup_accepts_codes_and_names() {
    common::init_device();
    let Some(assets) = common::require_assets() else {
        return;
    };
    // Upstream accepts both "en" and "english", case-insensitively.
    for spelling in ["en", "EN", "english", "English"] {
        let tk = get_tokenizer(true, 99, Some(spelling), Some("transcribe"), &assets)
            .unwrap_or_else(|e| panic!("get_tokenizer({spelling}) failed: {e}"));
        assert_eq!(
            tk.language.as_deref(),
            Some("en"),
            "{spelling} should normalise to the code \"en\""
        );
    }

    assert!(
        get_tokenizer(true, 99, Some("klingon"), Some("transcribe"), &assets).is_err(),
        "an unknown language must be rejected, not silently defaulted"
    );
}

/// A language outside the model's `num_languages` window must be rejected.
///
/// Upstream truncates the table first — `tuple(LANGUAGES.keys())[:num_languages]`
/// — so `langs.index(language)` raises `ValueError` for anything past the
/// window. `Tokenizer::new` used to search the full 100-entry `LANGUAGES` and
/// fall back to `unwrap_or(0)`.
///
/// The concrete failure that motivated this test: `yue` is index 99, so for a
/// 99-language model the port emitted `sot + 1 + 99 == 50358`, which is
/// `<|translate|>` — a task token silently standing in for a language token.
/// Any unknown code likewise collapsed to English rather than erroring.
#[test]
fn language_outside_num_languages_window_is_rejected() {
    common::init_device();
    let Some(assets) = common::require_assets() else {
        return;
    };
    let fx = common::json("tokenizer");
    let v = &fx["variants"]["multilingual_99_en_transcribe"];
    let translate = v["translate"].as_u64().unwrap() as u32;

    match get_tokenizer(true, 99, Some("yue"), Some("transcribe"), &assets) {
        Err(_) => {} // correct: upstream raises here
        Ok(tk) => {
            let lang_token = tk.sot_sequence[1];
            assert_ne!(
                lang_token, translate,
                "\"yue\" is language index 99, outside a 99-language model's window; \
                 emitting <|translate|> ({translate}) as its language token corrupts \
                 the sot sequence"
            );
            panic!(
                "get_tokenizer(num_languages=99, language=\"yue\") should have failed; \
                 it returned sot_sequence {:?}",
                tk.sot_sequence
            );
        }
    }
}

/// Every language code must round-trip to the token id derived from its index.
#[test]
fn every_language_code_maps_to_its_positional_token() {
    common::init_device();
    let Some(assets) = common::require_assets() else {
        return;
    };
    let fx = common::json("tokenizer");
    let sot = fx["variants"]["multilingual_100_en_transcribe"]["sot"]
        .as_u64()
        .unwrap() as u32;

    // A 100-language model covers the whole table, so every code is in-window.
    let tk = get_tokenizer(true, 100, Some("en"), Some("transcribe"), &assets)
        .expect("get_tokenizer");

    for (i, (code, _)) in LANGUAGES.iter().enumerate() {
        let want = sot + 1 + i as u32;
        let got = tk
            .to_language_token(code)
            .unwrap_or_else(|e| panic!("to_language_token({code}) failed: {e}"));
        assert_eq!(got, want, "language token for {code:?} (table index {i})");
    }
}
