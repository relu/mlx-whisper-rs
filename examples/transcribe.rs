/// End-to-end Whisper transcription example.
///
/// Usage:
///   cargo run --example transcribe -- <audio_file> [options]
///
/// Options:
///   --model  <path_or_hf_id>      default: mlx-community/whisper-large-v3-turbo
///   --language <lang_code>        default: auto-detect  (e.g. en, zh, ja)
///   --task   <transcribe|translate>  default: transcribe
///   --assets <dir>                default: ./assets
///   --verbose                     print segments as they are decoded
///   --audio-ctx <N>               truncate encoder context to N positions (whisper.cpp -ac);
///                                 default: full context
///
/// Prerequisites:
///   • ffmpeg must be installed (brew install ffmpeg)
///   • assets/ must contain multilingual.tiktoken (and gpt2.tiktoken for EN-only models)
///     Copy from: $(python -c "import mlx_whisper; print(mlx_whisper.__path__[0])")/assets/
///
/// Example:
///   cargo run --example transcribe -- audio.wav --language zh --verbose

use std::path::{Path, PathBuf};
use std::time::Instant;

use mlx_whisper_rs::{
    audio::{load_audio, log_mel_spectrogram, N_SAMPLES, SAMPLE_RATE},
    load_models::load_model,
    tokenizer::LANGUAGES,
    transcribe::{detect_language, transcribe, TranscribeOptions},
};

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();

    if args.is_empty() || args.iter().any(|a| a == "--help" || a == "-h") {
        print_help();
        return Ok(());
    }

    // ── Parse arguments ───────────────────────────────────────────────────────
    let mut audio_file: Option<String> = None;
    let mut model_id = "mlx-community/whisper-large-v3-turbo".to_string();
    let mut language: Option<String> = None;
    let mut task = "transcribe".to_string();
    let mut assets_dir = PathBuf::from("assets");
    let mut verbose = false;
    let mut audio_ctx: Option<usize> = None;

    let mut i = 0usize;
    while i < args.len() {
        // A flag given as the last argument used to index past the end and
        // panic with "index out of bounds" instead of printing usage.
        let mut take_value = |i: &mut usize| -> anyhow::Result<String> {
            let flag = args[*i].clone();
            *i += 1;
            args.get(*i)
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("{flag} requires a value"))
        };

        match args[i].as_str() {
            "--model" => model_id = take_value(&mut i)?,
            "--language" | "--lang" => language = Some(take_value(&mut i)?),
            "--task" => task = take_value(&mut i)?,
            "--assets" => assets_dir = PathBuf::from(take_value(&mut i)?),
            "--verbose" | "-v" => verbose = true,
            "--audio-ctx" => {
                let v = take_value(&mut i)?;
                audio_ctx = Some(v.parse().map_err(|_| {
                    anyhow::anyhow!("--audio-ctx expects a positive integer, got {v:?}")
                })?);
            }
            // Anything that does not start with `-` is the positional audio
            // file. Matching on `!starts_with("--")` instead used to swallow a
            // mistyped short flag (`-x`) as the audio file, so the *real* file
            // that followed it tripped the duplicate-file error below.
            s if !s.starts_with('-') => {
                if audio_file.is_some() {
                    anyhow::bail!("Only one audio file may be given (got a second: {s})");
                }
                audio_file = Some(s.to_string());
            }
            other => eprintln!("Unknown option: {other}"),
        }
        i += 1;
    }

    let audio_file = audio_file.ok_or_else(|| {
        anyhow::anyhow!(
            "Audio file required\n\n\
             usage: transcribe <audio> [--model ID] [--language CODE] [--task transcribe|translate] \
             [--assets DIR] [--verbose] [--audio-ctx N]"
        )
    })?;

    // ── Validate assets directory ─────────────────────────────────────────────
    check_assets(&assets_dir)?;

    // ── Load model ────────────────────────────────────────────────────────────
    eprintln!("Loading model: {model_id}");
    let t0 = Instant::now();
    let mut model = load_model(&model_id)?;
    eprintln!("Model loaded in {:.1}s", t0.elapsed().as_secs_f32());

    // ── Load audio ────────────────────────────────────────────────────────────
    eprintln!("Loading audio: {audio_file}");
    let audio = load_audio(&audio_file, SAMPLE_RATE)?;
    let duration_s = audio.shape()[0] as f32 / SAMPLE_RATE as f32;
    eprintln!("Audio duration: {duration_s:.1}s");

    // ── Detect language (if not specified) ────────────────────────────────────
    if language.is_none() && model.is_multilingual() {
        eprintln!("Detecting language…");
        // `detect_language` takes a *mel spectrogram*, not a waveform; passing
        // `audio` straight through fails with "mel must be 2-D or 3-D, got
        // 1-D". Pad exactly as `transcribe` does so the 30-second window fed to
        // the encoder here is the same one it will use internally.
        let mel = log_mel_spectrogram(audio.clone(), model.dims.n_mels, &assets_dir, N_SAMPLES)?;
        let (lang, probs) = detect_language(&mut model, &mel, &assets_dir)?;
        let lang_name = LANGUAGES.iter()
            .find(|(c, _)| *c == lang)
            .map(|(_, n)| *n)
            .unwrap_or(&lang);
        let confidence = probs.get(&lang).copied().unwrap_or(0.0);
        eprintln!("Detected language: {} ({:.0}%)", lang_name, confidence * 100.0);
        language = Some(lang);
    }

    // ── Transcribe ────────────────────────────────────────────────────────────
    let opts = TranscribeOptions {
        language,
        task,
        verbose,
        audio_ctx,
        ..Default::default()
    };

    eprintln!("Transcribing…");
    let t1 = Instant::now();
    let result = transcribe(audio, &mut model, &assets_dir, &opts)?;
    let elapsed = t1.elapsed().as_secs_f32();
    eprintln!(
        "Done in {elapsed:.1}s  (RTF: {:.2}x)",
        duration_s / elapsed
    );

    // ── Print result ──────────────────────────────────────────────────────────
    println!();
    println!("Language: {}", result.language);
    println!();

    if !verbose {
        // Print segments with timestamps.
        //
        // Blank and instantaneous segments are kept in `result.segments` with
        // their text cleared, matching upstream's return value; it is the
        // presentation layer that skips them, as `mlx_whisper`'s writers do.
        for seg in result.segments.iter().filter(|s| !s.text.trim().is_empty()) {
            println!(
                "[{} --> {}]  {}",
                fmt_ts(seg.start),
                fmt_ts(seg.end),
                seg.text
            );
        }
        println!();
    }

    println!("{}", result.text);

    Ok(())
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn fmt_ts(seconds: f32) -> String {
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

fn check_assets(dir: &Path) -> anyhow::Result<()> {
    if !dir.exists() {
        anyhow::bail!(
            "Assets directory not found: {}\n\
             Create it and copy *.tiktoken files from the mlx_whisper Python package:\n\
             cp $(python -c \"import mlx_whisper, os; print(os.path.dirname(mlx_whisper.__file__))\")\\
                /assets/*.tiktoken {}",
            dir.display(),
            dir.display()
        );
    }
    for name in &["multilingual.tiktoken", "gpt2.tiktoken"] {
        let p = dir.join(name);
        if !p.exists() {
            eprintln!(
                "Warning: {} not found in assets/. \
                 Copy from mlx_whisper/assets/ if you encounter tokenizer errors.",
                name
            );
        }
    }
    Ok(())
}

fn print_help() {
    eprintln!(
        "mlx-whisper-rs — Whisper speech recognition on Apple Silicon\n\
         \n\
         Usage: transcribe <audio_file> [options]\n\
         \n\
         Options:\n\
           --model  <path|hf_id>   Model path or HuggingFace repo  [mlx-community/whisper-large-v3-turbo]\n\
           --language <code>       Language code (en, zh, ja, …)   [auto-detect]\n\
           --task   <task>         transcribe or translate          [transcribe]\n\
           --assets <dir>          Directory with *.tiktoken files  [./assets]\n\
           --verbose, -v           Print segments as decoded\n\
           --audio-ctx <N>         Truncate encoder context to N positions (whisper.cpp -ac)  [full]\n\
           --help,   -h            Show this message\n\
         \n\
         Prerequisites:\n\
           brew install ffmpeg\n\
           cp <mlx_whisper>/assets/*.tiktoken ./assets/\n"
    );
}
