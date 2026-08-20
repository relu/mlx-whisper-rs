#!/usr/bin/env python3
"""Generate golden fixtures from the upstream Python `mlx_whisper` package.

These fixtures pin the Rust port's numerics to the reference implementation.
Regenerate them only when deliberately re-pinning to a new upstream version;
the pinned version is recorded in `tests/fixtures/manifest.json`.

Usage:

    python3 -m venv venv
    ./venv/bin/pip install mlx mlx-whisper numpy
    ./venv/bin/python tools/gen_fixtures.py

On Linux, MLX needs its CPU backend (`pip install mlx-cpu`); the fixtures are
device-independent, so generating them on Linux CPU is fine.

Outputs land in `tests/fixtures/`:

  * `*.f32`   raw little-endian float32 arrays (row-major)
  * `*.json`  shapes, integer token data, and scalar expectations
"""

import importlib.metadata as md
import json
import os
import pathlib
import sys
import zlib

import numpy as np
import mlx.core as mx

from mlx_whisper import audio as A
from mlx_whisper import decoding as D
from mlx_whisper import tokenizer as T
from mlx_whisper.whisper import sinusoids

OUT = pathlib.Path(__file__).resolve().parent.parent / "tests" / "fixtures"
OUT.mkdir(parents=True, exist_ok=True)


def write_f32(name: str, arr) -> dict:
    """Write a raw float32 array and return its descriptor."""
    a = np.ascontiguousarray(np.array(arr, dtype=np.float32))
    (OUT / f"{name}.f32").write_bytes(a.tobytes(order="C"))
    return {"file": f"{name}.f32", "shape": list(a.shape), "count": int(a.size)}


def write_json(name: str, obj) -> None:
    (OUT / f"{name}.json").write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


# --------------------------------------------------------------------------
# Deterministic synthetic inputs (no audio files, no model weights required)
# --------------------------------------------------------------------------

SR = A.SAMPLE_RATE


def synth_audio() -> np.ndarray:
    """2s @16k: 1s of 440 Hz at 0.5 amplitude, then 1s of silence.

    Chosen so the mel output exercises both a strong tonal band and the
    `max(log_spec, log_spec.max() - 8.0)` dynamic-range floor.
    """
    t = np.arange(SR, dtype=np.float32) / SR
    tone = (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    return np.concatenate([tone, np.zeros(SR, dtype=np.float32)]).astype(np.float32)


def synth_logits(rows: int, n_vocab: int, seed: int) -> np.ndarray:
    """Reproducible pseudo-random logits, generated identically in Rust.

    Uses a plain 64-bit LCG rather than numpy's Generator so the Rust test can
    reconstruct the exact same input without depending on an RNG crate.
    """
    out = np.empty(rows * n_vocab, dtype=np.float32)
    state = np.uint64(seed)
    mul = np.uint64(6364136223846793005)
    inc = np.uint64(1442695040888963407)
    with np.errstate(over="ignore"):
        for i in range(rows * n_vocab):
            state = np.uint64(state * mul + inc)
            # take the top 24 bits -> [0,1), then map to [-8, 8)
            u = float(np.uint64(state) >> np.uint64(40)) / float(1 << 24)
            out[i] = np.float32(u * 16.0 - 8.0)
    return out.reshape(rows, n_vocab)


# --------------------------------------------------------------------------
# Fixture groups
# --------------------------------------------------------------------------


def gen_audio() -> dict:
    sig = synth_audio()
    # The input itself is not committed: it is fully described by the formula
    # below and regenerated bit-identically in the Rust test.
    desc = {
        "input": {
            "kind": "tone440_then_silence",
            "samples": int(sig.size),
            "formula": "s[i] = 0.5*sin(2*pi*440*i/16000) for i in 0..16000, then 0.0 for i in 16000..32000",
            "checksum_sum": float(np.float64(sig).sum()),
            "checksum_abs_sum": float(np.abs(np.float64(sig)).sum()),
        }
    }

    for n_mels in (80, 128):
        mel = A.log_mel_spectrogram(mx.array(sig), n_mels=n_mels)
        mx.eval(mel)
        desc[f"mel_{n_mels}"] = write_f32(f"mel_{n_mels}", np.array(mel))

    # Hann window: catches periodic-vs-symmetric, the single most common
    # STFT porting error.
    desc["hanning_400"] = write_f32("hanning_400", np.hanning(A.N_FFT + 1)[:-1])

    # The mel filterbanks, unpacked from upstream's single `mel_filters.npz`
    # into the per-n_mels `.npy` layout this crate's loader expects. Writing
    # them as real .npy files means the tests exercise `mel_filters()` and the
    # hand-rolled .npy reader, not just the arithmetic downstream of them.
    filters = {}
    for n_mels in (80, 128):
        f = A.mel_filters(n_mels)
        mx.eval(f)
        arr = np.ascontiguousarray(np.array(f, dtype=np.float32))
        np.save(OUT / f"mel_filters_{n_mels}.npy", arr, allow_pickle=False)
        filters[f"mel_filters_{n_mels}"] = {
            "file": f"mel_filters_{n_mels}.npy",
            "shape": list(arr.shape),
            "sum": float(np.float64(arr).sum()),
        }
    desc["filters"] = filters

    # pad_or_trim cases, expressed as lengths so Rust can assert shape + content.
    desc["pad_or_trim"] = [
        {"in_len": 100, "target": 100, "out_len": 100},
        {"in_len": 100, "target": 250, "out_len": 250, "pads_with": 0.0},
        {"in_len": 400, "target": 250, "out_len": 250, "keeps": "prefix"},
    ]

    desc["constants"] = {
        "SAMPLE_RATE": A.SAMPLE_RATE,
        "N_FFT": A.N_FFT,
        "HOP_LENGTH": A.HOP_LENGTH,
        "CHUNK_LENGTH": A.CHUNK_LENGTH,
        "N_SAMPLES": A.N_SAMPLES,
        "N_FRAMES": A.N_FRAMES,
        "N_SAMPLES_PER_TOKEN": A.N_SAMPLES_PER_TOKEN,
        "FRAMES_PER_SECOND": A.FRAMES_PER_SECOND,
        "TOKENS_PER_SECOND": A.TOKENS_PER_SECOND,
    }
    return desc


def gen_tokenizer() -> dict:
    desc = {"n_languages_in_table": len(T.LANGUAGES)}
    desc["languages"] = [list(x) for x in T.LANGUAGES.items()]

    variants = {}
    for label, kwargs in {
        "multilingual_99_en_transcribe": dict(
            multilingual=True, num_languages=99, language="en", task="transcribe"
        ),
        "multilingual_100_en_transcribe": dict(
            multilingual=True, num_languages=100, language="en", task="transcribe"
        ),
        "multilingual_99_zh_translate": dict(
            multilingual=True, num_languages=99, language="zh", task="translate"
        ),
        "gpt2_en_transcribe": dict(
            multilingual=False, num_languages=99, language="en", task="transcribe"
        ),
    }.items():
        tk = T.get_tokenizer(**kwargs)
        v = {
            "kwargs": {k: val for k, val in kwargs.items()},
            "eot": tk.eot,
            "sot": tk.sot,
            "sot_prev": tk.sot_prev,
            "sot_lm": tk.sot_lm,
            "no_speech": tk.no_speech,
            "no_timestamps": tk.no_timestamps,
            "timestamp_begin": tk.timestamp_begin,
            "transcribe": tk.transcribe,
            "translate": tk.translate,
            "sot_sequence": list(tk.sot_sequence),
            "sot_sequence_including_notimestamps": list(
                tk.sot_sequence_including_notimestamps
            ),
            "non_speech_tokens": sorted(tk.non_speech_tokens),
            "encode_space": tk.encode(" "),
            "all_language_tokens_len": len(tk.all_language_tokens),
            "all_language_tokens_first8": list(tk.all_language_tokens[:8]),
            "all_language_codes_first8": list(tk.all_language_codes[:8]),
        }
        try:
            v["language_token"] = tk.language_token
        except Exception as e:  # pragma: no cover - only for gpt2 variant
            v["language_token_error"] = type(e).__name__
        variants[label] = v
    desc["variants"] = variants

    # Round-trip encode/decode samples across scripts, to catch BPE bugs.
    tk = T.get_tokenizer(
        multilingual=True, num_languages=99, language="en", task="transcribe"
    )
    samples = [
        # The tiktoken `pat_str` pre-tokenization regex splits contractions
        # before BPE runs. A raw-bytes BPE without that step merges "'T" into
        # one token; these three cases pin the difference.
        " DON'T STOP",
        " IT'S A TEST",
        " ALL CAPS SENTENCE WITH THE'S",
        # Controls that must keep passing once the regex is added.
        " Hello world, how are you today?",
        " 100,000,000",
        "Hello, world!",
        " leading space",
        "trailing space ",
        "中文測試",
        "Ünïcödé — em dash & emoji 🎧",
        "",
        "  ",
        "123 456.789",
        "\n\ttabs and newlines\n",
    ]
    desc["encode_decode"] = [
        {"text": s, "tokens": tk.encode(s), "decoded": tk.decode(tk.encode(s))}
        for s in samples
    ]

    # decode_with_timestamps over a synthetic token stream.
    ts = tk.timestamp_begin
    stream = [ts + 0] + tk.encode(" hello") + [ts + 50] + tk.encode(" world") + [ts + 100]
    desc["decode_with_timestamps"] = {
        "tokens": stream,
        "text": tk.decode_with_timestamps(stream),
        "plain": tk.decode(stream),
    }

    # split_to_word_tokens on both a spaced and an unspaced language.
    for lang, text in (("en", " hello there world"), ("zh", "今天天氣很好")):
        tkl = T.get_tokenizer(
            multilingual=True, num_languages=99, language=lang, task="transcribe"
        )
        toks = tkl.encode(text)
        words, wtoks = tkl.split_to_word_tokens(toks)
        desc.setdefault("split_to_word_tokens", {})[lang] = {
            "text": text,
            "tokens": toks,
            "words": words,
            "word_tokens": wtoks,
        }
    return desc


def gen_logit_filters() -> dict:
    """Drive upstream's LogitFilter classes on synthetic logits.

    Two logit rows are used, both derived from the same LCG stream so Rust can
    regenerate them without committing a 400 KiB blob:

      row 0 ("text_dominant")  timestamp region shifted DOWN, so the final
                               "sum of timestamp probability beats every text
                               token" rule does NOT fire;
      row 1 ("ts_dominant")    timestamp region shifted UP, so it DOES fire.

    Without this split the rule is vacuous on uniform random logits: the
    logsumexp over ~1500 timestamp logits almost always beats the single best
    text logit, every token ends up -inf, and the fixture stops discriminating.
    """
    tk = T.get_tokenizer(
        multilingual=True, num_languages=99, language="en", task="transcribe"
    )
    # Real Whisper multilingual vocab size.
    n_vocab = 51866
    sample_begin = len(tk.sot_sequence)
    ts_begin = tk.timestamp_begin

    desc = {
        "n_vocab": n_vocab,
        "sample_begin": sample_begin,
        "timestamp_begin": ts_begin,
        "eot": tk.eot,
        "no_timestamps": tk.no_timestamps,
        "lcg_seed": 12345,
        "lcg_note": (
            "state: u64 = seed; repeat: state = state*6364136223846793005 "
            "+ 1442695040888963407 (wrapping); value = ((state >> 40) as f32 "
            "/ 2^24) * 16.0 - 8.0. Fill row-major over [2, n_vocab]."
        ),
        "row_shift_note": (
            "after filling: row 0 gets -12.0 added to indices >= timestamp_begin; "
            "row 1 gets +6.0 added to indices >= timestamp_begin."
        ),
        "row_labels": ["text_dominant", "ts_dominant"],
    }

    logits_np = synth_logits(2, n_vocab, 12345)
    logits_np[0, ts_begin:] -= np.float32(12.0)
    logits_np[1, ts_begin:] += np.float32(6.0)

    # Record the discriminator the last rule keys on, per row.
    lg = mx.array(logits_np)
    lp = lg - mx.logsumexp(lg, axis=-1, keepdims=True)
    ts_lp = lp[:, ts_begin:].logsumexp(axis=-1, keepdims=True)
    tx_lp = lp[:, :ts_begin].max(axis=-1, keepdims=True)
    mx.eval(ts_lp, tx_lp)
    desc["timestamp_vs_text"] = [
        {
            "row": i,
            "timestamp_logprob": float(np.array(ts_lp)[i, 0]),
            "max_text_token_logprob": float(np.array(tx_lp)[i, 0]),
            "rule_fires": bool(np.array(ts_lp)[i, 0] > np.array(tx_lp)[i, 0]),
        }
        for i in range(2)
    ]

    # --- SuppressBlank -----------------------------------------------------
    sb = D.SuppressBlank(tk, sample_begin, n_vocab)
    desc["suppress_blank"] = {
        "suppressed_tokens": sorted(tk.encode(" ") + [tk.eot]),
        "applies_only_when_len_tokens_eq_sample_begin": True,
    }

    # --- SuppressTokens ----------------------------------------------------
    # Reproduce upstream's _get_suppress_tokens for the "-1" sentinel.
    def get_suppress_tokens(suppress_tokens):
        if isinstance(suppress_tokens, str):
            suppress_tokens = [int(t) for t in suppress_tokens.split(",")]
        if -1 in suppress_tokens:
            suppress_tokens = [t for t in suppress_tokens if t >= 0]
            suppress_tokens.extend(tk.non_speech_tokens)
        elif suppress_tokens is None or len(suppress_tokens) == 0:
            suppress_tokens = []
        else:
            assert isinstance(suppress_tokens, list)
        suppress_tokens.extend(
            [tk.transcribe, tk.translate, tk.sot, tk.sot_prev, tk.sot_lm]
        )
        if tk.no_speech is not None:
            suppress_tokens.append(tk.no_speech)
        return tuple(sorted(set(suppress_tokens)))

    desc["suppress_tokens"] = {
        "sentinel_minus1": list(get_suppress_tokens("-1")),
        "explicit_list": list(get_suppress_tokens([1, 2, 3])),
        "empty": list(get_suppress_tokens([])),
    }

    # --- ApplyTimestampRules ----------------------------------------------
    ts = tk.timestamp_begin
    text_tok = tk.encode(" hello")[0]
    cases = {
        # at sample_begin: only timestamps allowed, capped by max_initial_timestamp
        "initial_no_cap": (list(tk.sot_sequence), None),
        "initial_cap_50": (list(tk.sot_sequence), 50),
        # last token was a timestamp, penultimate was not -> must be text
        "after_single_timestamp": (list(tk.sot_sequence) + [ts + 10], None),
        # two timestamps in a row -> must be text next
        "after_paired_timestamps": (
            list(tk.sot_sequence) + [ts + 10, text_tok, ts + 30],
            None,
        ),
        # monotonicity: next timestamp must be >= last
        "monotonic": (
            list(tk.sot_sequence) + [ts + 10, text_tok, ts + 30, text_tok],
            None,
        ),
        # zero-length-segment guard: last_timestamp incremented when it is 0
        "zero_timestamp": (list(tk.sot_sequence) + [ts + 0, text_tok], None),
    }

    out_cases = {}
    for name, (toks, cap) in cases.items():
        rule = D.ApplyTimestampRules(tk, sample_begin, cap)
        tokens = mx.array([toks, toks])
        res = rule.apply(mx.array(logits_np), tokens)
        mx.eval(res)
        r = np.array(res)
        # Store, per row, the set of -inf'd indices (compressed to ranges) plus
        # a checksum of the finite part — the full 51866-wide rows would be
        # 400 KiB per case.
        rows = []
        for i in range(r.shape[0]):
            row = r[i]
            neg = np.flatnonzero(np.isneginf(row)).astype(np.int64)
            finite = row[np.isfinite(row)]
            rows.append(
                {
                    "row": i,
                    "label": desc["row_labels"][i],
                    "n_suppressed": int(neg.size),
                    "suppressed_ranges": ranges(neg),
                    "finite_count": int(finite.size),
                    "finite_sum": float(np.float64(finite).sum()),
                }
            )
        out_cases[name] = {
            "tokens": toks,
            "max_initial_timestamp_index": cap,
            "rows": rows,
        }
    desc["timestamp_rules"] = out_cases

    # ------------------------------------------------------------------
    # The same cases under openai/whisper's semantics.
    #
    # mlx_whisper 0.4.3's monotonicity clause diverges from openai/whisper in
    # three ways, and the combination makes the clause a silent no-op:
    #
    #   openai:  timestamps = sampled_tokens[sampled_tokens.ge(ts_begin)]
    #            -> collects token VALUES (>= ts_begin)
    #   mlx:     timestamps = [i for i, v in enumerate(seq) if v > ts_begin]
    #            -> collects INDICES, and with a strict > comparison
    #
    #   openai:  if last_was_timestamp and not penultimate_was_timestamp:
    #                timestamp_last = timestamps[-1]
    #            else:
    #                timestamp_last = timestamps[-1] + 1
    #   mlx:     if not last_timestamp or penultimate_was_timestamp:
    #                last_timestamp += 1
    #            (tests the numeric `last_timestamp`, not `last_was_timestamp`)
    #
    # Because mlx's `last_timestamp` is a position within `seq` (a small
    # integer) rather than a token id, the resulting slice
    # `mask[k, ts_begin : last_timestamp]` is empty whenever that position is
    # below ts_begin — i.e. essentially always. The "timestamps shouldn't
    # decrease" constraint therefore never fires in mlx_whisper.
    #
    # Both targets are emitted so the Rust port can pin to whichever the
    # project decides on. See tests/fixtures/README.md.
    # ------------------------------------------------------------------
    def openai_rules(toks, cap, logits):
        mask = np.zeros(logits.shape, np.float32)
        mask[:, tk.no_timestamps] = -np.inf
        for k in range(logits.shape[0]):
            seq = toks[sample_begin:]
            last_was_timestamp = len(seq) >= 1 and seq[-1] >= ts_begin
            penultimate_was_timestamp = len(seq) < 2 or seq[-2] >= ts_begin
            if last_was_timestamp:
                if penultimate_was_timestamp:
                    mask[k, ts_begin:] = -np.inf
                else:
                    mask[k, : tk.eot] = -np.inf
            timestamps = [v for v in seq if v >= ts_begin]
            if timestamps:
                if last_was_timestamp and not penultimate_was_timestamp:
                    timestamp_last = timestamps[-1]
                else:
                    timestamp_last = timestamps[-1] + 1
                mask[k, ts_begin:timestamp_last] = -np.inf
        if len(toks) == sample_begin:
            mask[:, :ts_begin] = -np.inf
            if cap is not None:
                mask[:, ts_begin + cap + 1 :] = -np.inf
        m = mx.array(mask)
        lgm = mx.array(logits) + m
        lgp = lgm - mx.logsumexp(lgm, axis=-1, keepdims=True)
        t_lp = lgp[:, ts_begin:].logsumexp(axis=-1, keepdims=True)
        x_lp = lgp[:, :ts_begin].max(axis=-1, keepdims=True)
        m[:, :ts_begin] = mx.where(t_lp > x_lp, -float("inf"), m[:, :ts_begin])
        res = mx.array(logits) + m
        mx.eval(res)
        return np.array(res)

    oa_cases = {}
    for name, (toks, cap) in cases.items():
        r = openai_rules(toks, cap, logits_np)
        rows = []
        for i in range(r.shape[0]):
            row = r[i]
            neg = np.flatnonzero(np.isneginf(row)).astype(np.int64)
            finite = row[np.isfinite(row)]
            rows.append(
                {
                    "row": i,
                    "label": desc["row_labels"][i],
                    "n_suppressed": int(neg.size),
                    "suppressed_ranges": ranges(neg),
                    "finite_count": int(finite.size),
                    "finite_sum": float(np.float64(finite).sum()),
                }
            )
        oa_cases[name] = {
            "tokens": toks,
            "max_initial_timestamp_index": cap,
            "rows": rows,
        }
    desc["timestamp_rules_openai_semantics"] = oa_cases
    return desc


def ranges(idx: np.ndarray):
    """Compress a sorted index array into [start, end_inclusive] pairs."""
    out = []
    if idx.size == 0:
        return out
    start = prev = int(idx[0])
    for v in map(int, idx[1:]):
        if v == prev + 1:
            prev = v
            continue
        out.append([start, prev])
        start = prev = v
    out.append([start, prev])
    return out


def gen_model_math() -> dict:
    desc = {}
    # Small case, fully committed.
    s = sinusoids(10, 8)
    mx.eval(s)
    desc["sinusoids_10x8"] = write_f32("sinusoids_10x8", np.array(s))
    # Real encoder shape, committed as a checksum + probe values (1500*384
    # floats would be 2.3 MB).
    big = sinusoids(1500, 384)
    mx.eval(big)
    b = np.array(big)
    desc["sinusoids_1500x384"] = {
        "shape": list(b.shape),
        "sum": float(np.float64(b).sum()),
        "abs_sum": float(np.abs(np.float64(b)).sum()),
        "probes": [
            {"row": r, "col": c, "value": float(b[r, c])}
            for r, c in [(0, 0), (0, 191), (0, 192), (1, 0), (749, 100), (1499, 383)]
        ],
    }
    return desc


def gen_text_metrics() -> dict:
    """compression_ratio, which gates the temperature fallback."""
    samples = [
        "",
        "hello",
        "the quick brown fox jumps over the lazy dog",
        "a" * 200,
        "repeat repeat repeat repeat repeat repeat repeat repeat",
        "中文中文中文中文中文中文中文中文中文中文",
    ]
    out = []
    for s in samples:
        b = s.encode("utf-8")
        if len(b) == 0:
            out.append({"text": s, "utf8_len": 0, "ratio": None, "note": "div-by-zero"})
            continue
        out.append(
            {
                "text": s,
                "utf8_len": len(b),
                "zlib_len": len(zlib.compress(b)),
                "ratio": len(b) / len(zlib.compress(b)),
            }
        )
    return {"compression_ratio": out, "zlib_level": "default (-1 / level 6)"}


def main() -> int:
    manifest = {
        "generated_by": "tools/gen_fixtures.py",
        "upstream": {
            "mlx_whisper": md.version("mlx-whisper"),
            "mlx": md.version("mlx"),
            "numpy": np.__version__,
        },
        "note": "Regenerate only when deliberately re-pinning to a new upstream version.",
    }

    write_json("audio", gen_audio())
    write_json("tokenizer", gen_tokenizer())
    write_json("logit_filters", gen_logit_filters())
    write_json("model_math", gen_model_math())
    write_json("text_metrics", gen_text_metrics())
    write_json("manifest", manifest)

    total = sum(p.stat().st_size for p in OUT.iterdir())
    print(f"wrote {len(list(OUT.iterdir()))} files, {total/1024:.0f} KiB -> {OUT}")
    print(json.dumps(manifest["upstream"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
