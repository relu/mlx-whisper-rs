"""Reference implementation of the `transcribe` seek loop's window arithmetic.

`src/transcribe.rs` reaches its densest logic — carving a decoded window into
segments, and deciding how far `seek` advances — only with real model weights,
which no test has. Extracting that arithmetic into pure functions on the Rust
side makes it reachable; this module is the Python counterpart those functions
are pinned against, and `tests/fixtures/segmentation.json` is what it emits.

Transcribed clause by clause from `mlx_whisper/transcribe.py` at the version
recorded in `tests/fixtures/manifest.json` (0.4.3), specifically the
`while seek < seek_clip_end` body. Two reductions are deliberate:

* **`clip_timestamps` is fixed at its default.** Upstream's default `"0"` makes
  `seek_clips == [(0, content_frames)]`, so `seek_clip_end` is `content_frames`
  and `segment_size = min(N_FRAMES, content_frames - seek, seek_clip_end - seek)`
  collapses to `min(N_FRAMES, content_frames - seek)`. The port implements only
  that whole-file case, so `segment_size` is an input here rather than derived.
* **`word_timestamps` is off**, as it is in the port, so `new_segment`'s
  `words` key and the anomaly-scoring block are omitted.

Like `reference_rules.py` this imports nothing beyond the standard library — no
`mlx`, no `numpy` — so the fixture can be regenerated on any machine, not only
on the Apple Silicon box where `gen_fixtures.py` can run:

    python3 tools/reference_segmentation.py

`gen_fixtures.py` calls `build_fixture()` directly so the two paths cannot
produce different bytes.
"""

import json
import pathlib
import sys

# ── Constants (mlx_whisper/audio.py) ─────────────────────────────────────────

SAMPLE_RATE = 16000
HOP_LENGTH = 160
N_FRAMES = 3000

# The real multilingual layout: `<|0.00|>` sits at 50364 and `<|endoftext|>` at
# 50257. Hardcoded here only to build fixture inputs — the port derives both
# positionally from the tokenizer.
TS_BEGIN = 50364
EOT = 50257
INPUT_STRIDE = 2  # N_FRAMES // n_audio_ctx, i.e. 3000 // 1500


# ── The seek-loop body ───────────────────────────────────────────────────────


def split_window(tokens, ts_begin, seek, segment_size, input_stride):
    """Carve one decoded window into segments and compute the seek advance.

    Returns `(segments, seek_delta)`. `seek_delta` is upstream's raw advance:
    it can be `0`, which is upstream's infinite-loop case and where the port
    deliberately differs (see `anti_stall` in the emitted fixture).
    """
    time_offset = float(seek * HOP_LENGTH / SAMPLE_RATE)
    time_precision = input_stride * HOP_LENGTH / SAMPLE_RATE
    segment_duration = segment_size * HOP_LENGTH / SAMPLE_RATE

    timestamp_tokens = [t >= ts_begin for t in tokens]
    # `timestamp_tokens[-2:].tolist() == [False, True]` — note this is False for
    # a window of fewer than two tokens, because the slice is then too short to
    # compare equal.
    single_timestamp_ending = timestamp_tokens[-2:] == [False, True]

    # `np.where(timestamp_tokens[:-1] & timestamp_tokens[1:])[0] + 1`
    consecutive = [
        i + 1
        for i in range(len(timestamp_tokens) - 1)
        if timestamp_tokens[i] and timestamp_tokens[i + 1]
    ]

    segments = []

    if len(consecutive) > 0:
        slices = list(consecutive)
        if single_timestamp_ending:
            slices.append(len(tokens))

        last_slice = 0
        for current_slice in slices:
            sliced_tokens = tokens[last_slice:current_slice]
            # Upstream has no empty-slice guard and would raise IndexError here.
            # It is unreachable — `consecutive` is strictly increasing and every
            # entry is >= 1 — but the port guards it, so mirror the guard rather
            # than crashing the generator if a case ever hits it.
            if not sliced_tokens:
                last_slice = current_slice
                continue
            # Plain integer subtraction: a slice that does not open on a
            # timestamp yields a *negative* offset, which is exactly what the
            # `negative_start_offset` case below pins.
            start_timestamp_pos = sliced_tokens[0] - ts_begin
            end_timestamp_pos = sliced_tokens[-1] - ts_begin
            segments.append(
                {
                    "start_pos": start_timestamp_pos,
                    "end_pos": end_timestamp_pos,
                    "start": time_offset + start_timestamp_pos * time_precision,
                    "end": time_offset + end_timestamp_pos * time_precision,
                    "tokens": list(sliced_tokens),
                }
            )
            last_slice = current_slice

        if single_timestamp_ending:
            # A single timestamp at the end means no speech after it.
            seek_delta = segment_size
        else:
            # Otherwise drop the unfinished tail and seek to the last timestamp.
            last_timestamp_pos = tokens[last_slice - 1] - ts_begin
            seek_delta = last_timestamp_pos * input_stride
    else:
        duration = segment_duration
        end_pos = None
        timestamps = [t for t in tokens if t >= ts_begin]
        if len(timestamps) > 0 and timestamps[-1] != ts_begin:
            # No consecutive timestamps but there is one; use the last.
            end_pos = timestamps[-1] - ts_begin
            duration = end_pos * time_precision

        segments.append(
            {
                # The whole window: it opens at the window origin, so its start
                # offset is 0 by construction rather than read off a token.
                "start_pos": 0,
                # `None` means "no bounding timestamp — `end` came from the
                # window duration", which is a distinct case from `end_pos: 0`.
                "end_pos": end_pos,
                "start": time_offset,
                "end": time_offset + duration,
                "tokens": list(tokens),
            }
        )
        seek_delta = segment_size

    return segments, seek_delta


def should_skip_window(no_speech_prob, avg_logprob, no_speech_threshold, logprob_threshold):
    """Upstream's no-voice-activity check, verbatim."""
    if no_speech_threshold is None:
        return False
    should_skip = no_speech_prob > no_speech_threshold
    if logprob_threshold is not None and avg_logprob > logprob_threshold:
        # Don't skip if the logprob is high enough, despite the no_speech_prob.
        should_skip = False
    return should_skip


def should_skip_window_port(
    no_speech_prob, avg_logprob, no_speech_threshold, logprob_threshold
):
    """The port's version of the same check.

    `src/transcribe.rs` folds upstream's "clear the skip when the logprob is
    high enough" into a single expression that tests `avg_logprob < threshold`
    rather than negating `avg_logprob > threshold`. The two agree everywhere
    except at exact float equality, where upstream skips and the port does not.
    Recorded in `PARITY.md` under the unnumbered transcribe RISKs; emitted here
    so the divergence is pinned rather than latent.
    """
    if no_speech_threshold is None:
        return False
    return no_speech_prob > no_speech_threshold and (
        logprob_threshold is None or avg_logprob < logprob_threshold
    )


def needs_fallback(
    compression_ratio,
    avg_logprob,
    no_speech_prob,
    compression_ratio_threshold,
    logprob_threshold,
    no_speech_threshold,
):
    """Upstream's `decode_with_fallback` quality gate, verbatim.

    The ordering matters: the no-speech clause runs last and *clears* the flag,
    so a silent window is never retried at a higher temperature no matter how
    badly it scored on the other two.
    """
    needs = False
    if compression_ratio_threshold is not None and compression_ratio > compression_ratio_threshold:
        needs = True  # too repetitive
    if logprob_threshold is not None and avg_logprob < logprob_threshold:
        needs = True  # average log probability is too low
    if no_speech_threshold is not None and no_speech_prob > no_speech_threshold:
        needs = False  # silence
    return needs


def text_tokens(tokens, eot):
    """`new_segment`'s `[token for token in tokens if token < tokenizer.eot]`."""
    return [t for t in tokens if t < eot]


# ── Cases ────────────────────────────────────────────────────────────────────

# `(tokens, seek, segment_size)`. Timestamp ids are `TS_BEGIN + n`, so
# `TS_BEGIN + 50` is `<|1.00|>` at 0.02 s precision.
T = TS_BEGIN
SPLIT_CASES = {
    # No tokens at all: upstream still emits one whole-window segment, which
    # the caller then clears as blank.
    "empty": ([], 0, N_FRAMES),
    "single_text_token": ([1000], 0, N_FRAMES),
    "no_timestamps": ([1000, 2000, 3000], 0, N_FRAMES),
    # A lone trailing timestamp with no pair: no consecutive pair, so the
    # whole-window branch runs and the last timestamp overrides the duration.
    "trailing_timestamp_only": ([T, 1000, 2000, T + 50], 0, N_FRAMES),
    # `<|0.00|>` last: upstream explicitly declines to override the duration.
    "last_timestamp_is_zero": ([T, 1000, 2000, T], 0, N_FRAMES),
    "single_timestamp_token": ([T + 50], 0, N_FRAMES),
    # One consecutive pair plus an unpaired trailing timestamp, so `slices`
    # picks up `len(tokens)` and the advance is a whole window.
    "one_pair_single_ending": ([T, 1000, 2000, T + 50, T + 50, 3000, T + 100], 0, N_FRAMES),
    # One pair, no unpaired ending: the tail is dropped and seek lands on the
    # last consumed timestamp instead of the window edge.
    "one_pair_no_single_ending": ([T, 1000, T + 50, T + 50, 2000, T + 100, 3000], 0, N_FRAMES),
    "two_pairs": (
        [T, 1000, T + 50, T + 50, 2000, T + 100, T + 100, 3000, T + 150],
        0,
        N_FRAMES,
    ),
    # Three timestamps in a row produce two overlapping pairs and therefore
    # two adjacent single-token slices.
    "three_consecutive_timestamps": ([T, T + 50, T + 100, 1000, T + 150], 0, N_FRAMES),
    # The stall: the last consumed timestamp is `<|0.00|>`, so upstream's
    # advance is 0 and its loop spins on this window forever.
    "stall_zero_advance": ([T, T, 1000], 0, N_FRAMES),
    # Special tokens between EOT and `timestamp_begin` — a language id and
    # `<|notimestamps|>` — must survive in `tokens` but vanish from the text.
    "specials_between_eot_and_timestamps": (
        [T, 50259, 1000, 50363, T + 50, T + 50, 2000, EOT],
        0,
        N_FRAMES,
    ),
    # A slice that does not open on a timestamp: upstream's subtraction goes
    # negative and the segment starts before the file does.
    "negative_start_offset": ([1000, T + 50, T + 50, 2000], 0, N_FRAMES),
    # The final, short window of a file.
    "partial_window": ([1000], 0, 1234),
    "partial_window_with_timestamps": ([T, 1000, T + 50, T + 50, 2000], 0, 1234),
    # Second window of a long file: everything shifts by `time_offset`.
    "mid_file_offset": ([T, 1000, T + 50, T + 50, 2000, T + 100], N_FRAMES, N_FRAMES),
}

# `(no_speech_prob, avg_logprob, no_speech_threshold, logprob_threshold)`
SILENCE_CASES = {
    "speech": (0.1, -0.3, 0.6, -1.0),
    "silence_and_low_logprob": (0.9, -2.0, 0.6, -1.0),
    "silence_but_confident": (0.9, -0.5, 0.6, -1.0),
    # The one case where the port and upstream disagree.
    "logprob_exactly_at_threshold": (0.9, -1.0, 0.6, -1.0),
    "no_speech_exactly_at_threshold": (0.6, -2.0, 0.6, -1.0),
    "no_speech_threshold_disabled": (0.9, -2.0, None, -1.0),
    "logprob_threshold_disabled": (0.9, -2.0, 0.6, None),
}

# `(compression_ratio, avg_logprob, no_speech_prob, crt, lpt, nst)`
FALLBACK_CASES = {
    "clean": (1.5, -0.3, 0.1, 2.4, -1.0, 0.6),
    "too_repetitive": (3.0, -0.3, 0.1, 2.4, -1.0, 0.6),
    "logprob_too_low": (1.5, -2.0, 0.1, 2.4, -1.0, 0.6),
    "both_bad": (3.0, -2.0, 0.1, 2.4, -1.0, 0.6),
    # Silence clears the flag even when both quality gates failed.
    "silence_overrides_both": (3.0, -2.0, 0.9, 2.4, -1.0, 0.6),
    "compression_exactly_at_threshold": (2.4, -0.3, 0.1, 2.4, -1.0, 0.6),
    "logprob_exactly_at_threshold": (1.5, -1.0, 0.1, 2.4, -1.0, 0.6),
    "no_speech_exactly_at_threshold": (3.0, -0.3, 0.6, 2.4, -1.0, 0.6),
    "all_thresholds_disabled": (9.0, -9.0, 0.9, None, None, None),
}

TEXT_TOKEN_CASES = {
    "empty": [],
    "text_only": [1000, 2000],
    "timestamps_stripped": [T, 1000, T + 50],
    "eot_stripped": [1000, EOT],
    "language_and_notimestamps_stripped": [T, 50259, 1000, 50363, T + 50],
    "boundary_ids": [EOT - 1, EOT, EOT + 1],
}


def build_fixture() -> dict:
    split = {}
    for name, (tokens, seek, segment_size) in SPLIT_CASES.items():
        segments, seek_delta = split_window(tokens, TS_BEGIN, seek, segment_size, INPUT_STRIDE)
        split[name] = {
            "tokens": list(tokens),
            "seek": seek,
            "segment_size": segment_size,
            "segments": segments,
            "seek_delta": seek_delta,
            # What the port actually advances by. Upstream spins on a zero
            # advance; the port jumps a whole window and drops that audio.
            "seek_delta_port": segment_size if seek_delta == 0 else seek_delta,
        }

    silence = {}
    for name, (nsp, alp, nst, lpt) in SILENCE_CASES.items():
        silence[name] = {
            "no_speech_prob": nsp,
            "avg_logprob": alp,
            "no_speech_threshold": nst,
            "logprob_threshold": lpt,
            "upstream": should_skip_window(nsp, alp, nst, lpt),
            "port": should_skip_window_port(nsp, alp, nst, lpt),
        }

    fallback = {}
    for name, (cr, alp, nsp, crt, lpt, nst) in FALLBACK_CASES.items():
        fallback[name] = {
            "compression_ratio": cr,
            "avg_logprob": alp,
            "no_speech_prob": nsp,
            "compression_ratio_threshold": crt,
            "logprob_threshold": lpt,
            "no_speech_threshold": nst,
            "needs_fallback": needs_fallback(cr, alp, nsp, crt, lpt, nst),
        }

    text = {
        name: {"tokens": toks, "text_tokens": text_tokens(toks, EOT)}
        for name, toks in TEXT_TOKEN_CASES.items()
    }

    return {
        "constants": {
            "sample_rate": SAMPLE_RATE,
            "hop_length": HOP_LENGTH,
            "n_frames": N_FRAMES,
            "timestamp_begin": TS_BEGIN,
            "eot": EOT,
            "input_stride": INPUT_STRIDE,
            "time_precision": INPUT_STRIDE * HOP_LENGTH / SAMPLE_RATE,
        },
        "split_window": split,
        "silence_skip": silence,
        "needs_fallback": fallback,
        "text_tokens": text,
        "source": "mlx_whisper/transcribe.py 0.4.3, `while seek < seek_clip_end` body",
    }


def main() -> int:
    out = pathlib.Path(__file__).resolve().parent.parent / "tests" / "fixtures"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "segmentation.json"
    path.write_text(json.dumps(build_fixture(), indent=2, sort_keys=True) + "\n")
    print(f"wrote {path} ({path.stat().st_size / 1024:.1f} KiB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
