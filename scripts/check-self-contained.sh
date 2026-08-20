#!/usr/bin/env bash
#
# Checks whether a release build of this crate is genuinely a self-contained
# single binary on macOS, which is the whole point of comparing against
# whisper.cpp for a single-binary deployment.
#
# Two things can silently break "self-contained" and neither shows up the
# same way:
#
#   1. Homebrew (or any other non-system) dylib gets linked in. `otool -L`
#      catches this directly, and `otool -l`'s LC_RPATH commands catch the
#      sneakier variant where the linked name is `@rpath/libfoo.dylib` and
#      the rpath itself points into a Homebrew prefix.
#
#   2. MLX is a Metal-backed library: its GPU kernels are normally shipped as
#      compiled shaders in a `.metallib` file. `mlx-sys` builds MLX's Metal
#      backend by default (`metal` is a default feature, forwarded from
#      mlx-rs's Cargo.toml down to mlx-sys's `MLX_BUILD_METAL` CMake define),
#      but *how* the resulting shader library is packaged -- embedded in the
#      static lib, or a loose file the binary looks for at a path baked in
#      at build time -- is decided entirely inside MLX's own CMake, which is
#      fetched from GitHub during the build and never vendored into mlx-sys.
#      Nothing in this repo or in mlx-sys settles that question by reading
#      source; it can only be settled by trying to run the binary with
#      nothing beside it. That is what step 5 below actually does, and it is
#      the one check that matters most -- the others can look clean while
#      this one still fails.
#
# This script cannot be run on the machine that wrote it (Linux; mlx-sys
# links Foundation/objc/Metal/Accelerate unconditionally, so nothing in this
# crate even compiles here). Run it on an Apple Silicon Mac from the repo
# root:
#
#   ./scripts/check-self-contained.sh --audio /path/to/sample.wav
#
# See "Self-containment check" in README.md for the full picture.

set -euo pipefail

# ── Argument parsing ─────────────────────────────────────────────────────
#
# The model gets a sensible default: a small mlx-community model keeps the
# download and the transcription step fast, which is all this check needs --
# it is proving the binary runs standalone, not benchmarking accuracy. The
# audio file gets no default on purpose: the repo does not ship a playable
# WAV (tests/fixtures/*.f32 and *.json are numeric arrays, not audio), so any
# default would either be wrong or silently fetch something over the
# network. Per the task this must fail loudly rather than skip the most
# important check, so it is a required argument.
AUDIO_FILE=""
MODEL="mlx-community/whisper-tiny-mlx"
ASSETS_DIR=""

usage() {
  cat <<'EOF'
Usage: scripts/check-self-contained.sh --audio <file.wav> [options]

Required:
  --audio <file>       A real WAV file to transcribe in the isolated run.
                        No default -- the repo does not ship a sample WAV.

Options:
  --model <id>          HuggingFace model id or local path.
                         Default: mlx-community/whisper-tiny-mlx (small, so
                         the download and the run stay fast).
  --assets <dir>         Path to the populated assets/ directory (see
                         tools/extract_assets.py). Default: <repo>/assets

The isolated run (step 5) needs network access on its first invocation to
download <model> into the normal HuggingFace cache (~/.cache/huggingface),
which is unaffected by the environment-scrubbing this script does for the
DYLD_* variables -- that cache is expected external state, the same way a
model file would be for whisper.cpp.
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --audio|--model|--assets)
      # Guard the value lookup before shifting -- a flag given as the last
      # argument with nothing after it used to hit "shift count out of
      # range" instead of a clean usage message.
      if [ $# -lt 2 ]; then
        echo "$1 requires a value" >&2
        usage >&2
        exit 1
      fi
      case "$1" in
        --audio) AUDIO_FILE="$2" ;;
        --model) MODEL="$2" ;;
        --assets) ASSETS_DIR="$2" ;;
      esac
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)

if [ -z "$ASSETS_DIR" ]; then
  ASSETS_DIR="$REPO_ROOT/assets"
fi

if [ -z "$AUDIO_FILE" ]; then
  echo "ERROR: --audio <file.wav> is required -- the most important check" >&2
  echo "(step 5, running the binary alone in an empty directory) needs a" >&2
  echo "real audio file and cannot be skipped silently." >&2
  echo >&2
  usage >&2
  exit 1
fi

if [ ! -f "$AUDIO_FILE" ]; then
  echo "ERROR: audio file not found: $AUDIO_FILE" >&2
  exit 1
fi
AUDIO_FILE=$(cd "$(dirname "$AUDIO_FILE")" && pwd)/$(basename "$AUDIO_FILE")

if [ ! -d "$ASSETS_DIR" ] || [ -z "$(ls -A "$ASSETS_DIR" 2>/dev/null || true)" ]; then
  echo "ERROR: assets dir '$ASSETS_DIR' is missing or empty." >&2
  echo "Run 'python3 tools/extract_assets.py' first (see README.md)." >&2
  exit 1
fi
ASSETS_DIR=$(cd "$ASSETS_DIR" && pwd)

if [ "$(uname -s)" != "Darwin" ]; then
  echo "ERROR: this check only makes sense on macOS (otool, DYLD_*)." >&2
  exit 1
fi

# ── Bookkeeping ───────────────────────────────────────────────────────────
#
# Every check reports its own PASS/FAIL/INFO and the script keeps going --
# a fail counter, not `set -e`, decides the exit status, so one broken check
# never hides the results of the others.
FAIL_COUNT=0
pass() { echo "  PASS: $*"; }
fail() { echo "  FAIL: $*"; FAIL_COUNT=$((FAIL_COUNT + 1)); }
info() { echo "  INFO: $*"; }
section() { echo; echo "== $* =="; }

WORKDIR=$(mktemp -d "${TMPDIR:-/tmp}/mlx-whisper-self-contained.XXXXXX")
cleanup() { rm -rf "$WORKDIR"; }
trap cleanup EXIT

# ── Step 1: build ────────────────────────────────────────────────────────
section "Building examples/transcribe in release mode"
( cd "$REPO_ROOT" && cargo build --release --example transcribe )

BIN="$REPO_ROOT/target/release/examples/transcribe"
if [ ! -x "$BIN" ]; then
  echo "ERROR: expected binary not found at $BIN" >&2
  exit 1
fi
info "binary: $BIN"

# ── Step 2: otool -L -- linked dylibs/frameworks ────────────────────────
section "otool -L: linked libraries"
#
# The first line of `otool -L` output is the binary's own install name, not
# a dependency -- skip it. System paths are /usr/lib/** and
# /System/Library/**; everything else (Homebrew's /opt/homebrew or
# /usr/local, a bare relative name, or an @rpath/@loader_path/@executable_path
# entry that isn't obviously a system framework) counts as non-system and is
# a failure for the self-contained claim.
NONSYSTEM_LIBS=$(mktemp "$WORKDIR/nonsystem_libs.XXXXXX")
: > "$NONSYSTEM_LIBS"
otool -L "$BIN" | tail -n +2 | while IFS= read -r line; do
  lib=$(echo "$line" | sed -E 's/^[[:space:]]*([^[:space:]]+).*/\1/')
  case "$lib" in
    /usr/lib/*|/System/Library/*)
      echo "  system:     $lib"
      ;;
    *)
      echo "  NON-SYSTEM: $lib"
      echo "$lib" >> "$NONSYSTEM_LIBS"
      ;;
  esac
done

if [ -s "$NONSYSTEM_LIBS" ]; then
  fail "otool -L found $(wc -l < "$NONSYSTEM_LIBS" | tr -d ' ') non-system linked entr(y/ies) -- see above"
else
  pass "every linked library is under /usr/lib or /System/Library"
fi

# ── Step 3: otool -l -- LC_RPATH ────────────────────────────────────────
section "otool -l: LC_RPATH load commands"
#
# A clean `otool -L` can still be defeated by an rpath that resolves into a
# Homebrew prefix at runtime for an @rpath-relative library. @loader_path and
# @executable_path are self-relative (they resolve next to the binary
# itself) and are reported for visibility, not failed automatically.
RPATHS=$(otool -l "$BIN" | awk '/cmd LC_RPATH/{getline; getline; print $2}')
if [ -z "$RPATHS" ]; then
  info "no LC_RPATH load commands present"
else
  echo "$RPATHS" | while IFS= read -r rp; do
    [ -z "$rp" ] && continue
    case "$rp" in
      *"/opt/homebrew"*|*"/usr/local"*|*"linuxbrew"*)
        echo "  NON-SYSTEM rpath: $rp"
        ;;
      @loader_path*|@executable_path*)
        echo "  self-relative rpath: $rp"
        ;;
      *)
        echo "  other rpath: $rp"
        ;;
    esac
  done
  BAD_RPATHS=$(echo "$RPATHS" | grep -Ec '(/opt/homebrew|/usr/local|linuxbrew)' || true)
  if [ "${BAD_RPATHS:-0}" -gt 0 ]; then
    fail "$BAD_RPATHS LC_RPATH entr(y/ies) point into a Homebrew/local prefix"
  else
    pass "no LC_RPATH entry points into a Homebrew/local prefix"
  fi
fi

# ── Step 4: metallib presence (informational) ───────────────────────────
section "Metal shader library (.metallib) -- informational only"
#
# This is deliberately not a PASS/FAIL: an embedded shader blob typically
# lives in an ordinary __TEXT/__DATA section, not a dedicated Mach-O
# segment, so the *absence* of a "__METAL" segment or a metallib string
# proves nothing either way. Step 5 (moving aside any found .metallib and
# then actually running the binary) is what actually answers the question;
# this step just surfaces what there is to look at.
METALLIB_FILES=$(find "$REPO_ROOT/target" -name "*.metallib" 2>/dev/null || true)
if [ -n "$METALLIB_FILES" ]; then
  info "found .metallib file(s) under target/:"
  echo "$METALLIB_FILES" | while IFS= read -r f; do
    [ -z "$f" ] && continue
    info "  $f"
  done
else
  info "no .metallib file found anywhere under target/"
fi

if otool -l "$BIN" | grep -q "__METAL"; then
  info "binary has a __METAL Mach-O segment (some MLX/Metal shader blob is embedded)"
else
  info "no __METAL Mach-O segment (does not rule out an embedded blob elsewhere)"
fi

METALLIB_STRINGS=$(strings "$BIN" 2>/dev/null | grep -i '\.metallib' || true)
if [ -n "$METALLIB_STRINGS" ]; then
  info "binary contains string references to a .metallib path:"
  echo "$METALLIB_STRINGS" | sed 's/^/  /'
else
  info "no .metallib path strings found in the binary"
fi

# What does MLX's own (fetched) CMake actually say about this build? The
# source lives under target/**/build/mlx-sys-*/out/build/_deps/mlx-src after
# a build, having been FetchContent-cloned by mlx-c's CMakeLists.txt (see
# src/mlx-c/CMakeLists.txt in the mlx-sys crate) -- it is never vendored
# into mlx-sys itself, which is why this could not be answered from source
# on the machine that wrote this script (see README.md).
MLX_SRC_DIR=$(find "$REPO_ROOT/target" -type d -name "mlx-src" 2>/dev/null | head -n 1 || true)
if [ -n "$MLX_SRC_DIR" ]; then
  info "found fetched MLX source at: $MLX_SRC_DIR"
  MLX_CMAKE_HITS=$(grep -rniE "MLX_METAL_JIT|MLX_METAL_PATH|\.metallib" "$MLX_SRC_DIR" 2>/dev/null \
    | grep -iE '(CMakeLists\.txt|\.cmake):' | head -n 40 || true)
  if [ -n "$MLX_CMAKE_HITS" ]; then
    info "MLX's own CMake references to the metal build/metallib:"
    echo "$MLX_CMAKE_HITS" | sed 's/^/  /'
  else
    info "no MLX_METAL_JIT / MLX_METAL_PATH / .metallib references found in MLX's CMake files"
  fi
else
  info "fetched MLX source not found under target/ (unexpected after a successful build)"
fi

# ── Step 5: run the binary alone in an empty directory ──────────────────
section "Running the binary alone, with a scrubbed environment"
#
# This is the actual test. Everything above is static inspection; this is
# what happens when the binary is the only thing that exists.
#
# Sequence:
#   a) Run once normally first. This warms the HuggingFace model cache (so
#      the isolated run below is not timed out by a slow download) and
#      establishes a functional baseline -- if this fails, the isolated run
#      failing too tells you nothing about self-containment specifically.
#   b) Move aside any .metallib found under target/ (step 4), so the
#      isolated run cannot succeed merely by finding a build-tree file that
#      will not exist on another machine. Restored on exit via trap,
#      success or failure.
#   c) Copy *only* the binary into a fresh empty directory, cd there, and
#      run it with DYLD_* unset and PATH reduced to system directories plus
#      (if present) ffmpeg's own directory -- ffmpeg is a documented,
#      separate runtime dependency of load_audio() (see README.md
#      Prerequisites #2), not something this binary links, so stripping it
#      from PATH would fail the run for an unrelated reason and muddy the
#      result.
info "baseline run (normal environment, also warms the HF model cache)..."
BASELINE_LOG="$WORKDIR/baseline.log"
set +e
"$BIN" "$AUDIO_FILE" --model "$MODEL" --assets "$ASSETS_DIR" >"$BASELINE_LOG" 2>&1
BASELINE_STATUS=$?
set -e
if [ "$BASELINE_STATUS" -eq 0 ]; then
  pass "baseline run succeeded (see $BASELINE_LOG)"
else
  fail "baseline run failed (exit $BASELINE_STATUS) -- see $BASELINE_LOG; the isolated run below cannot be trusted if this failed for an unrelated reason"
  tail -n 20 "$BASELINE_LOG" | sed 's/^/  | /'
fi

# The isolated run below has to answer "does the binary need a .metallib beside
# it?", so any metallib the build produced is moved out of the way first. That
# means this script renames files in the caller's target/ directory, and it must
# put them back no matter how it exits -- if it does not, the tree is left in
# exactly the broken state this check is trying to detect, and every subsequent
# build inherits it.
#
# MOVED_METALLIBS is therefore recorded BEFORE the first mv, not after. With
# `set -e`, a partially-completed move loop aborts the script, and an assignment
# placed after the loop would never run: restore_metallibs would return early on
# an empty list and strand the renamed files.
MOVED_METALLIBS=""
restore_metallibs() {
  [ -z "$MOVED_METALLIBS" ] && return 0
  # Never let a restore failure abort the EXIT trap before cleanup runs, and
  # never let the loop's last iteration decide the trap's exit status.
  echo "$MOVED_METALLIBS" | while IFS= read -r f; do
    [ -z "$f" ] && continue
    if [ -e "${f}.self-contained-check-bak" ]; then
      mv "${f}.self-contained-check-bak" "$f" || echo "  WARNING: could not restore $f" >&2
    fi
  done
  return 0
}
trap 'restore_metallibs; cleanup' EXIT

if [ -n "$METALLIB_FILES" ]; then
  MOVED_METALLIBS="$METALLIB_FILES"
  echo "$METALLIB_FILES" | while IFS= read -r f; do
    [ -z "$f" ] && continue
    mv "$f" "${f}.self-contained-check-bak"
  done
  info "moved aside $(echo "$METALLIB_FILES" | wc -l | tr -d ' ') .metallib file(s) under target/ for the isolated run"
fi

ISODIR=$(mktemp -d "${TMPDIR:-/tmp}/mlx-whisper-isolated.XXXXXX")
cp "$BIN" "$ISODIR/transcribe"

FFMPEG_DIR=""
if command -v ffmpeg >/dev/null 2>&1; then
  FFMPEG_DIR=$(dirname "$(command -v ffmpeg)")
fi
SAFE_PATH="/usr/bin:/bin:/usr/sbin:/sbin"
if [ -n "$FFMPEG_DIR" ]; then
  SAFE_PATH="$SAFE_PATH:$FFMPEG_DIR"
fi

info "isolated run (empty dir, scrubbed DYLD_*, Homebrew off PATH except ffmpeg)..."
ISOLATED_LOG="$WORKDIR/isolated.log"
set +e
(
  cd "$ISODIR" && \
  env -u DYLD_LIBRARY_PATH -u DYLD_FALLBACK_LIBRARY_PATH -u DYLD_INSERT_LIBRARIES \
      -u DYLD_FRAMEWORK_PATH -u DYLD_FALLBACK_FRAMEWORK_PATH \
      PATH="$SAFE_PATH" \
      ./transcribe "$AUDIO_FILE" --model "$MODEL" --assets "$ASSETS_DIR"
) >"$ISOLATED_LOG" 2>&1
ISOLATED_STATUS=$?
set -e

rm -rf "$ISODIR"

if [ "$ISOLATED_STATUS" -eq 0 ]; then
  pass "isolated run succeeded with nothing beside the binary (see $ISOLATED_LOG)"
else
  fail "isolated run failed (exit $ISOLATED_STATUS) -- see $ISOLATED_LOG"
  tail -n 30 "$ISOLATED_LOG" | sed 's/^/  | /'
  if [ "$BASELINE_STATUS" -eq 0 ]; then
    info "baseline passed but the isolated run failed: the delta is isolation itself"
    info "(scrubbed DYLD_* / restricted PATH / the .metallib moved aside) -- read isolated.log above"
  fi
fi

# ── Summary ───────────────────────────────────────────────────────────────
section "Summary"
if [ "$FAIL_COUNT" -eq 0 ]; then
  echo "All checks passed: this build appears to be a genuinely self-contained"
  echo "single binary -- only system frameworks/dylibs, no non-system rpath,"
  echo "and it ran correctly with nothing beside it and no MLX .metallib in"
  echo "reach."
else
  echo "$FAIL_COUNT check(s) failed. This build is NOT confirmed self-contained."
  echo "See the FAIL lines above for exactly what it needs beside it (a"
  echo "specific dylib, an rpath into Homebrew, or the .metallib that was"
  echo "moved aside for the isolated run)."
fi
exit "$FAIL_COUNT"
