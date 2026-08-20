#!/usr/bin/env python3
"""Populate `assets/` with the tokenizer and mel-filter files this crate needs.

Upstream `mlx_whisper` ships the mel filterbanks as a **single** archive,
`assets/mel_filters.npz`, with keys `mel_80` and `mel_128`. This crate's
`audio::mel_filters()` instead reads separate `assets/mel_filters_{n}.npy`
files, so a plain `cp .../assets/*.npy assets/` copies nothing. This script
does the unpacking.

Two sources are supported, tried in order:

  1. an installed `mlx_whisper` Python package, if importable;
  2. the `mlx_whisper` wheel downloaded straight from PyPI (no install, no MLX
     required — only the standard library plus `numpy` for the `.npy` write).

Usage:

    python3 tools/extract_assets.py                 # auto
    python3 tools/extract_assets.py --version 0.4.3 # pin the wheel version
    python3 tools/extract_assets.py --out assets    # choose the output dir

Produces:

    assets/multilingual.tiktoken
    assets/gpt2.tiktoken
    assets/mel_filters_80.npy
    assets/mel_filters_128.npy
"""

import argparse
import io
import json
import pathlib
import sys
import urllib.request
import zipfile

REPO = pathlib.Path(__file__).resolve().parent.parent
WANT_TIKTOKEN = ("multilingual.tiktoken", "gpt2.tiktoken")


def from_installed():
    try:
        import mlx_whisper  # noqa: F401
    except Exception:
        return None
    d = pathlib.Path(mlx_whisper.__file__).parent / "assets"
    if not d.is_dir():
        return None
    print(f"source: installed mlx_whisper at {d}")
    return {p.name: p.read_bytes() for p in d.iterdir() if p.is_file()}


def from_pypi(version=None):
    url = "https://pypi.org/pypi/mlx-whisper/json"
    if version:
        url = f"https://pypi.org/pypi/mlx-whisper/{version}/json"
    print(f"source: PyPI ({url})")
    with urllib.request.urlopen(url, timeout=30) as r:
        meta = json.load(r)
    wheels = [u for u in meta["urls"] if u["packagetype"] == "bdist_wheel"]
    if not wheels:
        raise SystemExit("no wheel found on PyPI for mlx-whisper")
    w = wheels[0]
    print(f"  downloading {w['filename']} ({w['size']/1024:.0f} KiB)")
    with urllib.request.urlopen(w["url"], timeout=120) as r:
        blob = r.read()
    out = {}
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        for name in z.namelist():
            # Ignore the wheel's stale `build/lib/...` duplicates.
            if not name.startswith("mlx_whisper/assets/"):
                continue
            out[pathlib.PurePosixPath(name).name] = z.read(name)
    print(f"  version: {meta['info']['version']}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--version", help="pin a specific mlx-whisper release")
    ap.add_argument("--out", default=str(REPO / "assets"), help="output directory")
    ap.add_argument(
        "--force", action="store_true", help="overwrite files that already exist"
    )
    args = ap.parse_args()

    files = None
    if not args.version:
        files = from_installed()
    if files is None:
        files = from_pypi(args.version)

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    written = []

    for name in WANT_TIKTOKEN:
        if name not in files:
            raise SystemExit(f"upstream assets are missing {name}")
        dest = out / name
        if dest.exists() and not args.force:
            print(f"  keep   {dest} (exists; --force to overwrite)")
            continue
        dest.write_bytes(files[name])
        written.append(dest)

    if "mel_filters.npz" not in files:
        raise SystemExit("upstream assets are missing mel_filters.npz")

    try:
        import numpy as np
    except ImportError:
        raise SystemExit(
            "numpy is required to unpack mel_filters.npz "
            "(pip install numpy), or extract mel_80.npy / mel_128.npy from the "
            "npz by hand and rename them to mel_filters_{80,128}.npy"
        )

    with zipfile.ZipFile(io.BytesIO(files["mel_filters.npz"])) as z:
        for n_mels in (80, 128):
            member = f"mel_{n_mels}.npy"
            if member not in z.namelist():
                raise SystemExit(f"mel_filters.npz has no member {member}")
            arr = np.load(io.BytesIO(z.read(member)), allow_pickle=False)
            arr = np.ascontiguousarray(arr.astype(np.float32))
            dest = out / f"mel_filters_{n_mels}.npy"
            if dest.exists() and not args.force:
                print(f"  keep   {dest} (exists; --force to overwrite)")
                continue
            np.save(dest, arr, allow_pickle=False)
            written.append(dest)

    for p in written:
        print(f"  wrote  {p}  ({p.stat().st_size/1024:.0f} KiB)")
    print(f"\n{len(written)} file(s) written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
