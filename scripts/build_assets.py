#!/usr/bin/env python3
"""
FBA-Bench Asset Builder

- Validates required press assets exist.
- Optionally optimizes PNGs (optipng) and SVGs (svgo) if available on PATH.
- Rebuilds site/assets/press/press-kit.zip deterministically, excluding itself.
- Idempotent and clear output; exits 0 on success, 1 on failure.

This script uses only the Python standard library.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import zipfile
from pathlib import Path


def _print_header(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def _find_tool(name: str) -> str | None:
    return shutil.which(name)


def _run(cmd: list[str]) -> int:
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.stdout and proc.stdout.strip():
            print(proc.stdout.strip())
        if proc.returncode != 0 and proc.stderr and proc.stderr.strip():
            print(proc.stderr.strip())
        return proc.returncode
    except Exception as e:
        print(f"Failed to execute {' '.join(cmd)}: {e}")
        return 1


def validate_assets(press_dir: Path, required: list[str]) -> bool:
    ok = True
    for rel in required:
        p = press_dir / rel
        if not p.exists():
            print(f"Missing required asset: {p.as_posix()}")
            ok = False
    return ok


def optimize_images(press_dir: Path) -> None:
    _print_header("Optimizing images (optional)")

    # PNG optimization via optipng (if present)
    optipng = _find_tool("optipng")
    pngs = sorted(press_dir.glob("*.png"))
    if optipng:
        if pngs:
            print(f"optipng found at {optipng}; optimizing {len(pngs)} PNG files")
            for f in pngs:
                # Reasonable optimization level, quiet output; do not fail build on non-zero
                rc = _run([optipng, "-o2", "-quiet", f.as_posix()])
                if rc != 0:
                    print(f"optipng non-zero exit on {f.name} (continuing)")
        else:
            print("No PNG files to optimize")
    else:
        print("optipng not found; skipping PNG optimization")

    # SVG optimization via svgo (if present)
    svgo = _find_tool("svgo")
    svgs = sorted(press_dir.glob("*.svg"))
    if svgo:
        if svgs:
            print(f"svgo found at {svgo}; optimizing {len(svgs)} SVG files")
            for f in svgs:
                # svgo v2 CLI: -i input -o output --multipass
                rc = _run([svgo, "-i", f.as_posix(), "-o", f.as_posix(), "--multipass"])
                if rc != 0:
                    print(f"svgo non-zero exit on {f.name} (continuing)")
        else:
            print("No SVG files to optimize")
    else:
        print("svgo not found; skipping SVG optimization")


def create_press_kit(press_dir: Path, zip_path: Path) -> int:
    _print_header("Creating press-kit.zip")

    # Collect candidate files (exclude the zip itself), include png/svg only
    candidates = []
    for ext in (".png", ".svg"):
        candidates.extend(sorted(press_dir.glob(f"*{ext}")))
    files = [p for p in candidates if p.is_file() and p.name != zip_path.name]
    # Stable deterministic order by lowercased filename
    files = sorted(files, key=lambda p: p.name.lower())

    if not files:
        print("No assets to include. Skipping ZIP creation.")
        return 0

    tmp_zip = zip_path.with_suffix(".zip.tmp")

    # Deterministic zip settings
    EPOCH = (1980, 1, 1, 0, 0, 0)  # Minimum DOS timestamp supported by ZIP
    compression = zipfile.ZIP_DEFLATED

    try:
        with zipfile.ZipFile(tmp_zip, "w", compression=compression) as zf:
            for f in files:
                data = f.read_bytes()
                info = zipfile.ZipInfo(filename=f.name, date_time=EPOCH)
                info.compress_type = compression
                # Set permissions to 0644 for files
                info.external_attr = 0o100644 << 16
                zf.writestr(info, data)

        # Atomic replace of the final zip
        tmp_zip.replace(zip_path)

        size = zip_path.stat().st_size if zip_path.exists() else 0
        print(f"Created {zip_path.as_posix()} ({size} bytes) with {len(files)} files")
        return 0
    finally:
        if tmp_zip.exists():
            try:
                tmp_zip.unlink()
            except Exception:
                # Non-fatal cleanup failure
                pass


def main() -> int:
    # Resolve repo root from this file (assumes scripts/ directory)
    repo_root = Path(__file__).resolve().parents[1]
    press_dir = repo_root / "site" / "assets" / "press"
    zip_path = press_dir / "press-kit.zip"

    _print_header("FBA-Bench Asset Builder")

    # Validate required assets exist
    _print_header("Validating required assets")
    required = [
        "logo-light.svg",
        "logo-dark.svg",
        "logo.png",
        "og-image.png",
    ]
    if not validate_assets(press_dir, required):
        print("One or more required assets are missing.")
        return 1

    # Optional optimization (no hard failure if tools unavailable)
    optimize_images(press_dir)

    # Rebuild ZIP deterministically (exclude press-kit.zip itself)
    rc = create_press_kit(press_dir, zip_path)
    if rc != 0:
        print("Failed to create press-kit.zip")
        return 1

    _print_header("Asset pipeline completed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
