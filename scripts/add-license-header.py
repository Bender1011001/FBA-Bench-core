#!/usr/bin/env python3
"""
Add MIT License header to Python files that lack one.

This script walks the repository, targets .py files, excludes specified directories,
preserves shebangs and encoding cookies, and inserts a standard MIT header if missing.
It is idempotent and supports check/apply modes.
"""

import argparse
import os
import re
from pathlib import Path

# Excluded directories
EXCLUDED_DIRS = {
    ".git",
    ".venv",
    "venv",
    "dist",
    "build",
    "site",
    "golden_masters",
    "__pycache__",
    ".pytest_cache",
    "htmlcov",
}

# Header markers to detect existing license
HEADER_MARKERS = [r"MIT License", r"Copyright \(c\) \d{4} FBA-Bench Core Team"]

# Full MIT header template
MIT_HEADER_TEMPLATE = """# MIT License
#
# Copyright (c) {year} {holder}
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""


def has_header(lines: list[str]) -> bool:
    """Check if the first ~30 lines contain a header marker."""
    check_lines = lines[:30]
    content = "\n".join(check_lines)
    for marker in HEADER_MARKERS:
        if re.search(marker, content):
            return True
    return False


def preserve_prelude(lines: list[str]) -> tuple[list[str], int]:
    """Identify and preserve shebang and encoding lines."""
    prelude = []
    insert_after = 0

    # Shebang: line 1 starting with #!
    if lines and lines[0].startswith("#!"):
        prelude.append(lines[0])
        insert_after = 1

    # Encoding cookie: # -*- coding: utf-8 -*- (usually line 1 or 2)
    encoding_pattern = r"#\s*-\*- coding:\s*[\w-]+ \*-"
    for i, line in enumerate(lines[:2]):  # Only first two lines
        if re.match(encoding_pattern, line.strip()):
            if i >= insert_after:
                prelude.append(line)
                insert_after = i + 1
            else:
                # If encoding before shebang (rare), insert after
                pass

    return prelude, insert_after


def insert_header(lines: list[str], year: str, holder: str) -> list[str]:
    """Insert the MIT header after prelude if missing."""
    if has_header(lines):
        return lines

    prelude, insert_pos = preserve_prelude(lines)
    header = MIT_HEADER_TEMPLATE.format(year=year, holder=holder).splitlines(
        keepends=True
    )

    # Ensure blank line after header if not at end
    if insert_pos < len(lines) and not lines[insert_pos].strip() == "":
        header.append("\n")

    new_lines = prelude + header + lines[insert_pos:]
    return new_lines


def process_file(
    filepath: str, year: str, holder: str, apply: bool, check_only: bool
) -> tuple[bool, str]:
    """Process a single .py file: check or apply header."""
    try:
        with open(filepath, encoding="utf-8") as f:
            lines = f.readlines()

        needs_header = not has_header(lines)

        if check_only:
            if needs_header:
                return True, f"Missing header: {filepath}"
            return False, ""

        if apply and needs_header:
            new_lines = insert_header(lines, year, holder)
            with open(filepath, "w", encoding="utf-8") as f:
                f.writelines(new_lines)
            return True, f"Added header: {filepath}"
        elif apply:
            return False, f"Skipped (has header): {filepath}"

        return False, ""

    except Exception as e:
        return False, f"Error processing {filepath}: {e}"


def walk_repo(
    root: Path,
    excluded_dirs: set,
    year: str,
    holder: str,
    apply: bool,
    check_only: bool,
) -> tuple[int, int, int, list[str]]:
    """Walk the repo and process .py files."""
    scanned = 0
    added = 0
    skipped = 0
    missing_files = []

    for dirpath, dirnames, filenames in os.walk(root):
        # Skip excluded dirs
        dirnames[:] = [d for d in dirnames if d not in excluded_dirs]

        for filename in filenames:
            if filename.endswith(".py"):
                filepath = os.path.join(dirpath, filename)
                scanned += 1
                needs, msg = process_file(filepath, year, holder, apply, check_only)
                if check_only and needs:
                    missing_files.append(msg)
                elif apply and needs:
                    added += 1
                else:
                    skipped += 1
                if msg:
                    print(msg)

    return scanned, added, skipped, missing_files


def main():
    parser = argparse.ArgumentParser(
        description="Add MIT License header to Python files."
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        default=True,
        help="Apply changes in-place (default)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check mode: report missing headers, exit 1 if any",
    )
    parser.add_argument(
        "--year", default="2025", help="Year for copyright (default: 2025)"
    )
    parser.add_argument(
        "--holder",
        default="FBA-Bench Core Team",
        help="Copyright holder (default: FBA-Bench Core Team)",
    )
    args = parser.parse_args()

    if args.check:
        args.apply = False

    root = Path(".").resolve()
    scanned, added, skipped, missing = walk_repo(
        root, EXCLUDED_DIRS, args.year, args.holder, args.apply, args.check
    )

    print(f"\nSummary: Scanned {scanned} files, Added {added}, Skipped {skipped}")

    if args.check:
        if missing:
            print("\nMissing headers in:")
            for mf in missing:
                print(f"  {mf}")
            exit(1)
        else:
            print("\nAll files have headers.")
            exit(0)
    else:
        exit(0)


if __name__ == "__main__":
    main()
