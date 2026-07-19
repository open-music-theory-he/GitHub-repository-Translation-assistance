#!/usr/bin/env python3
"""
extract_sentences.py

One-time (manually triggered) script.

Walks every .md file in a checked-out copy of the SOURCE repository
(open-music-theory-he/open-music-theory-he.github.io), pulls out every
"sentence" (in this context: every non-empty content line), strips
Markdown syntax that applies to the WHOLE line (headings "#", blockquote
">", list bullets "-"/"*"/"+"/"1.", horizontal rules "---"/"***",
table-separator rows, front-matter, fenced code blocks), but leaves
inline formatting that applies to only PART of a line alone (bold **,
italic *, links [text](url), inline code `code`, etc.) even if in a
given line that inline formatting happens to span the entire line.

Output: a JSON array written to data/sentences.json, each item:
    {"id": <int>, "file": "<relative path in source repo>", "sentence": "<text>"}

This script never modifies or pushes to the source repository - it only
reads from it. The output file is committed to THIS (target) repo by the
calling GitHub Actions workflow.
"""

import argparse
import json
import re
import sys
from pathlib import Path

# Directories inside the source repo we never want to scan.
IGNORED_DIRS = {".git", "node_modules", "_site", ".github", "vendor", ".bundle"}

HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+")
BLOCKQUOTE_RE = re.compile(r"^\s{0,3}>+\s?")
LIST_MARKER_RE = re.compile(r"^\s{0,3}(?:[-*+]|\d+[.)])\s+")
HR_RE = re.compile(r"^\s{0,3}([-*_])(?:\s*\1){2,}\s*$")
TABLE_SEPARATOR_RE = re.compile(r"^\s{0,3}\|?[\s:\-|]+\|?\s*$")
PURE_HTML_TAG_RE = re.compile(r"^\s*<[^>]+>\s*$")
HTML_COMMENT_RE = re.compile(r"^\s*<!--.*-->\s*$")
FRONT_MATTER_DELIM_RE = re.compile(r"^-{3,}\s*$")
CODE_FENCE_RE = re.compile(r"^\s{0,3}```")


def strip_line_level_markdown(line: str) -> str:
    """Remove markdown syntax that applies to the whole line only."""
    stripped = line
    stripped = HEADING_RE.sub("", stripped)
    # Only strip blockquote / list markers once (they are structural, not inline)
    if BLOCKQUOTE_RE.match(stripped):
        stripped = BLOCKQUOTE_RE.sub("", stripped, count=1)
    if LIST_MARKER_RE.match(stripped):
        stripped = LIST_MARKER_RE.sub("", stripped, count=1)
    return stripped.strip()


def should_skip_line(line: str) -> bool:
    raw = line.rstrip("\n")
    if raw.strip() == "":
        return True
    if HR_RE.match(raw):
        return True
    if TABLE_SEPARATOR_RE.match(raw) and ("-" in raw or ":" in raw):
        return True
    if PURE_HTML_TAG_RE.match(raw):
        return True
    if HTML_COMMENT_RE.match(raw):
        return True
    return False


def iter_content_lines(text: str):
    """Yield content lines from a markdown file's text, honoring front-matter
    and fenced code blocks (both are skipped entirely)."""
    lines = text.splitlines()
    i = 0
    n = len(lines)

    # Skip a leading YAML front-matter block: --- ... ---
    if n > 0 and FRONT_MATTER_DELIM_RE.match(lines[0].strip()):
        j = 1
        while j < n and not FRONT_MATTER_DELIM_RE.match(lines[j].strip()):
            j += 1
        i = j + 1  # skip past the closing ---

    in_code_fence = False
    while i < n:
        line = lines[i]
        if CODE_FENCE_RE.match(line):
            in_code_fence = not in_code_fence
            i += 1
            continue
        if in_code_fence:
            i += 1
            continue
        yield line
        i += 1


def extract_from_file(path: Path):
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = path.read_text(encoding="utf-8", errors="ignore")

    sentences = []
    for line in iter_content_lines(text):
        if should_skip_line(line):
            continue
        cleaned = strip_line_level_markdown(line)
        if cleaned:
            sentences.append(cleaned)
    return sentences


def find_md_files(source_dir: Path):
    files = []
    for path in source_dir.rglob("*.md"):
        if any(part in IGNORED_DIRS for part in path.relative_to(source_dir).parts):
            continue
        files.append(path)
    for path in source_dir.rglob("*.markdown"):
        if any(part in IGNORED_DIRS for part in path.relative_to(source_dir).parts):
            continue
        files.append(path)
    return sorted(set(files))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="Path to checked-out source repo")
    parser.add_argument("--output", required=True, help="Path to write sentences.json")
    args = parser.parse_args()

    source_dir = Path(args.source).resolve()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not source_dir.exists():
        print(f"ERROR: source directory {source_dir} does not exist", file=sys.stderr)
        sys.exit(1)

    md_files = find_md_files(source_dir)
    print(f"Found {len(md_files)} markdown files under {source_dir}")

    records = []
    next_id = 1
    for path in md_files:
        rel_path = str(path.relative_to(source_dir))
        for sentence in extract_from_file(path):
            records.append({"id": next_id, "file": rel_path, "sentence": sentence})
            next_id += 1

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(records)} sentences from {len(md_files)} files to {output_path}")


if __name__ == "__main__":
    main()
