#!/usr/bin/env python3
"""
push_translations.py

One-time (manually triggered) script.

Reads the final, de-duplicated translations file (default:
data/translations_deduped.json) and writes the translations directly into
a LOCAL, already-checked-out copy of the source repo (default:
./source-repo). This script itself does NOT touch git at all - it only
edits files on disk. The calling GitHub Actions workflow is responsible for
committing those changes to a new branch and opening a Pull Request against
the source repo (rather than pushing directly to its main branch), so a
human can review the diff before it goes live.

Algorithm (per the spec):
    for each source file (grouped by the "file" field, sorted for
    deterministic, repeatable runs):
        read the file's current text
        for each translation unit, in ascending id order:
            find the source_sentence text in the current file text
            if found: replace ONLY THE FIRST occurrence with
                      translated_sentence, then continue with the next unit
                      (searching in the now-updated text)
            if not found: log a warning and skip this unit (file is left
                      untouched for that particular sentence)
        if anything changed, write the file back
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path


def apply_file_translations(path: Path, units: list):
    text = path.read_text(encoding="utf-8")
    replaced = 0
    unchanged = 0
    not_found = 0

    for unit in sorted(units, key=lambda u: u["id"]):
        source_sentence = unit["source_sentence"]
        translated_sentence = unit["translated_sentence"]

        if source_sentence == translated_sentence:
            # Nothing to do (e.g. script 2 kept the original text as-is,
            # such as for code/non-prose lines flagged by the model).
            unchanged += 1
            continue

        idx = text.find(source_sentence)
        if idx == -1:
            not_found += 1
            print(
                f"[id={unit['id']}] WARNING: source text not found in {path} - skipping.\n"
                f"  EXPECTED: {source_sentence}"
            )
            continue

        text = text[:idx] + translated_sentence + text[idx + len(source_sentence):]
        replaced += 1
        print(
            f"[id={unit['id']}] file={unit['file']}\n"
            f"  SOURCE     : {source_sentence}\n"
            f"  TRANSLATED : {translated_sentence}"
        )

    return text, replaced, unchanged, not_found


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--translations",
        default="data/translations_deduped.json",
        help="Path to the final (de-duplicated) translations JSON",
    )
    parser.add_argument(
        "--source", default="./source-repo", help="Path to the checked-out, writable source repo"
    )
    args = parser.parse_args()

    translations_path = Path(args.translations)
    source_dir = Path(args.source)

    if not translations_path.exists():
        raise SystemExit(
            f"ERROR: {translations_path} not found. Run script 3 (de-duplication) first, "
            "or pass --translations to point at a different file."
        )
    if not source_dir.exists():
        raise SystemExit(f"ERROR: source directory {source_dir} does not exist")

    with translations_path.open(encoding="utf-8") as f:
        data = json.load(f)

    if not data.get("completed", False):
        print(
            "WARNING: the translations file is not marked completed=true - "
            "this run will only push whatever translations exist so far."
        )

    by_file = defaultdict(list)
    for unit in data.get("translations", []):
        by_file[unit["file"]].append(unit)

    total_replaced = 0
    total_unchanged = 0
    total_not_found = 0
    files_written = 0

    for file_path in sorted(by_file.keys()):
        full_path = source_dir / file_path
        if not full_path.exists():
            print(f"WARNING: {full_path} does not exist in the source repo - skipping this file entirely.")
            continue

        new_text, replaced, unchanged, not_found = apply_file_translations(full_path, by_file[file_path])
        total_replaced += replaced
        total_unchanged += unchanged
        total_not_found += not_found

        if replaced > 0:
            full_path.write_text(new_text, encoding="utf-8")
            files_written += 1
            print(f"Wrote {replaced} translation(s) into {file_path} ({not_found} not found, {unchanged} unchanged).")
        else:
            print(f"No changes for {file_path} ({not_found} not found, {unchanged} unchanged).")

    print(
        f"\nDone. Files written: {files_written}. "
        f"Total replaced: {total_replaced}. Total unchanged (identical): {total_unchanged}. "
        f"Total not found: {total_not_found}."
    )
    if total_not_found:
        print(
            "NOTE: some source sentences were not found verbatim in their file - this can happen "
            "if the source file changed since sentences.json was generated. Review the WARNING "
            "lines above."
        )


if __name__ == "__main__":
    main()
