#!/usr/bin/env python3
"""
remove_duplicate_terms.py

One-time (manually triggered) script.

Reads the final translations file (default: data/translations.json) and,
for every SOURCE file (grouped by the "file" field), looks for English
terms in round parentheses - e.g. "(Sub-Dominant)" - inside the Hebrew
translated_sentence text (this is exactly what translation rule 5 in the
prompt produces: "term in Hebrew (Original English Term)").

A parenthetical expression is eligible for de-duplication only if ALL of
these hold:
    a. It uses round parentheses ( ... ) - not [ ], { }, etc.
    b. The exact same expression (including the parentheses) appears at
       least twice within the same source file.
    c. There are NO Hebrew letters anywhere inside the parentheses.
    d. There are at least 4 Latin letters inside the parentheses (spaces
       and punctuation don't count towards this).
    e. It is not a link - i.e. not the "(url)" part of a Markdown link
       "[text](url)", and the content itself doesn't look like a URL.

For every eligible expression, every occurrence EXCEPT THE FIRST one in
that file is deleted, together with the single space that precedes it (if
there is one).

Output (this script never modifies the input file or the source repo):
    - data/translations_deduped.json : a full COPY of the input data, with
      the above deletions applied to translated_sentence values.
    - data/dedup_log.txt : a human-readable log of every expression that
      was removed from every file (or "Nothing deleted in this file").
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

PAREN_RE = re.compile(r"\(([^()]*)\)")
HEBREW_RE = re.compile(r"[\u0590-\u05FF]")
LATIN_LETTER_RE = re.compile(r"[A-Za-z]")
URL_CONTENT_RE = re.compile(r"^\s*(https?://|www\.)", re.IGNORECASE)


def is_eligible(content: str, full_sentence: str, match_start: int) -> bool:
    """Check conditions (c), (d), (e) for a parenthetical match's content.
    (a) round parentheses and (b) count>=2 are handled by the caller."""
    if HEBREW_RE.search(content):
        return False
    if len(LATIN_LETTER_RE.findall(content)) < 4:
        return False
    if URL_CONTENT_RE.match(content):
        return False
    # If the char right before "(" is "]", this is the URL part of a
    # Markdown link "[text](url)" - always treat that as a link, regardless
    # of what the content looks like.
    if match_start > 0 and full_sentence[match_start - 1] == "]":
        return False
    return True


def compute_file_counts(units: list) -> dict:
    """First pass: count, per exact parenthetical expression (full text
    including the parentheses), how many eligible occurrences exist in
    this file. Only expressions with count >= 2 are candidates for
    deletion."""
    counts = defaultdict(int)
    for unit in units:
        text = unit.get("translated_sentence") or ""
        for match in PAREN_RE.finditer(text):
            full_expr = match.group(0)
            content = match.group(1)
            if is_eligible(content, text, match.start()):
                counts[full_expr] += 1
    return {expr: c for expr, c in counts.items() if c >= 2}


def dedupe_unit_text(text: str, qualifying_exprs: dict, seen: set, deletions: dict):
    """Second pass over a single unit's text: walk matches left-to-right and
    delete every occurrence of a qualifying expression except the first one
    ever seen in this file (tracked via the `seen` set, shared/mutated
    across all units of the same file, processed in id order)."""
    matches = list(PAREN_RE.finditer(text))
    if not matches:
        return text

    result_parts = []
    last_end = 0
    for match in matches:
        full_expr = match.group(0)
        content = match.group(1)
        start, end = match.span()

        if not is_eligible(content, text, start):
            continue
        if full_expr not in qualifying_exprs:
            continue

        if full_expr not in seen:
            # First occurrence in the whole file - keep it, don't touch.
            seen.add(full_expr)
            continue

        # A later occurrence - delete it (and a single preceding space, if any).
        del_start = start
        if del_start > last_end and text[del_start - 1] == " ":
            del_start -= 1
        result_parts.append(text[last_end:del_start])
        last_end = end
        deletions[full_expr] += 1

    result_parts.append(text[last_end:])
    return "".join(result_parts)


def process(data: dict):
    translations = data.get("translations", [])
    by_file = defaultdict(list)
    for unit in translations:
        by_file[unit["file"]].append(unit)

    log_lines = []
    for file_path in sorted(by_file.keys()):
        units = sorted(by_file[file_path], key=lambda u: u["id"])
        qualifying_exprs = compute_file_counts(units)

        seen = set()
        deletions = defaultdict(int)  # full_expr -> number of deletions in this file
        for unit in units:
            original = unit.get("translated_sentence") or ""
            updated = dedupe_unit_text(original, qualifying_exprs, seen, deletions)
            unit["translated_sentence"] = updated

        log_lines.append(file_path)
        if deletions:
            # Preserve first-seen order of expressions for a stable, readable log.
            ordered_exprs = [e for e in qualifying_exprs if e in deletions]
            for expr in ordered_exprs:
                log_lines.append(expr)
                log_lines.append(f" Deleted {deletions[expr]} times in this file")
        else:
            log_lines.append("Nothing deleted in this file")

    return data, log_lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", default="data/translations.json", help="Path to the final translations JSON"
    )
    parser.add_argument(
        "--output",
        default="data/translations_deduped.json",
        help="Path to write the de-duplicated COPY (input file is never modified)",
    )
    parser.add_argument(
        "--log", default="data/dedup_log.txt", help="Path to write the human-readable log"
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    log_path = Path(args.log)

    with input_path.open(encoding="utf-8") as f:
        data = json.load(f)

    updated_data, log_lines = process(data)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(updated_data, f, ensure_ascii=False, indent=2)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(log_lines) + "\n")

    total_deletions = sum(
        line.count("Deleted") for line in log_lines
    )  # rough count for the console summary
    print(f"Wrote de-duplicated translations to {output_path}")
    print(f"Wrote log ({len(log_lines)} lines, ~{total_deletions} expressions deleted) to {log_path}")


if __name__ == "__main__":
    main()
