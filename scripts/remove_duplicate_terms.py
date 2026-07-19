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

Additionally (small extra pass, unrelated to the parenthesis logic above):
every occurrence of the term "מייג'ור" in translated_sentence is normalized
to "מז'ור" (handles a few common apostrophe/geresh variants). This is
logged per file too.

Also: stray English solfège syllables (Do, Re, Mi, Fa, Sol, La, Si, Ti) left
untranslated inside translated_sentence are converted to their Hebrew
equivalent (Do -> דו, Sol -> סול, etc.), but ONLY when:
    - it's the whole word, not part of a longer word (e.g. never touches
      "Domain"), and
    - the nearest adjacent word (skipping over spaces/punctuation in either
      direction) is not already that same Hebrew equivalent - i.e. if the
      Hebrew translation is already sitting right next to it as a gloss
      (e.g. "Do (דו)"), it is left alone rather than duplicated.
This is also logged per file.

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

# Matches "מייג'ור" with any of the common apostrophe/geresh characters
# people (and LLMs) use: straight apostrophe ', Hebrew geresh ׳, curly ’.
MAJOR_TERM_RE = re.compile(r"מייג['\u05F3\u2019]ור")
MAJOR_TERM_REPLACEMENT = "מז'ור"

# Solfège syllables: exact capitalization as used in the translation prompt's
# own examples (Do -> דו, Sol -> סול). Matched as a whole word only (lookarounds
# reject a preceding/following Latin letter, so "Domain" is never touched).
SOLFEGE_MAP = {
    "Do": "דו",
    "Re": "רה",
    "Mi": "מי",
    "Fa": "פה",
    "Sol": "סול",
    "So": "סול",
    "La": "לה",
    "Si": "סי",
    "Ti": "סי",
}
SOLFEGE_RE = re.compile(
    r"(?<![A-Za-z])(?:" + "|".join(re.escape(k) for k in SOLFEGE_MAP) + r")(?![A-Za-z])"
)
WORD_CHARS_RE = re.compile(r"[A-Za-z\u0590-\u05FF]+")


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


def get_adjacent_word(text: str, pos: int, forward: bool):
    """Nearest word (Hebrew or Latin letters) before (forward=False) or
    after (forward=True) position `pos`, skipping over any amount of
    whitespace/punctuation in between. Returns None if there is none."""
    if forward:
        match = WORD_CHARS_RE.search(text, pos)
        return match.group(0) if match else None
    else:
        matches = list(WORD_CHARS_RE.finditer(text, 0, pos))
        return matches[-1].group(0) if matches else None


def convert_solfege_terms(text: str):
    """Convert whole-word English solfège syllables to Hebrew, unless the
    Hebrew equivalent is already sitting right next to it (before or after,
    ignoring punctuation/spaces) as a gloss."""
    if not text:
        return text, 0

    result_parts = []
    last_end = 0
    count = 0
    for match in SOLFEGE_RE.finditer(text):
        key = match.group(0)
        translation = SOLFEGE_MAP[key]
        start, end = match.span()

        preceding = get_adjacent_word(text, start, forward=False)
        following = get_adjacent_word(text, end, forward=True)
        if preceding == translation or following == translation:
            continue  # Hebrew gloss already present - leave the English as-is

        result_parts.append(text[last_end:start])
        result_parts.append(translation)
        last_end = end
        count += 1

    result_parts.append(text[last_end:])
    return "".join(result_parts), count


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

        # Small extra pass: normalize מייג'ור -> מז'ור (independent of the
        # parenthesis de-duplication below, and safe to run first).
        major_term_count = 0
        solfege_count = 0
        for unit in units:
            text = unit.get("translated_sentence") or ""
            text, n = MAJOR_TERM_RE.subn(MAJOR_TERM_REPLACEMENT, text)
            major_term_count += n
            text, n = convert_solfege_terms(text)
            solfege_count += n
            unit["translated_sentence"] = text

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

        if major_term_count:
            log_lines.append(
                f' Replaced "מייג\'ור" with "מז\'ור" {major_term_count} times in this file'
            )
        if solfege_count:
            log_lines.append(
                f" Converted {solfege_count} solfège term(s) (Do/Re/Mi/Fa/Sol/So/La/Si/Ti) to Hebrew in this file"
            )

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
