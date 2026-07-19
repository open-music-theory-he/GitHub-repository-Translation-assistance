#!/usr/bin/env python3
"""
translate_sentences.py

Runs on a schedule (every 2 hours, via GitHub Actions cron) plus can be
triggered manually. Reads data/sentences.json (produced once by
extract_sentences.py) and progressively translates each sentence into
Hebrew using the Gemini API, writing results to data/translations.json.

Behaviour, per spec:
- If translations.json already has "completed": true -> do nothing and exit.
- Otherwise, resume from the first sentence that has no translation yet.
- Sleep SLEEP_BETWEEN_SECONDS between successful calls (basic rate limiting).
- On an API error for the current sentence: retry the SAME sentence, with
  an increasing back-off delay. After 3 total failed attempts (no success
  in between) in this run, save progress and stop (the next scheduled run,
  2 hours later, will retry the same sentence).
- Any single successful call resets the error/attempt counter to 0.
- If the model determines a line is pure code / has no real prose to
  translate (see rule 9 in the prompt), it returns a sentinel token instead
  of a translation; this script detects that and stores the ORIGINAL text
  as the "translation" instead, printing a clear message about it.
- If TIME_BUDGET_SECONDS (default 1800 = 30 min) elapses, finish the
  sentence currently being processed and then stop, even with no errors.
- Sets "completed": true once every sentence has a translation, and on
  every subsequent run (completed == true) exits immediately without
  doing anything or touching the source repo.
- Never writes to / pushes to the SOURCE repository - it only ever reads
  data/sentences.json and writes data/translations.json, both inside
  this (target/translation-assistance) repo. Git commit + push is handled
  by the calling GitHub Actions workflow, not by this script.
"""

import functools
import json
import os
import sys
import time
from pathlib import Path

import requests

# Force every print() to flush immediately. Without this, stdout is
# block-buffered when it isn't a real terminal (which is the case inside
# GitHub Actions), so log lines can sit invisible in the buffer for a long
# time (or until the process exits) even though the script IS making
# progress. This is what looks like "the script is stuck" in the Actions UI.
print = functools.partial(print, flush=True)  # noqa: A001

# ---------------------------------------------------------------------------
# Configuration (overridable via environment variables set in the workflow)
# ---------------------------------------------------------------------------
SENTENCES_FILE = Path(os.environ.get("SENTENCES_FILE", "data/sentences.json"))
TRANSLATIONS_FILE = Path(os.environ.get("TRANSLATIONS_FILE", "data/translations.json"))

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
# gemini-3.5-flash: Google's current recommended default for new integrations
# (GA, fast, cost-effective, multimodal). NOTE: the whole Gemini 2.5 family
# (gemini-2.5-flash AND gemini-2.5-flash-lite) started hard-failing with a
# "This model ... is no longer available" 404 for ALL callers starting
# July 9 2026 - ahead of their officially announced Oct 16 2026 retirement
# date (see https://discuss.ai.google.dev, multiple reports). Do not switch
# back to any gemini-2.5-* model as a "fix" for this error. Override with
# GEMINI_MODEL if you want a different one, e.g.:
#   gemini-3.1-flash-lite -> cheapest current-generation option
#   gemini-3.5-flash      -> default here, good quality/cost balance
#   gemini-3.1-pro-preview / gemini-3-pro-preview -> highest quality, pricier
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-lite")
GEMINI_ENDPOINT = (
    f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
)

TIME_BUDGET_SECONDS = int(os.environ.get("TIME_BUDGET_SECONDS", 30 * 60))
SLEEP_BETWEEN_SECONDS = float(os.environ.get("SLEEP_BETWEEN_SECONDS", 4))
# Back-off delays after 1st and 2nd failed attempts on the same sentence.
# A 3rd consecutive failure gives up for this run.
RETRY_DELAYS_SECONDS = [
    int(x) for x in os.environ.get("RETRY_DELAYS_SECONDS", "60,300").split(",")
]
MAX_ATTEMPTS = len(RETRY_DELAYS_SECONDS) + 1  # e.g. 2 delays -> 3 total attempts

# If the model decides a line is pure code / not natural language at all, it
# is instructed to return exactly this token instead of attempting a
# translation. We then keep the original source text unchanged. Chosen to be
# an unambiguous, all-caps token that would never plausibly appear inside a
# real Hebrew (or English) translation.
NO_TRANSLATION_SENTINEL = "NO_TRANSLATION_NEEDED"

TRANSLATION_PROMPT_TEMPLATE = """You are an expert translator specializing in music theory literature and academic documents. Your task is to translate the provided text into Hebrew, strictly adhering to the formatting and terminology rules below.
CRITICAL RULES:
1. Translate ONLY the provided text at the end of this message. Do not include any introductory text, explanations, greetings, or conversational filler. Return ONLY the final translated output.
2. Preserve all Markdown formatting (such as headings #, bold **, italics *, links [], code blocks, and lists) exactly as they are in the original text. Do not translate code syntax, HTML tags, or URL paths.
3. Musical Notes: Keep the standard Latin letters (C, D, E, F, G, A, B) as English capital letters. Example: "A minor" -> "A מינור", "C major seventh" -> "C מייג'ור שבע".
4. Solfège Syllables: Translate spoken note degrees (Do, Re, Mi, Fa, Sol, La, Si/Ti) into Hebrew text. Example: "Do" -> "דו", "Sol" -> "סול".
5. Complex Terminology: For advanced music theory concepts, translate the term into Hebrew and include the original English term in parentheses next to it. Example: "Tritone Substitution" -> "חילופי טריטון (Tritone Substitution)".
6. Graphical Accidentals: Keep graphical symbols (such as ♯, ♭, ♮) exactly as they are.
7. Verbal Accidentals: Translate written accidentals into Hebrew. Example: "Flat" -> "במול", "Sharp" -> "דיאז", "Natural" -> "בקר".
8. Roman Numerals: Keep Roman numerals for chord analysis (I, iv, V7, etc.) exactly as written in Latin characters. Do not translate or change their case.
9. Non-Prose Lines: If, and ONLY if, the entire input line contains no human-readable prose whatsoever - for example it is purely code, a template/variable expression (such as Liquid `{{{{ ... }}}}` or `{{% ... %}}`), a raw shell command, a bare URL, or a standalone data value - respond with EXACTLY the single token {sentinel} and nothing else, no punctuation, no quotes. Use this ONLY when there is truly nothing to translate. If the line contains any actual word, phrase, or sentence a human reader would read as text (even a short one, even mixed in with code or a template tag), translate that part normally instead and do NOT return {sentinel}.
Strictly follow these rules. The text to translate is provided in the next line:
{text}"""


def load_sentences():
    if not SENTENCES_FILE.exists():
        print(
            f"ERROR: {SENTENCES_FILE} not found. Run extract_sentences.py (script 1) first.",
            file=sys.stderr,
        )
        sys.exit(1)
    with SENTENCES_FILE.open(encoding="utf-8") as f:
        return json.load(f)


def load_translations_state():
    if TRANSLATIONS_FILE.exists():
        with TRANSLATIONS_FILE.open(encoding="utf-8") as f:
            return json.load(f)
    return {"completed": False, "translations": []}


def save_translations_state(state):
    TRANSLATIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    # Rebuild dict so "completed" is always the first key, per spec.
    ordered = {"completed": state["completed"], "translations": state["translations"]}
    with TRANSLATIONS_FILE.open("w", encoding="utf-8") as f:
        json.dump(ordered, f, ensure_ascii=False, indent=2)


def call_gemini(sentence: str) -> str:
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY environment variable is not set")

    prompt = TRANSLATION_PROMPT_TEMPLATE.format(text=sentence, sentinel=NO_TRANSLATION_SENTINEL)
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.2},
    }
    headers = {"Content-Type": "application/json", "x-goog-api-key": GEMINI_API_KEY}

    response = requests.post(GEMINI_ENDPOINT, headers=headers, json=body, timeout=60)
    if response.status_code != 200:
        hint = ""
        if response.status_code == 404:
            hint = (
                f" Hint: 404 from Gemini for model '{GEMINI_MODEL}' usually means either "
                "(a) that specific model was retired/is temporarily unavailable (this hit the "
                "whole gemini-2.5-flash / gemini-2.5-flash-lite family for all users starting "
                "July 9 2026, well before their announced Oct 2026 retirement - if you're on a "
                "gemini-2.5-* model, switch to a gemini-3.x model instead), or (b) the API key "
                "itself is not a valid Generative Language API key (a real key from "
                "https://aistudio.google.com/apikey normally starts with 'AIzaSy...'). "
                "Verify by running: curl -H \"x-goog-api-key: $GEMINI_API_KEY\" "
                "https://generativelanguage.googleapis.com/v1beta/models "
                "and checking the key is listed there and generateContent is supported."
            )
        elif response.status_code in (401, 403):
            hint = " Hint: this usually means the API key is missing, invalid, or restricted."
        elif response.status_code == 429:
            hint = " Hint: rate limit exceeded - the retry back-off should handle this."
        raise RuntimeError(
            f"Gemini API returned {response.status_code}: {response.text[:500]}{hint}"
        )

    data = response.json()
    candidates = data.get("candidates") or []
    if not candidates:
        raise RuntimeError(f"Gemini API returned no candidates: {json.dumps(data)[:500]}")

    parts = candidates[0].get("content", {}).get("parts", [])
    text = "".join(p.get("text", "") for p in parts).strip()
    if not text:
        raise RuntimeError(f"Gemini API returned an empty translation: {json.dumps(data)[:500]}")
    return text


def is_no_translation_sentinel(text: str) -> bool:
    """True if the model's response is (essentially) just the sentinel token,
    signalling 'this line is pure code/non-prose, nothing to translate' -
    as opposed to a real translation that happens to contain the token
    somewhere inside a longer sentence (which we do NOT want to treat as
    a signal, to avoid ever bypassing a real translation)."""
    stripped = text.strip().strip("`'\"“”׳״.,;: \n\t")
    return stripped.upper() == NO_TRANSLATION_SENTINEL


def main():
    start_time = time.time()

    state = load_translations_state()
    if state.get("completed"):
        print("translations.json is already marked completed=true. Nothing to do.")
        return

    sentences = load_sentences()
    translations = state.get("translations", [])
    done_ids = {t["id"] for t in translations}

    pending = [s for s in sentences if s["id"] not in done_ids]
    print(
        f"Starting run: {len(pending)} sentences remaining "
        f"({len(done_ids)} already translated out of {len(sentences)} total)."
    )

    if not pending:
        state["completed"] = True
        save_translations_state(state)
        print("All sentences were already translated. Marked completed=true.")
        return

    consecutive_errors = 0

    for item in pending:
        elapsed = time.time() - start_time
        if elapsed >= TIME_BUDGET_SECONDS:
            print(f"Time budget of {TIME_BUDGET_SECONDS}s reached. Stopping before a new sentence.")
            break

        sentence_id = item["id"]
        source_file = item["file"]
        source_sentence = item["sentence"]

        attempt = 0
        translated_text = None
        while attempt < MAX_ATTEMPTS:
            attempt += 1
            try:
                translated_text = call_gemini(source_sentence)
                consecutive_errors = 0
                break
            except Exception as exc:  # noqa: BLE001
                consecutive_errors += 1
                print(
                    f"[id={sentence_id}] attempt {attempt}/{MAX_ATTEMPTS} failed: {exc}\n"
                    f"  INPUT : {source_sentence}"
                )
                if attempt >= MAX_ATTEMPTS:
                    print(
                        f"Giving up after {MAX_ATTEMPTS} consecutive failed attempts. "
                        "Saving progress and stopping this run."
                    )
                    save_translations_state(state)
                    sys.exit(1)
                delay = RETRY_DELAYS_SECONDS[attempt - 1]
                print(f"Waiting {delay}s before retrying the same sentence...")
                time.sleep(delay)

        if translated_text is None:
            # Should not happen (handled above), but guard anyway.
            break

        if is_no_translation_sentinel(translated_text):
            translated_text = source_sentence
            print(
                f"[id={sentence_id}] file={source_file}\n"
                f"  >>> MODEL FLAGGED THIS AS CODE/NON-PROSE - KEEPING ORIGINAL TEXT AS-IS <<<\n"
                f"  INPUT/OUTPUT (unchanged): {source_sentence}"
            )
        else:
            print(
                f"[id={sentence_id}] file={source_file}\n"
                f"  INPUT : {source_sentence}\n"
                f"  OUTPUT: {translated_text}"
            )

        translations.append(
            {
                "id": sentence_id,
                "file": source_file,
                "source_sentence": source_sentence,
                "translated_sentence": translated_text,
            }
        )
        state["translations"] = translations

        # Periodically persist progress so a crash mid-run doesn't lose work.
        save_translations_state(state)

        time.sleep(SLEEP_BETWEEN_SECONDS)

    all_done = len({t["id"] for t in translations}) == len(sentences)
    state["completed"] = all_done
    save_translations_state(state)

    if all_done:
        print("All sentences translated. Marked completed=true.")
    else:
        print(f"Run finished. {len(translations)}/{len(sentences)} sentences translated so far.")


if __name__ == "__main__":
    main()
