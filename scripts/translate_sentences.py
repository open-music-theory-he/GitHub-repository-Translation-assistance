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

import json
import os
import sys
import time
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Configuration (overridable via environment variables set in the workflow)
# ---------------------------------------------------------------------------
SENTENCES_FILE = Path(os.environ.get("SENTENCES_FILE", "data/sentences.json"))
TRANSLATIONS_FILE = Path(os.environ.get("TRANSLATIONS_FILE", "data/translations.json"))

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
# gemini-2.5-flash: strong balance of translation quality (handles the
# terminology/formatting rules below reliably) and cost, and is not on the
# deprecation list (unlike gemini-2.0-flash, shut down June 2026). Override
# with GEMINI_MODEL if you want to trade cost for quality:
#   gemini-2.5-flash-lite -> cheapest, use if budget is the top priority
#   gemini-2.5-flash      -> default, good quality/cost balance
#   gemini-3.5-flash      -> highest quality/context, more expensive
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
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

    prompt = TRANSLATION_PROMPT_TEMPLATE.format(text=sentence)
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.2},
    }
    headers = {"Content-Type": "application/json", "x-goog-api-key": GEMINI_API_KEY}

    response = requests.post(GEMINI_ENDPOINT, headers=headers, json=body, timeout=60)
    if response.status_code != 200:
        raise RuntimeError(f"Gemini API returned {response.status_code}: {response.text[:500]}")

    data = response.json()
    candidates = data.get("candidates") or []
    if not candidates:
        raise RuntimeError(f"Gemini API returned no candidates: {json.dumps(data)[:500]}")

    parts = candidates[0].get("content", {}).get("parts", [])
    text = "".join(p.get("text", "") for p in parts).strip()
    if not text:
        raise RuntimeError(f"Gemini API returned an empty translation: {json.dumps(data)[:500]}")
    return text


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
    print(f"{len(done_ids)} sentences already translated, {len(pending)} remaining.")

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

        translations.append(
            {
                "id": sentence_id,
                "file": source_file,
                "source_sentence": source_sentence,
                "translated_sentence": translated_text,
            }
        )
        state["translations"] = translations
        print(
            f"[id={sentence_id}] file={source_file}\n"
            f"  INPUT : {source_sentence}\n"
            f"  OUTPUT: {translated_text}"
        )

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
