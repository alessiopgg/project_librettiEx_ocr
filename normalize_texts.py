#!/usr/bin/env python3
"""Normalize OCR references and model outputs for normalized CER/WER evaluation."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent
GROUND_TRUTH_DIR = PROJECT_ROOT / "data" / "text_visionato"
RAW_OUTPUT_DIR = PROJECT_ROOT / "output"
NORMALIZED_ROOT = PROJECT_ROOT / "evaluation" / "normalized"
NORMALIZED_REFERENCES_DIR = NORMALIZED_ROOT / "references"
NORMALIZED_HYPOTHESES_DIR = NORMALIZED_ROOT / "hypotheses"

TRANSCRIPTION_PATTERN = "libretto_*.txt"

UNCLEAR_OPEN_RE = re.compile(r"<\s*unclear\s*>", flags=re.IGNORECASE)
UNCLEAR_CLOSE_RE = re.compile(r"<\s*/\s*unclear\s*>", flags=re.IGNORECASE)
EMPTY_UNREADABLE_TAG_RE = re.compile(
    r"<\s*(?:gap|illegible)\s*/?\s*>", flags=re.IGNORECASE
)


def normalize_text(text: str) -> str:
    """Apply the agreed normalization rules to one transcription."""

    # Technical normalization.
    text = text.replace("\ufeff", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = unicodedata.normalize("NFC", text)

    # Keep the content of <unclear>...</unclear>, but remove its markup.
    text = UNCLEAR_OPEN_RE.sub(" ", text)
    text = UNCLEAR_CLOSE_RE.sub(" ", text)

    # Remove placeholders that do not contain a readable transcription.
    text = EMPTY_UNREADABLE_TAG_RE.sub(" ", text)

    # Ignore upper/lower case while preserving accents and numbers.
    text = text.casefold()

    normalized_characters: list[str] = []
    for character in text:
        category = unicodedata.category(character)

        if character.isspace():
            normalized_characters.append(" ")
        elif category.startswith(("P", "S")):
            # Punctuation and symbols become separators rather than disappearing,
            # so that "12-06" becomes "12 06", not "1206".
            normalized_characters.append(" ")
        else:
            normalized_characters.append(character)

    # Collapse every whitespace sequence to one normal space and trim the text.
    return " ".join("".join(normalized_characters).split())


def normalize_file(source: Path, destination: Path) -> None:
    """Read a UTF-8 file, normalize it, and save it without a final newline."""

    raw_text = source.read_text(encoding="utf-8")
    normalized_text = normalize_text(raw_text)

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(normalized_text, encoding="utf-8", newline="\n")


def relative_to_project(path: Path) -> str:
    """Return a stable project-relative path for reports."""

    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def normalize_references() -> set[str]:
    """Normalize all manually reviewed ground-truth transcriptions."""

    reference_files = sorted(GROUND_TRUTH_DIR.glob(TRANSCRIPTION_PATTERN))
    if not reference_files:
        raise RuntimeError(
            f"No ground-truth files found in {GROUND_TRUTH_DIR} "
            f"with pattern {TRANSCRIPTION_PATTERN!r}."
        )

    for source in reference_files:
        destination = NORMALIZED_REFERENCES_DIR / source.name
        normalize_file(source, destination)

    return {path.name for path in reference_files}


def normalize_hypotheses(reference_names: set[str]) -> list[dict[str, Any]]:
    """Normalize every model transcription in every experiment folder."""

    model_reports: list[dict[str, Any]] = []
    experiment_dirs = sorted(
        path for path in RAW_OUTPUT_DIR.glob("experiment_*") if path.is_dir()
    )

    for experiment_dir in experiment_dirs:
        model_dirs = sorted(path for path in experiment_dir.iterdir() if path.is_dir())

        for model_dir in model_dirs:
            source_files = sorted(model_dir.glob(TRANSCRIPTION_PATTERN))
            if not source_files:
                continue

            destination_dir = (
                NORMALIZED_HYPOTHESES_DIR / experiment_dir.name / model_dir.name
            )

            for source in source_files:
                normalize_file(source, destination_dir / source.name)

            model_names = {path.name for path in source_files}
            model_reports.append(
                {
                    "experiment": experiment_dir.name,
                    "model": model_dir.name,
                    "normalized_files": len(source_files),
                    "missing_ground_truth_files": sorted(reference_names - model_names),
                    "extra_output_files": sorted(model_names - reference_names),
                }
            )

    return model_reports


def write_report(reference_names: set[str], model_reports: list[dict[str, Any]]) -> None:
    """Save the rules and a summary of the normalized material."""

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(PROJECT_ROOT),
        "sources": {
            "ground_truth": relative_to_project(GROUND_TRUTH_DIR),
            "model_outputs": relative_to_project(RAW_OUTPUT_DIR),
        },
        "destinations": {
            "references": relative_to_project(NORMALIZED_REFERENCES_DIR),
            "hypotheses": relative_to_project(NORMALIZED_HYPOTHESES_DIR),
        },
        "normalization_rules": {
            "unicode": "NFC",
            "case": "casefold",
            "line_breaks_and_whitespace": "replace with spaces and collapse",
            "punctuation_and_symbols": "replace with spaces",
            "accents": "preserved",
            "numbers": "preserved",
            "word_order": "preserved",
            "unclear_tag": "remove markup and preserve enclosed text",
            "gap_and_illegible_tags": "remove",
            "model_introductions_repetitions_and_hallucinations": "preserved as words",
        },
        "reference_files": len(reference_names),
        "models": model_reports,
    }

    report_path = NORMALIZED_ROOT / "normalization_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
        newline="\n",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Normalize the reviewed ground truth and all OCR model outputs "
            "without modifying the original files."
        )
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete evaluation/normalized before regenerating all normalized files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not GROUND_TRUTH_DIR.is_dir():
        raise FileNotFoundError(f"Ground-truth directory not found: {GROUND_TRUTH_DIR}")
    if not RAW_OUTPUT_DIR.is_dir():
        raise FileNotFoundError(f"Output directory not found: {RAW_OUTPUT_DIR}")

    if args.clean and NORMALIZED_ROOT.exists():
        shutil.rmtree(NORMALIZED_ROOT)

    reference_names = normalize_references()
    model_reports = normalize_hypotheses(reference_names)
    write_report(reference_names, model_reports)

    normalized_model_files = sum(
        report["normalized_files"] for report in model_reports
    )

    print("Normalization completed")
    print(f"Ground-truth files: {len(reference_names)}")
    print(f"Model configurations: {len(model_reports)}")
    print(f"Model output files: {normalized_model_files}")
    print(f"Destination: {NORMALIZED_ROOT}")
    print(f"Report: {NORMALIZED_ROOT / 'normalization_report.json'}")


if __name__ == "__main__":
    main()
