#!/usr/bin/env python3
"""Compute normalized CER and WER from previously normalized OCR texts.

Expected project layout:

    evaluation/normalized/references/libretto_*.txt
    evaluation/normalized/hypotheses/experiment_*/<model>/libretto_*.txt

Outputs:

    evaluation/metrics/normalized/per_file_metrics.csv
    evaluation/metrics/normalized/model_summary.csv
    evaluation/metrics/normalized/metrics_report.json

The script does not modify references, hypotheses, or raw OCR outputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import statistics
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

try:
    from rapidfuzz.distance import Levenshtein
except ImportError as exc:  # pragma: no cover - depends on the runtime environment
    raise SystemExit(
        "Missing dependency 'rapidfuzz'. Install it with:\n"
        "  python3 -m pip install rapidfuzz\n"
        "Then run this script again."
    ) from exc


PROJECT_ROOT = Path(__file__).resolve().parent
NORMALIZED_ROOT = PROJECT_ROOT / "evaluation" / "normalized"
REFERENCES_DIR = NORMALIZED_ROOT / "references"
HYPOTHESES_DIR = NORMALIZED_ROOT / "hypotheses"
METRICS_DIR = PROJECT_ROOT / "evaluation" / "metrics" / "normalized"

TRANSCRIPTION_PATTERN = "libretto_*.txt"
FLOAT_DIGITS = 8


@dataclass(frozen=True)
class EditCounts:
    substitutions: int
    deletions: int
    insertions: int

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions


@dataclass(frozen=True)
class FileMetrics:
    experiment: str
    model: str
    file: str
    status: str
    reference_characters: int
    hypothesis_characters: int
    character_substitutions: int
    character_deletions: int
    character_insertions: int
    character_errors: int
    cer: float | None
    reference_words: int
    hypothesis_words: int
    word_substitutions: int
    word_deletions: int
    word_insertions: int
    word_errors: int
    wer: float | None
    exact_match: bool


PER_FILE_FIELDS = [
    "experiment",
    "model",
    "file",
    "status",
    "reference_characters",
    "hypothesis_characters",
    "character_substitutions",
    "character_deletions",
    "character_insertions",
    "character_errors",
    "cer",
    "reference_words",
    "hypothesis_words",
    "word_substitutions",
    "word_deletions",
    "word_insertions",
    "word_errors",
    "wer",
    "exact_match",
]

SUMMARY_FIELDS = [
    "experiment",
    "model",
    "reference_files",
    "found_hypotheses",
    "missing_hypotheses",
    "empty_hypotheses",
    "exact_matches",
    "exact_match_rate",
    "total_reference_characters",
    "total_hypothesis_characters",
    "character_substitutions",
    "character_deletions",
    "character_insertions",
    "character_errors",
    "micro_cer",
    "macro_cer_mean",
    "macro_cer_median",
    "macro_cer_population_stddev",
    "macro_cer_min",
    "macro_cer_max",
    "total_reference_words",
    "total_hypothesis_words",
    "word_substitutions",
    "word_deletions",
    "word_insertions",
    "word_errors",
    "micro_wer",
    "macro_wer_mean",
    "macro_wer_median",
    "macro_wer_population_stddev",
    "macro_wer_min",
    "macro_wer_max",
]


def read_text(path: Path) -> str:
    """Read a normalized UTF-8 transcription exactly as stored."""

    return path.read_text(encoding="utf-8")


def edit_counts(reference: Sequence[Any], hypothesis: Sequence[Any]) -> EditCounts:
    """Count unit-cost Levenshtein substitutions, deletions, and insertions."""

    operations = Levenshtein.editops(reference, hypothesis)
    counts = Counter(operation.tag for operation in operations)
    return EditCounts(
        substitutions=counts["replace"],
        deletions=counts["delete"],
        insertions=counts["insert"],
    )


def error_rate(error_count: int, reference_length: int, hypothesis_length: int) -> float | None:
    """Return errors/reference length; empty-reference cases are handled explicitly."""

    if reference_length > 0:
        return error_count / reference_length
    if hypothesis_length == 0:
        return 0.0
    return None


def evaluate_pair(
    experiment: str,
    model: str,
    filename: str,
    reference: str,
    hypothesis: str,
    status: str,
) -> FileMetrics:
    """Compute normalized character-level and word-level metrics for one file."""

    reference_words = reference.split()
    hypothesis_words = hypothesis.split()

    character_counts = edit_counts(reference, hypothesis)
    word_counts = edit_counts(reference_words, hypothesis_words)

    return FileMetrics(
        experiment=experiment,
        model=model,
        file=filename,
        status=status,
        reference_characters=len(reference),
        hypothesis_characters=len(hypothesis),
        character_substitutions=character_counts.substitutions,
        character_deletions=character_counts.deletions,
        character_insertions=character_counts.insertions,
        character_errors=character_counts.errors,
        cer=error_rate(character_counts.errors, len(reference), len(hypothesis)),
        reference_words=len(reference_words),
        hypothesis_words=len(hypothesis_words),
        word_substitutions=word_counts.substitutions,
        word_deletions=word_counts.deletions,
        word_insertions=word_counts.insertions,
        word_errors=word_counts.errors,
        wer=error_rate(word_counts.errors, len(reference_words), len(hypothesis_words)),
        exact_match=reference == hypothesis,
    )


def discover_references() -> dict[str, Path]:
    reference_files = sorted(REFERENCES_DIR.glob(TRANSCRIPTION_PATTERN))
    if not reference_files:
        raise RuntimeError(
            f"No normalized references found in {REFERENCES_DIR} "
            f"with pattern {TRANSCRIPTION_PATTERN!r}."
        )
    return {path.name: path for path in reference_files}


def discover_model_directories() -> list[tuple[str, str, Path]]:
    model_directories: list[tuple[str, str, Path]] = []

    experiment_directories = sorted(
        path for path in HYPOTHESES_DIR.glob("experiment_*") if path.is_dir()
    )
    for experiment_directory in experiment_directories:
        for model_directory in sorted(
            path for path in experiment_directory.iterdir() if path.is_dir()
        ):
            model_directories.append(
                (experiment_directory.name, model_directory.name, model_directory)
            )

    if not model_directories:
        raise RuntimeError(f"No normalized model directories found in {HYPOTHESES_DIR}.")

    return model_directories


def evaluate_model(
    experiment: str,
    model: str,
    model_directory: Path,
    references: dict[str, Path],
) -> tuple[list[FileMetrics], list[str]]:
    """Evaluate one experiment/model against the complete reference set."""

    hypothesis_paths = {
        path.name: path for path in model_directory.glob(TRANSCRIPTION_PATTERN)
    }
    rows: list[FileMetrics] = []

    for filename, reference_path in sorted(references.items()):
        reference = read_text(reference_path)
        hypothesis_path = hypothesis_paths.get(filename)

        if hypothesis_path is None:
            hypothesis = ""
            status = "missing_hypothesis"
        else:
            hypothesis = read_text(hypothesis_path)
            status = "empty_hypothesis" if hypothesis == "" else "ok"

        rows.append(
            evaluate_pair(
                experiment=experiment,
                model=model,
                filename=filename,
                reference=reference,
                hypothesis=hypothesis,
                status=status,
            )
        )

    extra_files = sorted(set(hypothesis_paths) - set(references))
    return rows, extra_files


def finite_values(rows: list[FileMetrics], attribute: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = getattr(row, attribute)
        if value is not None and math.isfinite(value):
            values.append(value)
    return values


def descriptive_statistics(values: list[float], prefix: str) -> dict[str, float | None]:
    if not values:
        return {
            f"{prefix}_mean": None,
            f"{prefix}_median": None,
            f"{prefix}_population_stddev": None,
            f"{prefix}_min": None,
            f"{prefix}_max": None,
        }

    return {
        f"{prefix}_mean": statistics.fmean(values),
        f"{prefix}_median": statistics.median(values),
        f"{prefix}_population_stddev": statistics.pstdev(values),
        f"{prefix}_min": min(values),
        f"{prefix}_max": max(values),
    }


def summarize_model(rows: list[FileMetrics]) -> dict[str, Any]:
    if not rows:
        raise ValueError("Cannot summarize an empty list of file metrics.")

    total_reference_characters = sum(row.reference_characters for row in rows)
    total_hypothesis_characters = sum(row.hypothesis_characters for row in rows)
    total_character_substitutions = sum(row.character_substitutions for row in rows)
    total_character_deletions = sum(row.character_deletions for row in rows)
    total_character_insertions = sum(row.character_insertions for row in rows)
    total_character_errors = sum(row.character_errors for row in rows)

    total_reference_words = sum(row.reference_words for row in rows)
    total_hypothesis_words = sum(row.hypothesis_words for row in rows)
    total_word_substitutions = sum(row.word_substitutions for row in rows)
    total_word_deletions = sum(row.word_deletions for row in rows)
    total_word_insertions = sum(row.word_insertions for row in rows)
    total_word_errors = sum(row.word_errors for row in rows)

    exact_matches = sum(row.exact_match for row in rows)

    summary: dict[str, Any] = {
        "experiment": rows[0].experiment,
        "model": rows[0].model,
        "reference_files": len(rows),
        "found_hypotheses": sum(row.status != "missing_hypothesis" for row in rows),
        "missing_hypotheses": sum(row.status == "missing_hypothesis" for row in rows),
        "empty_hypotheses": sum(row.status == "empty_hypothesis" for row in rows),
        "exact_matches": exact_matches,
        "exact_match_rate": exact_matches / len(rows),
        "total_reference_characters": total_reference_characters,
        "total_hypothesis_characters": total_hypothesis_characters,
        "character_substitutions": total_character_substitutions,
        "character_deletions": total_character_deletions,
        "character_insertions": total_character_insertions,
        "character_errors": total_character_errors,
        "micro_cer": (
            total_character_errors / total_reference_characters
            if total_reference_characters > 0
            else None
        ),
        "total_reference_words": total_reference_words,
        "total_hypothesis_words": total_hypothesis_words,
        "word_substitutions": total_word_substitutions,
        "word_deletions": total_word_deletions,
        "word_insertions": total_word_insertions,
        "word_errors": total_word_errors,
        "micro_wer": (
            total_word_errors / total_reference_words
            if total_reference_words > 0
            else None
        ),
    }

    summary.update(descriptive_statistics(finite_values(rows, "cer"), "macro_cer"))
    summary.update(descriptive_statistics(finite_values(rows, "wer"), "macro_wer"))
    return summary


def round_float(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, FLOAT_DIGITS)
    return value


def serializable_row(row: FileMetrics) -> dict[str, Any]:
    return {key: round_float(value) for key, value in asdict(row).items()}


def serializable_dict(data: dict[str, Any]) -> dict[str, Any]:
    return {key: round_float(value) for key, value in data.items()}


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def relative_to_project(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def write_report(
    references: dict[str, Path],
    all_rows: list[FileMetrics],
    summaries: list[dict[str, Any]],
    extra_outputs: list[dict[str, Any]],
) -> None:
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(PROJECT_ROOT),
        "evaluation_mode": "normalized",
        "inputs": {
            "references": relative_to_project(REFERENCES_DIR),
            "hypotheses": relative_to_project(HYPOTHESES_DIR),
            "reference_files": len(references),
        },
        "outputs": {
            "per_file_metrics": relative_to_project(
                METRICS_DIR / "per_file_metrics.csv"
            ),
            "model_summary": relative_to_project(METRICS_DIR / "model_summary.csv"),
        },
        "definitions": {
            "character_unit": "one Unicode character in the normalized text, including single spaces",
            "word_unit": "one token obtained with normalized_text.split()",
            "edit_distance": "unit-cost Levenshtein substitutions + deletions + insertions",
            "cer": "character_errors / reference_characters",
            "wer": "word_errors / reference_words",
            "micro": "sum of errors across files / sum of reference units across files",
            "macro": "descriptive statistics of per-file error rates",
            "missing_hypothesis": "evaluated as an empty hypothesis",
            "empty_reference": "per-file rate is 0 only when hypothesis is also empty; otherwise null",
            "rates_can_exceed_one": True,
        },
        "model_summaries": [serializable_dict(summary) for summary in summaries],
        "extra_output_files": extra_outputs,
        "status_counts": dict(Counter(row.status for row in all_rows)),
    }

    report_path = METRICS_DIR / "metrics_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
        newline="\n",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute normalized CER and WER for every experiment/model using "
            "the texts produced by normalize_texts.py."
        )
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete evaluation/metrics/normalized before writing new results.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not REFERENCES_DIR.is_dir():
        raise FileNotFoundError(
            f"Normalized references directory not found: {REFERENCES_DIR}\n"
            "Run normalize_texts.py first."
        )
    if not HYPOTHESES_DIR.is_dir():
        raise FileNotFoundError(
            f"Normalized hypotheses directory not found: {HYPOTHESES_DIR}\n"
            "Run normalize_texts.py first."
        )

    if args.clean and METRICS_DIR.exists():
        shutil.rmtree(METRICS_DIR)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    references = discover_references()
    model_directories = discover_model_directories()

    all_rows: list[FileMetrics] = []
    summaries: list[dict[str, Any]] = []
    extra_outputs: list[dict[str, Any]] = []

    for experiment, model, model_directory in model_directories:
        rows, extras = evaluate_model(
            experiment=experiment,
            model=model,
            model_directory=model_directory,
            references=references,
        )
        all_rows.extend(rows)
        summaries.append(summarize_model(rows))

        if extras:
            extra_outputs.append(
                {
                    "experiment": experiment,
                    "model": model,
                    "files": extras,
                }
            )

    per_file_rows = [serializable_row(row) for row in all_rows]
    summary_rows = [serializable_dict(summary) for summary in summaries]

    write_csv(
        METRICS_DIR / "per_file_metrics.csv",
        per_file_rows,
        PER_FILE_FIELDS,
    )
    write_csv(
        METRICS_DIR / "model_summary.csv",
        summary_rows,
        SUMMARY_FIELDS,
    )
    write_report(references, all_rows, summaries, extra_outputs)

    print("Normalized metric evaluation completed")
    print(f"Reference files: {len(references)}")
    print(f"Model configurations: {len(model_directories)}")
    print(f"File comparisons: {len(all_rows)}")
    print(f"Per-file results: {METRICS_DIR / 'per_file_metrics.csv'}")
    print(f"Model summary: {METRICS_DIR / 'model_summary.csv'}")
    print(f"Report: {METRICS_DIR / 'metrics_report.json'}")


if __name__ == "__main__":
    main()
