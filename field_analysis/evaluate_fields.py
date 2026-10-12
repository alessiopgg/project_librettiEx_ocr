#!/usr/bin/env python3
"""Evaluate Libretti OCR at field level.

The parser is run independently on OCR and ground truth using the same known
YAML template. Evaluation is performed on normalized text only.

Primary outputs:
* documents/libretto_XXX.csv  -> one row per key/value pair in a document
* all_fields.csv              -> all document rows
* field_summary.csv           -> aggregate by semantic field
* document_summary.csv        -> aggregate by document
* category_summary.csv        -> final metrics by category (coverage/exact/error severity)

PRE/POST are tracked for both keys and values.

KEY PRE  = normalized OCR key exactly as observed by the parser.
KEY POST = canonical template key only when the key was actually detected and its
           normalized OCR text differs from the template expected text. Missing
           keys are never filled from the template.

VALUE PRE  = normalized OCR value before category-specific correction.
VALUE POST = normalized OCR value after category-specific correction.

For human-readable row outputs, key PRE/POST text columns are populated only
when a key correction is actually applied; the legacy ``ocr_key`` column always
keeps the original normalized OCR key. Internally the effective POST key is
retained for metrics and summaries.

Ground truth is used only here, never by parse_fields.py.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

import yaml
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein

from parse_fields import TRANSCRIPTION_PATTERN, load_templates, normalize_text, parse_file


STAGES = ("pre", "post")
VALUE_REPORT_STAGES = {
    "dictionary": STAGES,
    "numeric": ("pre",),
    "string": ("pre",),
}
METRIC_COUNTS = (
    "ref_chars", "hyp_chars", "char_errors",
    "char_substitutions", "char_deletions", "char_insertions",
    "ref_words", "hyp_words", "word_errors",
    "word_substitutions", "word_deletions", "word_insertions",
)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def edit_counts(reference: Sequence[Any], hypothesis: Sequence[Any]) -> dict[str, int]:
    counts = Counter(op.tag for op in Levenshtein.editops(reference, hypothesis))
    return {
        "substitutions": counts["replace"],
        "deletions": counts["delete"],
        "insertions": counts["insert"],
    }


def metrics(reference: str, hypothesis: str) -> dict[str, Any]:
    char = edit_counts(reference, hypothesis)
    ref_words, hyp_words = reference.split(), hypothesis.split()
    word = edit_counts(ref_words, hyp_words)
    char_errors, word_errors = sum(char.values()), sum(word.values())

    return {
        "ref_chars": len(reference),
        "hyp_chars": len(hypothesis),
        "char_errors": char_errors,
        "cer": char_errors / len(reference) if reference else (0.0 if not hypothesis else None),
        "ref_words": len(ref_words),
        "hyp_words": len(hyp_words),
        "word_errors": word_errors,
        "wer": word_errors / len(ref_words) if ref_words else (0.0 if not hypothesis else None),
        "char_substitutions": char["substitutions"],
        "char_deletions": char["deletions"],
        "char_insertions": char["insertions"],
        "word_substitutions": word["substitutions"],
        "word_deletions": word["deletions"],
        "word_insertions": word["insertions"],
    }


def empty_metrics() -> dict[str, Any]:
    return {name: "" for name in (*METRIC_COUNTS[:3], "cer", *METRIC_COUNTS[6:9], "wer",
                                    *METRIC_COUNTS[3:6], *METRIC_COUNTS[9:])}


def evaluate_value(
    reference: str,
    hypothesis: str,
    evaluable: bool,
    key_correct: bool,
) -> tuple[dict[str, Any], bool | str, bool | str]:
    if not evaluable:
        return empty_metrics(), "", ""
    correct = reference == hypothesis
    return metrics(reference, hypothesis), correct, key_correct and correct


def _add_metrics(bucket: dict[str, int], data: dict[str, Any]) -> None:
    if data.get("ref_chars", "") == "":
        return
    for name in METRIC_COUNTS:
        bucket[name] += int(data[name])


def _final_metrics(bucket: dict[str, int]) -> dict[str, Any]:
    result = {name: int(bucket.get(name, 0)) for name in METRIC_COUNTS}
    result["cer"] = (
        result["char_errors"] / result["ref_chars"]
        if result["ref_chars"] else (0.0 if not result["hyp_chars"] else None)
    )
    result["wer"] = (
        result["word_errors"] / result["ref_words"]
        if result["ref_words"] else (0.0 if not result["hyp_words"] else None)
    )
    return result


def _aggregate_non_exact(
    rows: list[dict[str, Any]],
    metric_key: str,
    correct_key: str,
    evaluable_key: str | None = None,
) -> tuple[int, dict[str, Any]]:
    """Micro-aggregate edit metrics only over evaluable non-exact items.

    Exact items are excluded so CER/WER describe the severity of errors when an
    error occurs. If ``evaluable_key`` is provided, non-evaluable items are also
    excluded; for keys this lets missing keys be measured by detection rate
    rather than being converted into a 100% text error.
    """
    bucket: dict[str, int] = defaultdict(int)
    count = 0
    for row in rows:
        if evaluable_key is not None and not row[evaluable_key]:
            continue
        if row[correct_key] is not False:
            continue
        count += 1
        _add_metrics(bucket, row[metric_key])

    result = _final_metrics(bucket)
    if count == 0:
        result["cer"] = None
        result["wer"] = None
    return count, result


# ---------------------------------------------------------------------------
# Optional POST-processing
# ---------------------------------------------------------------------------

def load_dictionaries(path: Path | None) -> dict[str, list[str]]:
    """Load and normalize the external dictionaries used for POST correction."""
    if path is None or not path.exists():
        return {}

    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    raw = data.get("dictionaries", data)
    result: dict[str, list[str]] = {}
    for name, values in raw.items():
        if isinstance(values, list):
            normalized = [normalize_text(str(value)) for value in values]
            result[str(name)] = [value for value in normalized if value]
    return result


def empty_correction(dictionary_name: str | None = None) -> dict[str, Any]:
    return {
        "dictionary": dictionary_name or "",
        "correction_applied": False,
        "correction_candidate": "",
        "correction_score": "",
        "correction_second_score": "",
        "correction_margin": "",
    }


def empty_local_recovery(region_text: str = "") -> dict[str, Any]:
    return {
        "local_recovery_applied": False,
        "local_recovery_region": region_text,
        "local_recovery_match": "",
        "local_recovery_candidate": "",
        "local_recovery_score": "",
        "local_recovery_second_score": "",
        "local_recovery_margin": "",
    }


def dictionary_correction(
    text: str,
    dictionary_name: str | None,
    dictionaries: dict[str, list[str]],
    threshold: float,
    min_margin: float,
) -> tuple[str, dict[str, Any]]:
    """Correct a non-empty value only when the dictionary match is convincing."""
    info = empty_correction(dictionary_name)
    if not text or not dictionary_name:
        return text, info

    candidates = dictionaries.get(dictionary_name, [])
    if not candidates:
        return text, info

    ranked = sorted(
        ((float(fuzz.ratio(text, candidate)), candidate) for candidate in candidates),
        reverse=True,
    )
    best_score, best = ranked[0]
    second_score = ranked[1][0] if len(ranked) > 1 else 0.0
    margin = best_score - second_score
    info.update({
        "correction_candidate": best,
        "correction_score": round(best_score, 2),
        "correction_second_score": round(second_score, 2),
        "correction_margin": round(margin, 2),
    })

    if best_score >= threshold and margin >= min_margin:
        info["correction_applied"] = best != text
        return best, info
    return text, info


def dictionary_local_recovery(
    region_text: str,
    dictionary_name: str | None,
    dictionaries: dict[str, list[str]],
    threshold: float,
    min_margin: float,
) -> tuple[str, str, dict[str, Any]]:
    """Recover a dictionary value from the local region around a missing key."""
    info = empty_local_recovery(region_text)
    if not region_text or not dictionary_name:
        return "", "", info

    candidates = dictionaries.get(dictionary_name, [])
    region_tokens = region_text.split()
    if not candidates or not region_tokens:
        return "", "", info

    ranked: list[tuple[float, str, str]] = []
    for candidate in candidates:
        n = len(candidate.split())
        if not n:
            continue
        sizes = [n] if n == 1 else range(max(1, n - 1), n + 2)
        best_score, best_phrase = -1.0, ""
        for size in sizes:
            for i in range(max(0, len(region_tokens) - size + 1)):
                phrase = " ".join(region_tokens[i:i + size])
                score = float(fuzz.ratio(phrase, candidate))
                if score > best_score:
                    best_score, best_phrase = score, phrase
        if best_score >= 0:
            ranked.append((best_score, candidate, best_phrase))

    if not ranked:
        return "", "", info

    ranked.sort(key=lambda x: (x[0], x[1]), reverse=True)
    best_score, best_candidate, best_phrase = ranked[0]
    second_score = ranked[1][0] if len(ranked) > 1 else 0.0
    margin = best_score - second_score
    info.update({
        "local_recovery_match": best_phrase,
        "local_recovery_candidate": best_candidate,
        "local_recovery_score": round(best_score, 2),
        "local_recovery_second_score": round(second_score, 2),
        "local_recovery_margin": round(margin, 2),
    })

    if best_score >= threshold and margin >= min_margin:
        info["local_recovery_applied"] = True
        return best_phrase, best_candidate, info
    return "", "", info



def postprocess_key(
    text: str,
    expected: str,
    key_status: str,
) -> tuple[str, dict[str, Any]]:
    """Canonicalize a detected key using the known template text.

    The template is already the structural source used by the parser. Once a key
    has been detected (found/recovered/local), its OCR spelling may be replaced
    by the normalized template spelling for POST evaluation. Missing keys are
    never synthesized. Ground truth is not consulted here.
    """
    expected_normalized = normalize_text(expected)
    info = {
        "key_correction_applied": False,
        "key_correction_source": "",
    }

    if key_status == "missing" or not text:
        return text, info

    if text == expected_normalized:
        return text, info

    info["key_correction_applied"] = True
    info["key_correction_source"] = "template_expected"
    return expected_normalized, info

def postprocess_value(
    text: str,
    category: str,
    dictionary_name: str | None,
    dictionaries: dict[str, list[str]],
    dictionary_threshold: float,
    dictionary_margin: float,
) -> tuple[str, dict[str, Any]]:
    if category != "dictionary":
        return text, empty_correction(dictionary_name)
    return dictionary_correction(
        text, dictionary_name, dictionaries, dictionary_threshold, dictionary_margin
    )


def process_value(
    ocr_field: dict[str, Any],
    category: str,
    key_status: str,
    dictionaries: dict[str, list[str]],
    dictionary_threshold: float,
    dictionary_margin: float,
) -> dict[str, Any]:
    """Build PRE/POST value state, including optional dictionary recovery."""
    value = ocr_field["value"]
    dictionary_name = value.get("dictionary")
    boundary_status = value["boundary_status"]
    evaluable_pre = boundary_status == "ok"
    evaluable_post = evaluable_pre
    skip_pre = "" if evaluable_pre else boundary_status
    skip_post = skip_pre
    value_pre = value["normalized"]
    value_post = value_pre
    source_pre = "normal" if evaluable_pre else "unavailable"
    source_post = source_pre
    local_recovery = empty_local_recovery()

    if category == "dictionary" and key_status == "missing" and not evaluable_pre:
        observed, recovered, local_recovery = dictionary_local_recovery(
            value.get("search_region_normalized", ""),
            dictionary_name,
            dictionaries,
            dictionary_threshold,
            dictionary_margin,
        )
        if local_recovery["local_recovery_applied"]:
            value_post = recovered
            evaluable_post = True
            skip_post = ""
            source_post = "dictionary_local_recovery"
            correction = {
                "dictionary": dictionary_name or "",
                "correction_applied": observed != recovered,
                "correction_candidate": recovered,
                "correction_score": local_recovery["local_recovery_score"],
                "correction_second_score": local_recovery["local_recovery_second_score"],
                "correction_margin": local_recovery["local_recovery_margin"],
            }
        else:
            value_post, correction = postprocess_value(
                value_pre, category, dictionary_name, dictionaries,
                dictionary_threshold, dictionary_margin,
            )
    else:
        value_post, correction = postprocess_value(
            value_pre, category, dictionary_name, dictionaries,
            dictionary_threshold, dictionary_margin,
        )

    return {
        "boundary_status": boundary_status,
        "value_evaluable_pre": evaluable_pre,
        "value_evaluable_post": evaluable_post,
        "value_source_pre": source_pre,
        "value_source_post": source_post,
        "skip_reason_pre": skip_pre,
        "skip_reason_post": skip_post,
        "ocr_value_pre": value_pre,
        "ocr_value_post": value_post,
        "local_recovery": local_recovery,
        "correction": correction,
    }


# ---------------------------------------------------------------------------
# Build one concrete row per template field
# ---------------------------------------------------------------------------

def evaluate_document(
    ocr_path: Path,
    gt_path: Path,
    templates: list[dict[str, Any]],
    dictionaries: dict[str, list[str]],
    global_threshold: float,
    recovery_threshold: float,
    local_fuzzy_threshold: float,
    dictionary_threshold: float,
    dictionary_margin: float,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    ocr = parse_file(
        ocr_path, templates, global_threshold, recovery_threshold, local_fuzzy_threshold
    )
    gt = parse_file(
        gt_path, templates, global_threshold, recovery_threshold, local_fuzzy_threshold
    )

    gt_missing = [f["key"]["id"] for f in gt["fields"] if f["key"]["status"] == "missing"]
    if gt_missing:
        raise RuntimeError(
            f"Ground-truth parsing failed for {gt_path.name}; missing keys: "
            f"{', '.join(gt_missing)}. Review the template before evaluating OCR outputs."
        )

    ocr_fields = {field["key"]["id"]: field for field in ocr["fields"]}
    rows: list[dict[str, Any]] = []

    for gt_field in gt["fields"]:
        key_id = gt_field["key"]["id"]
        ocr_field = ocr_fields[key_id]

        # Key comparison and template-based POST canonicalization.
        gt_key = gt_field["key"]["normalized"]
        ocr_key = ocr_field["key"]["normalized"]
        key_status = ocr_field["key"]["status"]
        key_detected = key_status != "missing"

        key_pre = ocr_key
        key_post, key_correction = postprocess_key(
            key_pre,
            ocr_field["key"]["expected"],
            key_status,
        )

        key_correct_pre = gt_key == key_pre
        key_correct_post = gt_key == key_post
        key_pre_m = metrics(gt_key, key_pre)
        key_post_m = metrics(gt_key, key_post)

        # Backward-compatible aliases: historical key_* fields keep PRE semantics.
        key_correct = key_correct_pre
        key_m = key_pre_m

        category = gt_field["value"]["category"]
        gt_value = gt_field["value"]["normalized"]
        value = process_value(
            ocr_field, category, key_status, dictionaries,
            dictionary_threshold, dictionary_margin,
        )

        value_pre_m, value_correct_pre, field_correct_pre = evaluate_value(
            gt_value,
            value["ocr_value_pre"],
            value["value_evaluable_pre"],
            key_correct_pre,
        )
        value_post_m, value_correct_post, field_correct_post = evaluate_value(
            gt_value,
            value["ocr_value_post"],
            value["value_evaluable_post"],
            key_correct_post,
        )

        row = {
            "document": gt["document"],
            "template": gt["template"],
            "order": gt_field["order"],
            "field_id": gt_field["value"]["id"],
            "category": category,
            # Key: expected/GT/OCR side by side. Legacy columns retain PRE semantics.
            "key_id": key_id,
            "key_expected": gt_field["key"]["expected"],
            "gt_key": gt_key,
            "ocr_key": ocr_key,
            "ocr_key_pre": key_pre if key_correction["key_correction_applied"] else "",
            "ocr_key_post": key_post if key_correction["key_correction_applied"] else "",
            "key_status": key_status,
            "key_detected": key_detected,
            "key_correct": key_correct,
            "key_correct_pre": key_correct_pre,
            "key_correct_post": key_correct_post,
            "key_cer": key_m["cer"],
            "key_wer": key_m["wer"],
            "key_cer_pre": key_pre_m["cer"],
            "key_wer_pre": key_pre_m["wer"],
            "key_cer_post": key_post_m["cer"],
            "key_wer_post": key_post_m["wer"],
            **key_correction,
            # Value: GT/PRE/POST side by side.
            "gt_value": gt_value,
            "ocr_value_pre": value["ocr_value_pre"],
            "ocr_value_post": value["ocr_value_post"],
            "boundary_status": value["boundary_status"],
            "value_evaluable_pre": value["value_evaluable_pre"],
            "value_evaluable_post": value["value_evaluable_post"],
            "value_source_pre": value["value_source_pre"],
            "value_source_post": value["value_source_post"],
            "skip_reason_pre": value["skip_reason_pre"],
            "skip_reason_post": value["skip_reason_post"],
            **value["local_recovery"],
            "value_correct_pre": value_correct_pre,
            "value_correct_post": value_correct_post,
            "field_correct_pre": field_correct_pre,
            "field_correct_post": field_correct_post,
            # Value POST-processing diagnostics.
            **value["correction"],
            # Value metrics.
            "value_cer_pre": value_pre_m["cer"],
            "value_wer_pre": value_pre_m["wer"],
            "value_cer_post": value_post_m["cer"],
            "value_wer_post": value_post_m["wer"],
            # Internal counts used for aggregation. _key_metrics is a PRE alias.
            "_key_metrics": key_pre_m,
            "_key_pre_metrics": key_pre_m,
            "_key_post_metrics": key_post_m,
            "_value_pre_metrics": value_pre_m,
            "_value_post_metrics": value_post_m,
        }
        rows.append(row)

    return ocr, gt, rows


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    clean = [{k: v for k, v in row.items() if not k.startswith("_")} for row in rows]
    fieldnames = list(dict.fromkeys(name for row in clean for name in row))
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(clean)


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _count_results(rows: list[dict[str, Any]]) -> dict[str, int]:
    key_correct_pre = sum(r["key_correct_pre"] is True for r in rows)
    key_correct_post = sum(r["key_correct_post"] is True for r in rows)
    key_corrected = sum(bool(r["key_correction_applied"]) for r in rows)
    value_corrected = sum(bool(r["correction_applied"]) for r in rows)
    return {
        "key_detected": sum(bool(r["key_detected"]) for r in rows),
        # Historical aliases retain PRE semantics.
        "key_correct": key_correct_pre,
        "key_correct_pre": key_correct_pre,
        "key_correct_post": key_correct_post,
        "key_corrected": key_corrected,
        "value_evaluable_pre": sum(bool(r["value_evaluable_pre"]) for r in rows),
        "value_evaluable_post": sum(bool(r["value_evaluable_post"]) for r in rows),
        "value_correct_pre": sum(r["value_correct_pre"] is True for r in rows),
        "value_correct_post": sum(r["value_correct_post"] is True for r in rows),
        "field_correct_pre": sum(r["field_correct_pre"] is True for r in rows),
        "field_correct_post": sum(r["field_correct_post"] is True for r in rows),
        # Historical 'corrected' counter is value-only.
        "corrected": value_corrected,
        "value_corrected": value_corrected,
        "total_corrected": key_corrected + value_corrected,
    }


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------

def build_field_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate results by semantic field_id across templates/documents."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["field_id"]].append(row)

    output: list[dict[str, Any]] = []
    for field_id, items in sorted(groups.items()):
        occurrences = len(items)
        category = items[0]["category"]
        c = _count_results(items)

        key_non_exact, key_error_metrics = _aggregate_non_exact(
            items,
            "_key_pre_metrics",
            "key_correct_pre",
            "key_detected",
        )
        pre_non_exact, pre_error_metrics = _aggregate_non_exact(
            items,
            "_value_pre_metrics",
            "value_correct_pre",
            "value_evaluable_pre",
        )
        post_non_exact, post_error_metrics = _aggregate_non_exact(
            items,
            "_value_post_metrics",
            "value_correct_post",
            "value_evaluable_post",
        )

        output.append({
            "field_id": field_id,
            "category": category,
            "key_expected": " | ".join(sorted({str(r["key_expected"]) for r in items})),
            "templates": " | ".join(sorted({str(r["template"]) for r in items})),
            "occurrences": occurrences,

            # Key quality is evaluated on PRE OCR text only.
            "key_detected": c["key_detected"],
            "key_missing": occurrences - c["key_detected"],
            "key_detection_rate": _rate(c["key_detected"], occurrences),
            "key_exact": c["key_correct_pre"],
            "key_exact_rate": _rate(c["key_correct_pre"], occurrences),
            "key_non_exact_detected": key_non_exact,
            "key_cer_non_exact": key_error_metrics["cer"],
            "key_corrections_applied": c["key_corrected"],

            # Values: coverage separates structural evaluability from OCR accuracy.
            "value_evaluable_pre": c["value_evaluable_pre"],
            "value_skipped_pre": occurrences - c["value_evaluable_pre"],
            "value_coverage_pre": _rate(c["value_evaluable_pre"], occurrences),
            "value_exact_pre": c["value_correct_pre"],
            "value_exact_rate_pre": _rate(c["value_correct_pre"], c["value_evaluable_pre"]),
            "value_non_exact_pre": pre_non_exact,
            "value_cer_non_exact_pre": pre_error_metrics["cer"],
            "value_wer_non_exact_pre": pre_error_metrics["wer"] if category == "string" else None,

            "value_evaluable_post": c["value_evaluable_post"],
            "value_skipped_post": occurrences - c["value_evaluable_post"],
            "value_coverage_post": _rate(c["value_evaluable_post"], occurrences),
            "value_exact_post": c["value_correct_post"],
            "value_exact_rate_post": _rate(c["value_correct_post"], c["value_evaluable_post"]),
            "value_non_exact_post": post_non_exact,
            "value_cer_non_exact_post": post_error_metrics["cer"],
            "value_wer_non_exact_post": post_error_metrics["wer"] if category == "string" else None,

            # POST-processing diagnostics.
            "value_corrections_applied": c["value_corrected"],
            "total_corrections_applied": c["total_corrected"],

            # Complete field success: key AND value correct.
            "field_correct_pre": c["field_correct_pre"],
            "field_accuracy_conditional_pre": _rate(
                c["field_correct_pre"], c["value_evaluable_pre"]
            ),
            "field_success_rate_pre": _rate(c["field_correct_pre"], occurrences),
            "field_correct_post": c["field_correct_post"],
            "field_accuracy_conditional_post": _rate(
                c["field_correct_post"], c["value_evaluable_post"]
            ),
            "field_success_rate_post": _rate(c["field_correct_post"], occurrences),
        })
    return output


def build_document_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate structural/exact success by document; no redundant CER/WER here."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["document"]].append(row)

    output: list[dict[str, Any]] = []
    for document, items in sorted(groups.items()):
        total = len(items)
        c = _count_results(items)
        output.append({
            "document": document,
            "template": items[0]["template"],
            "fields_total": total,
            "keys_detected": c["key_detected"],
            "keys_missing": total - c["key_detected"],
            "key_detection_rate": _rate(c["key_detected"], total),
            "keys_exact": c["key_correct_pre"],
            "key_exact_rate": _rate(c["key_correct_pre"], total),
            "key_corrections_applied": c["key_corrected"],

            "values_evaluable_pre": c["value_evaluable_pre"],
            "values_skipped_pre": total - c["value_evaluable_pre"],
            "value_coverage_pre": _rate(c["value_evaluable_pre"], total),
            "values_exact_pre": c["value_correct_pre"],
            "value_exact_rate_pre": _rate(c["value_correct_pre"], c["value_evaluable_pre"]),

            "values_evaluable_post": c["value_evaluable_post"],
            "values_skipped_post": total - c["value_evaluable_post"],
            "value_coverage_post": _rate(c["value_evaluable_post"], total),
            "values_exact_post": c["value_correct_post"],
            "value_exact_rate_post": _rate(c["value_correct_post"], c["value_evaluable_post"]),

            "fields_correct_pre": c["field_correct_pre"],
            "field_accuracy_conditional_pre": _rate(
                c["field_correct_pre"], c["value_evaluable_pre"]
            ),
            "field_success_rate_pre": _rate(c["field_correct_pre"], total),
            "fields_correct_post": c["field_correct_post"],
            "field_accuracy_conditional_post": _rate(
                c["field_correct_post"], c["value_evaluable_post"]
            ),
            "field_success_rate_post": _rate(c["field_correct_post"], total),

            "value_corrections_applied": c["value_corrected"],
            "total_corrections_applied": c["total_corrected"],
        })
    return output


def build_category_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build the compact metrics table used in the results section.

    Final reporting contract:
    * key: detection rate, exact rate, CER on detected non-exact keys;
    * dictionary: coverage, exact rate, CER on non-exact values, PRE and POST;
    * numeric: coverage, exact rate, CER on non-exact values;
    * string: coverage, exact rate, CER and WER on non-exact values.

    Value exact rates are conditional on evaluability. Missing/unreliable values
    are represented by coverage instead of being mixed into OCR content accuracy.
    """
    output: list[dict[str, Any]] = []

    total_keys = len(rows)
    detected_keys = sum(bool(r["key_detected"]) for r in rows)
    exact_keys = sum(r["key_correct_pre"] is True for r in rows)
    key_statuses = Counter(str(r["key_status"]) for r in rows)
    key_non_exact, key_error_metrics = _aggregate_non_exact(
        rows,
        "_key_pre_metrics",
        "key_correct_pre",
        "key_detected",
    )
    output.append({
        "category": "key",
        "stage": "pre",
        "items_total": total_keys,
        "items_detected": detected_keys,
        "items_missing": key_statuses["missing"],
        "items_recovered": key_statuses["recovered"],
        "items_local": key_statuses["local"],
        "items_found": key_statuses["found"],
        "detection_rate": _rate(detected_keys, total_keys),
        "items_exact": exact_keys,
        "exact_rate": _rate(exact_keys, total_keys),
        "items_non_exact": key_non_exact,
        "cer_non_exact": key_error_metrics["cer"],
        "wer_non_exact": None,
    })

    for category in ("dictionary", "numeric", "string"):
        category_rows = [r for r in rows if r["category"] == category]
        for stage in VALUE_REPORT_STAGES[category]:
            evaluable_key = f"value_evaluable_{stage}"
            correct_key = f"value_correct_{stage}"
            metric_key = f"_value_{stage}_metrics"

            total = len(category_rows)
            evaluated = sum(bool(r[evaluable_key]) for r in category_rows)
            exact = sum(r[correct_key] is True for r in category_rows)
            non_exact, error_metrics = _aggregate_non_exact(
                category_rows,
                metric_key,
                correct_key,
                evaluable_key,
            )

            output.append({
                "category": category,
                "stage": stage,
                "items_total": total,
                "items_evaluated": evaluated,
                "items_skipped": total - evaluated,
                "coverage": _rate(evaluated, total),
                "items_exact": exact,
                "exact_rate": _rate(exact, evaluated),
                "items_non_exact": non_exact,
                "cer_non_exact": error_metrics["cer"],
                "wer_non_exact": error_metrics["wer"] if category == "string" else None,
            })

    return output


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate normalized OCR key/value fields and create readable summaries."
    )
    parser.add_argument("ocr_dir", type=Path, help="Directory containing OCR libretto_*.txt files.")
    parser.add_argument(
        "--gt-dir", type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "text_visionato",
    )
    parser.add_argument(
        "--templates", type=Path,
        default=Path(__file__).with_name("templates"),
    )
    parser.add_argument(
        "--dictionaries", type=Path,
        default=Path(__file__).with_name("templates") / "dictionaries.yaml",
    )
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("results"))
    parser.add_argument("--global-threshold", type=float, default=88.0)
    parser.add_argument("--recovery-threshold", type=float, default=72.0)
    parser.add_argument("--local-fuzzy-threshold", type=float, default=82.0)
    parser.add_argument("--dictionary-threshold", type=float, default=70.0)
    parser.add_argument(
        "--dictionary-margin", type=float, default=15.0,
        help="Minimum score gap between the best and second dictionary candidate.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    templates = load_templates(args.templates)
    dictionaries = load_dictionaries(args.dictionaries)

    ocr_files = {p.name: p for p in sorted(args.ocr_dir.glob(TRANSCRIPTION_PATTERN))}
    gt_files = {p.name: p for p in sorted(args.gt_dir.glob(TRANSCRIPTION_PATTERN))}
    common = sorted(ocr_files.keys() & gt_files.keys())
    if not common:
        raise SystemExit("No matching OCR/ground-truth transcription files found.")

    all_rows: list[dict[str, Any]] = []
    parsed_ocr: dict[str, Any] = {}
    parsed_gt: dict[str, Any] = {}
    for filename in common:
        ocr, gt, rows = evaluate_document(
            ocr_files[filename], gt_files[filename], templates, dictionaries,
            args.global_threshold, args.recovery_threshold, args.local_fuzzy_threshold,
            args.dictionary_threshold, args.dictionary_margin,
        )
        parsed_ocr[filename], parsed_gt[filename] = ocr, gt
        all_rows.extend(rows)

    args.output.mkdir(parents=True, exist_ok=True)
    docs_dir = args.output / "documents"
    docs_dir.mkdir(parents=True, exist_ok=True)

    write_csv(args.output / "all_fields.csv", all_rows)
    by_document: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in all_rows:
        by_document[row["document"]].append(row)
    for document, rows in sorted(by_document.items()):
        write_csv(docs_dir / f"{document}.csv", rows)

    field_summary = build_field_summary(all_rows)
    document_summary = build_document_summary(all_rows)
    category_summary = build_category_summary(all_rows)
    write_csv(args.output / "field_summary.csv", field_summary)
    write_csv(args.output / "document_summary.csv", document_summary)
    write_csv(args.output / "category_summary.csv", category_summary)

    write_json(args.output / "parsed_ocr.json", parsed_ocr)
    write_json(args.output / "parsed_gt.json", parsed_gt)
    report = {
        "ocr_dir": str(args.ocr_dir),
        "gt_dir": str(args.gt_dir),
        "documents": len(common),
        "fields": len(all_rows),
        "parameters": {
            "global_threshold": args.global_threshold,
            "recovery_threshold": args.recovery_threshold,
            "local_fuzzy_threshold": args.local_fuzzy_threshold,
            "dictionary_threshold": args.dictionary_threshold,
            "dictionary_margin": args.dictionary_margin,
        },
        "dictionaries_file": str(args.dictionaries) if args.dictionaries.exists() else None,
        "metric_policy": {
            "key": ["detection_rate", "exact_rate", "cer_non_exact"],
            "dictionary": ["coverage", "exact_rate", "cer_non_exact"],
            "numeric": ["coverage", "exact_rate", "cer_non_exact"],
            "string": ["coverage", "exact_rate", "cer_non_exact", "wer_non_exact"],
            "value_exact_rate_denominator": "evaluable_items",
            "error_severity_subset": "evaluable_non_exact_items_only",
            "error_aggregation": "micro_sum_errors_over_sum_reference_units",
            "missing_keys_excluded_from_key_cer_non_exact": True,
        },
        "key_postprocessing": {
            "enabled": True,
            "source": "template_expected",
            "detected_keys_only": True,
            "fill_missing_keys": False,
            "public_pre_post_text_only_when_changed": True,
        },
        "field_summary": field_summary,
        "document_summary": document_summary,
        "category_summary": category_summary,
    }
    write_json(args.output / "report.json", report)

    total_keys = len(all_rows)
    detected_keys = sum(bool(row["key_detected"]) for row in all_rows)
    correct_keys_pre = sum(row["key_correct_pre"] is True for row in all_rows)
    correct_keys_post = sum(row["key_correct_post"] is True for row in all_rows)
    key_corrections = sum(bool(row["key_correction_applied"]) for row in all_rows)
    value_skipped_pre = sum(not bool(row["value_evaluable_pre"]) for row in all_rows)
    value_skipped_post = sum(not bool(row["value_evaluable_post"]) for row in all_rows)

    print(f"Documents evaluated: {len(common)}")
    print(f"Structural fields: {len(all_rows)}")
    print(
        f"OCR keys detected: {detected_keys}/{total_keys} "
        f"({100 * detected_keys / total_keys:.2f}%)"
    )
    print(
        f"OCR keys exactly correct: {correct_keys_pre}/{total_keys} "
        f"({100 * correct_keys_pre / total_keys:.2f}%)"
    )
    print(
        f"OCR keys exactly correct after template canonicalization: "
        f"{correct_keys_post}/{total_keys} ({100 * correct_keys_post / total_keys:.2f}%)"
    )
    print(f"Key canonicalizations applied from template: {key_corrections}")
    print(f"Value fields skipped PRE because of unreliable boundaries: {value_skipped_pre}")
    print(f"Value fields skipped POST after dictionary recovery: {value_skipped_post}")
    print(f"Output: {args.output}")
    print("Main files: all_fields.csv, field_summary.csv, document_summary.csv, category_summary.csv")



if __name__ == "__main__":
    main()
