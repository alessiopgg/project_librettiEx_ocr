#!/usr/bin/env python3
"""Parse Libretti OCR text into ordered template fields.

Pipeline:
1. find global keys;
2. keep the best order-consistent sequence;
3. recover missing global keys in local intervals;
4. find short/local keys between surrounding matches;
5. extract the value after each key.

Ground truth is never used by this module.
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path
from typing import Any

import yaml
from rapidfuzz import fuzz

TRANSCRIPTION_PATTERN = "libretto_*.txt"
UNCLEAR_TAG_RE = re.compile(r"<\s*/?\s*unclear\s*>", re.IGNORECASE)
EMPTY_TAG_RE = re.compile(r"<\s*(?:gap|illegible)\s*/?\s*>", re.IGNORECASE)


def prepare_text(text: str) -> str:
    """Technical cleanup only, preserving character offsets."""
    text = text.replace("\ufeff", "").replace("\r\n", "\n").replace("\r", "\n")
    text = unicodedata.normalize("NFC", text)
    for regex in (UNCLEAR_TAG_RE, EMPTY_TAG_RE):
        text = regex.sub(lambda match: " " * len(match.group(0)), text)
    return text


def tokenize(text: str) -> list[dict[str, Any]]:
    """Normalize text into tokens while retaining raw character offsets."""
    tokens: list[dict[str, Any]] = []
    current: list[str] = []
    start: int | None = None

    for index, char in enumerate(text):
        separator = char.isspace() or unicodedata.category(char).startswith(("P", "S"))
        if separator:
            if current and start is not None:
                tokens.append({"text": "".join(current), "raw_start": start, "raw_end": index})
                current = []
                start = None
            continue

        if start is None:
            start = index
        current.append(char.casefold())

    if current and start is not None:
        tokens.append({"text": "".join(current), "raw_start": start, "raw_end": len(text)})
    return tokens


def normalize_text(text: str) -> str:
    return " ".join(token["text"] for token in tokenize(prepare_text(text)))


def load_templates(template_dir: Path) -> list[dict[str, Any]]:
    templates: list[dict[str, Any]] = []
    for path in sorted(template_dir.glob("template*.yaml")):
        template = yaml.safe_load(path.read_text(encoding="utf-8"))["template"]
        template["_path"] = str(path)
        template["fields"] = sorted(template["fields"], key=lambda field: field["order"])
        templates.append(template)
    if not templates:
        raise RuntimeError(f"No template*.yaml files found in {template_dir}")
    return templates


def template_for_document(document: str, templates: list[dict[str, Any]]) -> dict[str, Any]:
    stem = Path(document).stem
    for template in templates:
        if stem in template.get("documents", []):
            return template
    raise KeyError(f"No template configured for {stem}")


def _key_tokens(field: dict[str, Any]) -> list[str]:
    return normalize_text(field["key"]["text"]).split()


def _candidate(
    field: dict[str, Any],
    tokens: list[dict[str, Any]],
    start: int,
    end: int,
    score: float,
    status: str,
) -> dict[str, Any]:
    return {
        "key_id": field["key"]["id"],
        "order": field["order"],
        "start_token": start,
        "end_token": end,
        "raw_start": tokens[start]["raw_start"],
        "raw_end": tokens[end - 1]["raw_end"],
        "score": round(float(score), 2),
        "status": status,
    }


def find_candidates(
    field: dict[str, Any],
    tokens: list[dict[str, Any]],
    start: int,
    end: int,
    threshold: float,
    max_candidates: int = 12,
    status: str = "found",
) -> list[dict[str, Any]]:
    """Find exact or fuzzy candidates for one key inside [start, end)."""
    key_tokens = _key_tokens(field)
    if not key_tokens or start >= end:
        return []

    target = " ".join(key_tokens)
    n = len(key_tokens)
    exact = field["key"]["detection"].get("match") == "exact"
    sizes = [n] if exact or n == 1 else list(range(max(1, n - 1), n + 2))
    candidates: list[dict[str, Any]] = []

    for size in sizes:
        for index in range(start, end - size + 1):
            window = tokens[index : index + size]
            phrase = " ".join(token["text"] for token in window)
            score = 100.0 if exact and [token["text"] for token in window] == key_tokens else (
                fuzz.ratio(target, phrase) if not exact else 0.0
            )
            if score >= threshold:
                candidates.append(_candidate(field, tokens, index, index + size, score, status))

    # n-1/n/n+1 fuzzy windows can overlap heavily: keep only strongest distinct ones.
    candidates.sort(key=lambda item: (-item["score"], item["start_token"], item["end_token"]))
    kept: list[dict[str, Any]] = []
    for candidate in candidates:
        if any(
            candidate["start_token"] == old["start_token"]
            or candidate["end_token"] == old["end_token"]
            for old in kept
        ):
            continue
        kept.append(candidate)
        if len(kept) == max_candidates:
            break
    return sorted(kept, key=lambda item: (item["start_token"], -item["score"]))


def find_global_keys(
    tokens: list[dict[str, Any]], fields: list[dict[str, Any]], threshold: float
) -> list[dict[str, Any]]:
    return [
        candidate
        for field in fields
        if field["key"]["detection"]["scope"] == "global"
        for candidate in find_candidates(field, tokens, 0, len(tokens), threshold)
    ]


def select_ordered_keys(candidates: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Maximum-score chain with increasing template order and text position."""
    if not candidates:
        return {}

    nodes = sorted(candidates, key=lambda item: (item["order"], item["start_token"], item["end_token"]))
    dp = [float(node["score"]) for node in nodes]
    previous = [-1] * len(nodes)

    for i, node in enumerate(nodes):
        for j, old in enumerate(nodes[:i]):
            if old["order"] >= node["order"] or old["end_token"] > node["start_token"]:
                continue
            score = dp[j] + node["score"]
            if score > dp[i]:
                dp[i] = score
                previous[i] = j

    index = max(range(len(nodes)), key=dp.__getitem__)
    chain: list[dict[str, Any]] = []
    while index != -1:
        chain.append(nodes[index])
        index = previous[index]
    return {match["key_id"]: match for match in reversed(chain)}


def _neighbor_bounds(
    field: dict[str, Any], matches: dict[str, dict[str, Any]], token_count: int
) -> tuple[int, int, bool, bool]:
    order = field["order"]
    left = max(
        (match for match in matches.values() if match["order"] < order),
        key=lambda match: match["order"],
        default=None,
    )
    right = min(
        (match for match in matches.values() if match["order"] > order),
        key=lambda match: match["order"],
        default=None,
    )
    return (
        left["end_token"] if left else 0,
        right["start_token"] if right else token_count,
        left is not None,
        right is not None,
    )


def recover_missing_global_keys(
    tokens: list[dict[str, Any]],
    fields: list[dict[str, Any]],
    matches: dict[str, dict[str, Any]],
    high_threshold: float,
    low_threshold: float,
) -> None:
    """Retry missing global keys only in the interval allowed by the template."""
    for field in fields:
        key = field["key"]
        if key["detection"]["scope"] != "global" or key["id"] in matches:
            continue

        start, end, has_left, has_right = _neighbor_bounds(field, matches, len(tokens))
        threshold = low_threshold if has_left and has_right else high_threshold
        candidates = find_candidates(
            field, tokens, start, end, threshold, max_candidates=1, status="recovered"
        )
        if candidates:
            matches[key["id"]] = max(candidates, key=lambda item: item["score"])


def find_local_keys(
    tokens: list[dict[str, Any]],
    fields: list[dict[str, Any]],
    matches: dict[str, dict[str, Any]],
    fuzzy_threshold: float,
) -> None:
    """Find short/ambiguous keys only between two detected structural keys."""
    for field in fields:
        key = field["key"]
        if key["detection"]["scope"] != "local" or key["id"] in matches:
            continue

        start, end, has_left, has_right = _neighbor_bounds(field, matches, len(tokens))
        if not (has_left and has_right) or start >= end:
            continue

        candidates = find_candidates(
            field, tokens, start, end, fuzzy_threshold, max_candidates=1, status="local"
        )
        if candidates:
            matches[key["id"]] = max(candidates, key=lambda item: item["score"])


def extract_fields(
    source: str,
    tokens: list[dict[str, Any]],
    fields: list[dict[str, Any]],
    matches: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Attach matched key text and the following value to every template field."""
    next_found: list[tuple[dict[str, Any] | None, int | None]] = [
        (None, None) for _ in fields
    ]
    next_match: dict[str, Any] | None = None
    next_order: int | None = None
    for index in range(len(fields) - 1, -1, -1):
        next_found[index] = (next_match, next_order)
        current = matches.get(fields[index]["key"]["id"])
        if current:
            next_match, next_order = current, fields[index]["order"]

    result: list[dict[str, Any]] = []
    for index, field in enumerate(fields):
        key = field["key"]
        value = field["value"]
        match = matches.get(key["id"])
        following, following_order = next_found[index]

        if match:
            key_raw = source[match["raw_start"] : match["raw_end"]]
            key_norm = " ".join(
                token["text"] for token in tokens[match["start_token"] : match["end_token"]]
            )
            value_end = following["start_token"] if following else len(tokens)
            raw_end = following["raw_start"] if following else len(source)
            value_raw = source[match["raw_end"] : raw_end].strip()
            value_norm = " ".join(
                token["text"] for token in tokens[match["end_token"] : value_end]
            )
            immediate_order = fields[index + 1]["order"] if index + 1 < len(fields) else None
            boundary_status = (
                "ok"
                if immediate_order is None or following_order == immediate_order
                else "next_key_missing"
            )
        else:
            key_raw = key_norm = value_raw = value_norm = ""
            boundary_status = "key_missing"

        region_raw = region_norm = ""
        has_left = has_right = False
        if not match:
            start, end, has_left, has_right = _neighbor_bounds(field, matches, len(tokens))
            if start < end and (has_left or has_right):
                raw_start = tokens[start]["raw_start"] if start < len(tokens) else len(source)
                raw_end = tokens[end - 1]["raw_end"] if end > 0 else 0
                region_raw = source[raw_start:raw_end].strip()
                region_norm = " ".join(token["text"] for token in tokens[start:end])

        result.append(
            {
                "order": field["order"],
                "key": {
                    "id": key["id"],
                    "expected": key["text"],
                    "scope": key["detection"]["scope"],
                    "status": match["status"] if match else "missing",
                    "score": match["score"] if match else None,
                    "raw": key_raw,
                    "normalized": key_norm,
                },
                "value": {
                    "id": value["id"],
                    "category": value["category"],
                    "format_hint": value.get("format_hint"),
                    "dictionary": value.get("dictionary"),
                    "boundary_status": boundary_status,
                    "raw": value_raw,
                    "normalized": value_norm,
                    "search_region_raw": region_raw,
                    "search_region_normalized": region_norm,
                    "search_region_has_left": has_left,
                    "search_region_has_right": has_right,
                },
            }
        )
    return result


def parse_text(
    text: str,
    template: dict[str, Any],
    global_threshold: float = 88.0,
    recovery_threshold: float = 72.0,
    local_fuzzy_threshold: float = 82.0,
) -> dict[str, Any]:
    source = prepare_text(text)
    tokens = tokenize(source)
    fields = template["fields"]

    matches = select_ordered_keys(find_global_keys(tokens, fields, global_threshold))
    recover_missing_global_keys(tokens, fields, matches, global_threshold, recovery_threshold)
    find_local_keys(tokens, fields, matches, local_fuzzy_threshold)
    parsed_fields = extract_fields(source, tokens, fields, matches)

    status_counts: dict[str, int] = {}
    for field in parsed_fields:
        status = field["key"]["status"]
        status_counts[status] = status_counts.get(status, 0) + 1

    return {
        "template": template["id"],
        "parameters": {
            "global_threshold": global_threshold,
            "recovery_threshold": recovery_threshold,
            "local_fuzzy_threshold": local_fuzzy_threshold,
        },
        "key_status_counts": status_counts,
        "fields": parsed_fields,
    }


def parse_file(
    path: Path,
    templates: list[dict[str, Any]],
    global_threshold: float,
    recovery_threshold: float,
    local_fuzzy_threshold: float,
) -> dict[str, Any]:
    result = parse_text(
        path.read_text(encoding="utf-8"),
        template_for_document(path.name, templates),
        global_threshold,
        recovery_threshold,
        local_fuzzy_threshold,
    )
    result.update(document=path.stem, source=str(path))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parse Libretti OCR text using known templates.")
    parser.add_argument(
        "input", type=Path, help="One transcription file or a directory of libretto_*.txt files."
    )
    parser.add_argument("--templates", type=Path, default=Path(__file__).with_name("templates"))
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("parsed"))
    parser.add_argument("--global-threshold", type=float, default=88.0)
    parser.add_argument("--recovery-threshold", type=float, default=72.0)
    parser.add_argument("--local-fuzzy-threshold", type=float, default=82.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    templates = load_templates(args.templates)
    paths = [args.input] if args.input.is_file() else sorted(args.input.glob(TRANSCRIPTION_PATTERN))
    if not paths:
        raise SystemExit(f"No transcription files found in {args.input}")

    args.output.mkdir(parents=True, exist_ok=True)
    found = total = 0
    for path in paths:
        result = parse_file(
            path,
            templates,
            args.global_threshold,
            args.recovery_threshold,
            args.local_fuzzy_threshold,
        )
        (args.output / f"{path.stem}.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        total += len(result["fields"])
        found += sum(field["key"]["status"] != "missing" for field in result["fields"])

    print(f"Parsed documents: {len(paths)}")
    print(f"Keys found: {found}/{total} ({100 * found / total:.2f}%)")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
