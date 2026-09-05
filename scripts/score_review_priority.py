"""Score mined drift candidates by structural review priority without assigning labels."""

import argparse
import ast
import json
import re
import statistics
import sys
import textwrap
from pathlib import Path

from filter_real_drift_candidates import diff_size as _raw_diff_size, relevance_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CANDIDATE_ROOT = PROJECT_ROOT / "data" / "real_world" / "mined_candidates"
WORD_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
CAPITALIZED_WORD = re.compile(r"^[A-Z][A-Za-z0-9_]*$")


# Weighting plan used for review_priority_score:
#   0.45 * vocabulary_divergence + 0.35 * dropped_identifier_signal
# + 0.12 * diff_size + 0.08 * docstring_length
# This intentionally gives the highest weight to the two signals that most
# closely proxy contract mismatch while still allowing larger edits and longer
# docstrings to move a candidate up the review queue.


def load_candidates(path: Path) -> list[dict]:
    """Load JSONL candidates while skipping malformed lines defensively."""
    records: list[dict] = []
    if not path.exists():
        return records

    with path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            try:
                candidate = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"Warning: skipping malformed JSON at {path}:{line_number}: {exc}", file=sys.stderr)
                continue
            if isinstance(candidate, dict):
                records.append(candidate)
            else:
                print(f"Warning: skipping non-object JSON at {path}:{line_number}", file=sys.stderr)
    return records


def normalized_docstring(candidate: dict) -> str:
    """Return the non-empty docstring side for a candidate."""
    for key in ("docstring_before", "docstring_after"):
        value = candidate.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return ""


def function_names_in_code(source: str) -> set[str]:
    """Extract identifier names from a code snippet while ignoring parse failures."""
    if not source or not source.strip():
        return set()
    try:
        tree = ast.parse(textwrap.dedent(source))
    except (SyntaxError, ValueError):
        return set()

    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.arg):
            names.add(node.arg)
    return names


def docstring_identifiers(docstring: str, code_before: str, code_after: str) -> set[str]:
    """Collect identifiers from docstrings that are likely to be meaningful signals."""
    if not docstring:
        return set()

    before_names = function_names_in_code(code_before)
    after_names = function_names_in_code(code_after)
    names = set()

    for token in WORD_PATTERN.findall(docstring):
        if token in before_names or token in after_names:
            names.add(token)
        elif CAPITALIZED_WORD.fullmatch(token):
            names.add(token)
    return names


def dropped_identifier_signal(candidate: dict) -> int:
    """Return 1 when a docstring-mentioned identifier disappears or appears in code."""
    before = candidate.get("code_before", "") or ""
    after = candidate.get("code_after", "") or ""
    docstring = normalized_docstring(candidate)
    if not docstring:
        return 0

    before_names = function_names_in_code(before)
    after_names = function_names_in_code(after)
    doc_identifiers = docstring_identifiers(docstring, before, after)
    if not doc_identifiers:
        return 0

    for identifier in doc_identifiers:
        if (identifier in before_names and identifier not in after_names) or (
            identifier in after_names and identifier not in before_names
        ):
            return 1
    return 0


def normalize_0_1(value: float, scale: float) -> float:
    """Clamp a raw score into the [0, 1] range using a fixed scaling factor."""
    if scale <= 0:
        return 0.0
    return max(0.0, min(1.0, value / scale))


def score_candidate(candidate: dict) -> dict:
    """Compute a priority score and preserve all existing fields."""
    raw_relevance = relevance_score(candidate)
    raw_diff = _raw_diff_size(candidate)
    doc_before = candidate.get("docstring_before") or ""
    doc_after = candidate.get("docstring_after") or ""
    doc_length = max(len(doc_before), len(doc_after))

    vocabulary_divergence = normalize_0_1(float(raw_relevance), 5.0)
    diff_size_signal = normalize_0_1(float(raw_diff), 20.0)
    docstring_length_signal = normalize_0_1(float(doc_length), 200.0)
    dropped_identifier = float(dropped_identifier_signal(candidate))

    score = (
        0.45 * vocabulary_divergence
        + 0.35 * dropped_identifier
        + 0.12 * diff_size_signal
        + 0.08 * docstring_length_signal
    )

    candidate["vocabulary_divergence"] = round(vocabulary_divergence, 6)
    candidate["diff_size"] = round(diff_size_signal, 6)
    candidate["dropped_identifier_signal"] = float(dropped_identifier)
    candidate["docstring_length"] = round(docstring_length_signal, 6)
    candidate["review_priority_score"] = round(score, 6)
    return candidate


def main() -> int:
    parser = argparse.ArgumentParser(description="Score filtered drift candidates by review priority.")
    parser.add_argument("--repo", required=True, help="Repository folder name")
    args = parser.parse_args()

    input_path = CANDIDATE_ROOT / f"{args.repo}_filtered.jsonl"
    output_path = CANDIDATE_ROOT / f"{args.repo}_scored.jsonl"
    if not input_path.is_file():
        parser.error(f"Filtered candidate file not found: {input_path}")

    scored: list[dict] = []
    for candidate in load_candidates(input_path):
        try:
            scored.append(score_candidate(candidate))
        except (TypeError, ValueError, SyntaxError, AttributeError) as exc:
            print(f"Warning: skipping malformed candidate {candidate.get('function_name', 'unknown')}: {exc}", file=sys.stderr)

    scored.sort(key=lambda record: record.get("review_priority_score", 0.0), reverse=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        for candidate in scored:
            output_file.write(json.dumps(candidate, ensure_ascii=False) + "\n")

    values = [candidate.get("review_priority_score", 0.0) for candidate in scored]
    if values:
        minimum = min(values)
        maximum = max(values)
        median = statistics.median(values)
        print(f"Total candidates scored: {len(scored)}")
        print(f"Score range: min={minimum:.6f}, max={maximum:.6f}, median={median:.6f}")
    else:
        print(f"Total candidates scored: 0")
        print("Score range: min=0.000000, max=0.000000, median=0.000000")
    return 0


if __name__ == "__main__":
    sys.exit(main())
