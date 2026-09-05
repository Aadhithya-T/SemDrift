"""Filter and rank mined real-world documentation-drift candidates."""

import argparse
import ast
import json
import logging
import re
import sys
import tokenize
import textwrap
from io import StringIO
from pathlib import Path


LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CANDIDATE_ROOT = PROJECT_ROOT / "data" / "real_world" / "mined_candidates"
IGNORED_TOKEN_TYPES = {
    tokenize.ENCODING,
    tokenize.NL,
    tokenize.NEWLINE,
    tokenize.INDENT,
    tokenize.DEDENT,
    tokenize.COMMENT,
}
WORD_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def load_candidates(path: Path) -> list[dict]:
    """Load non-empty JSONL records from a candidate file."""
    candidates = []
    with path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            try:
                candidates.append(json.loads(line))
            except json.JSONDecodeError as exc:
                LOGGER.warning("Skipping malformed JSON on line %d: %s", line_number, exc)
    return candidates


def significant_tokens(source: str) -> list[tuple[int, str]]:
    """Return token type/string pairs excluding layout and comments."""
    return [
        (token.type, token.string)
        for token in tokenize.generate_tokens(StringIO(textwrap.dedent(source)).readline)
        if token.type not in IGNORED_TOKEN_TYPES
    ]


def is_formatting_only(candidate: dict) -> bool:
    """Return whether the code differs only in whitespace or comments."""
    before = candidate.get("code_before", "")
    after = candidate.get("code_after", "")
    return significant_tokens(before) == significant_tokens(after)


def normalized_ast(source: str) -> str:
    """Dump an AST with local variable and parameter names normalized."""
    tree = ast.parse(textwrap.dedent(source))
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            node.id = "__NAME__"
        elif isinstance(node, ast.arg):
            node.arg = "__ARG__"
    return ast.dump(tree, annotate_fields=True, include_attributes=False)


def identifier_names(source: str) -> set[str]:
    """Collect variable and parameter identifiers from a function source."""
    tree = ast.parse(textwrap.dedent(source))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.arg):
            names.add(node.arg)
    return names


def is_rename_only(candidate: dict) -> bool:
    """Return whether a name-only change is unreferenced by the documentation."""
    before = candidate.get("code_before", "")
    after = candidate.get("code_after", "")
    if significant_tokens(before) == significant_tokens(after):
        return False
    if normalized_ast(before) != normalized_ast(after):
        return False

    changed_names = identifier_names(before) ^ identifier_names(after)
    docstring = " ".join(
        (
            candidate.get("docstring_before") or "",
            candidate.get("docstring_after") or "",
        )
    )
    return not any(
        re.search(rf"\b{re.escape(name)}\b", docstring)
        for name in changed_names
    )


def has_meaningful_docstring(candidate: dict) -> bool:
    """Return whether the present docstring is at least 20 characters long."""
    docstring = candidate.get("docstring_before") or candidate.get("docstring_after") or ""
    return len(docstring.strip()) >= 20


def has_nontrivial_code(candidate: dict) -> bool:
    """Return whether both code snapshots contain at least three lines."""
    before_lines = candidate.get("code_before", "").splitlines()
    after_lines = candidate.get("code_after", "").splitlines()
    return len(before_lines) >= 3 and len(after_lines) >= 3


def diff_size(candidate: dict) -> int:
    """Measure changed source lines between the two code snapshots."""
    before = candidate.get("code_before", "").splitlines()
    after = candidate.get("code_after", "").splitlines()
    common = 0
    before_counts = {}
    for line in before:
        before_counts[line] = before_counts.get(line, 0) + 1
    for line in after:
        if before_counts.get(line, 0):
            before_counts[line] -= 1
            common += 1
    return (len(before) - common) + (len(after) - common)


def relevance_score(candidate: dict) -> int:
    """Score words that appear on only one side of the code/doc relationship."""
    doc_words = {
        word.lower()
        for word in WORD_PATTERN.findall(
            (candidate.get("docstring_before") or "") + " " + (candidate.get("docstring_after") or "")
        )
    }
    before_words = {word.lower() for word in WORD_PATTERN.findall(candidate.get("code_before", ""))}
    after_words = {word.lower() for word in WORD_PATTERN.findall(candidate.get("code_after", ""))}
    return len(doc_words & (before_words ^ after_words))


def filter_candidates(candidates: list[dict]) -> tuple[list[dict], dict[str, int]]:
    """Apply filters in order and return survivors plus funnel counts."""
    removed = {
        "formatting_only": 0,
        "rename_only": 0,
        "short_docstring": 0,
        "short_code": 0,
        "duplicates": 0,
    }
    current = []

    for candidate in candidates:
        try:
            if is_formatting_only(candidate):
                removed["formatting_only"] += 1
                continue
            if is_rename_only(candidate):
                removed["rename_only"] += 1
                continue
            if not has_meaningful_docstring(candidate):
                removed["short_docstring"] += 1
                continue
            if not has_nontrivial_code(candidate):
                removed["short_code"] += 1
                continue
            candidate["_diff_size"] = diff_size(candidate)
            current.append(candidate)
        except (SyntaxError, tokenize.TokenError, ValueError, TypeError) as exc:
            LOGGER.warning("Skipping malformed candidate for %s: %s", candidate.get("function_name", "unknown"), exc)

    largest_by_function = {}
    for candidate in current:
        key = (candidate.get("file_path", ""), candidate.get("function_name", ""))
        previous = largest_by_function.get(key)
        if previous is None or candidate["_diff_size"] > previous["_diff_size"]:
            largest_by_function[key] = candidate

    removed["duplicates"] = len(current) - len(largest_by_function)
    survivors = list(largest_by_function.values())
    for candidate in survivors:
        candidate["_relevance_score"] = relevance_score(candidate)
    return survivors, removed


def main() -> int:
    parser = argparse.ArgumentParser(description="Filter mined real-world drift candidates.")
    parser.add_argument("--repo", required=True, help="Repository folder name")
    parser.add_argument("--max-output", type=int, default=500, help="Maximum filtered candidates to write")
    args = parser.parse_args()

    input_path = CANDIDATE_ROOT / f"{args.repo}_candidates.jsonl"
    output_path = CANDIDATE_ROOT / f"{args.repo}_filtered.jsonl"
    if not input_path.is_file():
        LOGGER.error("Candidate file not found: %s", input_path)
        return 1
    if args.max_output < 0:
        LOGGER.error("--max-output must be non-negative")
        return 1

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    candidates = load_candidates(input_path)
    filtered, removed = filter_candidates(candidates)
    filtered.sort(key=lambda candidate: candidate["_relevance_score"], reverse=True)
    selected = filtered[:args.max_output]

    with output_path.open("w", encoding="utf-8") as output_file:
        for candidate in selected:
            candidate = {key: value for key, value in candidate.items() if not key.startswith("_")}
            output_file.write(json.dumps(candidate, ensure_ascii=False) + "\n")

    print(f"Starting count: {len(candidates)}")
    print(f"Removed by filter a (formatting-only): {removed['formatting_only']}")
    print(f"Removed by filter b (rename-only): {removed['rename_only']}")
    print(f"Removed by filter c (short docstring): {removed['short_docstring']}")
    print(f"Removed by filter d (short code): {removed['short_code']}")
    print(f"Removed by filter e (duplicates): {removed['duplicates']}")
    print(f"Final count after dedup: {len(filtered)}")
    print(f"Final count after top N: {len(selected)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
