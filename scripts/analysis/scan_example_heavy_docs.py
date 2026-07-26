"""
Scans data/labeled/semdrift_labeled.jsonl and flags docstrings that are
dominated by REPL-example / numeric-table output rather than actual prose
explanation of behavior.

Why this matters: a docstring that's mostly a printed DataFrame/array dump
(e.g. pandas' `>>> df.eval(...)` examples) carries much less semantic
"this is what the function does" signal than plain English explanation --
even though it's completely real, human-written, professional documentation.
This script doesn't decide whether to filter these out; it just measures
how common the pattern is, so that's an informed decision rather than a
gut call based on one example.

Heuristic (deliberately simple and inspectable, not a black box):
  - A line counts as "example-like" if it contains a '>>>' REPL prompt,
    OR if it's mostly digits/whitespace/punctuation (a table row) with
    very few actual word characters.
  - A docstring is "example-heavy" if more than EXAMPLE_HEAVY_THRESHOLD
    fraction of its non-blank lines are example-like.
"""

import os
import json
import re

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH = os.path.join(BASE_DIR, "..", "data", "labeled", "semdrift_labeled.jsonl")
FLAGGED_OUTPUT_PATH = os.path.join(BASE_DIR, "..", "data", "labeled", "example_heavy_flagged.jsonl")

EXAMPLE_HEAVY_THRESHOLD = 0.4  # >40% of lines are example-like -> flagged

# A line is "table-like" if, after stripping, it's mostly digits/whitespace/
# punctuation with few word characters -- catches printed DataFrame/array rows
# like "0   1   10   10" or "A   B   C&C" without needing to parse real tables.
WORD_CHAR_RE = re.compile(r"[A-Za-z]{3,}")  # real words are 3+ letters


def is_example_like_line(line):
    stripped = line.strip()
    if not stripped:
        return False
    if ">>>" in stripped or stripped.startswith("..."):
        return True
    # Count "real word" tokens (3+ letter runs) vs total tokens
    words = WORD_CHAR_RE.findall(stripped)
    tokens = stripped.split()
    if not tokens:
        return False
    # If fewer than 30% of tokens are real words, treat as a numeric/table row
    return (len(words) / len(tokens)) < 0.3


def analyze_docstring(docstring):
    lines = [l for l in docstring.split("\n") if l.strip()]
    if not lines:
        return 0.0, 0, 0
    example_lines = sum(1 for l in lines if is_example_like_line(l))
    fraction = example_lines / len(lines)
    return fraction, example_lines, len(lines)


def main():
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    print(f"Loaded {len(records)} records from {INPUT_PATH}")

    flagged = []
    fractions = []

    for r in records:
        fraction, example_lines, total_lines = analyze_docstring(r["docstring"])
        fractions.append(fraction)
        if fraction >= EXAMPLE_HEAVY_THRESHOLD:
            flagged.append({
                "function_id": r["function_id"],
                "repo": r["repo"],
                "function_name": r["function_name"],
                "label": r["label"],
                "example_line_fraction": round(fraction, 3),
                "example_lines": example_lines,
                "total_lines": total_lines,
                "docstring_preview": r["docstring"][:200],
            })

    flagged.sort(key=lambda x: -x["example_line_fraction"])

    with open(FLAGGED_OUTPUT_PATH, "w", encoding="utf-8") as out_f:
        for item in flagged:
            out_f.write(json.dumps(item) + "\n")

    # --- Summary stats ---
    avg_fraction = sum(fractions) / len(fractions) if fractions else 0
    print(f"\n=== Example-heavy docstring analysis ===")
    print(f"Threshold: docstrings with >= {EXAMPLE_HEAVY_THRESHOLD:.0%} example-like lines are flagged")
    print(f"Average example-like-line fraction across all docstrings: {avg_fraction:.1%}")
    print(f"Flagged records: {len(flagged)} / {len(records)} ({len(flagged)/len(records):.1%})")

    # Breakdown by repo, since pandas/numpy are likely the main offenders
    by_repo = {}
    for item in flagged:
        by_repo[item["repo"]] = by_repo.get(item["repo"], 0) + 1
    print(f"\nFlagged records by repo:")
    for repo, count in sorted(by_repo.items(), key=lambda x: -x[1]):
        print(f"  {repo}: {count}")

    print(f"\nTop 5 most example-heavy docstrings (function_id, fraction):")
    for item in flagged[:5]:
        print(f"  {item['function_id']}  ({item['example_line_fraction']:.0%} example-like)")

    print(f"\nFull flagged list saved to: {FLAGGED_OUTPUT_PATH}")
    print("Review a sample of these with your Research Lead to decide whether")
    print("to keep as-is (real-world diversity) or add a filtering step that")
    print("trims/limits REPL-example blocks in build_dataset.py's input.")


if __name__ == "__main__":
    main()