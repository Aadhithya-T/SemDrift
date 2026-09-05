"""Interactively review top-priority mined drift candidates."""

import argparse
import json
import sys
from pathlib import Path

try:
    import msvcrt
except ImportError:
    msvcrt = None


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CANDIDATE_ROOT = PROJECT_ROOT / "data" / "real_world" / "mined_candidates"
VERIFIED_ROOT = PROJECT_ROOT / "data" / "real_world" / "verified"
REVIEWED_ROOT = PROJECT_ROOT / "data" / "real_world" / "rejected"


LABELING_GUIDE = """Labeling guide:
- drift: the docstring no longer accurately/completely describes what the code does after this change (wrong param, wrong return, wrong behavior description)
- no_drift: the change is real but the docstring's claims still hold, or the docstring was already vague enough to still cover the new behavior
- ambiguous: genuinely unclear without more context, or reasonable people could disagree
"""


def load_jsonl(path: Path) -> list[dict]:
    """Load non-empty JSON objects from a JSONL file."""
    records = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"Warning: skipping malformed JSON at {path}:{line_number}: {exc}", file=sys.stderr)
    return records


def candidate_key(candidate: dict) -> tuple[str, str]:
    """Return the stable review identity requested by the workflow."""
    return candidate.get("commit_hash", ""), candidate.get("function_name", "")


def load_reviewed_keys(repo_name: str) -> set[tuple[str, str]]:
    """Load candidates already labeled as verified or rejected."""
    reviewed = set()
    for path in (
        VERIFIED_ROOT / f"{repo_name}_verified.jsonl",
        REVIEWED_ROOT / f"{repo_name}_rejected.jsonl",
    ):
        reviewed.update(candidate_key(candidate) for candidate in load_jsonl(path))
    return reviewed


def load_or_create_sample(repo_name: str, candidates: list[dict], top_n: int) -> list[dict]:
    """Load a fixed sample, creating or expanding it from top-N scored candidates."""
    sample_path = CANDIDATE_ROOT / f"{repo_name}_review_sample.jsonl"
    if sample_path.exists():
        existing = load_jsonl(sample_path)
        if len(existing) >= top_n:
            return existing[:top_n]

    sample = candidates[:top_n]
    CANDIDATE_ROOT.mkdir(parents=True, exist_ok=True)
    with sample_path.open("w", encoding="utf-8") as sample_file:
        for candidate in sample:
            sample_file.write(json.dumps(candidate, ensure_ascii=False) + "\n")
    return sample


def read_choice() -> str:
    """Read one review command without requiring Enter on Windows."""
    if msvcrt is not None:
        choice = msvcrt.getwch()
        print(choice)
        return choice.lower()
    return input().strip().lower()[:1]


def print_truncated(label: str, text: str, max_lines: int = 40) -> None:
    """Print a labeled code block, truncating long content for readability."""
    lines = text.splitlines()
    print(f"{label}:")
    if not lines:
        print("(none)")
        return
    print("\n".join(lines[:max_lines]))
    if len(lines) > max_lines:
        print(f"... ({len(lines) - max_lines} more lines truncated)")


def display_candidate(candidate: dict, position: int, total: int) -> None:
    """Print one candidate in the review format."""
    print("=" * 64)
    print(
        f"[{position} of {total}]  {candidate.get('repo_name', '')} / "
        f"{candidate.get('file_path', '')} :: {candidate.get('function_name', '')}"
    )
    print(f"Commit: {candidate.get('commit_hash', '')[:16]}...")
    print("-" * 64)
    print_truncated("DOCSTRING (before)", candidate.get("docstring_before", "") or "(none)", 40)
    print()
    print_truncated("DOCSTRING (after)", candidate.get("docstring_after", "") or "(none)", 40)
    print("-" * 64)
    print_truncated("CODE (before)", candidate.get("code_before", "") or "(none)", 40)
    print()
    print_truncated("CODE (after)", candidate.get("code_after", "") or "(none)", 40)
    print("=" * 64)
    print("[d] drift  [n] no_drift  [a] ambiguous  [s] skip  [q] quit")
    print("Choice: ", end="", flush=True)


def append_labeled(repo_name: str, candidate: dict, label: str, reviewer: str, notes: str) -> None:
    """Append a full candidate plus review metadata to its destination file."""
    record = dict(candidate)
    record.update({
        "drift_label": label,
        "verified_by": reviewer,
        "verification_notes": notes,
    })
    destination_root = REVIEWED_ROOT if label == "no_drift" else VERIFIED_ROOT
    destination_root.mkdir(parents=True, exist_ok=True)
    destination = destination_root / (
        f"{repo_name}_rejected.jsonl" if label == "no_drift" else f"{repo_name}_verified.jsonl"
    )
    with destination.open("a", encoding="utf-8") as output_file:
        output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        output_file.flush()


def main() -> int:
    parser = argparse.ArgumentParser(description="Review top-priority mined drift candidates.")
    parser.add_argument("--repo", required=True, help="Repository folder name")
    parser.add_argument("--reviewer", required=True, help="Reviewer name")
    parser.add_argument("--top-n", type=int, default=50, help="Number of top-priority candidates to review")
    args = parser.parse_args()

    if args.top_n <= 0:
        parser.error("--top-n must be greater than 0")

    input_path = CANDIDATE_ROOT / f"{args.repo}_scored.jsonl"
    if not input_path.is_file():
        parser.error(
            f"Scored candidate file not found: {input_path}\n"
            f"Please run 'python scripts/score_review_priority.py --repo {args.repo}' first."
        )

    candidates = load_jsonl(input_path)
    sample = load_or_create_sample(args.repo, candidates, args.top_n)
    reviewed_keys = load_reviewed_keys(args.repo)
    pending = [candidate for candidate in sample if candidate_key(candidate) not in reviewed_keys]

    print(LABELING_GUIDE)
    print(f"Selected top {len(sample)} of {len(candidates)} scored candidates")
    print(f"Already reviewed: {len(sample) - len(pending)}")

    reviewed_this_session = 0
    try:
        for position, candidate in enumerate(pending, start=1):
            display_candidate(candidate, position, len(pending))
            while True:
                choice = read_choice()
                if choice in {"d", "n", "a", "s", "q"}:
                    break
                print("Invalid choice; use d, n, a, s, or q.")
                print("Choice: ", end="", flush=True)
            if choice == "q":
                break
            if choice == "s":
                continue

            print("Verification notes (optional): ", end="", flush=True)
            notes = input().strip()
            label = {"d": "drift", "n": "no_drift", "a": "ambiguous"}[choice]
            append_labeled(args.repo, candidate, label, args.reviewer, notes)
            reviewed_keys.add(candidate_key(candidate))
            reviewed_this_session += 1
    except (KeyboardInterrupt, EOFError):
        print("\nReview interrupted; progress already written.")

    remaining = sum(candidate_key(candidate) not in reviewed_keys for candidate in sample)
    print(f"Session summary: reviewed this session: {reviewed_this_session}, remaining in sample: {remaining}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
