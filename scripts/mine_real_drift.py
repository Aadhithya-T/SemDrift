"""Mine git history for possible real-world code/documentation drift."""

import argparse
import json
import logging
import re
import subprocess
import sys
import time
from pathlib import Path


LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_REPOS_ROOT = PROJECT_ROOT / "data" / "raw_repos"
OUTPUT_ROOT = PROJECT_ROOT / "data" / "real_world" / "mined_candidates"

if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

COMMIT_MARKER = "__SEMDIRFT_COMMIT__"
COMMIT_PATTERN = re.compile(rf"^{COMMIT_MARKER}([0-9a-f]{{40}})$", re.MULTILINE)


def run_git(repo_path: Path, *args: str) -> str:
	"""Run a git command and return text output."""
	completed = subprocess.run(
		["git", *args],
		cwd=repo_path,
		check=True,
		capture_output=True,
		text=True,
		encoding="utf-8",
		errors="replace",
	)
	return completed.stdout


def tracked_python_files(repo_path: Path) -> list[str]:
	"""Return version-controlled Python files in the repository."""
	output = run_git(repo_path, "ls-files", "*.py")
	return [line.strip() for line in output.splitlines() if line.strip()]


def history_commits(repo_path: Path, file_path: str) -> list[str]:
	"""Run the oldest-first history for the current path and extract commit hashes."""
	history = run_git(
		repo_path,
		"log",
		"-p",
		"--reverse",
		f"--format={COMMIT_MARKER}%H",
		"--",
		file_path,
	)
	return COMMIT_PATTERN.findall(history)


def commit_parent(repo_path: Path, commit_hash: str) -> str | None:
	"""Return the sole parent, or None for roots and merge commits."""
	parents = run_git(repo_path, "rev-list", "--parents", "-n", "1", commit_hash)
	hashes = parents.split()
	if len(hashes) != 2:
		return None
	return hashes[1]


def show_file(repo_path: Path, commit_hash: str, file_path: str) -> str:
	"""Read a file snapshot from a particular commit."""
	return run_git(repo_path, "show", f"{commit_hash}:{file_path}")


def parse_functions(source: str, file_path: str):
	"""Parse source with the repository's existing AST-based extractor."""
	from semdrift.parser.ast_parser import ASTParser

	parser = ASTParser(skip_test_files=False)
	return parser._parse_source(source, file_path)


def function_map(source: str, file_path: str) -> dict[tuple[str, str | None], object]:
	"""Index extracted functions by class and function name."""
	return {
		(function.class_name or "", function.name): function
		for function in parse_functions(source, file_path)
	}


def mine_file(
    repo_path: Path,
    repo_name: str,
    file_path: str,
    output_file,
) -> tuple[int, set[str]]:
	"""Mine one tracked Python file and return candidates and processed commits."""
	candidate_count = 0
	processed_commits: set[str] = set()

	try:
		commits = history_commits(repo_path, file_path)
	except (OSError, subprocess.SubprocessError) as exc:
		LOGGER.warning("Could not read history for %s: %s", file_path, exc)
		return candidate_count, processed_commits

	for commit_hash in commits:
		try:
			parent_hash = commit_parent(repo_path, commit_hash)
			if parent_hash is None:
				LOGGER.warning("Skipping root or merge commit %s for %s", commit_hash, file_path)
				continue

			before = show_file(repo_path, parent_hash, file_path)
			after = show_file(repo_path, commit_hash, file_path)
			before_functions = function_map(before, file_path)
			after_functions = function_map(after, file_path)
			processed_commits.add(commit_hash)
		except (OSError, subprocess.SubprocessError, SyntaxError, ValueError, ImportError) as exc:
			LOGGER.warning("Skipping commit %s for %s: %s", commit_hash, file_path, exc)
			continue

		for key in before_functions.keys() & after_functions.keys():
			before_function = before_functions[key]
			after_function = after_functions[key]
			code_changed = before_function.source_code != after_function.source_code
			docstring_changed = before_function.docstring != after_function.docstring

			# TODO: Add a conservative AST-aware filter for formatting-only diffs.
			if code_changed == docstring_changed:
				continue

			candidate = {
				"repo_name": repo_name,
				"file_path": file_path,
				"function_name": after_function.name,
				"commit_hash": commit_hash,
				"code_before": before_function.source_code,
				"code_after": after_function.source_code,
				"docstring_before": before_function.docstring,
				"docstring_after": after_function.docstring,
			}
			output_file.write(json.dumps(candidate, ensure_ascii=False) + "\n")
			output_file.flush()
			candidate_count += 1

	return candidate_count, processed_commits


def mine_repo(repo_name: str) -> tuple[int, int, int, float]:
	"""Mine one repository and write its candidate JSONL output."""
	started = time.perf_counter()
	repo_path = (RAW_REPOS_ROOT / repo_name).resolve()
	if repo_path.parent != RAW_REPOS_ROOT.resolve() or not repo_path.is_dir():
		raise ValueError(f"Repository does not exist under data/raw_repos/: {repo_name}")

	try:
		files = tracked_python_files(repo_path)
	except (OSError, subprocess.SubprocessError) as exc:
		raise RuntimeError(f"Could not list tracked Python files: {exc}") from exc

	output_path = OUTPUT_ROOT / f"{repo_name}_candidates.jsonl"
	progress_path = OUTPUT_ROOT / f"{repo_name}_progress.txt"
	completed_files = set()
	if progress_path.exists():
		completed_files = {
			line.strip()
			for line in progress_path.read_text(encoding="utf-8").splitlines()
			if line.strip()
		}

	candidate_count = 0
	processed_commits: set[str] = set()
	OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
	with output_path.open("a", encoding="utf-8") as output_file, progress_path.open("a", encoding="utf-8") as progress_file:
		for file_path in files:
			if file_path in completed_files:
				continue
			file_candidate_count, file_commits = mine_file(
				repo_path, repo_name, file_path, output_file
			)
			candidate_count += file_candidate_count
			processed_commits.update(file_commits)
			progress_file.write(file_path + "\n")
			progress_file.flush()

	return len(files), len(processed_commits), candidate_count, time.perf_counter() - started


def main() -> int:
	parser = argparse.ArgumentParser(description="Mine real-world documentation drift candidates from one repository.")
	parser.add_argument("--repo", required=True, help="Repository folder name under data/raw_repos/")
	args = parser.parse_args()

	logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
	try:
		files, commits, candidates, elapsed = mine_repo(args.repo)
	except (OSError, RuntimeError, ValueError) as exc:
		LOGGER.error("Mining failed: %s", exc)
		return 1

	print(f"Total files scanned: {files}")
	print(f"Total commits processed: {commits}")
	print(f"Total candidates found: {candidates}")
	print(f"Run time: {elapsed:.2f} seconds")
	return 0


if __name__ == "__main__":
	sys.exit(main())
