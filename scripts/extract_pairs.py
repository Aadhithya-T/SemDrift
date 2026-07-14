import ast
import json
import os


def is_dev_tooling_function(node):
    """
    Returns True if this function is decorated with CLI/dev-tooling
    decorators (click, spin, etc.) rather than being core library code.
    """
    dev_tooling_markers = ("click", "spin")
    for decorator in node.decorator_list:
        try:
            decorator_source = ast.unparse(decorator)
        except Exception:
            continue
        for marker in dev_tooling_markers:
            if marker in decorator_source:
                return True
    return False


def extract_pairs_from_file(filepath, repo_name):
    """Extract (function, docstring) pairs from a single .py file."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            source = f.read()
        tree = ast.parse(source)
    except (SyntaxError, UnicodeDecodeError):
        return []
    pairs = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            docstring = ast.get_docstring(node)
            if docstring and not is_dev_tooling_function(node):
                try:
                    code = ast.unparse(node)
                except Exception:
                    continue
                pairs.append({
                    "repo": repo_name,
                    "file": filepath,
                    "function_name": node.name,
                    "code": code,
                    "docstring": docstring,
                    "lineno": node.lineno,  # NEW: line number of the 'def' line
                })
    return pairs


def walk_repo(repo_path, repo_name):
    """Walk every .py file in a repo, skipping tests/examples/migrations/dev-tooling dirs."""
    skip_dirs = {
        "tests", "test", "examples", "migrations", "docs", ".git",
        "tools", "_build_utils", "benchmarks", "doc", "build",
        "vendor", "ci", "release", "scripts", ".spin",
    }
    all_pairs = []
    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        for file in files:
            if file.endswith(".py"):
                filepath = os.path.join(root, file)
                all_pairs.extend(extract_pairs_from_file(filepath, repo_name))
    return all_pairs


if __name__ == "__main__":
    raw_repos_dir = "data/raw_repos"
    output_file = "data/extracted_pairs.jsonl"
    all_pairs = []
    for repo_name in os.listdir(raw_repos_dir):
        repo_path = os.path.join(raw_repos_dir, repo_name)
        if os.path.isdir(repo_path):
            print(f"Extracting from {repo_name}...")
            pairs = walk_repo(repo_path, repo_name)
            print(f"  -> found {len(pairs)} pairs")
            all_pairs.extend(pairs)
    with open(output_file, "w", encoding="utf-8") as f:
        for pair in all_pairs:
            f.write(json.dumps(pair) + "\n")
    print(f"\nTotal pairs extracted: {len(all_pairs)}")
    print(f"Saved to {output_file}")