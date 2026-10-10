"""
SemDrift Dataset Expansion - Step 2: Candidate Mining
Mines real-world function candidates across expansion repositories with
exact immutable commit SHAs, AST properties, and docstring specifications.
"""

import os
import sys
import re
import ast
import json
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional

def parse_docstring(docstring: str) -> Dict[str, Any]:
    if not docstring:
        return {"params": [], "param_types": {}, "returns": False, "raises": [], "defaults": {}}
    
    params = []
    param_types = {}
    defaults = {}
    raises = []
    has_returns = False
    
    # Sphinx format: :param name: or :type name:
    sphinx_param = re.findall(r":param\s+(?:([\w\[\], ]+)\s+)?(\w+):", docstring)
    for p_type, p_name in sphinx_param:
        params.append(p_name)
        if p_type:
            param_types[p_name] = p_type.strip()
            
    sphinx_type = re.findall(r":type\s+(\w+):\s*([\w\[\], ]+)", docstring)
    for p_name, p_type in sphinx_type:
        param_types[p_name] = p_type.strip()

    if any(k in docstring for k in [":return:", ":returns:", ":rtype:"]):
        has_returns = True

    # Google format: Args:\n    name (type): ...
    google_args_match = re.search(r"(?:Args|Arguments|Parameters):\s*\n((?:\s+\w+.*?\n)+)", docstring, re.MULTILINE)
    if google_args_match:
        arg_lines = google_args_match.group(1).split("\n")
        for line in arg_lines:
            m = re.match(r"\s+([a-zA-Z_]\w*)(?:\s*\((.*?)\))?\s*:", line)
            if m:
                p_name = m.group(1)
                p_type = m.group(2)
                if p_name not in params:
                    params.append(p_name)
                if p_type:
                    param_types[p_name] = p_type.strip()
                    # Check if default is mentioned in doc (e.g. default=None)
                    def_match = re.search(r"default(?:s to|\s*[:=])\s*([^,\)]+)", p_type, re.IGNORECASE)
                    if def_match:
                        defaults[p_name] = def_match.group(1).strip()

    if re.search(r"(?:Returns|Return|Yields):\s*\n", docstring):
        has_returns = True

    # NumPy format
    numpy_params_match = re.search(r"Parameters\s*\n\s*-+\s*\n((?:[^\n]+\n)+)", docstring)
    if numpy_params_match:
        for line in numpy_params_match.group(1).split("\n"):
            m = re.match(r"([a-zA-Z_]\w*)\s*(?::\s*(.*))?", line.strip())
            if m and m.group(1) and not m.group(1).startswith("-"):
                p_name = m.group(1)
                p_type = m.group(2)
                if p_name not in params:
                    params.append(p_name)
                if p_type:
                    param_types[p_name] = p_type.strip()
                    if "default" in p_type.lower():
                        def_match = re.search(r"default(?:s to|\s*[:=])\s*([^,\)]+)", p_type, re.IGNORECASE)
                        if def_match:
                            defaults[p_name] = def_match.group(1).strip()

    if re.search(r"Returns\s*\n\s*-+\s*\n", docstring):
        has_returns = True

    # Raises detection
    raises_matches = re.findall(r"(?:Raises|raises|raise)\s+(?:(\w+Error|\w+Exception))", docstring)
    for r in raises_matches:
        if r not in raises:
            raises.append(r)

    return {
        "params": list(dict.fromkeys(params)),
        "param_types": param_types,
        "defaults": defaults,
        "returns": has_returns,
        "raises": raises
    }


def extract_ast(node: ast.FunctionDef) -> Dict[str, Any]:
    args = []
    arg_types = {}
    arg_defaults = {}
    
    # Non-kwonly args
    non_kw_defaults = [None] * (len(node.args.args) - len(node.args.defaults)) + list(node.args.defaults)
    for arg, default_node in zip(node.args.args, non_kw_defaults):
        if arg.arg not in ("self", "cls"):
            args.append(arg.arg)
            if arg.annotation:
                try:
                    arg_types[arg.arg] = ast.unparse(arg.annotation)
                except Exception:
                    pass
            if default_node:
                try:
                    arg_defaults[arg.arg] = ast.unparse(default_node)
                except Exception:
                    pass

    # Kwonly args
    for arg, default_node in zip(node.args.kwonlyargs, node.args.kw_defaults):
        args.append(arg.arg)
        if arg.annotation:
            try:
                arg_types[arg.arg] = ast.unparse(arg.annotation)
            except Exception:
                pass
        if default_node:
            try:
                arg_defaults[arg.arg] = ast.unparse(default_node)
            except Exception:
                pass

    if node.args.vararg:
        args.append(node.args.vararg.arg)
    if node.args.kwarg:
        args.append(node.args.kwarg.arg)

    has_return = False
    return_type = None
    if node.returns:
        try:
            return_type = ast.unparse(node.returns)
            has_return = True
        except Exception:
            pass

    for child in ast.walk(node):
        if isinstance(child, ast.Return) and child.value is not None:
            has_return = True

    raises = []
    for child in ast.walk(node):
        if isinstance(child, ast.Raise):
            if child.exc and isinstance(child.exc, ast.Call) and hasattr(child.exc.func, "id"):
                raises.append(child.exc.func.id)
            elif child.exc and isinstance(child.exc, ast.Name):
                raises.append(child.exc.id)

    is_async = isinstance(node, ast.AsyncFunctionDef)

    return {
        "args": args,
        "arg_types": arg_types,
        "arg_defaults": arg_defaults,
        "has_return": has_return,
        "return_type": return_type,
        "raises": list(set(raises)),
        "is_async": is_async
    }


def mine_repo(repo_path: Path, repo_info: Dict[str, Any]) -> List[Dict[str, Any]]:
    candidates = []
    repo_name = repo_info["repository"]
    sha = repo_info["immutable_commit_sha"]

    py_files = []
    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in [".git", ".venv", "venv", "build", "dist", "docs", "tests", "test", "__pycache__", "site-packages"]]
        for f in files:
            if f.endswith(".py"):
                py_files.append(Path(root) / f)

    for pf in py_files:
        try:
            content = pf.read_text(encoding="utf-8", errors="ignore")
            tree = ast.parse(content)
        except Exception:
            continue

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                docstring = ast.get_docstring(node)
                if not docstring or len(docstring.strip()) < 20:
                    continue
                if node.name.startswith("__") and node.name.endswith("__"):
                    continue
                if node.name.startswith("test_"):
                    continue

                try:
                    code_seg = ast.get_source_segment(content, node)
                    if not code_seg or len(code_seg.strip()) < 30:
                        continue
                except Exception:
                    continue

                ast_info = extract_ast(node)
                doc_info = parse_docstring(docstring)

                rel_path = str(pf.relative_to(repo_path)).replace("\\", "/")

                candidates.append({
                    "repository": repo_name,
                    "url": repo_info["url"],
                    "license": repo_info["license"],
                    "commit_hash": sha,
                    "filepath": rel_path,
                    "function_name": node.name,
                    "code": code_seg,
                    "docstring": docstring.strip(),
                    "ast_info": ast_info,
                    "doc_info": doc_info
                })

    return candidates


def main():
    base_dir = Path(__file__).resolve().parents[2]
    manifest_file = base_dir / "data" / "real_dataset_expansion_v1" / "manifests" / "repository_manifest.json"
    output_file = base_dir / "data" / "real_dataset_expansion_v1" / "candidates" / "raw_candidates.jsonl"
    
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if not manifest_file.exists():
        print(f"[-] Manifest {manifest_file} not found. Run 01_discover_repositories.py first.", flush=True)
        return

    with open(manifest_file, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    all_candidates = []
    print(f"[*] Starting candidate extraction across {len(manifest)} expansion repositories...", flush=True)

    for repo_info in manifest:
        repo_name = repo_info["repository"]
        repo_path = Path(repo_info["local_path"])
        if not repo_path.exists():
            print(f"[-] Repository path {repo_path} missing. Skipping.", flush=True)
            continue

        cands = mine_repo(repo_path, repo_info)
        print(f"    [+] {repo_name}: Extracted {len(cands)} verified candidate functions.", flush=True)
        all_candidates.extend(cands)

    with open(output_file, "w", encoding="utf-8") as f:
        for c in all_candidates:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    print(f"\n[+] Total raw candidates extracted: {len(all_candidates):,}", flush=True)
    print(f"[+] Saved to {output_file}", flush=True)

if __name__ == "__main__":
    main()
