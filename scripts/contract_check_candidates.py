"""Deterministic, rule-based contract checking for mined drift candidates.

Performs structural AST and docstring analysis without machine learning to flag:
1. parameter_contract_violation: documented parameter added/removed from signature
2. return_contract_violation: return type annotation changed vs documented return
3. raises_contract_violation: documented exception raise statement disappeared
4. default_contract_violation: parameter default changed vs docstring-stated default
"""

import argparse
import ast
import json
import re
import sys
import textwrap
from pathlib import Path
from typing import Optional, Set, Dict, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CANDIDATE_ROOT = PROJECT_ROOT / "data" / "real_world" / "mined_candidates"

# Regular expressions for docstring parameter detection
DOC_PARAM_SPHINX = re.compile(
    r":(?:param|arg|argument)\s+(?:[a-zA-Z_]\w+\s+)?([a-zA-Z_]\w*)\s*:",
    re.IGNORECASE,
)
DOC_PARAM_GOOGLE = re.compile(
    r"^\s*([a-zA-Z_]\w*)\s*(?:\([^)]+\))?\s*:\s*\S+",
    re.MULTILINE,
)
DOC_PARAM_NUMPY = re.compile(
    r"^\s*([a-zA-Z_]\w*)\s*:\s*\S+",
    re.MULTILINE,
)
DOC_PARAM_QUOTED = re.compile(
    r"(?:the\s+)?['`\"]([a-zA-Z_]\w*)['`\"]\s+(?:parameter|argument|field)",
    re.IGNORECASE,
)
DOC_PARAM_QUOTED_REV = re.compile(
    r"(?:parameter|argument|field)\s+['`\"]([a-zA-Z_]\w*)['`\"]",
    re.IGNORECASE,
)

# Regular expressions for docstring sections
RE_SECTION_ARGS = re.compile(
    r"(?:Args|Arguments|Parameters|Params)\s*:\s*\n(.*?)(?=\n\s*(?:Returns?|Raises?|Yields?|Note|Notes|Example|Examples|Warns?):|\Z)",
    re.IGNORECASE | re.DOTALL,
)
RE_SECTION_NUMPY_ARGS = re.compile(
    r"Parameters\s*\n\s*-+\s*\n(.*?)(?=\n\s*(?:Returns?|Raises?|Yields?|See Also|Notes?|Examples?)\s*\n\s*-+|\Z)",
    re.IGNORECASE | re.DOTALL,
)
RE_SECTION_RETURNS = re.compile(
    r"(?:Returns?|Return)\s*:\s*\n(.*?)(?=\n\s*(?:Raises?|Yields?|Note|Notes|Example|Examples|Warns?):|\Z)",
    re.IGNORECASE | re.DOTALL,
)
RE_SECTION_NUMPY_RETURNS = re.compile(
    r"Returns\s*\n\s*-+\s*\n(.*?)(?=\n\s*(?:Raises?|Yields?|See Also|Notes?|Examples?)\s*\n\s*-+|\Z)",
    re.IGNORECASE | re.DOTALL,
)
RE_SECTION_RAISES = re.compile(
    r"(?:Raises?|Raise)\s*:\s*\n(.*?)(?=\n\s*(?:Returns?|Yields?|Note|Notes|Example|Examples|Warns?):|\Z)",
    re.IGNORECASE | re.DOTALL,
)
RE_SECTION_NUMPY_RAISES = re.compile(
    r"Raises\s*\n\s*-+\s*\n(.*?)(?=\n\s*(?:Returns?|Yields?|See Also|Notes?|Examples?)\s*\n\s*-+|\Z)",
    re.IGNORECASE | re.DOTALL,
)
RE_DOC_RAISES_SPHINX = re.compile(
    r":(?:raises?|exception)\s+([a-zA-Z_]\w*)\s*:",
    re.IGNORECASE,
)
RE_DOC_RAISES_INLINE = re.compile(
    r"\b(?:raises?|raising)\s+(?:an?\s+)?([A-Z]\w*(?:Error|Exception|Warning))",
    re.IGNORECASE,
)


def load_candidates(path: Path) -> list[dict]:
    """Load JSONL candidates while skipping malformed lines defensively."""
    records: list[dict] = []
    if not path.exists():
        return records

    with path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            line_str = line.strip()
            if not line_str:
                continue
            try:
                candidate = json.loads(line_str)
                if isinstance(candidate, dict):
                    records.append(candidate)
                else:
                    print(f"Warning: skipping non-object JSON at {path}:{line_number}", file=sys.stderr)
            except json.JSONDecodeError as exc:
                print(f"Warning: skipping malformed JSON at {path}:{line_number}: {exc}", file=sys.stderr)
                continue
    return records


def parse_ast_function(source: str, expected_name: Optional[str] = None) -> Optional[ast.FunctionDef | ast.AsyncFunctionDef]:
    """Parse a code snippet with ast and locate the target function definition."""
    if not source or not source.strip():
        return None
    try:
        tree = ast.parse(textwrap.dedent(source))
    except (SyntaxError, ValueError):
        return None

    functions: list[ast.FunctionDef | ast.AsyncFunctionDef] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions.append(node)

    if not functions:
        return None

    if expected_name:
        for func in functions:
            if func.name == expected_name:
                return func

    return functions[0]


def extract_parameter_names(func_node: Optional[ast.FunctionDef | ast.AsyncFunctionDef]) -> Set[str]:
    """Extract argument names from a function definition node, excluding self/cls/dunder."""
    if func_node is None:
        return set()

    names: Set[str] = set()
    args_obj = func_node.args

    for arg in args_obj.posonlyargs:
        names.add(arg.arg)
    for arg in args_obj.args:
        names.add(arg.arg)
    if args_obj.vararg:
        names.add(args_obj.vararg.arg)
    for arg in args_obj.kwonlyargs:
        names.add(arg.arg)
    if args_obj.kwarg:
        names.add(args_obj.kwarg.arg)

    # Filter standard receiver parameters and dunder self variables
    return {name for name in names if name not in {"self", "cls", "__self", "mcls"}}


def extract_documented_parameters(docstring: str) -> Set[str]:
    """Extract parameter names mentioned in Google, NumPy, Sphinx, or inline docstrings."""
    if not docstring or not docstring.strip():
        return set()

    params: Set[str] = set()

    # 1. Sphinx style: :param <name>: or :arg <name>:
    for match in DOC_PARAM_SPHINX.finditer(docstring):
        params.add(match.group(1))

    # 2. Google style: Args: / Parameters: section
    for sec_match in RE_SECTION_ARGS.finditer(docstring):
        section_text = sec_match.group(1)
        for line_match in DOC_PARAM_GOOGLE.finditer(section_text):
            params.add(line_match.group(1))

    # 3. NumPy style: Parameters\n--- section
    for sec_match in RE_SECTION_NUMPY_ARGS.finditer(docstring):
        section_text = sec_match.group(1)
        for line_match in DOC_PARAM_NUMPY.finditer(section_text):
            params.add(line_match.group(1))

    # 4. Explicit parameter references: 'param_name' parameter or parameter 'param_name'
    for match in DOC_PARAM_QUOTED.finditer(docstring):
        params.add(match.group(1))
    for match in DOC_PARAM_QUOTED_REV.finditer(docstring):
        params.add(match.group(1))

    return {p for p in params if p not in {"self", "cls", "__self"}}


def extract_parameter_defaults(func_node: Optional[ast.FunctionDef | ast.AsyncFunctionDef]) -> Dict[str, str]:
    """Map parameter names to their unparsed string default values."""
    if func_node is None:
        return {}

    defaults_map: Dict[str, str] = {}
    args_obj = func_node.args

    # Positional args and their defaults (aligned from the right)
    pos_args = [a.arg for a in args_obj.posonlyargs + args_obj.args]
    pos_defaults = args_obj.defaults
    if pos_defaults:
        offset = len(pos_args) - len(pos_defaults)
        for idx, default_node in enumerate(pos_defaults):
            arg_name = pos_args[offset + idx]
            try:
                defaults_map[arg_name] = ast.unparse(default_node).strip()
            except Exception:
                pass

    # Keyword-only args and their defaults
    for arg, default_node in zip(args_obj.kwonlyargs, args_obj.kw_defaults):
        if default_node is not None:
            try:
                defaults_map[arg.arg] = ast.unparse(default_node).strip()
            except Exception:
                pass

    return defaults_map


def extract_return_annotation(func_node: Optional[ast.FunctionDef | ast.AsyncFunctionDef]) -> Optional[str]:
    """Extract stringified return type annotation from function definition node."""
    if func_node is None or func_node.returns is None:
        return None
    try:
        return ast.unparse(func_node.returns).strip()
    except Exception:
        return None


def extract_raised_exceptions(func_node: Optional[ast.FunctionDef | ast.AsyncFunctionDef]) -> Set[str]:
    """Extract exception class names from `raise X(...)` or `raise X` statements in function body."""
    if func_node is None:
        return set()

    exceptions: Set[str] = set()
    for node in ast.walk(func_node):
        if isinstance(node, ast.Raise) and node.exc is not None:
            exc = node.exc
            if isinstance(exc, ast.Name):
                exceptions.add(exc.id)
            elif isinstance(exc, ast.Call):
                if isinstance(exc.func, ast.Name):
                    exceptions.add(exc.func.id)
                elif isinstance(exc.func, ast.Attribute):
                    exceptions.add(exc.func.attr)
            elif isinstance(exc, ast.Attribute):
                exceptions.add(exc.attr)
    return exceptions


def extract_documented_exceptions(docstring: str) -> Set[str]:
    """Extract exception class names from docstring Raises section or inline mentions."""
    if not docstring or not docstring.strip():
        return set()

    exceptions: Set[str] = set()

    # 1. Google / NumPy Raises sections
    for sec_match in RE_SECTION_RAISES.finditer(docstring):
        for token in re.findall(r"\b([A-Z]\w*(?:Error|Exception|Warning))\b", sec_match.group(1)):
            exceptions.add(token)
    for sec_match in RE_SECTION_NUMPY_RAISES.finditer(docstring):
        for token in re.findall(r"\b([A-Z]\w*(?:Error|Exception|Warning))\b", sec_match.group(1)):
            exceptions.add(token)

    # 2. Sphinx :raises Exception:
    for match in RE_DOC_RAISES_SPHINX.finditer(docstring):
        exceptions.add(match.group(1))

    # 3. Inline docstring mentions like "Raise a MultiPartParserError if..."
    for match in RE_DOC_RAISES_INLINE.finditer(docstring):
        exceptions.add(match.group(1))

    return exceptions


def check_parameter_contract_violation(
    params_before: Set[str],
    params_after: Set[str],
    doc_params: Set[str],
) -> bool:
    """Flag True if a documented parameter was removed from code_before to code_after (or vice versa)."""
    if not doc_params:
        return False

    removed_params = params_before - params_after
    added_params = params_after - params_before

    # Documented parameter was deleted from code signature
    if any(p in doc_params for p in removed_params):
        return True

    # Documented parameter in docstring only existed in code_after and was absent from code_before
    if any(p in doc_params for p in added_params):
        return True

    return False


def check_return_contract_violation(
    ret_before: Optional[str],
    ret_after: Optional[str],
    docstring: str,
) -> bool:
    """Flag True if return annotation changed and docstring has a Returns section that doesn't mention the new type."""
    if ret_before == ret_after:
        return False

    # Check for Returns section in docstring
    returns_text = ""
    google_match = RE_SECTION_RETURNS.search(docstring)
    if google_match:
        returns_text += " " + google_match.group(1)
    numpy_match = RE_SECTION_NUMPY_RETURNS.search(docstring)
    if numpy_match:
        returns_text += " " + numpy_match.group(1)
    sphinx_matches = re.findall(r":(?:return|returns|rtype)\s*:\s*(.*?)(?=\n\s*:|\Z)", docstring, re.IGNORECASE | re.DOTALL)
    if sphinx_matches:
        returns_text += " " + " ".join(sphinx_matches)

    # Top-level docstring sentence explicitly stating return (must start at line beginning)
    top_level_return = re.search(r"^\s*(?:Returns?|Return)\s+([^\n.]+)", docstring, re.IGNORECASE | re.MULTILINE)
    if top_level_return and not google_match and not numpy_match and not sphinx_matches:
        returns_text += " " + top_level_return.group(1)

    if not returns_text.strip():
        return False

    # If annotation changed to or from something specific
    new_type_str = ret_after or ret_before or ""
    # Extract identifiers from the type annotation (e.g., 'Iterator', 'Tuple', 'None', 'int', 'dict')
    type_tokens = set(re.findall(r"\b([A-Za-z_]\w*)\b", new_type_str))
    # Exclude typing prefixes like 't' or 'typing' or 'Optional'
    meaningful_tokens = {t for t in type_tokens if t not in {"t", "typing", "Optional", "Union"}}

    if not meaningful_tokens:
        return False

    # If none of the meaningful type tokens from the new annotation appear in the Returns documentation
    returns_lower = returns_text.lower()
    matches_new = any(t.lower() in returns_lower for t in meaningful_tokens)
    if not matches_new:
        return True

    return False


def check_raises_contract_violation(
    raises_before: Set[str],
    raises_after: Set[str],
    doc_exceptions: Set[str],
) -> bool:
    """Flag True if an exception documented in docstring was raised in code_before but disappeared in code_after."""
    if not doc_exceptions:
        return False

    disappeared_raises = raises_before - raises_after
    return any(exc in doc_exceptions for exc in disappeared_raises)


def check_default_contract_violation(
    defaults_before: Dict[str, str],
    defaults_after: Dict[str, str],
    docstring: str,
) -> bool:
    """Flag True if a parameter default changed and docstring contains the specific old default."""
    if not docstring or not defaults_before:
        return False

    common_params = set(defaults_before.keys()) & set(defaults_after.keys())
    for param in common_params:
        old_val = defaults_before[param]
        new_val = defaults_after[param]
        if old_val != new_val:
            # Clean quotes if it was a string literal
            clean_old = old_val.strip("\"'")
            if not clean_old:
                continue

            # Check if docstring documents the old default explicitly:
            # e.g., "defaults to True", "default is None", "default 'django.conf'", or "param ... default ... old_val"
            pattern_default = re.compile(
                r"defaults?\s*(?:is|to|:|=)?\s*[`'\"]?" + re.escape(clean_old) + r"[`'\"]?",
                re.IGNORECASE,
            )
            if pattern_default.search(docstring):
                return True

            # Also check if param is mentioned near the old default
            param_near_old = re.compile(
                r"\b" + re.escape(param) + r"\b[^.\n]{0,80}\b(?:default|is)[^.\n]{0,30}\b[`'\"]?" + re.escape(clean_old) + r"[`'\"]?",
                re.IGNORECASE,
            )
            if param_near_old.search(docstring):
                return True

    return False


def check_candidate_contracts(candidate: dict) -> dict:
    """Execute the four deterministic contract checks on a single candidate."""
    code_before = candidate.get("code_before", "") or ""
    code_after = candidate.get("code_after", "") or ""
    func_name = candidate.get("function_name", "")

    # Normalize docstring from either after or before
    doc_before = candidate.get("docstring_before", "") or ""
    doc_after = candidate.get("docstring_after", "") or ""
    docstring = doc_after.strip() or doc_before.strip()

    # Parse ASTs
    func_before = parse_ast_function(code_before, func_name)
    func_after = parse_ast_function(code_after, func_name)

    # 1. Parameter contract check
    params_before = extract_parameter_names(func_before)
    params_after = extract_parameter_names(func_after)
    doc_params = extract_documented_parameters(docstring)
    param_violation = check_parameter_contract_violation(params_before, params_after, doc_params)

    # 2. Return contract check
    ret_before = extract_return_annotation(func_before)
    ret_after = extract_return_annotation(func_after)
    return_violation = check_return_contract_violation(ret_before, ret_after, docstring)

    # 3. Raises contract check
    raises_before = extract_raised_exceptions(func_before)
    raises_after = extract_raised_exceptions(func_after)
    doc_exceptions = extract_documented_exceptions(docstring)
    raises_violation = check_raises_contract_violation(raises_before, raises_after, doc_exceptions)

    # 4. Default contract check
    defaults_before = extract_parameter_defaults(func_before)
    defaults_after = extract_parameter_defaults(func_after)
    default_violation = check_default_contract_violation(defaults_before, defaults_after, docstring)

    count = int(param_violation) + int(return_violation) + int(raises_violation) + int(default_violation)

    candidate["parameter_contract_violation"] = param_violation
    candidate["return_contract_violation"] = return_violation
    candidate["raises_contract_violation"] = raises_violation
    candidate["default_contract_violation"] = default_violation
    candidate["contract_violation_count"] = count

    return candidate


def main() -> int:
    parser = argparse.ArgumentParser(description="Deterministic rule-based contract checking for mined drift candidates.")
    parser.add_argument("--repo", required=True, help="Repository folder name (e.g. click, fastapi, django)")
    args = parser.parse_args()

    # Fallback order: _scored.jsonl -> _filtered.jsonl
    scored_path = CANDIDATE_ROOT / f"{args.repo}_scored.jsonl"
    filtered_path = CANDIDATE_ROOT / f"{args.repo}_filtered.jsonl"

    if scored_path.is_file():
        input_path = scored_path
    elif filtered_path.is_file():
        input_path = filtered_path
    else:
        parser.error(f"Neither scored nor filtered candidate file found for '{args.repo}' in {CANDIDATE_ROOT}")

    output_path = CANDIDATE_ROOT / f"{args.repo}_contract_checked.jsonl"

    print(f"Reading candidates from {input_path.name}...")
    candidates = load_candidates(input_path)
    if not candidates:
        print(f"No candidates found in {input_path}")
        return 0

    checked: list[dict] = []
    for idx, candidate in enumerate(candidates, start=1):
        try:
            checked.append(check_candidate_contracts(candidate))
        except Exception as exc:
            fn = candidate.get("function_name", "unknown")
            print(f"Warning: error contract-checking candidate #{idx} ({fn}): {exc}", file=sys.stderr)
            checked.append(candidate)

    # Sort descending by contract_violation_count, breaking ties by review_priority_score
    checked.sort(
        key=lambda c: (
            c.get("contract_violation_count", 0),
            c.get("review_priority_score", 0.0),
        ),
        reverse=True,
    )

    # Write output JSONL
    with output_path.open("w", encoding="utf-8") as out_file:
        for item in checked:
            out_file.write(json.dumps(item, ensure_ascii=False) + "\n")

    # Metrics computation
    total = len(checked)
    ge_1 = sum(1 for c in checked if c.get("contract_violation_count", 0) >= 1)
    ge_2 = sum(1 for c in checked if c.get("contract_violation_count", 0) >= 2)
    param_cnt = sum(1 for c in checked if c.get("parameter_contract_violation"))
    return_cnt = sum(1 for c in checked if c.get("return_contract_violation"))
    raises_cnt = sum(1 for c in checked if c.get("raises_contract_violation"))
    default_cnt = sum(1 for c in checked if c.get("default_contract_violation"))

    print("=" * 60)
    print(f"Contract Check Summary: {args.repo.upper()}")
    print("=" * 60)
    print(f"Total candidates analyzed:           {total}")
    print(f"Candidates with violations (>= 1):    {ge_1} ({ge_1 / total * 100:.1f}%)")
    print(f"Candidates with violations (>= 2):    {ge_2} ({ge_2 / total * 100:.1f}%)")
    print("-" * 60)
    print(f"Individual violation check breakdown:")
    print(f"  - Parameter contract violations:   {param_cnt} ({param_cnt / total * 100:.1f}%)")
    print(f"  - Return contract violations:      {return_cnt} ({return_cnt / total * 100:.1f}%)")
    print(f"  - Raises contract violations:      {raises_cnt} ({raises_cnt / total * 100:.1f}%)")
    print(f"  - Default contract violations:     {default_cnt} ({default_cnt / total * 100:.1f}%)")
    print("=" * 60)
    print(f"Output saved to: {output_path.name}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
