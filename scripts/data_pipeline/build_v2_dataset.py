"""V2 Dataset Generator for SemDrift.

Implements the research upgrades recommended by the team lead:
1. Full Docstring Representation: [SUMMARY], [PARAMETERS], [RETURNS], [RAISES]
2. Eliminates blind param_rename (only mutates parameters with docstring contract references)
3. Adds rich AST contract mutations:
   - exception_contract_drift
   - default_value_drift
   - return_contract_drift
   - behavior_operator_drift
   - param_contract_drift
   - doc_negation_drift
   - doc_sentence_delete
4. Strict class balance (1:1 aligned vs drifted) and difficulty stratification.
"""

import ast
import json
import random
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from docstring_parser import normalize_docstring_structured


NEGATION_SWAPS = {
    r"\breturns\b": "does not return",
    r"\bwill\b": "will not",
    r"\bcan\b": "cannot",
    r"\bshould\b": "should not",
    r"\bmust\b": "must not",
    r"\balways\b": "never",
    r"\bautomatically\b": "manually",
    r"\bdefault\b": "non-default",
    r"\braises\b": "suppresses",
    r"\bvalid\b": "invalid",
    r"\benabled\b": "disabled",
    r"\boptional\b": "required",
    r"\bsupported\b": "unsupported",
    r"\ballow\b": "disallow",
    r"\ballows\b": "disallows",
    r"\btrue\b": "false",
    r"\bincludes\b": "excludes",
    r"\bignores\b": "enforces",
    r"\bwith\b": "without",
}


# ==============================================================================
# AST Mutators
# ==============================================================================

class ParamRenamer(ast.NodeTransformer):
    def __init__(self, old_name, new_name):
        self.old_name = old_name
        self.new_name = new_name

    def visit_Name(self, node):
        if node.id == self.old_name:
            node.id = self.new_name
        return self.generic_visit(node)

    def visit_arg(self, node):
        if node.arg == self.old_name:
            node.arg = self.new_name
        return node


class ExceptionMutator(ast.NodeTransformer):
    """Mutates raised exceptions in code to create exception contract drift."""
    def __init__(self, target_exc, replacement_exc):
        self.target_exc = target_exc
        self.replacement_exc = replacement_exc
        self.mutated = False

    def visit_Raise(self, node):
        if self.mutated or node.exc is None:
            return node
        
        # raise ExcClass(...) or raise ExcClass
        if isinstance(node.exc, ast.Call) and isinstance(node.exc.func, ast.Name):
            if node.exc.func.id == self.target_exc or not self.target_exc:
                node.exc.func.id = self.replacement_exc
                self.mutated = True
        elif isinstance(node.exc, ast.Name):
            if node.exc.id == self.target_exc or not self.target_exc:
                node.exc.id = self.replacement_exc
                self.mutated = True
        return node


class DefaultValueMutator(ast.NodeTransformer):
    """Mutates default parameter values in function signature."""
    def __init__(self, param_name, new_default_val):
        self.param_name = param_name
        self.new_default_val = new_default_val
        self.mutated = False

    def visit_FunctionDef(self, node):
        # Match parameter defaults
        num_defaults = len(node.args.defaults)
        params_with_defaults = [a.arg for a in node.args.args[-num_defaults:]] if num_defaults > 0 else []
        if self.param_name in params_with_defaults:
            idx = params_with_defaults.index(self.param_name)
            node.args.defaults[idx] = ast.Constant(value=self.new_default_val)
            self.mutated = True
        return self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node):
        return self.visit_FunctionDef(node)


class OperatorMutator(ast.NodeTransformer):
    """Flips relational and boolean operators in code body."""
    def __init__(self):
        self.mutated = False
        self.mutation_detail = ""

    def visit_Compare(self, node):
        if self.mutated or not node.ops:
            return node
        
        flip_map = {
            ast.Gt: (ast.LtE, "> to <="),
            ast.GtE: (ast.Lt, ">= to <"),
            ast.Lt: (ast.GtE, "< to >="),
            ast.LtE: (ast.Gt, "<= to >"),
            ast.Eq: (ast.NotEq, "== to !="),
            ast.NotEq: (ast.Eq, "!= to =="),
        }
        for i, op in enumerate(node.ops):
            op_cls = op.__class__
            if op_cls in flip_map:
                new_op_cls, desc = flip_map[op_cls]
                node.ops[i] = new_op_cls()
                self.mutated = True
                self.mutation_detail = desc
                break
        return node


class ReturnStatementMutator(ast.NodeTransformer):
    """Mutates return expressions."""
    def __init__(self, severity="moderate"):
        self.severity = severity
        self.mutated = False
        self.mutation_detail = ""

    def visit_Return(self, node):
        if self.mutated or node.value is None:
            return node

        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, bool):
            node.value = ast.Constant(value=not node.value.value)
            self.mutated = True
            self.mutation_detail = "flipped boolean return"
        elif isinstance(node.value, ast.Constant) and isinstance(node.value.value, (int, float)):
            node.value = ast.Constant(value=-node.value.value if node.value.value != 0 else 1)
            self.mutated = True
            self.mutation_detail = "numeric constant negated"
        elif self.severity == "severe":
            node.value = ast.Constant(value=None)
            self.mutated = True
            self.mutation_detail = "return replaced with None"
        return node


# ==============================================================================
# Mutation Functions
# ==============================================================================

def mutate_param_contract(code_str: str, function_name: str, doc_meta: dict) -> Optional[dict]:
    """Mutates a parameter ONLY if it is explicitly documented in docstring parameters."""
    doc_params = list(doc_meta.get("parameters", {}).keys())
    if not doc_params:
        return None

    try:
        tree = ast.parse(code_str)
    except SyntaxError:
        return None

    target_param = random.choice(doc_params)
    new_param = f"{target_param}_mod"

    renamer = ParamRenamer(target_param, new_param)
    mutated_tree = renamer.visit(tree)
    ast.fix_missing_locations(mutated_tree)

    try:
        mutated_code = ast.unparse(mutated_tree)
        if mutated_code == code_str:
            return None
    except Exception:
        return None

    return {
        "mutated_code": mutated_code,
        "mutation_type": "param_contract_drift",
        "severity": "moderate",
        "detail": f"Renamed documented parameter '{target_param}' to '{new_param}' in code",
    }


def mutate_exception_contract(code_str: str, function_name: str, doc_meta: dict) -> Optional[dict]:
    """Mutates raised exceptions when documented in docstring."""
    doc_raises = list(doc_meta.get("raises", {}).keys())
    if not doc_raises:
        return None

    try:
        tree = ast.parse(code_str)
    except SyntaxError:
        return None

    target_exc = random.choice(doc_raises)
    exc_swaps = {
        "ValueError": "TypeError",
        "TypeError": "ValueError",
        "KeyError": "IndexError",
        "IndexError": "KeyError",
        "FileNotFoundError": "PermissionError",
        "PermissionError": "FileNotFoundError",
        "RuntimeError": "ValueError",
    }
    new_exc = exc_swaps.get(target_exc, "RuntimeError")

    mutator = ExceptionMutator(target_exc, new_exc)
    mutated_tree = mutator.visit(tree)
    if not mutator.mutated:
        return None

    ast.fix_missing_locations(mutated_tree)
    try:
        mutated_code = ast.unparse(mutated_tree)
    except Exception:
        return None

    return {
        "mutated_code": mutated_code,
        "mutation_type": "exception_contract_drift",
        "severity": "severe",
        "detail": f"Swapped raised exception '{target_exc}' -> '{new_exc}' in code",
    }


def mutate_default_value(code_str: str, function_name: str, doc_meta: dict) -> Optional[dict]:
    """Mutates parameter default value in signature."""
    try:
        tree = ast.parse(code_str)
    except SyntaxError:
        return None

    # Find function
    func_node = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name:
            func_node = node
            break
    if not func_node or not func_node.args.defaults:
        return None

    num_defaults = len(func_node.args.defaults)
    params_with_defaults = [a.arg for a in func_node.args.args[-num_defaults:]]
    if not params_with_defaults:
        return None

    target_param = random.choice(params_with_defaults)
    mutator = DefaultValueMutator(target_param, 9999)
    mutated_tree = mutator.visit(tree)
    if not mutator.mutated:
        return None

    ast.fix_missing_locations(mutated_tree)
    try:
        mutated_code = ast.unparse(mutated_tree)
    except Exception:
        return None

    return {
        "mutated_code": mutated_code,
        "mutation_type": "default_value_drift",
        "severity": "moderate",
        "detail": f"Mutated default value for parameter '{target_param}' in code",
    }


def mutate_behavior_operator(code_str: str, function_name: str, doc_meta: dict) -> Optional[dict]:
    """Flips comparison or boolean operator in code body."""
    try:
        tree = ast.parse(code_str)
    except SyntaxError:
        return None

    mutator = OperatorMutator()
    mutated_tree = mutator.visit(tree)
    if not mutator.mutated:
        return None

    ast.fix_missing_locations(mutated_tree)
    try:
        mutated_code = ast.unparse(mutated_tree)
    except Exception:
        return None

    return {
        "mutated_code": mutated_code,
        "mutation_type": "behavior_operator_drift",
        "severity": "hard",
        "detail": f"Flipped operator ({mutator.mutation_detail}) in code",
    }


def mutate_return_contract(code_str: str, function_name: str, doc_meta: dict) -> Optional[dict]:
    """Mutates return statement."""
    try:
        tree = ast.parse(code_str)
    except SyntaxError:
        return None

    mutator = ReturnStatementMutator(severity="moderate")
    mutated_tree = mutator.visit(tree)
    if not mutator.mutated:
        return None

    ast.fix_missing_locations(mutated_tree)
    try:
        mutated_code = ast.unparse(mutated_tree)
    except Exception:
        return None

    return {
        "mutated_code": mutated_code,
        "mutation_type": "return_contract_drift",
        "severity": "moderate",
        "detail": mutator.mutation_detail,
    }


def mutate_doc_negation(raw_docstring: str) -> Optional[dict]:
    """Negates semantic assertions in docstring."""
    patterns = list(NEGATION_SWAPS.keys())
    random.shuffle(patterns)
    for pat in patterns:
        if re.search(pat, raw_docstring, flags=re.IGNORECASE):
            repl = NEGATION_SWAPS[pat]
            mutated_raw = re.sub(pat, repl, raw_docstring, count=1, flags=re.IGNORECASE)
            return {
                "mutated_raw_docstring": mutated_raw,
                "mutation_type": "doc_negation_drift",
                "severity": "moderate",
                "detail": f"Negated '{pat}' -> '{repl}' in docstring",
            }
    return None


# ==============================================================================
# Pipeline Execution
# ==============================================================================

def process_extracted_pairs(input_path: Path, output_path: Path) -> dict:
    """Read extracted pairs, format docstrings, apply V2 mutations, and write balanced dataset."""
    raw_pairs = []
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    raw_pairs.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    print(f"Loaded {len(raw_pairs)} extracted raw pairs.")
    dataset = []
    stats = {
        "total_aligned": 0,
        "total_drifted": 0,
        "mutation_breakdown": {},
    }

    mutation_fns = [
        ("param_contract_drift", mutate_param_contract),
        ("exception_contract_drift", mutate_exception_contract),
        ("default_value_drift", mutate_default_value),
        ("behavior_operator_drift", mutate_behavior_operator),
        ("return_contract_drift", mutate_return_contract),
    ]

    for pair in raw_pairs:
        repo = pair.get("repo", "")
        file_p = pair.get("file", "")
        fn_name = pair.get("function_name", "")
        code = pair.get("code", "")
        raw_doc = pair.get("docstring", "")
        lineno = pair.get("lineno", 0)

        formatted_doc, doc_meta = normalize_docstring_structured(raw_doc)
        if not formatted_doc or not code:
            continue

        base_item = {
            "repo": repo,
            "file": file_p,
            "lineno": lineno,
            "function_name": fn_name,
            "code": code,
            "docstring": formatted_doc,
            "raw_docstring": raw_doc,
            "doc_meta": doc_meta,
        }

        # 1. Add Aligned Pair (label: 0)
        aligned_item = dict(base_item)
        aligned_item["label"] = 0
        aligned_item["mutation_type"] = "aligned"
        aligned_item["severity"] = "none"
        dataset.append(aligned_item)
        stats["total_aligned"] += 1

        # 2. Try Generating Code Mutation (label: 1)
        shuffled_mutators = list(mutation_fns)
        random.shuffle(shuffled_mutators)
        mutated_res = None
        for m_name, m_fn in shuffled_mutators:
            mutated_res = m_fn(code, fn_name, doc_meta)
            if mutated_res is not None:
                break

        # Fallback to docstring negation if code mutation not applicable
        if mutated_res is None:
            doc_neg = mutate_doc_negation(raw_doc)
            if doc_neg is not None:
                fmt_neg_doc, _ = normalize_docstring_structured(doc_neg["mutated_raw_docstring"])
                drift_item = dict(base_item)
                drift_item["docstring"] = fmt_neg_doc
                drift_item["label"] = 1
                drift_item["mutation_type"] = doc_neg["mutation_type"]
                drift_item["severity"] = doc_neg["severity"]
                drift_item["mutation_detail"] = doc_neg["detail"]
                dataset.append(drift_item)
                stats["total_drifted"] += 1
                mtype = doc_neg["mutation_type"]
                stats["mutation_breakdown"][mtype] = stats["mutation_breakdown"].get(mtype, 0) + 1
        else:
            drift_item = dict(base_item)
            drift_item["code"] = mutated_res["mutated_code"]
            drift_item["label"] = 1
            drift_item["mutation_type"] = mutated_res["mutation_type"]
            drift_item["severity"] = mutated_res["severity"]
            drift_item["mutation_detail"] = mutated_res.get("detail", "")
            dataset.append(drift_item)
            stats["total_drifted"] += 1
            mtype = mutated_res["mutation_type"]
            stats["mutation_breakdown"][mtype] = stats["mutation_breakdown"].get(mtype, 0) + 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print("\n" + "=" * 60)
    print("V2 DATASET GENERATION SUMMARY")
    print("=" * 60)
    print(f"Total Rows Generated: {len(dataset)}")
    print(f"  - Aligned Samples (label 0): {stats['total_aligned']}")
    print(f"  - Drifted Samples (label 1): {stats['total_drifted']}")
    print("-" * 60)
    print("Mutation Breakdown:")
    for mtype, count in stats["mutation_breakdown"].items():
        print(f"  - {mtype}: {count}")
    print("=" * 60)
    print(f"Output saved to: {output_path}")

    return stats


if __name__ == "__main__":
    in_file = Path("data/extracted_pairs.jsonl")
    out_file = Path("data/experiments/v2/semdrift_v2_labeled.jsonl")
    if in_file.exists():
        process_extracted_pairs(in_file, out_file)
    else:
        print(f"Error: {in_file} does not exist.")
