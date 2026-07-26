import ast
import json
import random
import re


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


def split_sentences(text):
    parts = re.split(r'(?<=[.])\s+', text.strip())
    return [p for p in parts if p.strip()]


def mutate_doc_sentence_delete(docstring, severity="moderate"):
    sentences = split_sentences(docstring)
    if len(sentences) < 2:
        return None

    if severity == "mild":
        idx = len(sentences) - 1
    elif severity == "severe":
        idx = 0
    else:
        idx = len(sentences) // 2

    removed = sentences.pop(idx)
    mutated_doc = " ".join(sentences)

    return {
        "mutated_docstring": mutated_doc,
        "mutation_type": "doc_sentence_delete",
        "severity": severity,
        "removed_sentence": removed,
    }


def mutate_doc_negation(docstring, severity="moderate"):
    severity_priority = {
        "mild": [r"\bdefault\b", r"\bvalid\b", r"\balways\b", r"\benabled\b", r"\boptional\b", r"\bsupported\b", r"\bwith\b"],
        "moderate": [r"\bcan\b", r"\bshould\b", r"\bmust\b", r"\bwill\b", r"\ballow\b", r"\ballows\b", r"\btrue\b", r"\bincludes\b"],
        "severe": [r"\breturns\b", r"\braises\b", r"\bautomatically\b", r"\bignores\b"],
    }

    candidates = list(severity_priority.get(severity, []))
    random.shuffle(candidates)

    # First search prioritized severity candidates
    for pattern in candidates:
        if re.search(pattern, docstring, flags=re.IGNORECASE):
            replacement = NEGATION_SWAPS[pattern]
            mutated_doc = re.sub(pattern, replacement, docstring, count=1, flags=re.IGNORECASE)
            matched_word = re.search(pattern, docstring, flags=re.IGNORECASE).group(0)
            return {
                "mutated_docstring": mutated_doc,
                "mutation_type": "doc_negation",
                "severity": severity,
                "original_word": matched_word,
                "replaced_with": replacement,
            }

    # Fallback search across all negation patterns if prioritized candidates fail
    all_patterns = list(NEGATION_SWAPS.keys())
    random.shuffle(all_patterns)
    for pattern in all_patterns:
        if re.search(pattern, docstring, flags=re.IGNORECASE):
            replacement = NEGATION_SWAPS[pattern]
            mutated_doc = re.sub(pattern, replacement, docstring, count=1, flags=re.IGNORECASE)
            matched_word = re.search(pattern, docstring, flags=re.IGNORECASE).group(0)
            return {
                "mutated_docstring": mutated_doc,
                "mutation_type": "doc_negation",
                "severity": severity,
                "original_word": matched_word,
                "replaced_with": replacement,
            }

    return None


def mutate_doc(docstring, severity="moderate"):
    result = mutate_doc_negation(docstring, severity)
    if result is not None:
        return result
    return mutate_doc_sentence_delete(docstring, severity)


class ParamRenamer(ast.NodeTransformer):
    """Renames a parameter everywhere it's used inside a function body (and signature)."""

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


def get_function_params(func_node):
    return [a.arg for a in func_node.args.args if a.arg not in ("self", "cls")]


def pick_replacement_name(old_name, severity):
    mild_map = {
        "filename": "file_name", "path": "file_path", "value": "val",
        "timeout": "time_limit", "data": "payload", "encoding": "enc",
        "directory": "folder", "resource": "asset", "endpoint": "route",
        "options": "settings", "callback": "handler", "response": "reply",
        "request": "req", "config": "settings_dict", "key": "identifier",
    }
    severe_map = {
        "filename": "user_id", "path": "index", "value": "count",
        "timeout": "retries", "data": "response", "encoding": "mode",
        "directory": "username", "resource": "session_token", "endpoint": "password",
        "options": "credentials", "callback": "error_code", "response": "request",
        "request": "cache_key", "config": "auth_token", "key": "file_size",
    }
    mild_fallback_pool = ["item", "target", "source", "entry", "record", "field"]
    severe_fallback_pool = ["session", "token", "index", "flag", "cache", "handle"]

    if severity == "mild":
        return mild_map.get(old_name, random.choice(mild_fallback_pool))
    elif severity == "moderate":
        return "param_" + str(random.randint(1, 99))
    elif severity == "severe":
        return severe_map.get(old_name, random.choice(severe_fallback_pool))


def mutate_param_rename(code_str, function_name, severity="moderate"):
    try:
        tree = ast.parse(code_str)
    except SyntaxError:
        return None

    func_node = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name:
            func_node = node
            break
    if func_node is None:
        return None

    params = get_function_params(func_node)
    if not params:
        return None

    old_name = random.choice(params)
    new_name = pick_replacement_name(old_name, severity)

    renamer = ParamRenamer(old_name, new_name)
    mutated_tree = renamer.visit(tree)
    ast.fix_missing_locations(mutated_tree)

    try:
        mutated_code = ast.unparse(mutated_tree)
    except Exception:
        return None

    return {
        "mutated_code": mutated_code,
        "mutation_type": "param_rename",
        "severity": severity,
        "old_param": old_name,
        "new_param": new_name,
    }


def get_type_hint_placeholder(annotation_node):
    if annotation_node is None:
        return None

    if isinstance(annotation_node, ast.Name):
        type_map = {
            "bool": ast.Constant(value=False),
            "str": ast.Constant(value=""),
            "int": ast.Constant(value=0),
            "float": ast.Constant(value=0.0),
            "list": ast.List(elts=[], ctx=ast.Load()),
            "dict": ast.Dict(keys=[], values=[]),
        }
        return type_map.get(annotation_node.id)

    if isinstance(annotation_node, ast.Subscript) and isinstance(annotation_node.value, ast.Name):
        base = annotation_node.value.id
        if base in ("List", "list"):
            return ast.List(elts=[], ctx=ast.Load())
        if base in ("Dict", "dict"):
            return ast.Dict(keys=[], values=[])

    return None


class ReturnValueMutator(ast.NodeTransformer):
    def __init__(self, severity, return_annotation=None, param_names=None):
        self.severity = severity
        self.mutated = False
        self.mutation_detail = None
        self.return_annotation = return_annotation
        self.param_names = param_names or []

    def visit_Return(self, node):
        if self.mutated or node.value is None:
            return node

        if self.severity == "mild":
            if isinstance(node.value, ast.Constant) and isinstance(node.value.value, bool):
                original = node.value.value
                node.value = ast.Constant(value=not original)
                self.mutated = True
                self.mutation_detail = f"boolean flipped: {original} -> {not original}"
            elif isinstance(node.value, ast.Compare):
                op = node.value.ops[0]
                flip_map = {
                    ast.Lt: ast.GtE, ast.Gt: ast.LtE,
                    ast.Eq: ast.NotEq, ast.NotEq: ast.Eq,
                    ast.LtE: ast.Gt, ast.GtE: ast.Lt,
                }
                for old_op, new_op in flip_map.items():
                    if isinstance(op, old_op):
                        node.value.ops[0] = new_op()
                        self.mutated = True
                        self.mutation_detail = f"comparison flipped: {old_op.__name__} -> {new_op.__name__}"
                        break
            if not self.mutated and isinstance(node.value, ast.Constant) and \
               isinstance(node.value.value, (int, float)) and not isinstance(node.value.value, bool):
                original = node.value.value
                node.value = ast.Constant(value=original + 1)
                self.mutated = True
                self.mutation_detail = f"numeric off-by-one: {original} -> {node.value.value}"

        elif self.severity == "moderate":
            strategies = []

            if isinstance(node.value, ast.Constant) and \
               (isinstance(node.value.value, (int, float)) and not isinstance(node.value.value, bool)
                or isinstance(node.value.value, str)):
                strategies.append("constant_swap")

            if get_type_hint_placeholder(self.return_annotation) is not None:
                strategies.append("type_hint_swap")

            if self.param_names:
                strategies.append("wrong_param")

            strategies.append("not_wrap")

            chosen = random.choice(strategies)

            if chosen == "constant_swap":
                if isinstance(node.value.value, (int, float)) and not isinstance(node.value.value, bool):
                    original = node.value.value
                    new_value = -original if original != 0 else original + 10
                    node.value = ast.Constant(value=new_value)
                    self.mutation_detail = f"numeric constant changed: {original} -> {new_value}"
                else:
                    original = node.value.value
                    node.value = ast.Constant(value=original + "_modified")
                    self.mutation_detail = f"string constant changed: '{original}' -> '{node.value.value}'"

            elif chosen == "type_hint_swap":
                placeholder = get_type_hint_placeholder(self.return_annotation)
                original_desc = ast.unparse(node.value)
                node.value = placeholder
                self.mutation_detail = f"substituted wrong value matching declared return type (was: {original_desc})"

            elif chosen == "wrong_param":
                original_desc = ast.unparse(node.value)
                wrong_param = random.choice(self.param_names)
                node.value = ast.Name(id=wrong_param, ctx=ast.Load())
                self.mutation_detail = f"returned unrelated parameter '{wrong_param}' instead (was: {original_desc})"

            else:  # not_wrap
                node.value = ast.UnaryOp(op=ast.Not(), operand=node.value)
                self.mutation_detail = "wrapped return value in logical NOT"

            self.mutated = True

        elif self.severity == "severe":
            original_desc = ast.unparse(node.value)
            node.value = ast.Constant(value=None)
            self.mutated = True
            self.mutation_detail = f"return value replaced with None (was: {original_desc})"

        return node


def mutate_return_value(code_str, function_name, severity="moderate"):
    try:
        tree = ast.parse(code_str)
    except SyntaxError:
        return None

    func_node = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name:
            func_node = node
            break
    if func_node is None:
        return None

    return_annotation = func_node.returns
    param_names = [a.arg for a in func_node.args.args if a.arg not in ("self", "cls")]

    mutator = ReturnValueMutator(severity, return_annotation=return_annotation, param_names=param_names)
    mutated_func = mutator.visit(func_node)

    if not mutator.mutated:
        return None

    ast.fix_missing_locations(tree)
    try:
        mutated_code = ast.unparse(tree)
    except Exception:
        return None

    return {
        "mutated_code": mutated_code,
        "mutation_type": "return_value_change",
        "severity": severity,
        "detail": mutator.mutation_detail,
    }


def build_dataset(input_file, output_file, mutation_fraction=0.5, seed=42):
    random.seed(seed)

    with open(input_file, "r", encoding="utf-8") as f:
        pairs = [json.loads(line) for line in f]

    random.shuffle(pairs)
    split_point = int(len(pairs) * mutation_fraction)
    to_mutate = pairs[:split_point]
    to_keep_aligned = pairs[split_point:]

    final_records = []
    skipped = 0

    # Aligned (clean) examples -> label 0
    # NOTE: "file" and "lineno" now carried through from the upstream pair.
    # Using .get() with a None default so this script doesn't hard-crash on
    # older filtered_pairs.jsonl files generated before lineno was added.
    for pair in to_keep_aligned:
        final_records.append({
            "repo": pair["repo"],
            "file": pair.get("file"),
            "function_name": pair["function_name"],
            "code": pair["code"],
            "docstring": pair["docstring"],
            "lineno": pair.get("lineno"),
            "label": 0,
            "mutation_type": None,
            "severity": None,
        })

    mutation_counts = {"param_rename": 0, "return_value_change": 0, "doc": 0}

    def try_param_rename(pair, severity):
        r = mutate_param_rename(pair["code"], pair["function_name"], severity=severity)
        if r is None:
            return None
        return ("code", r)

    def try_return_value(pair, severity):
        r = mutate_return_value(pair["code"], pair["function_name"], severity=severity)
        if r is None:
            return None
        return ("code", r)

    def try_doc(pair, severity):
        r = mutate_doc(pair["docstring"], severity=severity)
        if r is None:
            return None
        return ("doc", r)

    for pair in to_mutate:
        severity = random.choice(["mild", "moderate", "severe"])

        options = [try_param_rename, try_return_value, try_doc]
        random.shuffle(options)

        result = None
        applied_to = None
        for option_fn in options:
            outcome = option_fn(pair, severity)
            if outcome is not None:
                applied_to, result = outcome
                break

        if result is None:
            final_records.append({
                "repo": pair["repo"],
                "file": pair.get("file"),
                "function_name": pair["function_name"],
                "code": pair["code"],
                "docstring": pair["docstring"],
                "lineno": pair.get("lineno"),
                "label": 0,
                "mutation_type": None,
                "severity": None,
            })
            skipped += 1
            continue

        if applied_to == "code":
            mutation_counts[result["mutation_type"]] = mutation_counts.get(result["mutation_type"], 0) + 1
            final_records.append({
                "repo": pair["repo"],
                "file": pair.get("file"),
                "function_name": pair["function_name"],
                "code": result["mutated_code"],
                "docstring": pair["docstring"],
                "lineno": pair.get("lineno"),
                "label": 1,
                "mutation_type": result["mutation_type"],
                "severity": result["severity"],
                "old_param": result.get("old_param"),
                "new_param": result.get("new_param"),
                "detail": result.get("detail"),
            })
        else:
            mutation_counts["doc"] += 1
            final_records.append({
                "repo": pair["repo"],
                "file": pair.get("file"),
                "function_name": pair["function_name"],
                "code": pair["code"],
                "docstring": result["mutated_docstring"],
                "lineno": pair.get("lineno"),
                "label": 1,
                "mutation_type": result["mutation_type"],
                "severity": result["severity"],
            })

    random.shuffle(final_records)

    with open(output_file, "w", encoding="utf-8") as f:
        for record in final_records:
            f.write(json.dumps(record) + "\n")

    label_0 = sum(1 for r in final_records if r["label"] == 0)
    label_1 = sum(1 for r in final_records if r["label"] == 1)
    missing_file = sum(1 for r in final_records if not r.get("file"))
    missing_lineno = sum(1 for r in final_records if r.get("lineno") is None)

    print(f"Total records: {len(final_records)}")
    print(f"  label 0 (aligned): {label_0}")
    print(f"  label 1 (drifted): {label_1}")
    print(f"    - param_rename: {mutation_counts.get('param_rename', 0)}")
    print(f"    - return_value_change: {mutation_counts.get('return_value_change', 0)}")
    print(f"    - doc mutations: {mutation_counts.get('doc', 0)}")
    print(f"  fell back to aligned (neither mutation applicable): {skipped}")
    print(f"  records missing 'file': {missing_file}")
    print(f"  records missing 'lineno': {missing_lineno}")
    print(f"Saved to {output_file}")


if __name__ == "__main__":
    import argparse
    import os
    parser = argparse.ArgumentParser(description="Inject synthetic semantic mutations to generate training dataset.")
    parser.add_argument("--input", default="data/filtered_pairs.jsonl", help="Input filtered JSONL path")
    parser.add_argument("--output", default="data/mutated_dataset.jsonl", help="Output mutated JSONL path")
    parser.add_argument("--fraction", type=float, default=0.5, help="Fraction of mutated pairs")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    build_dataset(
        input_file=args.input,
        output_file=args.output,
        mutation_fraction=args.fraction,
    )