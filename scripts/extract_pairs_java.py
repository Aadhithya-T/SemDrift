import re
import html
import tree_sitter_java as tsjava
from tree_sitter import Language, Parser
import json
import os
import sys

JAVA_LANGUAGE = Language(tsjava.language())
parser = Parser(JAVA_LANGUAGE)


def get_node_text(node, source_bytes):
    return source_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def clean_javadoc(raw_comment_text):
    """Strip /** */ markers, leading *, javadoc block tags, inline {@...} tags, and HTML."""
    text = raw_comment_text.strip()
    if text.startswith("/**"):
        text = text[3:]
    elif text.startswith("/*"):
        text = text[2:]
    if text.endswith("*/"):
        text = text[:-2]

    lines = []
    for line in text.split("\n"):
        line = line.strip()
        if line.startswith("*"):
            line = line[1:].strip()
        if line.startswith("@"):
            break  # hit first tag -> everything from here on is tags/continuations, stop collecting
        lines.append(line)

    cleaned = "\n".join(lines).strip()

    # drop {@snippet ...} code-example blocks entirely
    cleaned = re.sub(r"\{@snippet\b.*?\}", " ", cleaned, flags=re.DOTALL)

    # inline {@code X} / {@literal X} -> X, padded
    cleaned = re.sub(r"\{@(?:code|literal)\s+([^}]*)\}", r" \1 ", cleaned)

    # {@link ...} / {@linkplain ...} -> visible label text, padded
    def _link_repl(m):
        inner = m.group(1).strip()
        if ")" in inner:
            idx = inner.rfind(")")
            label = inner[idx + 1:].strip()
        else:
            parts = inner.split(None, 1)
            label = parts[1] if len(parts) == 2 else ""

        if label:
            return f" {label} "
        return " "

    cleaned = re.sub(r"\{@link(?:plain)?\s+([^}]*)\}", _link_repl, cleaned)

    # strip embedded HTML tags
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)

    cleaned = html.unescape(cleaned)

    # final safety sweep: any stray { or } left over
    cleaned = re.sub(r"[{}]", " ", cleaned)

    # collapse whitespace/newlines to single spaces
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    # remove stray space before punctuation
    cleaned = re.sub(r"\s+([.,;:!?])", r"\1", cleaned)

    return cleaned


def extract_methods_from_class_body(class_body_node, source_bytes, class_name, file_path, results):
    pending_comment = None

    for child in class_body_node.children:
        if child.type == "block_comment":
            raw_text = get_node_text(child, source_bytes)
            if raw_text.strip().startswith("/**"):
                pending_comment = raw_text
            else:
                pending_comment = None  # non-doc comment, discard
            continue

        if child.type == "method_declaration":
            name_node = child.child_by_field_name("name")
            body_node = child.child_by_field_name("body")

            if name_node is None or body_node is None:
                pending_comment = None
                continue

            method_name = get_node_text(name_node, source_bytes)
            full_code = get_node_text(child, source_bytes)
            docstring = clean_javadoc(pending_comment) if pending_comment else ""

            results.append({
                "function_id": f"{file_path}::{class_name}::{method_name}::{child.start_point[0]}",
                "repo": None,  # filled in by caller
                "file": file_path,
                "function_name": method_name,
                "code": full_code,
                "docstring": docstring,
                "lineno": child.start_point[0] + 1,
                "language": "java",
            })

            pending_comment = None
            continue

        # recurse into nested classes/interfaces if present
        if child.type in ("class_declaration", "interface_declaration"):
            nested_body = child.child_by_field_name("body")
            nested_name_node = child.child_by_field_name("name")
            nested_name = get_node_text(nested_name_node, source_bytes) if nested_name_node else class_name
            if nested_body:
                extract_methods_from_class_body(nested_body, source_bytes, nested_name, file_path, results)
            pending_comment = None


def extract_from_file(file_path, repo_name):
    with open(file_path, "rb") as f:
        source_bytes = f.read()

    tree = parser.parse(source_bytes)
    root = tree.root_node
    results = []

    def walk_top_level(node):
        for child in node.children:
            if child.type in ("class_declaration", "interface_declaration"):
                name_node = child.child_by_field_name("name")
                class_name = get_node_text(name_node, source_bytes) if name_node else "Unknown"
                body_node = child.child_by_field_name("body")
                if body_node:
                    extract_methods_from_class_body(body_node, source_bytes, class_name, file_path, results)
            else:
                walk_top_level(child)

    walk_top_level(root)

    for r in results:
        r["repo"] = repo_name

    return results


def extract_from_repo(repo_dir, repo_name, output_path):
    all_results = []
    for dirpath, _, filenames in os.walk(repo_dir):
        for fname in filenames:
            if fname.endswith(".java"):
                full_path = os.path.join(dirpath, fname)
                rel_path = os.path.relpath(full_path, repo_dir)
                try:
                    records = extract_from_file(full_path, repo_name)
                    all_results.extend(records)
                except Exception as e:
                    print(f"  [SKIP] {rel_path}: {e}", file=sys.stderr)

    with open(output_path, "w", encoding="utf-8") as f:
        for r in all_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Extracted {len(all_results)} method records from {repo_name} -> {output_path}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python extract_pairs_java.py <repo_dir> <repo_name> <output_jsonl>")
        sys.exit(1)

    repo_dir, repo_name, output_path = sys.argv[1], sys.argv[2], sys.argv[3]
    extract_from_repo(repo_dir, repo_name, output_path)