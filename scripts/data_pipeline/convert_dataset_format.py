"""
Converts data/mutated_dataset.jsonl into the Research Lead's target schema:

{
  "function_id": "...",
  "repo": "...",
  "file": "...",
  "function_name": "...",
  "code": "...",
  "docstring": "...",
  "lineno": <int or null>,
  "label": "aligned" | "drifted",
  "drift_type": "<mutation_type>" | null
}

No join needed anymore -- extract_pairs.py now captures lineno, and
build_dataset.py now carries file+lineno through into every output record
(both aligned and drifted). This script just relabels and builds function_id.
"""

import os
import json
from collections import defaultdict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MUTATED_PATH = os.path.join(BASE_DIR, "..", "data", "mutated_dataset.jsonl")
OUTPUT_PATH = os.path.join(BASE_DIR, "..", "data", "labeled", "semdrift_labeled.jsonl")


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def make_function_id(repo, file_path, function_name, counter):
    if file_path:
        base = os.path.splitext(os.path.basename(file_path))[0]
        file_slug = f"{base}_py"
    else:
        file_slug = "unknownfile"
    return f"{repo}_{file_slug}_{function_name}_{counter:03d}"


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Convert mutated dataset to target schema.")
    parser.add_argument("--input", default=MUTATED_PATH, help="Path to input mutated_dataset.jsonl")
    parser.add_argument("--output", default=OUTPUT_PATH, help="Path to output semdrift_labeled.jsonl")
    args = parser.parse_args()

    records = load_jsonl(args.input)
    print(f"Loaded {len(records)} records from {args.input}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    function_id_counters = defaultdict(int)
    missing_file = 0
    missing_lineno = 0

    with open(args.output, "w", encoding="utf-8") as out_f:
        for r in records:
            repo = r["repo"]
            file_path = r.get("file")
            function_name = r["function_name"]

            if not file_path:
                missing_file += 1
            if r.get("lineno") is None:
                missing_lineno += 1

            counter_key = (repo, file_path, function_name)
            function_id_counters[counter_key] += 1
            function_id = make_function_id(
                repo, file_path, function_name, function_id_counters[counter_key]
            )

            label_str = "aligned" if r["label"] == 0 else "drifted"
            drift_type = r.get("mutation_type") if r["label"] == 1 else None

            converted = {
                "function_id": function_id,
                "repo": repo,
                "file": file_path,
                "function_name": function_name,
                "code": r["code"],
                "docstring": r["docstring"],
                "lineno": r.get("lineno"),
                "label": label_str,
                "drift_type": drift_type,
                "severity": r.get("severity"),
            }

            out_f.write(json.dumps(converted) + "\n")

    print(f"\nWrote {len(records)} converted records to {args.output}")
    print(f"Records missing 'file': {missing_file}")
    print(f"Records missing 'lineno': {missing_lineno}")
    if missing_file == 0 and missing_lineno == 0:
        print("\nAll records have both file and lineno populated. Clean output.")
    else:
        print("\nSome records are missing file/lineno -- likely from data extracted")
        print("before this pipeline update. Re-run the full pipeline from")
        print("extract_pairs.py onward to guarantee every record is complete.")


if __name__ == "__main__":
    main()