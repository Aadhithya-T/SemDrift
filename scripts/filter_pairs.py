import json
import argparse
import os


def is_useful(pair):
    docstring = pair["docstring"].strip()
    function_name = pair["function_name"]

    # Skip too-short docstrings
    if len(docstring.split()) < 10:
        return False

    # Skip dunder/private methods
    if function_name.startswith("_") and not docstring:
        return False

    # Skip trivial one-line functions (rough heuristic: code is very short)
    code_lines = pair["code"].strip().split("\n")
    if len(code_lines) <= 2:
        return False

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter extracted function-docstring pairs.")
    parser.add_argument("--input", default="data/extracted_pairs.jsonl", help="Input extracted pairs JSONL file")
    parser.add_argument("--output", default="data/filtered_pairs.jsonl", help="Output filtered pairs JSONL file")
    args = parser.parse_args()

    input_file = args.input
    output_file = args.output

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)

    kept = 0
    dropped = 0

    with open(input_file, "r", encoding="utf-8") as fin, \
         open(output_file, "w", encoding="utf-8") as fout:
        for line in fin:
            pair = json.loads(line)
            if is_useful(pair):
                fout.write(json.dumps(pair) + "\n")
                kept += 1
            else:
                dropped += 1

    print(f"Kept: {kept}")
    print(f"Dropped: {dropped}")
    print(f"Saved to {output_file}")