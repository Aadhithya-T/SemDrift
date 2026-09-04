import json
import re

def extract_docstring_summary(docstring: str) -> str:
    if not docstring:
        return ""
    lines = docstring.strip().split("\n")
    summary_lines = []
    for line in lines:
        stripped = line.strip()
        if (
            not stripped
            or stripped.startswith(">>>")
            or stripped.startswith("...")
            or stripped.startswith("Parameters")
            or stripped.startswith("Returns")
            or stripped.startswith("Examples")
            or stripped.startswith("See Also")
            or stripped.startswith("Notes")
            or stripped.startswith("Raises")
            or stripped.startswith("Warnings")
            or stripped.startswith("References")
        ):
            break
        summary_lines.append(stripped)

    cleaned = " ".join(summary_lines).strip()
    if len(cleaned) >= 10:
        return cleaned
    return lines[0].strip()

# Negation words and their original counterparts
NEG_PAIRS = [
    ("does not return", "returns"),
    ("will not", "will"),
    ("cannot", "can"),
    ("should not", "should"),
    ("must not", "must"),
    ("never", "always"),
    ("manually", "automatically"),
    ("non-default", "default"),
    ("suppresses", "raises"),
    ("invalid", "valid"),
    ("disabled", "enabled"),
    ("required", "optional"),
    ("unsupported", "supported"),
    ("disallow", "allow"),
    ("disallows", "allows"),
    ("false", "true"),
    ("excludes", "includes"),
    ("enforces", "ignores"),
    ("without", "with"),
]

with open("data/experiments/v2/test.jsonl", encoding="utf-8") as f:
    test_rows = [json.loads(line) for line in f]

neg_rows = [r for r in test_rows if r.get("drift_type") == "doc_negation"]

with open("data/experiments/v2/joint_focal_controlled/predictions_joint_encoder.jsonl", encoding="utf-8") as f:
    preds = {p["function_id"]: p for p in [json.loads(line) for line in f]}

print(f"Total doc_negation in test: {len(neg_rows)}")

in_raw = 0
in_cleaned = 0
cut_off = 0

details = []

for r in neg_rows:
    fid = r["function_id"]
    pred = preds[fid]
    raw_doc = r["docstring"]
    cleaned_doc = extract_docstring_summary(raw_doc)
    
    # Find which negation word was inserted
    found_mut = None
    for mut_word, orig_word in NEG_PAIRS:
        pattern = r"\b" + re.escape(mut_word) + r"\b"
        if re.search(pattern, raw_doc, re.IGNORECASE):
            found_mut = mut_word
            break
            
    if found_mut:
        in_raw += 1
        survived = bool(re.search(r"\b" + re.escape(found_mut) + r"\b", cleaned_doc, re.IGNORECASE))
        if survived:
            in_cleaned += 1
        else:
            cut_off += 1
            details.append({
                "fid": fid,
                "mut_word": found_mut,
                "pred": pred["predicted_label"],
                "conf": pred["confidence"],
                "cleaned": cleaned_doc,
                "raw_first_300": raw_doc[:300]
            })

print(f"Negation pattern identified in raw docstring: {in_raw}/{len(neg_rows)}")
print(f"Negation SURVIVED in cleaned docstring (seen by model): {in_cleaned}/{in_raw} ({in_cleaned/in_raw*100:.1f}%)")
print(f"Negation CUT OFF by clean_docstrings (INVISIBLE to model): {cut_off}/{in_raw} ({cut_off/in_raw*100:.1f}%)")

print("\n--- Accuracy on SURVIVED (model actually saw the negation) vs CUT OFF (model saw unmutated text) ---")
survived_tps = sum(1 for r in neg_rows if preds[r["function_id"]]["predicted_label"] == "drifted" and any(re.search(r"\b" + re.escape(m) + r"\b", extract_docstring_summary(r["docstring"]), re.IGNORECASE) for m, _ in NEG_PAIRS))
survived_total = in_cleaned
cutoff_tps = sum(1 for r in neg_rows if preds[r["function_id"]]["predicted_label"] == "drifted" and not any(re.search(r"\b" + re.escape(m) + r"\b", extract_docstring_summary(r["docstring"]), re.IGNORECASE) for m, _ in NEG_PAIRS))
cutoff_total = cut_off

print(f"When mutation SURVIVED in summary: Recalled {survived_tps}/{survived_total} ({survived_tps/survived_total*100:.1f}%)")
print(f"When mutation was CUT OFF in body: Recalled {cutoff_tps}/{cutoff_total} ({cutoff_tps/max(1, cutoff_total)*100:.1f}%)")
