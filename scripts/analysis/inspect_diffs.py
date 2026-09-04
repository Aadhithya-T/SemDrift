import json
import difflib

with open("data/experiments/v2/filtered_pairs.jsonl", encoding="utf-8") as f:
    orig_pairs = {}
    for line in f:
        r = json.loads(line)
        norm_file = r["file"].replace("\\", "/")
        orig_pairs[(r["repo"], norm_file, r["function_name"])] = r["docstring"]

with open("data/experiments/v2/joint_focal_controlled/predictions_joint_encoder.jsonl", encoding="utf-8") as f:
    preds = [json.loads(line) for line in f]

neg_fns = [p for p in preds if p.get("drift_type") == "doc_negation" and p["predicted_label"] == "aligned"]

print(f"Total doc_negation FNs: {len(neg_fns)}")
inspected = 0
for p in neg_fns:
    key = (p["repo"], p["file"].replace("\\", "/"), p["function_name"])
    if key in orig_pairs:
        orig = orig_pairs[key]
        mut = p["docstring"]
        diff = [l for l in difflib.ndiff(orig.splitlines(), mut.splitlines()) if l.startswith("- ") or l.startswith("+ ")]
        inspected += 1
        print(f"\n================ Case #{inspected}: {p['function_name']} ({p['repo']}) P(drift)={p['confidence']:.4f} ================")
        print("Diff:", diff[:6])
        print("Mutated docstring (raw):")
        print(mut[:300])
        if inspected >= 10:
            break
