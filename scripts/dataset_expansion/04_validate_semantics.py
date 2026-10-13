"""
SemDrift Dataset Expansion - Step 4: Semantic Validation & Category Balancing
Processes raw candidates, applies expanded AST & semantic validation,
and generates validated pools of DRIFT and CONSISTENT samples.
"""

import os
import sys
import json
import random
from pathlib import Path
from typing import List, Dict, Any

# Ensure import of verification functions
sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify_candidates import build_consistent_sample, build_drift_sample, RANDOM_SEED

random.seed(RANDOM_SEED)

EXPANDED_CATEGORIES = [
    "parameter_missing",
    "parameter_obsolete",
    "parameter_rename",
    "parameter_default_mismatch",
    "parameter_constraint_mismatch",
    "parameter_order_mismatch",
    "return_type_mismatch",
    "return_none_behavior",
    "return_value_semantics",
    "exception_undocumented",
    "exception_contract_mismatch",
    "mutation_behavior",
    "async_execution_mismatch"
]

def main():
    base_dir = Path(__file__).resolve().parents[2]
    candidates_file = base_dir / "data" / "real_dataset_expansion_v1" / "candidates" / "raw_candidates.jsonl"
    drift_output = base_dir / "data" / "real_dataset_expansion_v1" / "evidence" / "validated_drift.jsonl"
    consistent_output = base_dir / "data" / "real_dataset_expansion_v1" / "evidence" / "validated_consistent.jsonl"

    drift_output.parent.mkdir(parents=True, exist_ok=True)

    if not candidates_file.exists():
        print(f"[-] Candidates file {candidates_file} not found. Run 02_mine_candidates.py first.", flush=True)
        return

    print("[*] Loading raw candidates for semantic validation...", flush=True)
    with open(candidates_file, "r", encoding="utf-8") as f:
        raw_candidates = [json.loads(line) for line in f if line.strip()]

    print(f"[*] Loaded {len(raw_candidates):,} raw candidates.", flush=True)

    # 1. Generate verified consistent samples
    print("[*] Verifying CONSISTENT samples across expansion candidates...", flush=True)
    consistent_pool = []
    for cand in raw_candidates:
        s = build_consistent_sample(cand)
        if s:
            consistent_pool.append(s)

    print(f"[+] Total verified consistent samples: {len(consistent_pool):,}", flush=True)

    # 2. Generate verified drift samples across expanded taxonomy
    print("[*] Constructing and verifying DRIFT samples across expanded taxonomy...", flush=True)
    drift_pool = []
    
    # Shuffle candidates
    shuffled_cands = list(raw_candidates)
    random.shuffle(shuffled_cands)

    target_drift_samples = 3000  # Aiming for 3,000+ new verified drift samples

    cat_index = 0
    for cand in shuffled_cands:
        if len(drift_pool) >= target_drift_samples * 2:
            break
        
        # Cycle through categories to ensure rich representation of new taxonomy
        cat = EXPANDED_CATEGORIES[cat_index % len(EXPANDED_CATEGORIES)]
        cat_index += 1

        d_sample = build_drift_sample(cand, cat)
        if d_sample:
            drift_pool.append(d_sample)

    print(f"[+] Total verified drift samples generated: {len(drift_pool):,}", flush=True)

    # Write validated intermediate pools
    with open(drift_output, "w", encoding="utf-8") as f:
        for s in drift_pool:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    with open(consistent_output, "w", encoding="utf-8") as f:
        for s in consistent_pool:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"[+] Validated drift saved to: {drift_output}", flush=True)
    print(f"[+] Validated consistent saved to: {consistent_output}", flush=True)

if __name__ == "__main__":
    main()
