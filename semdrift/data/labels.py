"""
semdrift.data.labels — Authoritative training label extraction and dataset invariants.

This module establishes ONE canonical contract for extracting binary training labels
(0 = aligned, 1 = drifted) across V2 and legacy V1 datasets. It eliminates silent
fallbacks to 0, prevents contradictory label annotations, and validates dataset
invariants before model initialization or training.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

LABEL_STR_MAP: Dict[int, str] = {0: "aligned", 1: "drifted"}


def _normalize_binary_val(val: Any, field_name: str, record_ctx: Optional[dict] = None) -> int:
    """Normalize a value to binary integer 0 or 1, or raise ValueError."""
    if isinstance(val, bool):
        return 1 if val else 0
    if isinstance(val, (int, float)):
        int_val = int(val)
        if int_val in (0, 1) and val == int_val:
            return int_val
        raise ValueError(
            f"Invalid numeric {field_name} '{val}': expected 0 or 1. "
            f"Record context: {record_ctx}"
        )
    if isinstance(val, str):
        s = val.strip().lower()
        if s in ("1", "drifted", "drift", "inconsistent"):
            return 1
        if s in ("0", "aligned", "clean", "non_drift", "consistent"):
            return 0
        raise ValueError(
            f"Invalid string {field_name} '{val}': expected binary ('0', '1') or ('aligned', 'drifted', 'consistent', 'inconsistent'). "
            f"Record context: {record_ctx}"
        )
    raise ValueError(
        f"Unsupported {field_name} type '{type(val).__name__}' with value '{val}'. "
        f"Record context: {record_ctx}"
    )


def extract_training_label(record: dict) -> int:
    """Extract canonical training label as binary integer (0=aligned, 1=drifted).

    Contract:
      1. If 'pseudo_label' is present (and not None):
         - Must be strictly binary (0 or 1).
         - If 'label' or categorical keys are ALSO present, verifies consistency.
           Contradictory annotations (e.g. pseudo_label=1 vs label=0) raise ValueError.
         - Returns authoritative pseudo_label (0 or 1).
         - Does NOT require 'label' to be present.
      2. If 'pseudo_label' is absent (or None):
         - Check 'label': parse numeric (0/1) or string ('aligned'/'drifted').
         - Check categorical keys ('verified_label', 'drift_label', 'filtered_label') as fallback.
      3. If no recognized label field is present (or all are None):
         - Raise ValueError with informative context. Never silently return 0.
    """
    has_pseudo = "pseudo_label" in record and record["pseudo_label"] is not None
    pseudo_val = None
    if has_pseudo:
        pseudo_val = _normalize_binary_val(record["pseudo_label"], "pseudo_label", record)

    # Check for 'label'
    has_label = "label" in record and record["label"] is not None
    label_val = None
    if has_label:
        label_val = _normalize_binary_val(record["label"], "label", record)

    # Check for categorical labels
    cat_val = None
    cat_key = None
    for k in ("verified_label", "drift_label", "filtered_label"):
        if k in record and record[k] is not None:
            c_val = _normalize_binary_val(record[k], k, record)
            if cat_val is None:
                cat_val = c_val
                cat_key = k

    # Verification of consistency when pseudo_label is present
    if has_pseudo:
        if has_label and pseudo_val != label_val:
            raise ValueError(
                f"Conflicting label annotations in record: "
                f"pseudo_label={record['pseudo_label']} ({pseudo_val}) conflicts with "
                f"label={record['label']} ({label_val}). Record: {record}"
            )
        if cat_val is not None and pseudo_val != cat_val:
            raise ValueError(
                f"Conflicting label annotations in record: "
                f"pseudo_label={record['pseudo_label']} ({pseudo_val}) conflicts with "
                f"{cat_key}={record[cat_key]} ({cat_val}). Record: {record}"
            )
        return pseudo_val

    # Verification of consistency when label is present without pseudo_label
    if has_label:
        if cat_val is not None and label_val != cat_val:
            raise ValueError(
                f"Conflicting label annotations in record: "
                f"label={record['label']} ({label_val}) conflicts with "
                f"{cat_key}={record[cat_key]} ({cat_val}). Record: {record}"
            )
        return label_val

    # Fallback to categorical keys if neither pseudo_label nor label is present
    if cat_val is not None:
        return cat_val

    raise ValueError(
        f"Missing authoritative label in record. Expected 'pseudo_label', 'label', "
        f"or categorical fields ('verified_label', 'drift_label', 'filtered_label'). "
        f"Available keys: {list(record.keys())}"
    )


def extract_label_tuple(record: dict) -> Tuple[int, str]:
    """Extract canonical training label as (label_int, label_str)."""
    lbl_int = extract_training_label(record)
    return lbl_int, LABEL_STR_MAP[lbl_int]


def validate_dataset_split(
    records_or_path: Union[str, Path, List[Dict[str, Any]]],
    split_name: str = "split",
    require_both_classes: bool = True,
) -> Dict[str, Any]:
    """Validate schema and class-distribution invariants for a dataset split.

    Invariants checked:
      1. Non-empty split (total > 0).
      2. Required fields present on every record:
         - code content ('code', 'code_after', or 'code_before') must be non-empty string.
         - docstring content ('docstring', 'docstring_after', or 'docstring_before') must be non-empty string.
         - authoritative label extractable via extract_training_label without conflict or invalid value.
      3. Class-presence invariant (if require_both_classes is True):
         - class_0_count > 0 and class_1_count > 0.
         - Neither all-0 nor all-1 splits are permitted.

    Returns:
      Dict with summary statistics: split, total, class_0_count, class_1_count,
      class_0_ratio, class_1_ratio.
    """
    records: List[Dict[str, Any]] = []
    source_desc = str(records_or_path)

    if isinstance(records_or_path, (str, Path)):
        p = Path(records_or_path)
        if not p.is_file():
            raise FileNotFoundError(f"Dataset split file not found: {p}")
        source_desc = str(p)
        with p.open("r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, start=1):
                s_line = line.strip()
                if not s_line:
                    continue
                try:
                    records.append(json.loads(s_line))
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Malformed JSON on line {line_idx} of {source_desc}: {exc}"
                    ) from exc
    elif isinstance(records_or_path, list):
        records = records_or_path
    else:
        raise TypeError(
            f"Expected filepath (str/Path) or list of dicts, got {type(records_or_path).__name__}"
        )

    total = len(records)
    if total == 0:
        raise ValueError(f"Dataset split '{split_name}' ({source_desc}) is empty (0 samples).")

    class_0_count = 0
    class_1_count = 0

    for idx, rec in enumerate(records, start=1):
        if not isinstance(rec, dict):
            raise ValueError(
                f"Record #{idx} in '{split_name}' is not a dict: {type(rec).__name__}"
            )

        # 1. Required code content
        raw_code = rec.get("code") or rec.get("code_after") or rec.get("code_before") or ""
        if not isinstance(raw_code, str) or not raw_code.strip():
            raise ValueError(
                f"Record #{idx} in '{split_name}' missing required code content. "
                f"Available keys: {list(rec.keys())}"
            )

        # 2. Required docstring content
        raw_doc = rec.get("docstring") or rec.get("docstring_after") or rec.get("docstring_before") or rec.get("raw_docstring") or ""
        if not isinstance(raw_doc, str) or not raw_doc.strip():
            raise ValueError(
                f"Record #{idx} in '{split_name}' missing required docstring content. "
                f"Available keys: {list(rec.keys())}"
            )

        # 3. Label correctness
        try:
            lbl_int = extract_training_label(rec)
        except Exception as exc:
            raise ValueError(
                f"Record #{idx} in '{split_name}' failed label extraction: {exc}"
            ) from exc

        if lbl_int == 0:
            class_0_count += 1
        elif lbl_int == 1:
            class_1_count += 1
        else:
            raise ValueError(
                f"Record #{idx} in '{split_name}' yielded unexpected integer label: {lbl_int}"
            )

    class_0_ratio = class_0_count / total if total > 0 else 0.0
    class_1_ratio = class_1_count / total if total > 0 else 0.0

    # 4. Class-presence invariant
    if require_both_classes:
        if class_0_count == 0 or class_1_count == 0:
            raise ValueError(
                f"Dataset split '{split_name}' ({source_desc}) violated class-presence invariant: "
                f"total={total}, class_0={class_0_count}, class_1={class_1_count}. "
                f"Both classes (0=aligned and 1=drifted) must be present."
            )

    return {
        "split": split_name,
        "total": total,
        "class_0_count": class_0_count,
        "class_1_count": class_1_count,
        "class_0_ratio": round(class_0_ratio, 4),
        "class_1_ratio": round(class_1_ratio, 4),
    }


def validate_dataset_pipeline(
    train_path: Union[str, Path],
    val_path: Union[str, Path],
    test_path: Optional[Union[str, Path]] = None,
    print_summary: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """Run preflight schema and invariant validation across train, val, and test splits."""
    results: Dict[str, Dict[str, Any]] = {}

    results["TRAIN"] = validate_dataset_split(train_path, split_name="TRAIN", require_both_classes=True)
    results["VAL"] = validate_dataset_split(val_path, split_name="VAL", require_both_classes=True)

    if test_path is not None:
        results["TEST"] = validate_dataset_split(test_path, split_name="TEST", require_both_classes=True)

    if print_summary:
        print("=" * 78, flush=True)
        print("PREFLIGHT DATASET INVARIANT VALIDATION SUMMARY", flush=True)
        print("=" * 78, flush=True)
        print(f"{'Split':<8} {'Total':<10} {'Class 0 (aligned)':<22} {'Class 1 (drifted)':<22} {'Ratio (0:1)':<12}", flush=True)
        print("-" * 78, flush=True)
        for split_name, stats in results.items():
            c0 = stats["class_0_count"]
            c1 = stats["class_1_count"]
            tot = stats["total"]
            p0 = stats["class_0_ratio"] * 100
            p1 = stats["class_1_ratio"] * 100
            ratio_str = f"{c0/c1:.2f} : 1" if c1 > 0 else "N/A"
            print(
                f"{split_name:<8} {tot:<10} {f'{c0} ({p0:.2f}%)':<22} {f'{c1} ({p1:.2f}%)':<22} {ratio_str:<12}",
                flush=True,
            )
        print("-" * 78, flush=True)
        print("  -> All schema and class-presence invariants VERIFIED.", flush=True)
        print("=" * 78, flush=True)

    return results
