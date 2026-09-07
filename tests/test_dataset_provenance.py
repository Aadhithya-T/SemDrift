#!/usr/bin/env python3
"""
test_dataset_provenance.py

Unit and integration tests for:
1. Two-layer cryptographic integrity gate (semdrift/data/integrity.py).
2. Tamper rejection (verifying that modified files raise DatasetIntegrityError).
3. End-to-end dataset provenance and mathematical disjointness.
"""

import json
import tempfile
import unittest
from pathlib import Path

from semdrift.data.integrity import (
    DatasetIntegrityError,
    compute_sha256,
    verify_dataset_integrity,
)
from scripts.data_pipeline.verify_dataset_provenance import (
    load_jsonl,
    verify_split_provenance,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "experiments" / "2026-09-07_clean_v2" / "dataset"


class TestDatasetProvenanceAndIntegrity(unittest.TestCase):
    
    def test_canonical_dataset_integrity_passes(self):
        """Live files in canonical dataset dir must match SHA256SUMS and manifest.yaml."""
        verified = verify_dataset_integrity(DATASET_DIR)
        self.assertIn("train.jsonl", verified)
        self.assertIn("val.jsonl", verified)
        self.assertIn("verified_test.jsonl", verified)

    def test_tampered_file_raises_integrity_error(self):
        """Altering a single byte in a dataset file must cause verify_dataset_integrity to fail."""
        with tempfile.TemporaryDirectory() as tmpdir:
            td = Path(tmpdir)
            
            # Create a mock file and correct SHA256SUMS
            fpath = td / "train.jsonl"
            fpath.write_text('{"label": 0}\n', encoding="utf-8")
            correct_hash = compute_sha256(fpath)
            
            sums_file = td / "SHA256SUMS"
            sums_file.write_text(f"{correct_hash}  train.jsonl\n", encoding="utf-8")
            
            # Verify clean check passes
            res = verify_dataset_integrity(td, required_files=["train.jsonl"])
            self.assertEqual(res["train.jsonl"], correct_hash)
            
            # Tamper with file
            fpath.write_text('{"label": 1}\n', encoding="utf-8")
            
            # Verify integrity gate refuses execution
            with self.assertRaises(DatasetIntegrityError):
                verify_dataset_integrity(td, required_files=["train.jsonl"])

    def test_end_to_end_dataset_provenance_audit(self):
        """Audit 100% of rows in canonical train, val, and verified_test splits."""
        train_rows = load_jsonl(DATASET_DIR / "train.jsonl")
        val_rows = load_jsonl(DATASET_DIR / "val.jsonl")
        test_rows = load_jsonl(DATASET_DIR / "verified_test.jsonl")
        
        train_lineages = verify_split_provenance("TRAIN", train_rows)
        val_lineages = verify_split_provenance("VAL", val_rows)
        test_lineages = verify_split_provenance("TEST", test_rows)
        
        # Invariants: Disjointness
        self.assertEqual(len(train_lineages.intersection(val_lineages)), 0)
        self.assertEqual(len(train_lineages.intersection(test_lineages)), 0)
        self.assertEqual(len(val_lineages.intersection(test_lineages)), 0)
        
        # Test set counts: >= 100 samples, >= 50 positives, >= 50 negatives
        self.assertGreaterEqual(len(test_rows), 100)
        test_drift = sum(1 for r in test_rows if r["label"] == 1)
        test_clean = sum(1 for r in test_rows if r["label"] == 0)
        self.assertGreaterEqual(test_drift, 50)
        self.assertGreaterEqual(test_clean, 50)


if __name__ == "__main__":
    unittest.main()
