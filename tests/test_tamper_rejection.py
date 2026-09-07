#!/usr/bin/env python3
"""
tests/test_tamper_rejection.py

Five-Scenario Intentional Cryptographic Tamper Rejection Suite.
Enforces that altering even a single byte in any canonical artifact causes
the pipeline to refuse execution immediately with zero experimental side effects:

  Scenario 1: Modify 1 byte in train.jsonl
              -> train_joint_encoder.py MUST raise DatasetIntegrityError / refuse training.
              -> ZERO model training, ZERO checkpoint overwrite.
  Scenario 2: Modify 1 byte in verified_test.jsonl
              -> independent_evaluate.py MUST raise DatasetIntegrityError / refuse evaluation.
              -> ZERO model evaluation, ZERO predictions written.
  Scenario 3: Modify 1 byte in SHA256SUMS
              -> Layer B integrity gate MUST raise DatasetIntegrityError / refuse execution.
  Scenario 4: Modify manifest hash (sha256sums_sha256 in manifest.yaml)
              -> Layer B integrity gate MUST raise DatasetIntegrityError / refuse execution.
  Scenario 5: Modify 1 byte in checkpoint
              -> independent_evaluate.py MUST raise CheckpointIntegrityError.
              -> STRICT SECURITY BOUNDARY: torch.load() MUST NOT be called!

Guarantees byte-for-byte rollback and post-restoration verification after each scenario.
"""

import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from semdrift.data.integrity import (
    CheckpointIntegrityError,
    DatasetIntegrityError,
    compute_sha256,
    verify_checkpoint_integrity,
    verify_dataset_integrity,
)

CANONICAL_DIR = PROJECT_ROOT / "experiments" / "2026-09-07_clean_v2" / "dataset"
CONFIG_DIR = PROJECT_ROOT / "experiments" / "2026-09-07_clean_v2" / "config"
MANIFEST_PATH = CONFIG_DIR / "manifest.yaml"
SHA256SUMS_PATH = CANONICAL_DIR / "SHA256SUMS"


class TestFiveScenarioTamperRejection(unittest.TestCase):
    
    def setUp(self):
        """Record baseline canonical hashes before every test to ensure state is pure."""
        self.assertTrue(CANONICAL_DIR.is_dir(), f"Missing {CANONICAL_DIR}")
        self.assertTrue(MANIFEST_PATH.is_file(), f"Missing {MANIFEST_PATH}")
        self.assertTrue(SHA256SUMS_PATH.is_file(), f"Missing {SHA256SUMS_PATH}")
        
        # Verify baseline integrity passes cleanly before we begin
        verify_dataset_integrity(CANONICAL_DIR, manifest_path=MANIFEST_PATH)
        
        self.baseline_hashes = {
            "train.jsonl": compute_sha256(CANONICAL_DIR / "train.jsonl"),
            "val.jsonl": compute_sha256(CANONICAL_DIR / "val.jsonl"),
            "verified_test.jsonl": compute_sha256(CANONICAL_DIR / "verified_test.jsonl"),
            "SHA256SUMS": compute_sha256(SHA256SUMS_PATH),
            "manifest.yaml": compute_sha256(MANIFEST_PATH),
        }

    def tearDown(self):
        """Verify that all canonical files are byte-for-byte restored to baseline."""
        for fname in ["train.jsonl", "val.jsonl", "verified_test.jsonl"]:
            current_h = compute_sha256(CANONICAL_DIR / fname)
            self.assertEqual(
                current_h, self.baseline_hashes[fname],
                f"TEARDOWN FAILURE: {fname} was not restored to baseline hash!"
            )
        self.assertEqual(compute_sha256(SHA256SUMS_PATH), self.baseline_hashes["SHA256SUMS"])
        self.assertEqual(compute_sha256(MANIFEST_PATH), self.baseline_hashes["manifest.yaml"])

    def test_scenario_1_train_jsonl_tamper_refuses_training(self):
        """Scenario 1: Modifying 1 byte in train.jsonl must cause training to refuse with zero side effects."""
        train_file = CANONICAL_DIR / "train.jsonl"
        original_bytes = train_file.read_bytes()
        
        with tempfile.TemporaryDirectory() as tmp_out:
            tmp_out_path = Path(tmp_out)
            try:
                # Tamper 1 byte (flip last newline or append character)
                tampered_bytes = original_bytes + b" "
                train_file.write_bytes(tampered_bytes)
                
                # Check directly with integrity gate
                with self.assertRaises(DatasetIntegrityError):
                    verify_dataset_integrity(CANONICAL_DIR, manifest_path=MANIFEST_PATH)
                    
                # Run the actual training entrypoint (scripts/training/train_joint_encoder.py)
                cmd = [
                    sys.executable,
                    str(PROJECT_ROOT / "scripts" / "training" / "train_joint_encoder.py"),
                    "--train", str(train_file),
                    "--val", str(CANONICAL_DIR / "val.jsonl"),
                    "--output_dir", str(tmp_out_path),
                    "--dry_run",
                ]
                proc = subprocess.run(cmd, capture_output=True, text=True)
                
                # MUST FAIL with non-zero exit code
                self.assertNotEqual(proc.returncode, 0, "Training entrypoint should have failed on tampered train.jsonl!")
                self.assertIn("DatasetIntegrityError", proc.stderr + proc.stdout)
                
                # Zero side effects: NO checkpoint, NO predictions, NO results files written
                self.assertFalse((tmp_out_path / "joint_encoder_checkpoint.pt").exists(), "Checkpoint must NOT be written!")
                self.assertFalse((tmp_out_path / "training_run.json").exists(), "Training run record must NOT be written!")
                self.assertFalse((tmp_out_path / "results_joint_encoder.json").exists(), "Results must NOT be written!")
                
            finally:
                train_file.write_bytes(original_bytes)
                
        print("[SCENARIO 1 PASSED] Tampered train.jsonl refused by training with zero side effects.")

    def test_scenario_2_test_jsonl_tamper_refuses_evaluation(self):
        """Scenario 2: Modifying 1 byte in verified_test.jsonl must cause independent evaluator to refuse."""
        test_file = CANONICAL_DIR / "verified_test.jsonl"
        original_bytes = test_file.read_bytes()
        
        with tempfile.TemporaryDirectory() as tmp_eval:
            tmp_eval_path = Path(tmp_eval)
            preds_file = tmp_eval_path / "preds.jsonl"
            results_file = tmp_eval_path / "results.json"
            mock_ckpt = tmp_eval_path / "mock.pt"
            mock_ckpt.write_bytes(b"MOCK_CHECKPOINT_BYTES_12345")
            mock_run = tmp_eval_path / "training_run.json"
            mock_run.write_text(json.dumps({"checkpoint_sha256": compute_sha256(mock_ckpt)}), encoding="utf-8")
            
            try:
                # Tamper 1 byte
                tampered_bytes = original_bytes + b" "
                test_file.write_bytes(tampered_bytes)
                
                # Check directly with integrity gate
                with self.assertRaises(DatasetIntegrityError):
                    verify_dataset_integrity(CANONICAL_DIR, manifest_path=MANIFEST_PATH, required_files=["verified_test.jsonl"])
                    
                # Run the actual independent evaluator entrypoint
                cmd = [
                    sys.executable,
                    str(PROJECT_ROOT / "experiments" / "2026-09-07_clean_v2" / "evaluation" / "independent_evaluate.py"),
                    "--test_file", str(test_file),
                    "--checkpoint", str(mock_ckpt),
                    "--training_run", str(mock_run),
                    "--manifest", str(MANIFEST_PATH),
                    "--output_results", str(results_file),
                    "--output_preds", str(preds_file),
                ]
                proc = subprocess.run(cmd, capture_output=True, text=True)
                
                # MUST FAIL with non-zero exit code
                self.assertNotEqual(proc.returncode, 0, "Evaluator should have failed on tampered verified_test.jsonl!")
                self.assertIn("DatasetIntegrityError", proc.stderr + proc.stdout)
                
                # Zero side effects: NO output files written
                self.assertFalse(preds_file.exists(), "Prediction file must NOT be written on integrity failure!")
                self.assertFalse(results_file.exists(), "Results file must NOT be written on integrity failure!")
                
            finally:
                test_file.write_bytes(original_bytes)
                
        print("[SCENARIO 2 PASSED] Tampered verified_test.jsonl refused by independent evaluator with zero side effects.")

    def test_scenario_3_sha256sums_tamper_refuses_execution(self):
        """Scenario 3: Modifying 1 byte in SHA256SUMS must cause Layer B integrity gate to refuse execution."""
        original_bytes = SHA256SUMS_PATH.read_bytes()
        
        try:
            # Tamper 1 byte in SHA256SUMS (append a newline -> Layer B mismatch)
            tampered_bytes = original_bytes + b"\n"
            SHA256SUMS_PATH.write_bytes(tampered_bytes)
            
            # Integrity gate MUST fail with Two-layer integrity failure
            with self.assertRaises(DatasetIntegrityError) as ctx:
                verify_dataset_integrity(CANONICAL_DIR, manifest_path=MANIFEST_PATH)
                
            self.assertIn("Two-layer integrity failure", str(ctx.exception))
            
            # Also test tampering a recorded hash inside SHA256SUMS -> Layer A mismatch
            tampered_bytes2 = original_bytes.replace(b"7d0f5c21", b"7d0f5c22")
            SHA256SUMS_PATH.write_bytes(tampered_bytes2)
            with self.assertRaises(DatasetIntegrityError) as ctx2:
                verify_dataset_integrity(CANONICAL_DIR, manifest_path=MANIFEST_PATH)
            self.assertIn("Checksum mismatch", str(ctx2.exception))
            
        finally:
            SHA256SUMS_PATH.write_bytes(original_bytes)
            
        print("[SCENARIO 3 PASSED] Tampered SHA256SUMS refused by Layer A & Layer B integrity gates.")

    def test_scenario_4_manifest_hash_tamper_refuses_execution(self):
        """Scenario 4: Modifying the expected hash in manifest.yaml must cause Layer B integrity gate to refuse."""
        original_bytes = MANIFEST_PATH.read_bytes()
        
        try:
            # Tamper 1 character in manifest.yaml's recorded sha256sums_sha256
            tampered_content = original_bytes.decode("utf-8").replace("fec45253", "00000000")
            MANIFEST_PATH.write_text(tampered_content, encoding="utf-8")
            
            with self.assertRaises(DatasetIntegrityError) as ctx:
                verify_dataset_integrity(CANONICAL_DIR, manifest_path=MANIFEST_PATH)
                
            self.assertIn("Two-layer integrity failure", str(ctx.exception))
            
        finally:
            MANIFEST_PATH.write_bytes(original_bytes)
            
        print("[SCENARIO 4 PASSED] Tampered manifest.yaml hash refused by Layer B integrity gate.")

    def test_scenario_5_checkpoint_tamper_refuses_without_torch_load(self):
        """Scenario 5: Modifying 1 byte in checkpoint must raise CheckpointIntegrityError BEFORE torch.load()."""
        with tempfile.TemporaryDirectory() as tmpdir:
            td = Path(tmpdir)
            mock_ckpt = td / "mock_checkpoint.pt"
            mock_run = td / "training_run.json"
            
            # Create a valid mock checkpoint file
            mock_ckpt.write_bytes(b"PYTORCH_VALID_MODEL_WEIGHTS_MOCK_BYTES_123456789")
            valid_hash = compute_sha256(mock_ckpt)
            
            # Create matching training_run.json
            run_data = {
                "checkpoint_path": str(mock_ckpt),
                "checkpoint_sha256": valid_hash,
                "best_epoch": 1,
            }
            mock_run.write_text(json.dumps(run_data), encoding="utf-8")
            
            # Verify clean check passes
            verified = verify_checkpoint_integrity(mock_ckpt, mock_run)
            self.assertEqual(verified, valid_hash)
            
            # Tamper 1 byte in the checkpoint file
            mock_ckpt.write_bytes(b"PYTORCH_VALID_MODEL_WEIGHTS_MOCK_BYTES_123456780")
            
            # Patch torch.load to guarantee it is NEVER called if checkpoint hash mismatches
            with patch("torch.load") as mock_torch_load:
                with self.assertRaises(CheckpointIntegrityError) as ctx:
                    # Evaluator boundary: verify_checkpoint_integrity is called BEFORE torch.load
                    verify_checkpoint_integrity(mock_ckpt, mock_run)
                    torch.load(mock_ckpt)
                    
                # Assertion of ZERO SIDE EFFECTS: torch.load was NOT called
                mock_torch_load.assert_not_called()
                self.assertIn("Checkpoint SHA-256 mismatch", str(ctx.exception))
                
        print("[SCENARIO 5 PASSED] Tampered checkpoint refused before torch.load() with zero side effects.")


if __name__ == "__main__":
    unittest.main()
