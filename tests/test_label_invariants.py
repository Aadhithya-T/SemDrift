"""
tests/test_label_invariants.py — Comprehensive regression and invariant tests for V2 label contracts.

Covers:
  - Canonical label extraction (pseudo_label=0/1, legacy label string/int, categorical keys).
  - Rejection of invalid, missing, and non-binary labels (never silently 0).
  - Conflict detection between pseudo_label and label.
  - Joint-Encoder and Dual-Encoder consistency.
  - Dataset split schema and class-presence invariants.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from semdrift.data.labels import (
    extract_training_label,
    extract_label_tuple,
    validate_dataset_split,
    validate_dataset_pipeline,
)
from semdrift.models.joint_encoder import SemDriftDataset as JointEncoderDataset
from semdrift.models.dual_encoder import DualEncoderDataset


class TestCanonicalLabelExtraction(unittest.TestCase):
    """Test extract_training_label and extract_label_tuple contract."""

    def test_pseudo_label_one_authoritative(self):
        """pseudo_label=1 alone must be valid and yield 1 / 'drifted'."""
        rec = {"pseudo_label": 1}
        self.assertEqual(extract_training_label(rec), 1)
        self.assertEqual(extract_label_tuple(rec), (1, "drifted"))

    def test_pseudo_label_zero_authoritative(self):
        """pseudo_label=0 alone must be valid and yield 0 / 'aligned'."""
        rec = {"pseudo_label": 0}
        self.assertEqual(extract_training_label(rec), 0)
        self.assertEqual(extract_label_tuple(rec), (0, "aligned"))

    def test_pseudo_label_string_binary(self):
        """String '0' and '1' in pseudo_label are coerced properly."""
        self.assertEqual(extract_training_label({"pseudo_label": "1"}), 1)
        self.assertEqual(extract_training_label({"pseudo_label": "0"}), 0)

    def test_pseudo_label_does_not_require_label(self):
        """V2 authoritative pseudo_label must NOT require label field to exist."""
        rec = {"code": "def foo(): pass", "docstring": "Foo function", "pseudo_label": 1}
        self.assertNotIn("label", rec)
        self.assertEqual(extract_training_label(rec), 1)

    def test_pseudo_label_consistent_with_label(self):
        """Consistent pseudo_label and label annotations succeed."""
        rec1 = {"pseudo_label": 1, "label": 1}
        rec2 = {"pseudo_label": 1, "label": "drifted"}
        rec3 = {"pseudo_label": 0, "label": 0}
        rec4 = {"pseudo_label": 0, "label": "aligned"}

        self.assertEqual(extract_training_label(rec1), 1)
        self.assertEqual(extract_training_label(rec2), 1)
        self.assertEqual(extract_training_label(rec3), 0)
        self.assertEqual(extract_training_label(rec4), 0)

    def test_conflicting_pseudo_label_and_label_raises(self):
        """Contradictory pseudo_label and label annotations must fail loudly."""
        # pseudo_label=1 vs label=0 / 'aligned'
        with self.assertRaises(ValueError) as ctx1:
            extract_training_label({"pseudo_label": 1, "label": 0})
        self.assertIn("Conflicting label annotations", str(ctx1.exception))

        with self.assertRaises(ValueError) as ctx2:
            extract_training_label({"pseudo_label": 1, "label": "aligned"})
        self.assertIn("Conflicting label annotations", str(ctx2.exception))

        # pseudo_label=0 vs label=1 / 'drifted'
        with self.assertRaises(ValueError) as ctx3:
            extract_training_label({"pseudo_label": 0, "label": 1})
        self.assertIn("Conflicting label annotations", str(ctx3.exception))

        with self.assertRaises(ValueError) as ctx4:
            extract_training_label({"pseudo_label": 0, "label": "drifted"})
        self.assertIn("Conflicting label annotations", str(ctx4.exception))

    def test_invalid_pseudo_label_raises(self):
        """Non-binary or malformed pseudo_label must fail loudly (never become 0)."""
        invalid_cases = [
            {"pseudo_label": 2},
            {"pseudo_label": -1},
            {"pseudo_label": "banana"},
            {"pseudo_label": 2.5},
            {"pseudo_label": "unknown"},
        ]
        for rec in invalid_cases:
            with self.subTest(rec=rec):
                with self.assertRaises(ValueError):
                    extract_training_label(rec)

    def test_legacy_label_parsing(self):
        """Legacy label strings and ints parse correctly when pseudo_label is absent."""
        self.assertEqual(extract_training_label({"label": "drifted"}), 1)
        self.assertEqual(extract_training_label({"label": "drift"}), 1)
        self.assertEqual(extract_training_label({"label": "aligned"}), 0)
        self.assertEqual(extract_training_label({"label": "clean"}), 0)
        self.assertEqual(extract_training_label({"label": "non_drift"}), 0)
        self.assertEqual(extract_training_label({"label": 1}), 1)
        self.assertEqual(extract_training_label({"label": 0}), 0)

    def test_invalid_legacy_label_raises(self):
        """Unknown legacy label values fail loudly."""
        invalid_labels = ["unknown", "maybe", 42, -5, "foo", [1]]
        for lbl in invalid_labels:
            with self.subTest(lbl=lbl):
                with self.assertRaises(ValueError):
                    extract_training_label({"label": lbl})

    def test_categorical_fallback_parsing(self):
        """Categorical keys (verified_label, drift_label, filtered_label) work if present."""
        self.assertEqual(extract_training_label({"verified_label": "drifted"}), 1)
        self.assertEqual(extract_training_label({"drift_label": "aligned"}), 0)
        self.assertEqual(extract_training_label({"filtered_label": "drift"}), 1)

    def test_missing_label_fails_loudly(self):
        """Records with no label fields must raise ValueError, never silently return 0."""
        empty_rec = {"code": "def foo(): pass", "docstring": "summary"}
        with self.assertRaises(ValueError) as ctx:
            extract_training_label(empty_rec)
        self.assertIn("Missing authoritative label in record", str(ctx.exception))

        none_rec = {"pseudo_label": None, "label": None}
        with self.assertRaises(ValueError):
            extract_training_label(none_rec)


class TestJointAndDualDatasetConsistency(unittest.TestCase):
    """Ensure JointEncoderDataset and DualEncoderDataset interpret records identically."""

    def test_joint_encoder_handles_pseudo_label_regression(self):
        """Verify Joint Encoder correctly extracts pseudo_label 1 and 0."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = Path(tmp_dir) / "test_data.jsonl"
            records = [
                {"code": "def f(): return 1", "docstring": "Returns one.", "pseudo_label": 1},
                {"code": "def g(): return 0", "docstring": "Returns zero.", "pseudo_label": 0},
            ]
            with file_path.open("w", encoding="utf-8") as f:
                for r in records:
                    f.write(json.dumps(r) + "\n")

            dataset = JointEncoderDataset(str(file_path), clean_docs=False)
            self.assertEqual(len(dataset), 2)
            # Item 0
            doc, code, lbl, meta = dataset[0]
            self.assertEqual(lbl, 1)
            self.assertEqual(meta["label_str"], "drifted")
            # Item 1
            doc, code, lbl, meta = dataset[1]
            self.assertEqual(lbl, 0)
            self.assertEqual(meta["label_str"], "aligned")

    def test_joint_and_dual_produce_identical_targets(self):
        """Dual and Joint datasets must produce identical labels for the same records."""
        synthetic_records = [
            {"code": "def f1(): pass", "docstring": "d1", "pseudo_label": 0},
            {"code": "def f2(): pass", "docstring": "d2", "pseudo_label": 1},
            {"code": "def f3(): pass", "docstring": "d3", "label": "drifted"},
            {"code": "def f4(): pass", "docstring": "d4", "label": "aligned"},
            {"code": "def f5(): pass", "docstring": "d5", "label": 1},
            {"code": "def f6(): pass", "docstring": "d6", "label": 0},
            {"code": "def f7(): pass", "docstring": "d7", "pseudo_label": 1, "label": 1},
            {"code": "def f8(): pass", "docstring": "d8", "pseudo_label": 0, "label": "aligned"},
            {"code": "def f9(): pass", "docstring": "d9", "filtered_label": "drift"},
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = Path(tmp_dir) / "mixed_data.jsonl"
            with file_path.open("w", encoding="utf-8") as f:
                for r in synthetic_records:
                    f.write(json.dumps(r) + "\n")

            joint_ds = JointEncoderDataset(str(file_path), clean_docs=False)
            dual_ds = DualEncoderDataset(str(file_path), clean_docs=False)

            self.assertEqual(len(joint_ds), len(dual_ds))

            for idx in range(len(joint_ds)):
                _, _, joint_lbl, joint_meta = joint_ds[idx]
                _, _, dual_lbl, dual_meta = dual_ds[idx]

                self.assertEqual(
                    joint_lbl, dual_lbl,
                    f"Mismatch at index {idx}: joint={joint_lbl}, dual={dual_lbl}"
                )
                self.assertEqual(
                    joint_meta["label_str"], dual_meta["label_str"],
                    f"Meta label_str mismatch at index {idx}: joint={joint_meta['label_str']}, dual={dual_meta['label_str']}"
                )


class TestDatasetSplitInvariants(unittest.TestCase):
    """Test split-level schema and class-presence invariants."""

    def test_valid_split_succeeds(self):
        """Mixed class dataset passes validation and returns correct stats."""
        records = [
            {"code": "def a(): pass", "docstring": "doc a", "pseudo_label": 0},
            {"code": "def b(): pass", "docstring": "doc b", "pseudo_label": 1},
            {"code": "def c(): pass", "docstring": "doc c", "pseudo_label": 0},
        ]
        stats = validate_dataset_split(records, split_name="test_split", require_both_classes=True)
        self.assertEqual(stats["total"], 3)
        self.assertEqual(stats["class_0_count"], 2)
        self.assertEqual(stats["class_1_count"], 1)
        self.assertEqual(stats["class_0_ratio"], round(2 / 3, 4))
        self.assertEqual(stats["class_1_ratio"], round(1 / 3, 4))

    def test_single_class_all_zero_fails(self):
        """Split with only class 0 fails class-presence invariant."""
        records = [
            {"code": "def a(): pass", "docstring": "doc a", "pseudo_label": 0},
            {"code": "def b(): pass", "docstring": "doc b", "pseudo_label": 0},
        ]
        with self.assertRaises(ValueError) as ctx:
            validate_dataset_split(records, split_name="all_zero", require_both_classes=True)
        self.assertIn("violated class-presence invariant", str(ctx.exception))

    def test_single_class_all_one_fails(self):
        """Split with only class 1 fails class-presence invariant."""
        records = [
            {"code": "def a(): pass", "docstring": "doc a", "pseudo_label": 1},
            {"code": "def b(): pass", "docstring": "doc b", "pseudo_label": 1},
        ]
        with self.assertRaises(ValueError) as ctx:
            validate_dataset_split(records, split_name="all_one", require_both_classes=True)
        self.assertIn("violated class-presence invariant", str(ctx.exception))

    def test_empty_dataset_fails(self):
        """Empty dataset fails validation."""
        with self.assertRaises(ValueError) as ctx:
            validate_dataset_split([], split_name="empty", require_both_classes=True)
        self.assertIn("is empty (0 samples)", str(ctx.exception))

    def test_invalid_label_fails(self):
        """Split containing an invalid label (e.g. 2) fails validation."""
        records = [
            {"code": "def a(): pass", "docstring": "doc a", "pseudo_label": 0},
            {"code": "def b(): pass", "docstring": "doc b", "pseudo_label": 2},
        ]
        with self.assertRaises(ValueError) as ctx:
            validate_dataset_split(records, split_name="invalid_label", require_both_classes=True)
        self.assertIn("failed label extraction", str(ctx.exception))

    def test_missing_required_fields_fails(self):
        """Records with empty code or docstrings fail validation."""
        missing_code = [{"docstring": "doc only", "pseudo_label": 0}]
        with self.assertRaises(ValueError) as ctx:
            validate_dataset_split(missing_code, split_name="no_code")
        self.assertIn("missing required code content", str(ctx.exception))

        missing_doc = [{"code": "def f(): pass", "pseudo_label": 1}]
        with self.assertRaises(ValueError) as ctx2:
            validate_dataset_split(missing_doc, split_name="no_doc")
        self.assertIn("missing required docstring content", str(ctx2.exception))


if __name__ == "__main__":
    unittest.main()
