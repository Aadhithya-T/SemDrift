import json
import unittest
from pathlib import Path
from collections import Counter
from typing import Dict, Any, List

from scripts.data_pipeline.setup_dataset_architecture import (
    get_function_lineage,
    norm_repo,
    norm_file_path,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


class TestDatasetArchitecture(unittest.TestCase):
    """Rigorous invariant checks for the two-generation dataset architecture."""

    def test_v1_synthetic_files_and_counts(self):
        v1_dir = DATA_DIR / "v1_synthetic"
        self.assertTrue(v1_dir.is_dir(), "v1_synthetic directory missing")

        # 1. Benchmark (1,205 samples: 597 aligned, 608 drift)
        bench_file = v1_dir / "benchmark" / "synthetic_dataset.jsonl"
        self.assertTrue(bench_file.is_file(), "benchmark/synthetic_dataset.jsonl missing")
        bench_rows = load_jsonl(bench_file)
        self.assertEqual(len(bench_rows), 1205)

        bench_labels = Counter(r.get("label") for r in bench_rows)
        self.assertEqual(bench_labels["aligned"], 597)
        self.assertEqual(bench_labels["drifted"], 608)

        # 2. Raw original aligned (597 samples)
        raw_file = v1_dir / "raw" / "original_aligned.jsonl"
        self.assertTrue(raw_file.is_file(), "raw/original_aligned.jsonl missing")
        raw_rows = load_jsonl(raw_file)
        self.assertEqual(len(raw_rows), 597)
        for r in raw_rows:
            self.assertEqual(r.get("label"), "aligned")

        # 3. Ablation splits (9,638 train / 1,259 val / 1,205 test)
        train_file = v1_dir / "ablation" / "train.jsonl"
        val_file = v1_dir / "ablation" / "val.jsonl"
        test_file = v1_dir / "ablation" / "test.jsonl"
        self.assertTrue(train_file.is_file())
        self.assertTrue(val_file.is_file())
        self.assertTrue(test_file.is_file())

        self.assertEqual(len(load_jsonl(train_file)), 9638)
        self.assertEqual(len(load_jsonl(val_file)), 1259)
        self.assertEqual(len(load_jsonl(test_file)), 1205)

        # 4. Metadata
        meta_file = v1_dir / "metadata" / "dataset_summary.json"
        self.assertTrue(meta_file.is_file())
        with meta_file.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        self.assertEqual(meta["benchmark_samples"], 1205)
        self.assertEqual(meta["aligned_count"], 597)
        self.assertEqual(meta["drifted_count"], 608)

    def test_v2_real_world_evaluation_isolation(self):
        v2_dir = DATA_DIR / "v2_real_world"
        self.assertTrue(v2_dir.is_dir(), "v2_real_world directory missing")

        # Verified test set (balanced diagnostic test set: >=100 samples, >=50 drift)
        eval_file = v2_dir / "evaluation" / "verified_test.jsonl"
        self.assertTrue(eval_file.is_file(), "evaluation/verified_test.jsonl missing")
        eval_rows = load_jsonl(eval_file)
        self.assertGreaterEqual(len(eval_rows), 100, "Verified test set must be >= 100 samples")

        eval_labels = Counter(r.get("label") for r in eval_rows)
        self.assertGreaterEqual(eval_labels[1], 50, "Verified drift positives must be >= 50")
        self.assertGreaterEqual(eval_labels[0], 50, "Verified clean negatives must be >= 50")

        # Lineages
        test_lineages = {get_function_lineage(r) for r in eval_rows}
        self.assertEqual(len(test_lineages), len(eval_rows), "All verified test samples must have unique function lineages")

    def test_v2_real_world_training_partitions_and_zero_leakage(self):
        v2_dir = DATA_DIR / "v2_real_world"
        train_file = v2_dir / "training" / "train.jsonl"
        val_file = v2_dir / "training" / "val.jsonl"
        eval_file = v2_dir / "evaluation" / "verified_test.jsonl"

        self.assertTrue(train_file.is_file())
        self.assertTrue(val_file.is_file())
        self.assertTrue(eval_file.is_file())

        train_rows = load_jsonl(train_file)
        val_rows = load_jsonl(val_file)
        eval_rows = load_jsonl(eval_file)

        # Dynamic dataset counts and non-emptiness
        self.assertGreater(len(train_rows), 0, "Train split must not be empty")
        self.assertGreater(len(val_rows), 0, "Val split must not be empty")
        self.assertGreaterEqual(len(eval_rows), 100, "Verified test set must be >= 100 instances")

        # Label presence checks
        train_labels = Counter(r.get("pseudo_label", r.get("label")) for r in train_rows)
        val_labels = Counter(r.get("pseudo_label", r.get("label")) for r in val_rows)

        self.assertGreater(train_labels[0], 0, "Train must contain clean samples")
        self.assertGreater(train_labels[1], 0, "Train must contain drift samples")
        self.assertGreater(val_labels[0], 0, "Val must contain clean samples")
        self.assertGreater(val_labels[1], 0, "Val must contain drift samples")

        # Zero-leakage mathematical assertions
        test_lineages = {get_function_lineage(r) for r in eval_rows}
        train_lineages = {get_function_lineage(r) for r in train_rows}
        val_lineages = {get_function_lineage(r) for r in val_rows}

        # 1. Train and Val must have ZERO overlapping lineages
        train_val_overlap = train_lineages.intersection(val_lineages)
        self.assertEqual(
            len(train_val_overlap), 0,
            f"Lineage leak detected between train and val: {train_val_overlap}"
        )

        # 2. Test must have ZERO overlapping lineages with Train
        train_test_overlap = train_lineages.intersection(test_lineages)
        self.assertEqual(
            len(train_test_overlap), 0,
            f"Lineage leak detected between train and test: {train_test_overlap}"
        )

        # 3. Test must have ZERO overlapping lineages with Val
        val_test_overlap = val_lineages.intersection(test_lineages)
        self.assertEqual(
            len(val_test_overlap), 0,
            f"Lineage leak detected between val and test: {val_test_overlap}"
        )

    def test_v2_provenance_and_metadata(self):
        v2_dir = DATA_DIR / "v2_real_world"

        # Check raw & generated pools exist
        raw_candidates_file = v2_dir / "raw" / "historical_candidates.jsonl"
        gen_drift_file = v2_dir / "generated" / "contract_grounded_drift.jsonl"

        self.assertTrue(raw_candidates_file.is_file())
        self.assertTrue(gen_drift_file.is_file())

        raw_candidates = load_jsonl(raw_candidates_file)
        gen_drift = load_jsonl(gen_drift_file)

        self.assertEqual(len(raw_candidates), 2367)
        self.assertEqual(len(gen_drift), 5133)

        # Check metadata
        meta_file = v2_dir / "metadata" / "dataset_summary.json"
        self.assertTrue(meta_file.is_file())
        with meta_file.open("r", encoding="utf-8") as f:
            summary = json.load(f)

        self.assertEqual(summary["total_instances"], 14796)
        self.assertEqual(summary["partitions"]["train"], 13366)
        self.assertEqual(summary["partitions"]["val"], 1430)
        self.assertEqual(summary["partitions"]["verified_test"], 101)
        self.assertEqual(summary["drift"]["authentic_historical_mined"], 2222)
        self.assertEqual(summary["drift"]["contract_grounded_generated"], 5122)
        self.assertEqual(summary["clean"]["total"], 7452)
        self.assertEqual(summary["zero_leakage_guarantee"]["candidates_purged_for_leakage"], 204)
        self.assertEqual(summary["zero_leakage_guarantee"]["train_val_overlap"], 0)
        self.assertEqual(summary["zero_leakage_guarantee"]["train_test_overlap"], 0)
        self.assertEqual(summary["zero_leakage_guarantee"]["val_test_overlap"], 0)

    def test_clean_v2_canonical_manifest_and_splits(self):
        """Verify the clean-slate V2 canonical dataset, manifest, and 104-instance test set."""
        import yaml
        exp_dir = PROJECT_ROOT / "experiments" / "2026-09-07_clean_v2"
        manifest_file = exp_dir / "config" / "manifest.yaml"
        self.assertTrue(manifest_file.is_file(), "Canonical manifest.yaml missing")

        with manifest_file.open("r", encoding="utf-8") as f:
            manifest = yaml.safe_load(f)

        d = manifest["dataset"]
        self.assertEqual(d["train_samples"], 24229)
        self.assertEqual(d["val_samples"], 2636)
        self.assertEqual(d["test_samples"], 104)
        self.assertEqual(d["test_labels"]["aligned"], 52)
        self.assertEqual(d["test_labels"]["drifted"], 52)
        self.assertEqual(d["provenance_distribution"]["test"]["authentic_historical_mined"], 52)

        # Check canonical test file on disk
        test_file = exp_dir / "dataset" / "verified_test.jsonl"
        self.assertTrue(test_file.is_file())
        test_rows = load_jsonl(test_file)
        self.assertEqual(len(test_rows), d["test_samples"])


if __name__ == "__main__":
    unittest.main()
