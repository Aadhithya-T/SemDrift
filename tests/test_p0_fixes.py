"""
tests/test_p0_fixes.py — Tests verifying all four P0 correctness fixes.

Coverage:
    - P0-1: Function lineage identity using qualified names.
    - P0-2: Inverted head/tail truncation fix in joint encoder collate function.
    - P0-3: Pipeline parse -> run_record -> divergence integration and NotImplementedError on run().
    - P0-4: Qualified-name aware AST function lookup in mutation generator.
"""

import ast
import tempfile
import unittest
from pathlib import Path

import pytest
import torch
from transformers import AutoTokenizer

from scripts.data_pipeline.build_v2_dataset import (
    find_function_node,
    mutate_default_value,
)
from scripts.data_pipeline.setup_dataset_architecture import get_function_lineage
from semdrift.models.joint_encoder import make_collate_fn
from semdrift.pipeline import Pipeline


# ==============================================================================
# P0-1: Function Lineage Identity
# ==============================================================================

class TestP0_1_FunctionLineage(unittest.TestCase):
    """Tests for P0-1: Function lineage qualified-name disambiguation."""

    def test_lineage_qualified_name_disambiguation(self):
        """Two classes in the same file with identical method names produce distinct lineages."""
        row_user = {
            "repo": "my_org/repo_a",
            "file": "services/auth.py",
            "function_name": "load",
            "qualified_name": "UserService.load",
        }
        row_admin = {
            "repo": "my_org/repo_a",
            "file": "services/auth.py",
            "function_name": "load",
            "qualified_name": "AdminService.load",
        }

        lineage_user = get_function_lineage(row_user)
        lineage_admin = get_function_lineage(row_admin)

        self.assertNotEqual(lineage_user, lineage_admin)
        self.assertEqual(lineage_user, "my_org/repo_a::services/auth.py::UserService.load")
        self.assertEqual(lineage_admin, "my_org/repo_a::services/auth.py::AdminService.load")

    def test_lineage_fallback_to_function_name(self):
        """When qualified_name is missing, lineage falls back cleanly to function_name."""
        row = {
            "repo": "my_org/repo_a",
            "file": "utils/helpers.py",
            "function_name": "compute_sum",
        }
        lineage = get_function_lineage(row)
        self.assertEqual(lineage, "my_org/repo_a::utils/helpers.py::compute_sum")

    def test_lineage_prefers_qualified_name(self):
        """qualified_name takes precedence over function_name and qualified_function_name."""
        row_all = {
            "repo": "repo",
            "file": "f.py",
            "function_name": "bare_name",
            "qualified_function_name": "Secondary.name",
            "qualified_name": "Primary.name",
        }
        self.assertEqual(get_function_lineage(row_all), "repo::f.py::Primary.name")

        row_qfn = {
            "repo": "repo",
            "file": "f.py",
            "function_name": "bare_name",
            "qualified_function_name": "Secondary.name",
        }
        self.assertEqual(get_function_lineage(row_qfn), "repo::f.py::Secondary.name")


# ==============================================================================
# P0-2: Inverted Head/Tail Truncation
# ==============================================================================

class TestP0_2_TruncationFix(unittest.TestCase):
    """Tests for P0-2: make_collate_fn head and tail truncation semantics."""

    @classmethod
    def setUpClass(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")

    def test_head_truncation_keeps_first_tokens(self):
        """'head' truncation strategy preserves prefix (beginning) of code tokens."""
        collate_fn = make_collate_fn(
            self.tokenizer,
            max_length=22,
            doc_max_tokens=4,
            truncation_strategy="head",
        )
        docstring = "doc"
        code = "FIRST_MARKER = 1\nx = 2\ny = 3\nz = 4\nw = 5\na = 6\nb = 7\nc = 8\nLAST_MARKER = 9"
        batch = [(docstring, code, 0, {})]

        inputs, _, _ = collate_fn(batch)
        decoded = self.tokenizer.decode(inputs["input_ids"][0])

        self.assertIn("FIRST_MARKER", decoded)
        self.assertNotIn("LAST_MARKER", decoded)

    def test_tail_truncation_keeps_last_tokens(self):
        """'tail' truncation strategy preserves suffix (ending) of code tokens."""
        collate_fn = make_collate_fn(
            self.tokenizer,
            max_length=22,
            doc_max_tokens=4,
            truncation_strategy="tail",
        )
        docstring = "doc"
        code = "FIRST_MARKER = 1\nx = 2\ny = 3\nz = 4\nw = 5\na = 6\nb = 7\nc = 8\nLAST_MARKER = 9"
        batch = [(docstring, code, 0, {})]

        inputs, _, _ = collate_fn(batch)
        decoded = self.tokenizer.decode(inputs["input_ids"][0])

        self.assertIn("LAST_MARKER", decoded)
        self.assertNotIn("FIRST_MARKER", decoded)

    def test_head_tail_unchanged(self):
        """'head_tail' preserves both prefix and suffix, inserting mask token."""
        collate_fn = make_collate_fn(
            self.tokenizer,
            max_length=28,
            doc_max_tokens=4,
            truncation_strategy="head_tail",
        )
        docstring = "doc"
        code = "FIRST_MARKER = 1\nx = 2\ny = 3\nz = 4\nw = 5\na = 6\nb = 7\nc = 8\nLAST_MARKER = 9"
        batch = [(docstring, code, 0, {})]

        inputs, _, _ = collate_fn(batch)
        decoded = self.tokenizer.decode(inputs["input_ids"][0])

        self.assertIn("FIRST_MARKER", decoded)
        self.assertIn("LAST_MARKER", decoded)
        self.assertIn(self.tokenizer.mask_token, decoded)

    def test_truncation_fallback_to_head(self):
        """Unspecified or unknown truncation strategy falls back to standard prefix (head)."""
        collate_fn = make_collate_fn(
            self.tokenizer,
            max_length=22,
            doc_max_tokens=4,
            truncation_strategy="unknown_mode",
        )
        docstring = "doc"
        code = "FIRST_MARKER = 1\nx = 2\ny = 3\nz = 4\nw = 5\na = 6\nb = 7\nc = 8\nLAST_MARKER = 9"
        batch = [(docstring, code, 0, {})]

        inputs, _, _ = collate_fn(batch)
        decoded = self.tokenizer.decode(inputs["input_ids"][0])

        self.assertIn("FIRST_MARKER", decoded)
        self.assertNotIn("LAST_MARKER", decoded)


# ==============================================================================
# P0-3: Pipeline API & Integration Flow
# ==============================================================================

class TestP0_3_Pipeline(unittest.TestCase):
    """Tests for P0-3: Pipeline parse -> run_record -> drift prediction flow."""

    def setUp(self):
        self.pipe = Pipeline({"threshold": 0.20, "device": "cpu"})

    def test_pipeline_parse_returns_records(self):
        """Pipeline.parse() extracts function records with code and docstring."""
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False, encoding="utf-8") as f:
            f.write(
                "def calculate_tax(income: float) -> float:\n"
                '    """Calculates income tax for standard bracket."""\n'
                "    return income * 0.2\n"
            )
            temp_path = f.name

        try:
            records = self.pipe.parse(temp_path)
            self.assertGreaterEqual(len(records), 1)
            rec = records[0]
            self.assertIn("code", rec)
            self.assertIn("docstring", rec)
            self.assertIn("function_id", rec)
            self.assertIn("calculate_tax", rec["code"])
            self.assertIn("Calculates income tax", rec["docstring"])
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_pipeline_run_record_integration(self):
        """Pipeline.run_record() accepts pre-parsed record and computes divergence & prediction."""
        record = {
            "function_id": "test::add",
            "qualified_name": "Math.add",
            "code": "def add(x, y):\n    return x + y",
            "docstring": "Add two numbers together.",
        }
        res = self.pipe.run_record(record)

        self.assertIn("divergence", res)
        self.assertIsInstance(res["divergence"], float)
        self.assertIn("prediction", res)
        self.assertIn(res["prediction"], ("aligned", "drifted"))
        self.assertEqual(res["label"], res["prediction"])
        self.assertIsInstance(res["is_drift"], bool)
        self.assertEqual(res["function_id"], "test::add")
        self.assertEqual(res["qualified_name"], "Math.add")

    def test_pipeline_run_records_batch(self):
        """Pipeline.run_records() processes multiple records in sequence."""
        records = [
            {
                "function_id": "f1",
                "code": "def get_pi():\n    return 3.14159",
                "docstring": "Return constant value of pi.",
            },
            {
                "function_id": "f2",
                "code": "def is_even(n):\n    return n % 2 == 0",
                "docstring": "Check if an integer is even.",
            },
        ]
        results = self.pipe.run_records(records)
        self.assertEqual(len(results), 2)
        for res in results:
            self.assertIn("divergence", res)
            self.assertIn("prediction", res)

    def test_pipeline_end_to_end(self):
        """Full pipeline: source file -> Pipeline.parse() -> Pipeline.run_record() -> drift result."""
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False, encoding="utf-8") as f:
            f.write(
                "def multiply(a: int, b: int) -> int:\n"
                '    """Multiplies two integers and returns the product."""\n'
                "    return a * b\n"
            )
            temp_path = f.name

        try:
            records = self.pipe.parse(temp_path)
            self.assertEqual(len(records), 1)

            result = self.pipe.run_record(records[0])
            self.assertIn("divergence", result)
            self.assertIn("prediction", result)
            self.assertIn(result["prediction"], ("aligned", "drifted"))
            self.assertGreaterEqual(result["divergence"], 0.0)
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_pipeline_commit_run_explicit_error(self):
        """Pipeline.run(repo, c1, c2) raises NotImplementedError with helpful message."""
        with self.assertRaises(NotImplementedError) as ctx:
            self.pipe.run("/path/to/repo", "hash_a", "hash_b")

        self.assertIn("commit", str(ctx.exception).lower())


# ==============================================================================
# P0-4: Mutation Targeting Qualified Names
# ==============================================================================

class TestP0_4_MutationTargeting(unittest.TestCase):
    """Tests for P0-4: Qualified-name aware AST function lookup in mutation generator."""

    def setUp(self):
        self.code = (
            "class UserService:\n"
            "    def execute(self, timeout=30):\n"
            "        return timeout\n"
            "\n"
            "class AdminService:\n"
            "    def execute(self, retries=3):\n"
            "        return retries\n"
        )

    def test_mutation_qualified_name_targeting(self):
        """mutate_default_value mutates ONLY the targeted class method when names collide."""
        res_admin = mutate_default_value(self.code, "AdminService.execute", {})
        self.assertIsNotNone(res_admin)
        mutated_admin = res_admin["mutated_code"]

        # AdminService.execute retries must be mutated to 9999
        self.assertIn("retries=9999", mutated_admin)
        # UserService.execute timeout=30 must remain completely untouched
        self.assertIn("timeout=30", mutated_admin)

        # Now test targeting UserService.execute
        res_user = mutate_default_value(self.code, "UserService.execute", {})
        self.assertIsNotNone(res_user)
        mutated_user = res_user["mutated_code"]

        self.assertIn("timeout=9999", mutated_user)
        self.assertIn("retries=3", mutated_user)

    def test_mutation_bare_name_backward_compat(self):
        """mutate_default_value works with bare function name for backward compatibility."""
        res = mutate_default_value(self.code, "execute", {})
        self.assertIsNotNone(res)
        self.assertEqual(res["mutation_type"], "default_value_drift")

    def test_find_function_node_nested(self):
        """find_function_node correctly resolves classes, nested classes, and bare names."""
        nested_code = (
            "class Outer:\n"
            "    class Inner:\n"
            "        def helper(self, val=1):\n"
            "            return val\n"
            "    def top_level(self, x=2):\n"
            "        return x\n"
        )
        tree = ast.parse(nested_code)

        node_nested = find_function_node(tree, "Outer.Inner.helper")
        self.assertIsNotNone(node_nested)
        self.assertEqual(node_nested.name, "helper")

        node_top = find_function_node(tree, "Outer.top_level")
        self.assertIsNotNone(node_top)
        self.assertEqual(node_top.name, "top_level")

        node_bare = find_function_node(tree, "helper")
        self.assertIsNotNone(node_bare)
        self.assertEqual(node_bare.name, "helper")

        node_missing = find_function_node(tree, "NonExistentClass.helper")
        # Falls back to searching for helper if nonexistent class prefix
        self.assertIsNotNone(node_bare)

        node_completely_missing = find_function_node(tree, "Outer.does_not_exist")
        self.assertIsNone(node_completely_missing)


if __name__ == "__main__":
    unittest.main()
