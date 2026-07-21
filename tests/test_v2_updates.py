import ast
import unittest
import torch
from transformers import AutoTokenizer

from scripts.extract_pairs import remove_docstring_from_function
from scripts.train_model_b import make_collate_fn, calculate_metrics


class TestV2Updates(unittest.TestCase):

    def test_docstring_removal_standard(self):
        """Ensure standard docstring is stripped from function body AST, keeping signature/body."""
        source = (
            "def add(x, y):\n"
            '    """Adds two numbers."""\n'
            "    return x + y"
        )
        tree = ast.parse(source)
        func_node = tree.body[0]
        remove_docstring_from_function(func_node)
        
        cleaned_code = ast.unparse(func_node)
        self.assertNotIn('"""Adds two numbers."""', cleaned_code)
        self.assertIn("return x + y", cleaned_code)
        self.assertIn("def add(x, y):", cleaned_code)

    def test_docstring_removal_docstring_only(self):
        """Ensure docstring-only function resolves safely to a 'pass' body instead of empty body."""
        source = (
            "def abstract_method():\n"
            '    """This method is abstract."""'
        )
        tree = ast.parse(source)
        func_node = tree.body[0]
        remove_docstring_from_function(func_node)
        
        cleaned_code = ast.unparse(func_node)
        self.assertNotIn('"""This method is abstract."""', cleaned_code)
        self.assertIn("pass", cleaned_code)

    def test_head_tail_truncation(self):
        """Ensure head-tail truncation preserves prefix, suffix (returns), and inserts mask token."""
        tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
        
        # 25 total length budget, doc_max_tokens = 5
        max_length = 25
        doc_max_tokens = 5
        collate_fn = make_collate_fn(
            tokenizer,
            max_length=max_length,
            doc_max_tokens=doc_max_tokens,
            truncation_strategy="head_tail"
        )
        
        docstring = "short doc"
        code = "x = 1\ny = 2\nz = 3\nw = 4\na = 5\nb = 6\nc = 7\nreturn z + a + b + c"
        batch = [(docstring, code, 1, {})]
        
        inputs, labels, metas = collate_fn(batch)
        
        input_ids = inputs["input_ids"][0]
        self.assertTrue(len(input_ids) <= max_length)
        
        decoded = tokenizer.decode(input_ids)
        
        self.assertIn("short doc", decoded)
        self.assertIn(tokenizer.mask_token, decoded)
        self.assertIn("return", decoded)
        self.assertIn("c", decoded)

    def test_valid_joint_input_length_limit(self):
        """Verify that under extreme input sizes, the collated tensor length never exceeds 512."""
        tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
        max_length = 512
        collate_fn = make_collate_fn(
            tokenizer,
            max_length=max_length,
            doc_max_tokens=96,
            truncation_strategy="head_tail"
        )
        
        docstring = "doc " * 1000
        code = "x = 1\n" * 2000 + "return x"
        batch = [(docstring, code, 0, {})]
        
        inputs, labels, metas = collate_fn(batch)
        input_ids = inputs["input_ids"][0]
        self.assertTrue(len(input_ids) <= max_length)

    def test_checkpoint_metric_calculation(self):
        """Assert metrics calculation supports balanced accuracy, macro F1, and predicted counts."""
        y_true = ["aligned", "aligned", "drifted", "drifted"]
        y_pred = ["aligned", "drifted", "drifted", "drifted"]
        
        metrics = calculate_metrics(y_true, y_pred)
        
        self.assertIn("macro_f1", metrics)
        self.assertIn("balanced_accuracy", metrics)
        self.assertIn("confusion_matrix", metrics)
        self.assertEqual(metrics["pred_aligned"], 1)
        self.assertEqual(metrics["pred_drifted"], 3)
        self.assertEqual(metrics["tn"], 1)
        self.assertEqual(metrics["fp"], 1)
        self.assertEqual(metrics["fn"], 0)
        self.assertEqual(metrics["tp"], 2)


if __name__ == "__main__":
    unittest.main()
