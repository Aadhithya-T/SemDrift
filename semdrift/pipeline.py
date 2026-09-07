"""
semdrift.pipeline — End-to-end Semantic Drift Detection Pipeline.

Ties together the parser and embedder stages into a single callable pipeline.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class Pipeline:
    """Orchestrates the semantic drift detection flow.

    Stages:
        1. Parser   — extracts function records ({code, docstring, ...}) from source.
        2. Embedder — generates vector embeddings from code and docstrings.
        3. Scorer   — computes divergence and predicts drift labels.
    """

    def __init__(self, config: Optional[dict] = None) -> None:
        self.config = config or {}
        self.model_name: str = self.config.get("model_name", "microsoft/codebert-base")
        self.device: str = self.config.get("device", "cpu")
        self.threshold: float = float(self.config.get("threshold", 0.15))
        self.pooling: str = self.config.get("pooling", "mean")
        self.max_token_length: int = int(self.config.get("max_token_length", 512))

    def parse(self, path: str, **kwargs: Any) -> List[dict]:
        """Parse source file or directory into a list of function records.

        Parameters
        ----------
        path : str
            Path to a source file or directory.
        **kwargs : Any
            Additional options passed to `parse_codebase`.

        Returns
        -------
        list[dict]
            Each dict has keys 'function_id', 'code', 'docstring', etc.
        """
        from semdrift.parser import parse_codebase

        return parse_codebase(path, **kwargs)

    def run_record(self, record: dict) -> dict:
        """Run drift detection on a single pre-parsed function record.

        Parameters
        ----------
        record : dict
            A dictionary containing at least 'code' and 'docstring' keys.

        Returns
        -------
        dict
            Result dictionary containing:
                - divergence (float): Semantic divergence score.
                - prediction (str): 'drifted' or 'aligned'.
                - label (str): Alias for prediction.
                - is_drift (bool): True if divergence >= threshold.
                - function_id (optional str): Function identifier.
                - qualified_name (optional str): Qualified name.
        """
        if not isinstance(record, dict) or "code" not in record or "docstring" not in record:
            raise ValueError("Record must be a dict containing 'code' and 'docstring' keys.")

        from semdrift.embedder.embed import (
            compute_divergence,
            embed_function_record,
            predict_drift,
        )

        code_emb, doc_emb = embed_function_record(
            record,
            max_token_length=self.max_token_length,
            model_name=self.model_name,
            device=self.device,
            pooling=self.pooling,
        )

        divergence = float(compute_divergence(code_emb, doc_emb))
        label = predict_drift(divergence, threshold=self.threshold)

        result = {
            "function_id": record.get("function_id"),
            "qualified_name": record.get("qualified_name"),
            "divergence": divergence,
            "prediction": label,
            "label": label,
            "is_drift": label == "drifted",
            "threshold": self.threshold,
        }
        return result

    def run_records(self, records: List[dict]) -> List[dict]:
        """Run drift detection on a list of pre-parsed function records.

        Parameters
        ----------
        records : list[dict]
            List of dictionaries, each containing 'code' and 'docstring'.

        Returns
        -------
        list[dict]
            List of result dictionaries.
        """
        return [self.run_record(record) for record in records]

    def run(self, repo_path: str, commit_a: str, commit_b: str) -> dict:
        """Run the full pipeline between two commits (Not Implemented).

        Git checkout and version resolution between two commits are outside
        the current single-snapshot parser scope. Use `Pipeline.parse()`
        and `Pipeline.run_record()` for record-level drift detection.
        """
        raise NotImplementedError(
            "Commit-based pipeline requires git checkout and commit resolution functionality "
            "that is not yet implemented. Use Pipeline.parse() followed by Pipeline.run_record() "
            "or Pipeline.run_records() on parsed records."
        )
