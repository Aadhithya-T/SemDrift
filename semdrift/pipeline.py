"""
semdrift.pipeline — End-to-end Semantic Drift Detection Pipeline.

Ties together the parser, embedder, and comparator stages into a
single callable pipeline.
"""

from semdrift.parser import *
from semdrift.embedder import *
from semdrift.comparator import *


class Pipeline:
    """Orchestrates the three-stage semantic drift detection flow.

    Stages:
        1. Parser   — extracts AST representations from source code.
        2. Embedder — generates vector embeddings from AST data.
        3. Comparator — compares embeddings and produces drift scores.
    """

    def __init__(self, config: dict | None = None):
        self.config = config or {}
        # Stage handles will be initialised by submodules once implemented.
        self.parser = None
        self.embedder = None
        self.comparator = None

    def run(self, repo_path: str, commit_a: str, commit_b: str) -> dict:
        """Run the full pipeline between two commits.

        Parameters
        ----------
        repo_path : str
            Path to the local git repository.
        commit_a : str
            The older commit hash.
        commit_b : str
            The newer commit hash.

        Returns
        -------
        dict
            A result dictionary containing drift scores and metadata.
        """
        # Step 1 — Parse
        ast_a = self._parse(repo_path, commit_a)
        ast_b = self._parse(repo_path, commit_b)

        # Step 2 — Embed
        emb_a = self._embed(ast_a)
        emb_b = self._embed(ast_b)

        # Step 3 — Compare
        result = self._compare(emb_a, emb_b)

        return result

    # ------------------------------------------------------------------
    # Private helpers (delegates to stage modules once implemented)
    # ------------------------------------------------------------------

    def _parse(self, repo_path: str, commit: str):
        """Extract AST representation for a given commit."""
        raise NotImplementedError("Parser stage not yet wired up.")

    def _embed(self, ast_data):
        """Generate embeddings from parsed AST data."""
        raise NotImplementedError("Embedder stage not yet wired up.")

    def _compare(self, emb_a, emb_b) -> dict:
        """Compare two sets of embeddings and return drift scores."""
        raise NotImplementedError("Comparator stage not yet wired up.")
