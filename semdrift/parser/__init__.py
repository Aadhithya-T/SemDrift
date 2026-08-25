"""
semdrift.parser — AST Parsing & Documentation Extraction Stage.

Responsible for extracting function source code and docstrings from
Python codebases, and formatting them as model-ready JSON input for
the CodeBERT divergence-detection model.

Quick usage::

    from semdrift.parser import parse_codebase

    # Parse a whole repo — returns list of model-input dicts.
    records = parse_codebase("/path/to/repo")
    # Each record: {"function_id": ..., "code": ..., "docstring": ...}

    # Parse a single file.
    records = parse_codebase("/path/to/file.py")

    # Write results to a JSON file.
    records = parse_codebase("/path/to/repo", output_path="output.json")
"""

import os
from typing import Optional

from semdrift.parser.ast_parser import ASTParser, FunctionInfo
from semdrift.parser.doc_extractor import DocExtractor, DocInfo
from semdrift.parser.formatter import ModelInputFormatter
from semdrift.parser.universal_parser import UniversalParser


def parse_codebase(
    path: str,
    output_path: Optional[str] = None,
    include_undocumented: bool = False,
    normalise_docstring: bool = True,
    skip_dunder: bool = False,
    skip_private: bool = False,
    skip_test_files: bool = True,
    max_file_size_kb: int = 500,
) -> list[dict]:
    """Parse a Python codebase and return model-ready JSON records.

    This is the primary convenience function that ties together the
    AST parser, documentation extractor, and model input formatter
    into a single call.

    Parameters
    ----------
    path : str
        Path to a single ``.py`` file or a directory containing Python
        source files.
    output_path : str or None
        If provided, write the JSON output to this file path.
    include_undocumented : bool
        Include functions without docstrings.  Default False.
    normalise_docstring : bool
        Normalise docstrings into flat text.  Default True.
    skip_dunder : bool
        Skip dunder methods (``__init__``, etc.).  Default False.
    skip_private : bool
        Skip private methods (``_helper``).  Default False.
    skip_test_files : bool
        Skip test files (``test_*.py``, ``conftest.py``).  Default True.
    max_file_size_kb : int
        Skip files larger than this (KB).  Default 500.

    Returns
    -------
    list[dict]
        Each dict has keys ``function_id``, ``code``, ``docstring`` —
        the format expected by the CodeBERT model.

    Examples
    --------
    >>> from semdrift.parser import parse_codebase
    >>> records = parse_codebase("./my_project")
    >>> print(records[0]["function_id"])
    'my_module_py_MyClass_my_method_001'
    """
    parser = ASTParser(
        max_file_size_kb=max_file_size_kb,
        skip_dunder=skip_dunder,
        skip_private=skip_private,
        skip_test_files=skip_test_files,
    )

    # Determine base_path for building relative function IDs.
    if os.path.isdir(path):
        base_path = path
        functions = parser.parse_directory(path)
    else:
        base_path = os.path.dirname(os.path.abspath(path)) or "."
        functions = parser.parse_file(path)

    formatter = ModelInputFormatter(
        base_path=base_path,
        include_undocumented=include_undocumented,
        normalise_docstring=normalise_docstring,
    )

    if output_path:
        formatter.format_to_json(functions, output_path=output_path)

    return formatter.format(functions)


__all__ = [
    "ASTParser",
    "FunctionInfo",
    "UniversalParser",
    "DocExtractor",
    "DocInfo",
    "ModelInputFormatter",
    "parse_codebase",
]
