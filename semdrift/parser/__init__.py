"""
semdrift.parser — Code Parsing & Documentation Extraction Stage.

Provides two parsing pipelines:

**V2 (Recommended) — Universal Parser**
    Uses tree-sitter for multi-language support (Python, Java).
    Outputs ``FunctionContract`` objects with full semantic contract info.

    >>> from semdrift.parser import parse_codebase_v2
    >>> contracts = parse_codebase_v2("/path/to/repo")

**V1 (Legacy) — AST Parser**
    Uses Python's built-in ``ast`` module.  Retained for Phase 1
    reproducibility.  Outputs ``FunctionInfo`` objects.

    >>> from semdrift.parser import parse_codebase
    >>> records = parse_codebase("/path/to/repo")
"""

import os
from typing import Optional, List

# --- V1 Legacy imports (unchanged for backward compatibility) ---
from semdrift.parser.ast_parser import ASTParser, FunctionInfo
from semdrift.parser.doc_extractor import DocExtractor, DocInfo
from semdrift.parser.formatter import ModelInputFormatter

# --- V2 Universal Parser imports ---
from semdrift.parser.contracts import (
    FunctionContract,
    Parameter,
    ParameterKind,
    ParseStatus,
    ExceptionContract,
    RaiseForm,
    DocContract,
)
from semdrift.parser.docstring_parser import DocstringParser
from semdrift.parser.universal_parser import UniversalParser
from semdrift.parser.contract_formatter import ContractFormatter


# ==================================================================
# V1 Legacy convenience function (unchanged)
# ==================================================================

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
    """[LEGACY] Parse a Python codebase using the V1 AST parser.

    Retained for Phase 1 reproducibility.  For new usage, prefer
    :func:`parse_codebase_v2`.

    Parameters
    ----------
    path : str
        Path to a single ``.py`` file or a directory.
    output_path : str or None
        If provided, write the JSON output to this file path.
    include_undocumented : bool
        Include functions without docstrings.
    normalise_docstring : bool
        Normalise docstrings into flat text.
    skip_dunder : bool
        Skip dunder methods.
    skip_private : bool
        Skip private methods.
    skip_test_files : bool
        Skip test files.
    max_file_size_kb : int
        Skip files larger than this (KB).

    Returns
    -------
    list[dict]
        Each dict has keys ``function_id``, ``code``, ``docstring``.
    """
    parser = ASTParser(
        max_file_size_kb=max_file_size_kb,
        skip_dunder=skip_dunder,
        skip_private=skip_private,
        skip_test_files=skip_test_files,
    )

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


# ==================================================================
# V2 Universal Parser convenience function
# ==================================================================

def parse_codebase_v2(
    path: str,
    output_path: Optional[str] = None,
    include_undocumented: bool = False,
    use_structured_docstring: bool = True,
    skip_dunder: bool = False,
    skip_private: bool = False,
    skip_test_files: bool = True,
    max_file_size_kb: int = 500,
) -> List[FunctionContract]:
    """Parse a codebase using the V2 Universal Parser (tree-sitter).

    Supports Python and Java.  Returns ``FunctionContract`` objects with
    full semantic contract information.

    Parameters
    ----------
    path : str
        Path to a single source file or a directory containing source files.
    output_path : str or None
        If provided, write JSON output to this file.
    include_undocumented : bool
        Include functions without docstrings.  Default False.
    use_structured_docstring : bool
        Use ``[SUMMARY]``, ``[PARAMETERS]``, etc. section markers in the
        docstring output.  Default True.
    skip_dunder : bool
        Skip Python dunder methods.  Default False.
    skip_private : bool
        Skip Python private methods.  Default False.
    skip_test_files : bool
        Skip test files.  Default True.
    max_file_size_kb : int
        Skip files larger than this (KB).  Default 500.

    Returns
    -------
    list[FunctionContract]
        Extracted function contracts with full metadata.
    """
    parser = UniversalParser(
        max_file_size_kb=max_file_size_kb,
        skip_dunder=skip_dunder,
        skip_private=skip_private,
        skip_test_files=skip_test_files,
    )

    if os.path.isdir(path):
        base_path = path
        contracts = parser.parse_directory(path)
    else:
        base_path = os.path.dirname(os.path.abspath(path)) or "."
        contracts = parser.parse_file(path)

    if output_path:
        formatter = ContractFormatter(
            base_path=base_path,
            include_undocumented=include_undocumented,
            use_structured_docstring=use_structured_docstring,
        )
        formatter.format_to_json(contracts, output_path=output_path)

    return contracts


__all__ = [
    # V1 Legacy
    "ASTParser",
    "FunctionInfo",
    "DocExtractor",
    "DocInfo",
    "ModelInputFormatter",
    "parse_codebase",
    # V2 Universal
    "UniversalParser",
    "FunctionContract",
    "Parameter",
    "ParameterKind",
    "ParseStatus",
    "ExceptionContract",
    "RaiseForm",
    "DocContract",
    "DocstringParser",
    "ContractFormatter",
    "parse_codebase_v2",
]
