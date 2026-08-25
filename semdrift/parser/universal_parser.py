"""
semdrift.parser.universal_parser — Multi-language function extractor using tree-sitter.

This module provides the UniversalParser class, which uses the `tree-sitter`
library to parse code into a syntax tree and extract function/method metadata
(including source code and docstrings) in a language-agnostic manner.

It outputs the exact same `FunctionInfo` objects as the python-specific `ASTParser`.
"""

import os
from typing import List, Optional

try:
    from tree_sitter import Language, Parser, Node
    import tree_sitter_python
except ImportError:
    Language = None
    Parser = None
    Node = None
    tree_sitter_python = None

from semdrift.parser.ast_parser import FunctionInfo


class UniversalParser:
    """Parses source files across multiple languages using tree-sitter.

    Outputs exactly the same `FunctionInfo` objects as `ASTParser`.
    Currently supports: Python (via tree-sitter-python).
    Can be easily extended to Java, JavaScript, etc.

    Parameters
    ----------
    max_file_size_kb : int
        Skip files larger than this (in kilobytes). Default 500.
    """

    _SKIP_DIRS: set[str] = {
        "__pycache__", ".git", ".tox", ".mypy_cache",
        ".pytest_cache", "node_modules", ".venv", "venv",
        "env", ".env", "site-packages", ".eggs", "build",
        "dist", "egg-info",
    }

    def __init__(self, max_file_size_kb: int = 500):
        if Parser is None:
            raise ImportError(
                "tree-sitter packages are not installed. "
                "Please run `pip install tree-sitter tree-sitter-python`."
            )
            
        self.max_file_size_kb = max_file_size_kb
        
        # Initialize parsers and queries for supported languages
        self.parsers = {}
        self.queries = {}
        
        # Set up Python
        if tree_sitter_python is not None:
            py_lang = Language(tree_sitter_python.language())
            parser = Parser(py_lang)
            self.parsers[".py"] = parser
            
            # Tree-sitter query to find functions and their potential docstrings
            query = py_lang.query('''
                (function_definition
                    name: (identifier) @func.name
                    body: (block
                        (expression_statement
                            (string) @func.docstring)?
                    )
                ) @func.def
            ''')
            self.queries[".py"] = query

    def parse_file(self, filepath: str) -> List[FunctionInfo]:
        """Parse a single file and extract its functions."""
        ext = os.path.splitext(filepath)[1].lower()
        if ext not in self.parsers:
            # Language not supported by this universal parser yet
            return []

        try:
            stat = os.stat(filepath)
            if stat.st_size > self.max_file_size_kb * 1024:
                return []
        except OSError:
            return []

        try:
            with open(filepath, "rb") as f:
                source_bytes = f.read()
        except OSError:
            return []

        parser = self.parsers[ext]
        query = self.queries[ext]
        
        tree = parser.parse(source_bytes)
        matches = query.matches(tree.root_node)
        
        functions = []
        # matches is a list of (match_id, captures)
        # where captures is a dict mapping capture_name -> [nodes]
        for match_id, captures in matches:
            if "func.def" not in captures or "func.name" not in captures:
                continue
                
            func_node = captures["func.def"][0]
            name_node = captures["func.name"][0]
            
            func_name = source_bytes[name_node.start_byte:name_node.end_byte].decode("utf-8")
            
            # 1-indexed lines
            line_start = func_node.start_point[0] + 1
            line_end = func_node.end_point[0] + 1
            
            docstring = ""
            has_docstring = False
            
            # Function source code in bytes
            source_code_bytes = source_bytes[func_node.start_byte:func_node.end_byte]
            
            if "func.docstring" in captures:
                doc_node = captures["func.docstring"][0]
                has_docstring = True
                
                # Extract the raw string literal
                docstring = source_bytes[doc_node.start_byte:doc_node.end_byte].decode("utf-8")
                
                # To get the code WITHOUT the docstring, we need to remove it from the source_bytes slice.
                # However, removing it precisely while keeping formatting requires slicing.
                
                # Start of function to start of docstring statement
                # The doc_node is just the string, its parent is the expression_statement which holds the newline
                expr_stmt = doc_node.parent
                if expr_stmt and expr_stmt.type == "expression_statement":
                    pre_doc = source_bytes[func_node.start_byte:expr_stmt.start_byte]
                    post_doc = source_bytes[expr_stmt.end_byte:func_node.end_byte]
                    
                    # Clean up trailing whitespace from pre_doc
                    pre_doc = pre_doc.rstrip(b" \t")
                    source_code_bytes = pre_doc + post_doc
            
            source_code = source_code_bytes.decode("utf-8")
            
            info = FunctionInfo(
                name=func_name,
                class_name=None, # Class name resolution would require more complex queries
                file_path=os.path.abspath(filepath),
                line_start=line_start,
                line_end=line_end,
                source_code=source_code,
                docstring=docstring,
                has_docstring=has_docstring,
                params=[],
                return_annotation=None,
                decorators=[],
                raises=[]
            )
            functions.append(info)
            
        return functions

    def parse_directory(self, dirpath: str) -> List[FunctionInfo]:
        """Recursively parse a directory, yielding all supported functions."""
        results = []
        for root, dirs, files in os.walk(dirpath):
            dirs[:] = [d for d in dirs if d not in self._SKIP_DIRS]
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in self.parsers:
                    filepath = os.path.join(root, file)
                    results.extend(self.parse_file(filepath))
        return results
