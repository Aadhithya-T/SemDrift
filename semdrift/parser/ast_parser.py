"""
semdrift.parser.ast_parser — Python AST-based function extractor.

Parses Python source files using the built-in ``ast`` module and extracts
a structured representation of every function and method, including the
raw source code (with docstring removed) and the raw docstring text.

This module is the code-side of the parser stage.  It produces
:class:`FunctionInfo` objects that downstream modules (doc extractor,
formatter) consume to build model-ready JSON input.
"""

import ast
import os
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class FunctionInfo:
    """Structured representation of a single parsed function/method.

    Attributes
    ----------
    name : str
        Function or method name.
    class_name : str or None
        Enclosing class name, if this is a method.
    file_path : str
        Absolute path to the source file.
    line_start : int
        1-indexed start line of the function definition.
    line_end : int
        1-indexed end line of the function definition.
    source_code : str
        Full function source code **with the docstring removed**.
    docstring : str
        Raw docstring text (empty string if no docstring).
    has_docstring : bool
        Whether the function had a docstring.
    params : list[str]
        Parameter names (excluding ``self`` and ``cls``).
    return_annotation : str or None
        Return type annotation as source text, if present.
    decorators : list[str]
        Decorator names (e.g. ``["staticmethod", "property"]``).
    raises : list[str]
        Exception class names from explicit ``raise`` statements.
    """

    name: str
    class_name: Optional[str]
    file_path: str
    line_start: int
    line_end: int
    source_code: str
    docstring: str
    has_docstring: bool
    params: List[str] = field(default_factory=list)
    return_annotation: Optional[str] = None
    decorators: List[str] = field(default_factory=list)
    raises: List[str] = field(default_factory=list)


class ASTParser:
    """Parses Python source files and extracts function/method metadata.

    The parser walks the AST of each file, finds every ``def`` and
    ``async def`` node, and builds a :class:`FunctionInfo` that
    cleanly separates the function's **code** from its **documentation**.
    This separation is exactly what the downstream CodeBERT model needs.

    Parameters
    ----------
    max_file_size_kb : int
        Skip files larger than this (in kilobytes).  Default 500.
    skip_dunder : bool
        If True, skip dunder methods (``__init__``, ``__repr__``, …).
        Default False.
    skip_private : bool
        If True, skip private methods (single leading ``_``, but not
        dunder).  Default False.
    skip_test_files : bool
        If True, skip files whose name starts with ``test_`` or is
        ``conftest.py``.  Default True.
    """

    # Directories that should never be entered.
    _SKIP_DIRS: set[str] = {
        "__pycache__", ".git", ".tox", ".mypy_cache",
        ".pytest_cache", "node_modules", ".venv", "venv",
        "env", ".env", "site-packages", ".eggs", "build",
        "dist", "egg-info",
    }

    # File-name prefixes to skip when ``skip_test_files`` is True.
    _SKIP_FILE_PREFIXES: tuple[str, ...] = ("test_", "conftest")

    def __init__(
        self,
        max_file_size_kb: int = 500,
        skip_dunder: bool = False,
        skip_private: bool = False,
        skip_test_files: bool = True,
    ) -> None:
        self.max_file_size_kb = max_file_size_kb
        self.skip_dunder = skip_dunder
        self.skip_private = skip_private
        self.skip_test_files = skip_test_files

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse_file(self, file_path: str) -> List[FunctionInfo]:
        """Parse a single Python file and return extracted functions.

        Parameters
        ----------
        file_path : str
            Path to the ``.py`` file.

        Returns
        -------
        list[FunctionInfo]
            Extracted function metadata.  Empty list if the file cannot
            be parsed or is skipped by the configured filters.
        """
        file_path = os.path.abspath(file_path)

        if not file_path.endswith(".py"):
            return []

        # Check file size.
        try:
            size_kb = os.path.getsize(file_path) / 1024
            if size_kb > self.max_file_size_kb:
                return []
        except OSError:
            return []

        # Read source.
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as fh:
                source = fh.read()
        except (OSError, UnicodeDecodeError):
            return []

        return self._parse_source(source, file_path)

    def parse_directory(
        self,
        dir_path: str,
        file_extensions: tuple[str, ...] = (".py",),
    ) -> List[FunctionInfo]:
        """Recursively parse all Python files in a directory tree.

        Parameters
        ----------
        dir_path : str
            Root directory to walk.
        file_extensions : tuple[str, ...]
            File extensions to include.  Default ``(".py",)``.

        Returns
        -------
        list[FunctionInfo]
            Aggregated list from every parsed file.
        """
        results: List[FunctionInfo] = []

        for root, dirs, files in os.walk(dir_path):
            # Prune directories we never want to enter (in-place).
            dirs[:] = [d for d in dirs if d not in self._SKIP_DIRS]

            for fname in sorted(files):
                if not any(fname.endswith(ext) for ext in file_extensions):
                    continue

                if self.skip_test_files and any(
                    fname.startswith(pfx) for pfx in self._SKIP_FILE_PREFIXES
                ):
                    continue

                full_path = os.path.join(root, fname)
                results.extend(self.parse_file(full_path))

        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parse_source(self, source: str, file_path: str) -> List[FunctionInfo]:
        """Parse a source-code string into a list of :class:`FunctionInfo`."""
        try:
            tree = ast.parse(source, filename=file_path)
        except SyntaxError:
            return []

        source_lines = source.splitlines()
        functions: List[FunctionInfo] = []
        self._walk_nodes(tree, source_lines, file_path, class_name=None, out=functions)
        return functions

    def _walk_nodes(
        self,
        node: ast.AST,
        source_lines: list[str],
        file_path: str,
        class_name: Optional[str],
        out: List[FunctionInfo],
    ) -> None:
        """Recursively walk AST nodes, extracting functions and methods."""
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                # Recurse into class body — methods get the class name.
                self._walk_nodes(
                    child, source_lines, file_path,
                    class_name=child.name, out=out,
                )
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Apply name-based skip filters.
                if self._should_skip(child.name):
                    continue

                info = self._extract_function(child, source_lines, file_path, class_name)
                if info is not None:
                    out.append(info)

    def _should_skip(self, name: str) -> bool:
        """Return True if the function name should be skipped."""
        is_dunder = name.startswith("__") and name.endswith("__")
        is_private = name.startswith("_") and not is_dunder

        if self.skip_dunder and is_dunder:
            return True
        if self.skip_private and is_private:
            return True
        return False

    def _extract_function(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        source_lines: list[str],
        file_path: str,
        class_name: Optional[str],
    ) -> Optional[FunctionInfo]:
        """Build a :class:`FunctionInfo` from a function AST node."""
        # --- Docstring ---
        raw_docstring = ast.get_docstring(node, clean=False) or ""
        has_docstring = bool(raw_docstring)

        # --- Source code (with docstring stripped out) ---
        code_without_doc = self._get_source_without_docstring(node, source_lines)

        # --- Parameters (excluding self/cls) ---
        params = self._extract_params(node)

        # --- Return annotation ---
        return_annotation = None
        if node.returns is not None:
            return_annotation = ast.unparse(node.returns)

        # --- Decorators ---
        decorators: list[str] = []
        for dec in node.decorator_list:
            try:
                decorators.append(ast.unparse(dec))
            except Exception:
                decorators.append("<unknown>")

        # --- Raised exceptions ---
        raises = self._extract_raises(node)

        return FunctionInfo(
            name=node.name,
            class_name=class_name,
            file_path=file_path,
            line_start=node.lineno,
            line_end=node.end_lineno or node.lineno,
            source_code=code_without_doc,
            docstring=raw_docstring,
            has_docstring=has_docstring,
            params=params,
            return_annotation=return_annotation,
            decorators=decorators,
            raises=raises,
        )

    def _get_source_without_docstring(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        source_lines: list[str],
    ) -> str:
        """Extract function source code with the docstring removed.

        The CodeBERT model receives the **code** and **docstring** as
        separate inputs.  This method ensures the code field contains
        only the executable logic — no docstring embedded within it.
        """
        start = node.lineno - 1          # 0-indexed inclusive
        end = node.end_lineno            # 0-indexed exclusive (past-the-end)

        # Identify the docstring AST node (first Expr whose value is a
        # string Constant).
        doc_node = None
        if (
            node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.value, str)
        ):
            doc_node = node.body[0]

        if doc_node is not None:
            doc_start = doc_node.lineno - 1   # 0-indexed inclusive
            doc_end = doc_node.end_lineno      # 0-indexed exclusive

            # Keep lines before the docstring + lines after it.
            lines = source_lines[start:doc_start] + source_lines[doc_end:end]
        else:
            lines = source_lines[start:end]

        # Strip trailing blank lines that may remain after removal.
        while lines and lines[-1].strip() == "":
            lines.pop()

        return "\n".join(line.rstrip() for line in lines)

    @staticmethod
    def _extract_params(node: ast.FunctionDef | ast.AsyncFunctionDef) -> List[str]:
        """Extract parameter names, excluding ``self`` and ``cls``."""
        params: List[str] = []

        # Positional / normal args.
        for arg in node.args.args:
            if arg.arg not in ("self", "cls"):
                params.append(arg.arg)

        # *args
        if node.args.vararg:
            params.append(f"*{node.args.vararg.arg}")

        # Keyword-only args.
        for arg in node.args.kwonlyargs:
            params.append(arg.arg)

        # **kwargs
        if node.args.kwarg:
            params.append(f"**{node.args.kwarg.arg}")

        return params

    @staticmethod
    def _extract_raises(node: ast.FunctionDef | ast.AsyncFunctionDef) -> List[str]:
        """Collect exception class names from ``raise`` statements.

        Walks the entire function body to find explicit ``raise``
        expressions and extracts the exception class name.
        """
        raises: List[str] = []

        for child in ast.walk(node):
            if not isinstance(child, ast.Raise) or child.exc is None:
                continue

            exc = child.exc

            # raise ExceptionClass(...)
            if isinstance(exc, ast.Call) and isinstance(exc.func, ast.Name):
                raises.append(exc.func.id)
            # raise ExceptionClass
            elif isinstance(exc, ast.Name):
                raises.append(exc.id)
            # raise module.ExceptionClass(...)
            elif isinstance(exc, ast.Call) and isinstance(exc.func, ast.Attribute):
                raises.append(exc.func.attr)
            # raise module.ExceptionClass
            elif isinstance(exc, ast.Attribute):
                raises.append(exc.attr)

        # Deduplicate while preserving order.
        seen: set[str] = set()
        unique: List[str] = []
        for name in raises:
            if name not in seen:
                seen.add(name)
                unique.append(name)
        return unique
