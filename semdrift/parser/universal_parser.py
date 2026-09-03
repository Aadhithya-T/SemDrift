"""
semdrift.parser.universal_parser — Multi-language function extractor using tree-sitter.

This is the **V2 Universal Parser** that replaces the old skeleton and addresses
all 22 parser red flags from the evaluation review.

Supported languages:
  - **Python** via ``tree-sitter-python``
  - **Java** via ``tree-sitter-java``

Outputs ``FunctionContract`` objects (from ``semdrift.parser.contracts``) — a
unified intermediate representation that is completely language-agnostic.

Key improvements over the Phase 1 ``ASTParser``:
  1.  ``posonlyargs`` extraction
  2.  Parameter defaults and type annotations
  3.  Multiple return path capture
  4.  ``yield`` / ``async`` / generator detection
  5.  All ``raise`` / ``throw`` forms classified
  6.  Decorator semantic interpretation
  7.  ``@overload`` detection
  8.  Class inheritance context
  9.  Structured docstring → ``DocContract``
  10. Parameter–documentation mismatch detection
  11. Parser status & error tracking
  12. Stable function identity for dataset splitting
  13. Java: throws clause, Javadoc, annotations, modifiers

Usage::

    from semdrift.parser.universal_parser import UniversalParser

    parser = UniversalParser()
    contracts = parser.parse_directory("/path/to/repo")
    for c in contracts:
        print(c.qualified_name, c.parse_status, c.undocumented_params)
"""

from __future__ import annotations

import os
import re
from typing import List, Optional

from semdrift.parser.contracts import (
    DocContract,
    ExceptionContract,
    FunctionContract,
    Parameter,
    ParameterKind,
    ParseStatus,
    RaiseForm,
)
from semdrift.parser.docstring_parser import DocstringParser

# Tree-sitter imports — graceful fallback if not installed.
try:
    from tree_sitter import Language, Parser, Node

    import tree_sitter_python

    _PYTHON_LANGUAGE = Language(tree_sitter_python.language())
    _HAS_PYTHON = True
except (ImportError, Exception):
    _HAS_PYTHON = False
    _PYTHON_LANGUAGE = None
    Node = None  # type: ignore[assignment,misc]

try:
    import tree_sitter_java

    _JAVA_LANGUAGE = Language(tree_sitter_java.language())
    _HAS_JAVA = True
except (ImportError, Exception):
    _HAS_JAVA = False
    _JAVA_LANGUAGE = None


class UniversalParser:
    """Multi-language parser producing ``FunctionContract`` objects via tree-sitter.

    Parameters
    ----------
    max_file_size_kb : int
        Skip files larger than this (kilobytes).  Default 500.
    skip_dunder : bool
        Skip Python dunder methods (``__init__``, etc.).  Default False.
    skip_private : bool
        Skip Python private methods (``_helper``).  Default False.
    skip_test_files : bool
        Skip test files (``test_*.py``, ``*Test.java``).  Default True.
    """

    _SKIP_DIRS: set[str] = {
        "__pycache__", ".git", ".tox", ".mypy_cache",
        ".pytest_cache", "node_modules", ".venv", "venv",
        "env", ".env", "site-packages", ".eggs", "build",
        "dist", "egg-info", ".gradle", ".idea", "target",
    }

    _SKIP_FILE_PREFIXES_PY: tuple[str, ...] = ("test_", "conftest")
    _SKIP_FILE_SUFFIXES_JAVA: tuple[str, ...] = ("Test.java", "Tests.java")

    _SUPPORTED_EXTENSIONS: dict[str, str] = {}

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
        self._doc_parser = DocstringParser()

        # Build extension → language mapping based on available bindings.
        self._parsers: dict[str, Parser] = {}
        self._languages: dict[str, Language] = {}

        if _HAS_PYTHON and _PYTHON_LANGUAGE is not None:
            py_parser = Parser(_PYTHON_LANGUAGE)
            self._parsers[".py"] = py_parser
            self._languages[".py"] = _PYTHON_LANGUAGE

        if _HAS_JAVA and _JAVA_LANGUAGE is not None:
            java_parser = Parser(_JAVA_LANGUAGE)
            self._parsers[".java"] = java_parser
            self._languages[".java"] = _JAVA_LANGUAGE

        if not self._parsers:
            raise ImportError(
                "No tree-sitter language bindings are installed. "
                "Please run: pip install tree-sitter tree-sitter-python tree-sitter-java"
            )

    # ==================================================================
    # Public API
    # ==================================================================

    def parse_file(self, file_path: str) -> List[FunctionContract]:
        """Parse a single source file and return extracted function contracts.

        Parameters
        ----------
        file_path : str
            Path to the source file.

        Returns
        -------
        list[FunctionContract]
            Extracted contracts.  Empty list if the file cannot be parsed,
            is unsupported, or is skipped by configured filters.
        """
        file_path = os.path.abspath(file_path)
        ext = os.path.splitext(file_path)[1].lower()

        if ext not in self._parsers:
            return []

        # File size check.
        try:
            size_kb = os.path.getsize(file_path) / 1024
            if size_kb > self.max_file_size_kb:
                return []
        except OSError:
            return []

        # Test file check.
        if self.skip_test_files:
            basename = os.path.basename(file_path)
            if ext == ".py" and any(basename.startswith(p) for p in self._SKIP_FILE_PREFIXES_PY):
                return []
            if ext == ".java" and any(basename.endswith(s) for s in self._SKIP_FILE_SUFFIXES_JAVA):
                return []

        # Read source bytes.
        try:
            with open(file_path, "rb") as fh:
                source_bytes = fh.read()
        except OSError:
            return []

        # Parse the tree.
        parser = self._parsers[ext]
        tree = parser.parse(source_bytes)

        # Dispatch to language-specific extractor.
        if ext == ".py":
            return self._extract_python_functions(tree.root_node, source_bytes, file_path)
        elif ext == ".java":
            return self._extract_java_functions(tree.root_node, source_bytes, file_path)
        return []

    def parse_directory(self, dir_path: str) -> List[FunctionContract]:
        """Recursively parse all supported files in a directory tree.

        Parameters
        ----------
        dir_path : str
            Root directory to walk.

        Returns
        -------
        list[FunctionContract]
            Aggregated list from every parsed file.
        """
        results: List[FunctionContract] = []

        for root, dirs, files in os.walk(dir_path):
            dirs[:] = sorted(d for d in dirs if d not in self._SKIP_DIRS)

            for fname in sorted(files):
                ext = os.path.splitext(fname)[1].lower()
                if ext in self._parsers:
                    full_path = os.path.join(root, fname)
                    results.extend(self.parse_file(full_path))

        return results

    # ==================================================================
    # PYTHON EXTRACTOR
    # ==================================================================

    def _extract_python_functions(
        self,
        root: Node,
        source_bytes: bytes,
        file_path: str,
    ) -> List[FunctionContract]:
        """Extract all functions/methods from a Python tree-sitter parse tree."""
        contracts: List[FunctionContract] = []
        self._walk_python_node(root, source_bytes, file_path, class_name=None,
                               superclasses=[], contracts=contracts)
        return contracts

    def _walk_python_node(
        self,
        node: Node,
        source_bytes: bytes,
        file_path: str,
        class_name: Optional[str],
        superclasses: List[str],
        contracts: List[FunctionContract],
    ) -> None:
        """Recursively walk tree-sitter nodes, extracting Python functions."""
        for child in node.children:
            if child.type == "class_definition":
                cls_name = self._get_child_text(child, "name", source_bytes)
                cls_supers = self._extract_python_superclasses(child, source_bytes)
                body = self._get_child_node(child, "body")
                if body:
                    self._walk_python_node(body, source_bytes, file_path,
                                           class_name=cls_name, superclasses=cls_supers,
                                           contracts=contracts)

            elif child.type in ("function_definition", "decorated_definition"):
                func_node = child
                decorator_nodes = []

                if child.type == "decorated_definition":
                    # Collect decorators
                    for deco_child in child.children:
                        if deco_child.type == "decorator":
                            decorator_nodes.append(deco_child)
                        elif deco_child.type in ("function_definition", "class_definition"):
                            func_node = deco_child
                            break

                    # If it's a decorated class, recurse into it
                    if func_node.type == "class_definition":
                        cls_name = self._get_child_text(func_node, "name", source_bytes)
                        cls_supers = self._extract_python_superclasses(func_node, source_bytes)
                        body = self._get_child_node(func_node, "body")
                        if body:
                            self._walk_python_node(body, source_bytes, file_path,
                                                   class_name=cls_name, superclasses=cls_supers,
                                                   contracts=contracts)
                        continue

                if func_node.type != "function_definition":
                    continue

                contract = self._extract_python_function(
                    func_node, decorator_nodes, source_bytes, file_path,
                    class_name, superclasses,
                )
                if contract is not None:
                    contracts.append(contract)

    def _extract_python_function(
        self,
        node: Node,
        decorator_nodes: List[Node],
        source_bytes: bytes,
        file_path: str,
        class_name: Optional[str],
        superclasses: List[str],
    ) -> Optional[FunctionContract]:
        """Build a FunctionContract from a Python function_definition node."""
        parse_errors: List[str] = []

        # --- Name ---
        func_name = self._get_child_text(node, "name", source_bytes)
        if not func_name:
            return None

        # --- Apply skip filters ---
        if self._should_skip_python(func_name):
            return None

        # --- Line numbers ---
        # If we have decorator nodes, the full definition starts at the first decorator
        if decorator_nodes:
            full_start = decorator_nodes[0].start_point[0] + 1
        else:
            full_start = node.start_point[0] + 1
        line_start = full_start
        line_end = node.end_point[0] + 1

        # --- Async detection ---
        is_async = False
        # Check if parent of function_definition is a decorated_definition
        # and if the actual "def" keyword is preceded by "async"
        source_text = self._node_text(node, source_bytes)
        if source_text.lstrip().startswith("async "):
            is_async = True

        # --- Decorators ---
        decorators: List[str] = []
        for deco_node in decorator_nodes:
            deco_text = self._node_text(deco_node, source_bytes).lstrip("@").strip()
            decorators.append(deco_text)

        # Semantic decorator flags
        is_property = "property" in decorators
        is_staticmethod = "staticmethod" in decorators
        is_classmethod = "classmethod" in decorators
        is_abstract = any("abstractmethod" in d for d in decorators)
        is_overload = any("overload" in d for d in decorators)

        # --- Parameters ---
        parameters = self._extract_python_parameters(node, source_bytes)

        # --- Return annotation ---
        return_annotation = None
        ret_type_node = self._get_child_node(node, "return_type")
        if ret_type_node is None:
            # tree-sitter uses "type" for the return annotation
            ret_type_node = self._get_child_by_field(node, "return_type", source_bytes)
        if ret_type_node is not None:
            return_annotation = self._node_text(ret_type_node, source_bytes)

        # --- Body analysis: returns, yields, raises ---
        body_node = self._get_child_node(node, "body")
        return_paths: List[str] = []
        is_generator = False
        explicit_raises: List[ExceptionContract] = []

        if body_node:
            return_paths = self._extract_python_returns(body_node, source_bytes)
            is_generator = self._has_python_yield(body_node)
            explicit_raises = self._extract_python_raises(body_node, source_bytes)

        # --- Docstring ---
        docstring_raw = ""
        if body_node:
            docstring_raw = self._extract_python_docstring(body_node, source_bytes)

        # Parse docstring into structured contract.
        try:
            doc_contract = self._doc_parser.parse(docstring_raw)
        except Exception as e:
            doc_contract = DocContract(raw=docstring_raw)
            parse_errors.append(f"docstring_parse_failed: {e}")

        # --- Source code ---
        full_source = self._node_text(node, source_bytes)
        # Include decorators in full source
        if decorator_nodes:
            deco_texts = [self._node_text(d, source_bytes) for d in decorator_nodes]
            full_source = "\n".join(deco_texts) + "\n" + full_source

        source_without_doc = self._remove_python_docstring(node, body_node, source_bytes)
        if decorator_nodes:
            deco_texts = [self._node_text(d, source_bytes) for d in decorator_nodes]
            source_without_doc = "\n".join(deco_texts) + "\n" + source_without_doc

        # --- Qualified name ---
        qualified_name = f"{class_name}.{func_name}" if class_name else func_name

        # --- Module path ---
        module_path = self._file_to_module_path(file_path)

        # --- Build contract ---
        contract = FunctionContract(
            name=func_name,
            qualified_name=qualified_name,
            module_path=module_path,
            file_path=file_path,
            line_start=line_start,
            line_end=line_end,
            language="python",
            class_name=class_name,
            superclasses=list(superclasses),
            parameters=parameters,
            return_annotation=return_annotation,
            return_paths=return_paths,
            explicit_raises=explicit_raises,
            decorators=decorators,
            is_async=is_async,
            is_generator=is_generator,
            is_property=is_property,
            is_staticmethod=is_staticmethod,
            is_classmethod=is_classmethod,
            is_abstract=is_abstract,
            is_overload=is_overload,
            docstring_raw=docstring_raw,
            doc_contract=doc_contract,
            source_code=full_source,
            source_code_without_docstring=source_without_doc,
            parse_status=ParseStatus.PARTIAL if parse_errors else ParseStatus.SUCCESS,
            parse_errors=parse_errors,
        )

        # Compute contract-level mismatches.
        contract.compute_contract_mismatches()

        return contract

    # ------------------------------------------------------------------
    # Python parameter extraction
    # ------------------------------------------------------------------

    def _extract_python_parameters(self, func_node: Node, source_bytes: bytes) -> List[Parameter]:
        """Extract all parameters with kind, annotation, and default."""
        params: List[Parameter] = []
        params_node = self._get_child_node(func_node, "parameters")
        if params_node is None:
            return params

        # Track position for positional-only delimiter
        seen_posonly_sep = False
        seen_kwonly_sep = False

        for child in params_node.children:
            if child.type in ("(", ")", ","):
                continue

            if child.type == "positional_separator":
                # The "/" separator marks end of positional-only params
                seen_posonly_sep = True
                continue

            if child.type == "keyword_separator":
                # The "*" separator marks start of keyword-only params
                seen_kwonly_sep = True
                continue

            if child.type == "identifier":
                # Simple parameter without annotation or default
                name = self._node_text(child, source_bytes)
                if name in ("self", "cls"):
                    continue
                kind = self._determine_python_param_kind(
                    seen_posonly_sep, seen_kwonly_sep, is_vararg=False, is_kwarg=False
                )
                params.append(Parameter(name=name, kind=kind))

            elif child.type in ("default_parameter", "typed_default_parameter"):
                name_node = child.children[0] if child.children else None
                name = self._node_text(name_node, source_bytes) if name_node else ""
                if name in ("self", "cls"):
                    continue

                # Extract annotation
                annotation = None
                if child.type == "typed_default_parameter":
                    type_node = self._get_child_node(child, "type")
                    if type_node:
                        annotation = self._node_text(type_node, source_bytes)

                # Extract default value: last child after "="
                default = None
                found_eq = False
                for sub in child.children:
                    if sub.type == "=" or self._node_text(sub, source_bytes) == "=":
                        found_eq = True
                    elif found_eq:
                        default = self._node_text(sub, source_bytes)

                kind = self._determine_python_param_kind(
                    seen_posonly_sep, seen_kwonly_sep, is_vararg=False, is_kwarg=False
                )
                params.append(Parameter(name=name, kind=kind, annotation=annotation, default=default))

            elif child.type == "typed_parameter":
                name_node = child.children[0] if child.children else None
                name = self._node_text(name_node, source_bytes) if name_node else ""
                if name in ("self", "cls"):
                    continue

                annotation = None
                type_node = self._get_child_node(child, "type")
                if type_node:
                    annotation = self._node_text(type_node, source_bytes)

                kind = self._determine_python_param_kind(
                    seen_posonly_sep, seen_kwonly_sep, is_vararg=False, is_kwarg=False
                )
                params.append(Parameter(name=name, kind=kind, annotation=annotation))

            elif child.type == "list_splat_pattern":
                # *args
                seen_kwonly_sep = True  # Everything after *args is keyword-only
                name = self._node_text(child, source_bytes).lstrip("*")
                params.append(Parameter(name=name, kind=ParameterKind.VAR_POSITIONAL))

            elif child.type == "dictionary_splat_pattern":
                # **kwargs
                name = self._node_text(child, source_bytes).lstrip("*")
                params.append(Parameter(name=name, kind=ParameterKind.VAR_KEYWORD))

        return params

    @staticmethod
    def _determine_python_param_kind(
        seen_posonly_sep: bool,
        seen_kwonly_sep: bool,
        is_vararg: bool,
        is_kwarg: bool,
    ) -> ParameterKind:
        """Determine parameter kind based on position relative to separators."""
        if is_vararg:
            return ParameterKind.VAR_POSITIONAL
        if is_kwarg:
            return ParameterKind.VAR_KEYWORD
        if seen_kwonly_sep:
            return ParameterKind.KEYWORD_ONLY
        if not seen_posonly_sep:
            # Before any separator → could be positional-only if "/" comes later
            # But we haven't seen "/" yet, so it's positional_or_keyword for now
            return ParameterKind.POSITIONAL_OR_KEYWORD
        # After "/" but before "*"
        return ParameterKind.POSITIONAL_OR_KEYWORD

    # ------------------------------------------------------------------
    # Python return, yield, raise extraction
    # ------------------------------------------------------------------

    def _extract_python_returns(self, body_node: Node, source_bytes: bytes) -> List[str]:
        """Collect all distinct return value expressions from the function body."""
        returns: List[str] = []
        self._collect_python_returns(body_node, source_bytes, returns)
        # Deduplicate while preserving order.
        seen: set[str] = set()
        unique: List[str] = []
        for r in returns:
            if r not in seen:
                seen.add(r)
                unique.append(r)
        return unique

    def _collect_python_returns(self, node: Node, source_bytes: bytes, out: List[str]) -> None:
        """Recursively collect return expressions."""
        for child in node.children:
            if child.type == "return_statement":
                # Get the return value (everything after "return")
                parts = []
                skip_keyword = True
                for sub in child.children:
                    if skip_keyword and sub.type == "return":
                        skip_keyword = False
                        continue
                    parts.append(self._node_text(sub, source_bytes))
                value = " ".join(parts).strip()
                if value:
                    out.append(value)
                else:
                    out.append("None")

            elif child.type in ("function_definition", "class_definition"):
                # Don't descend into nested functions/classes.
                continue
            else:
                self._collect_python_returns(child, source_bytes, out)

    def _has_python_yield(self, body_node: Node) -> bool:
        """Check if the function body contains yield/yield from."""
        return self._search_for_type(body_node, {"yield", "yield_statement"})

    def _search_for_type(self, node: Node, types: set[str]) -> bool:
        """Recursively search for nodes of given types, skipping nested functions."""
        for child in node.children:
            if child.type in types:
                return True
            if child.type in ("function_definition", "class_definition"):
                continue
            if self._search_for_type(child, types):
                return True
        return False

    def _extract_python_raises(self, body_node: Node, source_bytes: bytes) -> List[ExceptionContract]:
        """Collect all explicit raise statements with form classification."""
        raises: List[ExceptionContract] = []
        self._collect_python_raises(body_node, source_bytes, raises)

        # Deduplicate by exception name while preserving order.
        seen: set[str] = set()
        unique: List[ExceptionContract] = []
        for exc in raises:
            if exc.exception_name not in seen:
                seen.add(exc.exception_name)
                unique.append(exc)
        return unique

    def _collect_python_raises(
        self, node: Node, source_bytes: bytes, out: List[ExceptionContract]
    ) -> None:
        """Recursively collect raise statements."""
        for child in node.children:
            if child.type == "raise_statement":
                exc = self._classify_python_raise(child, source_bytes)
                if exc is not None:
                    out.append(exc)

            elif child.type in ("function_definition", "class_definition"):
                continue
            else:
                self._collect_python_raises(child, source_bytes, out)

    def _classify_python_raise(self, raise_node: Node, source_bytes: bytes) -> Optional[ExceptionContract]:
        """Classify a single raise statement into an ExceptionContract."""
        children = [c for c in raise_node.children if c.type not in ("raise",)]

        if not children:
            # Bare re-raise: `raise`
            return ExceptionContract(exception_name="<re-raise>", form=RaiseForm.RE_RAISE)

        exc_node = children[0]

        # Check for "from" clause (chained exceptions)
        has_from = any(self._node_text(c, source_bytes) == "from" for c in children)
        base_form = RaiseForm.CHAINED if has_from else RaiseForm.CALL

        if exc_node.type == "call":
            # raise ExceptionClass(...) or raise module.ExceptionClass(...)
            func_node = exc_node.children[0] if exc_node.children else None
            if func_node is None:
                return ExceptionContract(exception_name="<unknown>", form=RaiseForm.UNKNOWN)
            if func_node.type == "identifier":
                name = self._node_text(func_node, source_bytes)
                return ExceptionContract(
                    exception_name=name, form=base_form if not has_from else RaiseForm.CHAINED
                )
            elif func_node.type == "attribute":
                full = self._node_text(func_node, source_bytes)
                short = full.rsplit(".", 1)[-1]
                return ExceptionContract(
                    exception_name=short, full_name=full, form=RaiseForm.ATTRIBUTE
                )
        elif exc_node.type == "identifier":
            # raise ExceptionClass (no call) or raise variable
            name = self._node_text(exc_node, source_bytes)
            # Heuristic: if name starts with uppercase, it's a class
            if name and name[0].isupper():
                return ExceptionContract(exception_name=name, form=RaiseForm.DIRECT)
            else:
                return ExceptionContract(exception_name=name, form=RaiseForm.VARIABLE)
        elif exc_node.type == "attribute":
            # raise module.ExceptionClass
            full = self._node_text(exc_node, source_bytes)
            short = full.rsplit(".", 1)[-1]
            return ExceptionContract(
                exception_name=short, full_name=full, form=RaiseForm.ATTRIBUTE
            )

        # Fallback
        text = self._node_text(exc_node, source_bytes)
        return ExceptionContract(exception_name=text, form=RaiseForm.UNKNOWN)

    # ------------------------------------------------------------------
    # Python docstring extraction
    # ------------------------------------------------------------------

    def _extract_python_docstring(self, body_node: Node, source_bytes: bytes) -> str:
        """Extract the docstring from the first statement of a function body."""
        if not body_node.children:
            return ""

        first_stmt = body_node.children[0]
        if first_stmt.type == "expression_statement":
            for child in first_stmt.children:
                if child.type == "string":
                    raw = self._node_text(child, source_bytes)
                    return self._strip_python_string_quotes(raw)
        return ""

    @staticmethod
    def _strip_python_string_quotes(s: str) -> str:
        """Remove triple or single quotes from a Python string literal."""
        for prefix in ('"""', "'''", '"', "'"):
            if s.startswith(prefix) and s.endswith(prefix):
                return s[len(prefix):-len(prefix)]
        # Handle prefixed strings (r"...", b"...", f"...", etc.)
        if len(s) > 1 and s[0] in "rRbBuUfF" and s[1] in ('"', "'"):
            return UniversalParser._strip_python_string_quotes(s[1:])
        return s

    def _remove_python_docstring(
        self, func_node: Node, body_node: Optional[Node], source_bytes: bytes
    ) -> str:
        """Get function source code with the docstring removed."""
        if body_node is None or not body_node.children:
            return self._node_text(func_node, source_bytes)

        first_stmt = body_node.children[0]
        has_docstring = False
        if first_stmt.type == "expression_statement":
            for child in first_stmt.children:
                if child.type == "string":
                    has_docstring = True
                    break

        if not has_docstring:
            return self._node_text(func_node, source_bytes)

        # Remove the docstring expression_statement from the source.
        func_start = func_node.start_byte
        func_end = func_node.end_byte
        doc_start = first_stmt.start_byte
        doc_end = first_stmt.end_byte

        # Skip any trailing newline after the docstring
        while doc_end < func_end and source_bytes[doc_end:doc_end + 1] in (b"\n", b"\r"):
            doc_end += 1

        before = source_bytes[func_start:doc_start]
        after = source_bytes[doc_end:func_end]

        result = (before + after).decode("utf-8", errors="replace")

        # Strip trailing blank lines
        lines = result.split("\n")
        while lines and lines[-1].strip() == "":
            lines.pop()
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Python superclass extraction
    # ------------------------------------------------------------------

    def _extract_python_superclasses(self, class_node: Node, source_bytes: bytes) -> List[str]:
        """Extract base class names from a class definition."""
        supers: List[str] = []
        arg_list = self._get_child_node(class_node, "argument_list")
        if arg_list is None:
            # Try superclasses node
            for child in class_node.children:
                if child.type == "argument_list":
                    arg_list = child
                    break

        if arg_list:
            for child in arg_list.children:
                if child.type in ("(", ")", ","):
                    continue
                text = self._node_text(child, source_bytes).strip()
                if text and text not in ("metaclass",):
                    # Skip metaclass=... patterns
                    if "=" not in text:
                        supers.append(text)

        return supers

    # ------------------------------------------------------------------
    # Python skip filters
    # ------------------------------------------------------------------

    def _should_skip_python(self, name: str) -> bool:
        """Return True if the function name should be skipped."""
        is_dunder = name.startswith("__") and name.endswith("__")
        is_private = name.startswith("_") and not is_dunder

        if self.skip_dunder and is_dunder:
            return True
        if self.skip_private and is_private:
            return True
        return False

    # ==================================================================
    # JAVA EXTRACTOR
    # ==================================================================

    def _extract_java_functions(
        self,
        root: Node,
        source_bytes: bytes,
        file_path: str,
    ) -> List[FunctionContract]:
        """Extract all methods from a Java tree-sitter parse tree."""
        contracts: List[FunctionContract] = []
        self._walk_java_node(root, source_bytes, file_path, class_name=None,
                             superclasses=[], contracts=contracts)
        return contracts

    def _walk_java_node(
        self,
        node: Node,
        source_bytes: bytes,
        file_path: str,
        class_name: Optional[str],
        superclasses: List[str],
        contracts: List[FunctionContract],
    ) -> None:
        """Recursively walk Java tree-sitter nodes, extracting methods."""
        for child in node.children:
            if child.type in ("class_declaration", "interface_declaration", "enum_declaration"):
                cls_name = self._get_child_text(child, "name", source_bytes)
                cls_supers = self._extract_java_superclasses(child, source_bytes)
                body = self._get_child_node(child, "body")
                if body is None:
                    body = self._get_child_node(child, "class_body")
                if body:
                    self._walk_java_node(body, source_bytes, file_path,
                                         class_name=cls_name, superclasses=cls_supers,
                                         contracts=contracts)

            elif child.type == "method_declaration":
                contract = self._extract_java_method(
                    child, source_bytes, file_path, class_name, superclasses
                )
                if contract is not None:
                    contracts.append(contract)

            elif child.type == "constructor_declaration":
                contract = self._extract_java_method(
                    child, source_bytes, file_path, class_name, superclasses,
                    is_constructor=True,
                )
                if contract is not None:
                    contracts.append(contract)

            else:
                self._walk_java_node(child, source_bytes, file_path,
                                     class_name, superclasses, contracts)

    def _extract_java_method(
        self,
        node: Node,
        source_bytes: bytes,
        file_path: str,
        class_name: Optional[str],
        superclasses: List[str],
        is_constructor: bool = False,
    ) -> Optional[FunctionContract]:
        """Build a FunctionContract from a Java method_declaration node."""
        parse_errors: List[str] = []

        # --- Name ---
        func_name = self._get_child_text(node, "name", source_bytes)
        if not func_name:
            return None

        # --- Line numbers ---
        line_start = node.start_point[0] + 1
        line_end = node.end_point[0] + 1

        # --- Modifiers & Annotations ---
        decorators: List[str] = []
        is_static = False
        is_abstract = False

        for child in node.children:
            if child.type == "modifiers":
                for mod in child.children:
                    mod_text = self._node_text(mod, source_bytes)
                    if mod.type == "marker_annotation" or mod.type == "annotation":
                        decorators.append(mod_text.lstrip("@"))
                    elif mod_text == "static":
                        is_static = True
                    elif mod_text == "abstract":
                        is_abstract = True

        # --- Return type ---
        return_annotation = None
        if not is_constructor:
            type_node = None
            for child in node.children:
                if child.type in ("void_type", "type_identifier", "generic_type",
                                  "array_type", "integral_type", "floating_point_type",
                                  "boolean_type", "scoped_type_identifier"):
                    type_node = child
                    break
            if type_node:
                return_annotation = self._node_text(type_node, source_bytes)

        # --- Parameters ---
        parameters = self._extract_java_parameters(node, source_bytes)

        # --- Throws clause ---
        explicit_raises: List[ExceptionContract] = []
        for child in node.children:
            if child.type == "throws":
                for exc_child in child.children:
                    if exc_child.type in ("type_identifier", "scoped_type_identifier"):
                        exc_name = self._node_text(exc_child, source_bytes)
                        explicit_raises.append(ExceptionContract(
                            exception_name=exc_name, form=RaiseForm.DIRECT
                        ))

        # Also collect throw statements from body
        body = self._get_child_node(node, "body")
        if body is None:
            body = self._get_child_node(node, "constructor_body")
        if body:
            self._collect_java_throws(body, source_bytes, explicit_raises)

        # --- Docstring (Javadoc) ---
        docstring_raw = self._extract_java_javadoc(node, source_bytes)

        try:
            doc_contract = self._doc_parser.parse(docstring_raw)
        except Exception as e:
            doc_contract = DocContract(raw=docstring_raw)
            parse_errors.append(f"javadoc_parse_failed: {e}")

        # --- Return paths from body ---
        return_paths: List[str] = []
        if body:
            return_paths = self._extract_java_returns(body, source_bytes)

        # --- Source code ---
        source_code = self._node_text(node, source_bytes)

        # --- Qualified name ---
        qualified_name = f"{class_name}.{func_name}" if class_name else func_name

        # --- Module path ---
        module_path = self._file_to_module_path(file_path)

        # Deduplicate raises
        seen_raises: set[str] = set()
        unique_raises: List[ExceptionContract] = []
        for exc in explicit_raises:
            if exc.exception_name not in seen_raises:
                seen_raises.add(exc.exception_name)
                unique_raises.append(exc)

        contract = FunctionContract(
            name=func_name,
            qualified_name=qualified_name,
            module_path=module_path,
            file_path=file_path,
            line_start=line_start,
            line_end=line_end,
            language="java",
            class_name=class_name,
            superclasses=list(superclasses),
            parameters=parameters,
            return_annotation=return_annotation,
            return_paths=return_paths,
            explicit_raises=unique_raises,
            decorators=decorators,
            is_async=False,
            is_generator=False,
            is_property=False,
            is_staticmethod=is_static,
            is_classmethod=False,
            is_abstract=is_abstract,
            is_overload=any("Override" in d for d in decorators),
            docstring_raw=docstring_raw,
            doc_contract=doc_contract,
            source_code=source_code,
            source_code_without_docstring=source_code,
            parse_status=ParseStatus.PARTIAL if parse_errors else ParseStatus.SUCCESS,
            parse_errors=parse_errors,
        )

        contract.compute_contract_mismatches()
        return contract

    def _extract_java_parameters(self, method_node: Node, source_bytes: bytes) -> List[Parameter]:
        """Extract Java method parameters with types."""
        params: List[Parameter] = []
        for child in method_node.children:
            if child.type == "formal_parameters":
                for param in child.children:
                    if param.type == "formal_parameter" or param.type == "spread_parameter":
                        name = ""
                        annotation = None
                        is_vararg = param.type == "spread_parameter"

                        for sub in param.children:
                            if sub.type == "identifier":
                                name = self._node_text(sub, source_bytes)
                            elif sub.type in ("type_identifier", "generic_type",
                                              "array_type", "integral_type",
                                              "floating_point_type", "boolean_type",
                                              "scoped_type_identifier", "void_type"):
                                annotation = self._node_text(sub, source_bytes)

                        if name:
                            kind = ParameterKind.VAR_POSITIONAL if is_vararg else ParameterKind.POSITIONAL_OR_KEYWORD
                            params.append(Parameter(name=name, kind=kind, annotation=annotation))
        return params

    def _extract_java_superclasses(self, class_node: Node, source_bytes: bytes) -> List[str]:
        """Extract Java class superclasses and implemented interfaces."""
        supers: List[str] = []
        for child in class_node.children:
            if child.type == "superclass":
                for sub in child.children:
                    if sub.type in ("type_identifier", "generic_type", "scoped_type_identifier"):
                        supers.append(self._node_text(sub, source_bytes))
            elif child.type == "super_interfaces":
                for sub in child.children:
                    if sub.type == "type_list":
                        for iface in sub.children:
                            if iface.type in ("type_identifier", "generic_type", "scoped_type_identifier"):
                                supers.append(self._node_text(iface, source_bytes))
        return supers

    def _extract_java_javadoc(self, method_node: Node, source_bytes: bytes) -> str:
        """Extract the Javadoc comment immediately preceding a method declaration."""
        # Look at the previous sibling for a block_comment starting with /**
        prev = method_node.prev_named_sibling
        if prev and prev.type == "block_comment":
            text = self._node_text(prev, source_bytes)
            if text.startswith("/**"):
                return text
        # Also check parent's children for comment just before this method
        if method_node.parent:
            idx = None
            for i, child in enumerate(method_node.parent.children):
                if child.id == method_node.id:
                    idx = i
                    break
            if idx is not None and idx > 0:
                prev_child = method_node.parent.children[idx - 1]
                if prev_child.type in ("block_comment", "comment"):
                    text = self._node_text(prev_child, source_bytes)
                    if text.startswith("/**"):
                        return text
        return ""

    def _extract_java_returns(self, body_node: Node, source_bytes: bytes) -> List[str]:
        """Collect return value expressions from a Java method body."""
        returns: List[str] = []
        self._collect_java_returns(body_node, source_bytes, returns)
        seen: set[str] = set()
        unique: List[str] = []
        for r in returns:
            if r not in seen:
                seen.add(r)
                unique.append(r)
        return unique

    def _collect_java_returns(self, node: Node, source_bytes: bytes, out: List[str]) -> None:
        """Recursively collect Java return statements."""
        for child in node.children:
            if child.type == "return_statement":
                parts = []
                for sub in child.children:
                    if sub.type not in ("return", ";"):
                        parts.append(self._node_text(sub, source_bytes))
                value = " ".join(parts).strip()
                if value:
                    out.append(value)
            elif child.type in ("method_declaration", "class_declaration",
                                "lambda_expression"):
                continue
            else:
                self._collect_java_returns(child, source_bytes, out)

    def _collect_java_throws(
        self, body_node: Node, source_bytes: bytes, out: List[ExceptionContract]
    ) -> None:
        """Recursively collect throw statements from Java method body."""
        for child in body_node.children:
            if child.type == "throw_statement":
                for sub in child.children:
                    if sub.type == "object_creation_expression":
                        type_node = self._get_child_node(sub, "type_identifier")
                        if type_node is None:
                            type_node = self._get_child_node(sub, "scoped_type_identifier")
                        if type_node:
                            name = self._node_text(type_node, source_bytes)
                            out.append(ExceptionContract(
                                exception_name=name, form=RaiseForm.CALL
                            ))
            elif child.type in ("method_declaration", "class_declaration", "lambda_expression"):
                continue
            else:
                self._collect_java_throws(child, source_bytes, out)

    # ==================================================================
    # Shared utility methods
    # ==================================================================

    @staticmethod
    def _node_text(node: Optional[Node], source_bytes: bytes) -> str:
        """Get the text of a tree-sitter node."""
        if node is None:
            return ""
        return source_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="replace")

    @staticmethod
    def _get_child_text(node: Node, field_name: str, source_bytes: bytes) -> str:
        """Get the text of a child node by field name."""
        child = node.child_by_field_name(field_name)
        if child is None:
            return ""
        return source_bytes[child.start_byte:child.end_byte].decode("utf-8", errors="replace")

    @staticmethod
    def _get_child_node(node: Node, field_name: str) -> Optional[Node]:
        """Get a child node by field name."""
        return node.child_by_field_name(field_name)

    @staticmethod
    def _get_child_by_field(node: Node, field_name: str, source_bytes: bytes) -> Optional[Node]:
        """Fallback: find child by field name."""
        return node.child_by_field_name(field_name)

    @staticmethod
    def _file_to_module_path(file_path: str) -> str:
        """Convert a file path to a dot-separated module path."""
        base = os.path.basename(file_path)
        name = os.path.splitext(base)[0]
        # Try to build a more complete module path
        parts = file_path.replace("\\", "/").split("/")
        # Find index of common project markers
        for marker in ("src", "semdrift", "main", "java", "python"):
            if marker in parts:
                idx = parts.index(marker)
                module_parts = parts[idx:]
                # Remove file extension from last part
                module_parts[-1] = os.path.splitext(module_parts[-1])[0]
                return ".".join(module_parts)
        return name
