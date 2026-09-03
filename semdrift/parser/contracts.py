"""
semdrift.parser.contracts — Unified Intermediate Representation for parsed code.

Defines the ``FunctionContract`` dataclass and its supporting types.
Every language-specific extractor (Python, Java) outputs this same
representation, allowing the rest of the SemDrift pipeline to be
completely language-agnostic.

This module addresses parser red flags #1–#14 from the evaluation by
providing structured fields for:
  - All parameter kinds (positional-only, keyword-only, etc.)
  - Default values and type annotations
  - Multiple return paths
  - Explicit raises with AST form classification
  - Decorator semantic interpretation
  - async / generator / property / overload detection
  - Class inheritance context
  - Structured docstring contracts
  - Parser status and error tracking
  - Contract-level mismatch detection
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


# ======================================================================
# Enums
# ======================================================================

class ParameterKind(Enum):
    """Classification of function parameter binding behavior.

    Mirrors Python's ``inspect.Parameter.kind`` but is language-agnostic.
    """

    POSITIONAL_ONLY = "positional_only"
    POSITIONAL_OR_KEYWORD = "positional_or_keyword"
    VAR_POSITIONAL = "var_positional"          # *args
    KEYWORD_ONLY = "keyword_only"
    VAR_KEYWORD = "var_keyword"                # **kwargs


class ParseStatus(Enum):
    """Outcome of a parse attempt.

    ``PARTIAL`` means some information was extracted but errors occurred
    (e.g. docstring parsing failed).  ``FAILED`` means the function
    could not be extracted at all.
    """

    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"


class RaiseForm(Enum):
    """How an exception was raised in source code.

    Distinguishes syntactic forms so downstream consumers know whether
    the exception class name is reliable or merely a best-effort guess.
    """

    DIRECT = "direct"            # raise ExceptionClass
    CALL = "call"                # raise ExceptionClass(...)
    ATTRIBUTE = "attribute"      # raise module.ExceptionClass or module.ExceptionClass(...)
    VARIABLE = "variable"        # raise some_variable
    RE_RAISE = "re_raise"        # raise (bare re-raise)
    CHAINED = "chained"          # raise X from Y
    UNKNOWN = "unknown"


# ======================================================================
# Dataclasses
# ======================================================================

@dataclass
class Parameter:
    """A single function parameter with full contract information.

    Attributes
    ----------
    name : str
        Parameter name as it appears in source code.
    kind : ParameterKind
        Binding behavior (positional-only, keyword-only, etc.).
    annotation : str or None
        Type annotation as source text (e.g. ``"list[str]"``).
    default : str or None
        Default value as source text (e.g. ``"None"``, ``"10"``).
    doc_description : str or None
        Description from the docstring, if the parameter was documented.
    """

    name: str
    kind: ParameterKind = ParameterKind.POSITIONAL_OR_KEYWORD
    annotation: Optional[str] = None
    default: Optional[str] = None
    doc_description: Optional[str] = None


@dataclass
class ExceptionContract:
    """A single explicitly raised exception.

    Attributes
    ----------
    exception_name : str
        Exception class name (e.g. ``"ValueError"``).
    full_name : str or None
        Fully qualified name if available (e.g. ``"errors.ValueError"``).
    doc_description : str or None
        Description from the docstring, if documented.
    form : RaiseForm
        The syntactic form of the ``raise`` statement.
    """

    exception_name: str
    full_name: Optional[str] = None
    doc_description: Optional[str] = None
    form: RaiseForm = RaiseForm.CALL


@dataclass
class DocContract:
    """Structured representation of a parsed docstring.

    Supports Google, NumPy, Sphinx, Javadoc, and plain styles.
    All fields are populated regardless of the original style.

    Attributes
    ----------
    raw : str
        The original unprocessed docstring text.
    summary : str
        First sentence or paragraph — the high-level description.
    style : str
        Detected docstring convention: ``"google"``, ``"numpy"``,
        ``"sphinx"``, ``"javadoc"``, or ``"plain"``.
    param_descriptions : dict[str, str]
        Mapping of parameter name → description text.
    return_description : str
        Description of the return value.
    raises_descriptions : dict[str, str]
        Mapping of exception type → description text.
    examples : list[str]
        Code examples found in the docstring.
    deprecation : str
        Deprecation notice, if any.
    """

    raw: str = ""
    summary: str = ""
    style: str = "plain"
    param_descriptions: Dict[str, str] = field(default_factory=dict)
    return_description: str = ""
    raises_descriptions: Dict[str, str] = field(default_factory=dict)
    examples: List[str] = field(default_factory=list)
    deprecation: str = ""


@dataclass
class FunctionContract:
    """The master intermediate representation for a parsed function or method.

    Every language-specific extractor (Python tree-sitter, Java tree-sitter)
    outputs instances of this class.  The rest of SemDrift never needs to
    understand language-specific AST nodes.

    Attributes
    ----------
    name : str
        Function or method name.
    qualified_name : str
        Fully qualified name (e.g. ``"MyClass.my_method"``).
    module_path : str
        Dot-separated module path derived from the file system
        (e.g. ``"semdrift.parser.contracts"``).
    file_path : str
        Absolute path to the source file.
    line_start : int
        1-indexed start line of the function definition.
    line_end : int
        1-indexed end line of the function definition.
    language : str
        Source language: ``"python"`` or ``"java"``.
    class_name : str or None
        Enclosing class name, if this is a method.
    superclasses : list[str]
        Base classes / implemented interfaces of the enclosing class.
    parameters : list[Parameter]
        All parameters with kind, annotation, default, and doc info.
    return_annotation : str or None
        Return type annotation as source text.
    return_paths : list[str]
        All distinct return value expressions found in the function body.
    explicit_raises : list[ExceptionContract]
        All explicitly raised exceptions with form classification.
    decorators : list[str]
        Decorator names as source text.
    is_async : bool
        Whether this is an ``async def`` / ``async`` method.
    is_generator : bool
        Whether the function contains ``yield`` / ``yield from``.
    is_property : bool
        Whether the function is decorated with ``@property``.
    is_staticmethod : bool
        Whether the function is decorated with ``@staticmethod``.
    is_classmethod : bool
        Whether the function is decorated with ``@classmethod``.
    is_abstract : bool
        Whether the function is decorated with ``@abstractmethod``.
    is_overload : bool
        Whether the function is decorated with ``@overload``.
    docstring_raw : str
        Raw docstring text as written in source.
    doc_contract : DocContract
        Structured docstring parsed into components.
    source_code : str
        Full function source code.
    source_code_without_docstring : str
        Function source code with the docstring removed.
    parse_status : ParseStatus
        Overall parse outcome.
    parse_errors : list[str]
        Human-readable descriptions of any errors encountered.
    undocumented_params : list[str]
        Parameters present in code but missing from documentation.
    phantom_doc_params : list[str]
        Parameters documented but not present in the code signature.
    """

    # --- Identity ---
    name: str = ""
    qualified_name: str = ""
    module_path: str = ""
    file_path: str = ""
    line_start: int = 0
    line_end: int = 0
    language: str = "python"

    # --- Class context ---
    class_name: Optional[str] = None
    superclasses: List[str] = field(default_factory=list)

    # --- Parameters ---
    parameters: List[Parameter] = field(default_factory=list)

    # --- Returns ---
    return_annotation: Optional[str] = None
    return_paths: List[str] = field(default_factory=list)

    # --- Exceptions ---
    explicit_raises: List[ExceptionContract] = field(default_factory=list)

    # --- Modifiers / Decorators ---
    decorators: List[str] = field(default_factory=list)
    is_async: bool = False
    is_generator: bool = False
    is_property: bool = False
    is_staticmethod: bool = False
    is_classmethod: bool = False
    is_abstract: bool = False
    is_overload: bool = False

    # --- Documentation ---
    docstring_raw: str = ""
    doc_contract: DocContract = field(default_factory=DocContract)

    # --- Source ---
    source_code: str = ""
    source_code_without_docstring: str = ""

    # --- Parse status ---
    parse_status: ParseStatus = ParseStatus.SUCCESS
    parse_errors: List[str] = field(default_factory=list)

    # --- Contract mismatch detection ---
    undocumented_params: List[str] = field(default_factory=list)
    phantom_doc_params: List[str] = field(default_factory=list)

    @property
    def has_docstring(self) -> bool:
        """Whether the function has a non-empty docstring."""
        return bool(self.docstring_raw and self.docstring_raw.strip())

    @property
    def function_identity(self) -> str:
        """Stable, unique identity string for dataset splitting.

        Format: ``file_path::qualified_name::L{start}-{end}``

        Example: ``semdrift/parser/contracts.py::FunctionContract.has_docstring::L248-250``
        """
        return f"{self.file_path}::{self.qualified_name}::L{self.line_start}-{self.line_end}"

    def compute_contract_mismatches(self) -> None:
        """Detect parameter-documentation mismatches.

        Populates ``undocumented_params`` and ``phantom_doc_params``
        by comparing code parameters against docstring parameter descriptions.
        """
        code_param_names = {
            p.name for p in self.parameters
            if p.kind not in (ParameterKind.VAR_POSITIONAL, ParameterKind.VAR_KEYWORD)
        }
        doc_param_names = set(self.doc_contract.param_descriptions.keys())

        self.undocumented_params = sorted(code_param_names - doc_param_names)
        self.phantom_doc_params = sorted(doc_param_names - code_param_names)

        # Also attach doc descriptions to matching parameters.
        for param in self.parameters:
            if param.name in self.doc_contract.param_descriptions:
                param.doc_description = self.doc_contract.param_descriptions[param.name]

        # Attach raise descriptions.
        for exc in self.explicit_raises:
            if exc.exception_name in self.doc_contract.raises_descriptions:
                exc.doc_description = self.doc_contract.raises_descriptions[exc.exception_name]
