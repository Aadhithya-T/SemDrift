"""
semdrift.parser.doc_extractor — Docstring extraction and normalisation.

Extracts raw docstrings from Python functions and normalises them into
clean plain text suitable for model input.  Handles Google, NumPy,
Sphinx/reST, and plain unstructured docstring styles.

The normalised output becomes the ``docstring`` field in the JSON input
that is sent to the CodeBERT model.
"""

import re
import textwrap
from dataclasses import dataclass, field
from typing import Dict


@dataclass
class DocInfo:
    """Structured representation of a parsed docstring.

    Attributes
    ----------
    summary : str
        First sentence / paragraph of the docstring.
    params : dict[str, str]
        Mapping of parameter name → description.
    returns : str
        Description of the return value.
    raises : dict[str, str]
        Mapping of exception type → description.
    deprecation : str
        Deprecation notice, if any.
    raw : str
        The original unprocessed docstring text.
    """

    summary: str = ""
    params: Dict[str, str] = field(default_factory=dict)
    returns: str = ""
    raises: Dict[str, str] = field(default_factory=dict)
    deprecation: str = ""
    raw: str = ""


class DocExtractor:
    """Extracts and normalises Python docstrings.

    The extractor auto-detects the docstring convention (Google, NumPy,
    Sphinx, or plain) and parses it into structured components.  It also
    provides a :meth:`normalise` method that produces a clean flat text
    representation — the exact text that populates the ``docstring``
    field in the model-input JSON.

    Usage::

        extractor = DocExtractor()
        info = extractor.extract(raw_docstring)
        clean_text = extractor.normalise(info)
    """

    # ------------------------------------------------------------------
    # Compiled patterns for style detection
    # ------------------------------------------------------------------

    _GOOGLE_SECTION = re.compile(
        r"^(Args|Arguments|Parameters|Params|Returns?|Raises?|Yields?|"
        r"Examples?|Notes?|References?|Attributes?|Todo|Deprecated)\s*:",
        re.MULTILINE,
    )
    _NUMPY_SECTION = re.compile(
        r"^(Parameters|Returns?|Raises?|Yields?|See Also|Notes?|"
        r"References?|Examples?|Attributes?|Deprecated)\s*\n-{3,}",
        re.MULTILINE,
    )
    _SPHINX_TAG = re.compile(
        r"^\s*:(param|type|returns?|rtype|raises?|var|ivar|cvar)\s",
        re.MULTILINE,
    )

    # ------------------------------------------------------------------
    # Google-style parsing helpers
    # ------------------------------------------------------------------

    _GOOGLE_SECTION_BLOCK = re.compile(
        r"^(Args|Arguments|Parameters|Params|Returns?|Raises?|Deprecated)"
        r"\s*:\s*\n(.*?)(?=\n\S|\Z)",
        re.MULTILINE | re.DOTALL,
    )
    _GOOGLE_ENTRY = re.compile(
        r"^\s{2,4}(\*{0,2}\w+)\s*(?:\(.+?\))?\s*:\s*(.+?)(?=\n\s{2,4}\*{0,2}\w|\n\n|\n\S|\s*\Z)",
        re.MULTILINE | re.DOTALL,
    )

    # ------------------------------------------------------------------
    # NumPy-style parsing helpers
    # ------------------------------------------------------------------

    _NUMPY_BLOCK = re.compile(
        r"^(Parameters|Returns?|Raises?|Deprecated)\s*\n-{3,}\n(.*?)"
        r"(?=\n\S+\s*\n-{3,}|\Z)",
        re.MULTILINE | re.DOTALL,
    )
    _NUMPY_ENTRY = re.compile(
        r"^(\w+)\s*:.*?\n((?:\s{4,}.+\n?)*)",
        re.MULTILINE,
    )

    # ------------------------------------------------------------------
    # Sphinx / reST parsing helpers
    # ------------------------------------------------------------------

    _SPHINX_PARAM = re.compile(
        r":param\s+(\w+)\s*:\s*(.+?)(?=\n\s*:|$)",
        re.DOTALL,
    )
    _SPHINX_RETURNS = re.compile(
        r":returns?\s*:\s*(.+?)(?=\n\s*:|$)",
        re.DOTALL,
    )
    _SPHINX_RAISES = re.compile(
        r":raises?\s+(\w+)\s*:\s*(.+?)(?=\n\s*:|$)",
        re.DOTALL,
    )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, raw_docstring: str) -> DocInfo:
        """Parse a raw docstring into structured components.

        Auto-detects the convention and delegates to the appropriate
        style-specific parser.

        Parameters
        ----------
        raw_docstring : str
            The raw docstring text (as returned by
            ``ast.get_docstring``).

        Returns
        -------
        DocInfo
            Structured representation with summary, params, returns,
            raises, and deprecation fields populated.
        """
        if not raw_docstring or not raw_docstring.strip():
            return DocInfo(raw=raw_docstring or "")

        cleaned = textwrap.dedent(raw_docstring).strip()
        style = self._detect_style(cleaned)

        if style == "google":
            return self._parse_google(cleaned, raw_docstring)
        elif style == "numpy":
            return self._parse_numpy(cleaned, raw_docstring)
        elif style == "sphinx":
            return self._parse_sphinx(cleaned, raw_docstring)
        else:
            return self._parse_plain(cleaned, raw_docstring)

    def normalise(self, doc_info: DocInfo) -> str:
        """Produce a flat text representation of a parsed docstring.

        This is the text that goes into the ``"docstring"`` field of
        the model-input JSON.  It concatenates all structured parts
        into one readable string.

        Parameters
        ----------
        doc_info : DocInfo
            Output of :meth:`extract`.

        Returns
        -------
        str
            Clean normalised text suitable for model input.
        """
        parts: list[str] = []

        if doc_info.summary:
            parts.append(doc_info.summary)

        if doc_info.params:
            param_strs = [
                f"{name}: {desc}" for name, desc in doc_info.params.items()
            ]
            parts.append("Parameters: " + "; ".join(param_strs) + ".")

        if doc_info.returns:
            parts.append(f"Returns: {doc_info.returns}")

        if doc_info.raises:
            raise_strs = [
                f"{exc}: {desc}" for exc, desc in doc_info.raises.items()
            ]
            parts.append("Raises: " + "; ".join(raise_strs) + ".")

        if doc_info.deprecation:
            parts.append(f"Deprecated: {doc_info.deprecation}")

        return " ".join(parts).strip()

    # ------------------------------------------------------------------
    # Style detection
    # ------------------------------------------------------------------

    def _detect_style(self, docstring: str) -> str:
        """Detect the docstring convention.

        Returns one of ``"google"``, ``"numpy"``, ``"sphinx"``, or
        ``"plain"``.
        """
        # NumPy is checked first because its section headers are the
        # most distinctive (word followed by a dashed underline).
        if self._NUMPY_SECTION.search(docstring):
            return "numpy"
        if self._SPHINX_TAG.search(docstring):
            return "sphinx"
        if self._GOOGLE_SECTION.search(docstring):
            return "google"
        return "plain"

    # ------------------------------------------------------------------
    # Style-specific parsers
    # ------------------------------------------------------------------

    def _parse_google(self, cleaned: str, raw: str) -> DocInfo:
        """Parse a Google-style docstring."""
        info = DocInfo(raw=raw)

        # Summary = everything before the first section header.
        first_section = self._GOOGLE_SECTION.search(cleaned)
        if first_section:
            info.summary = cleaned[: first_section.start()].strip()
        else:
            info.summary = cleaned.strip()

        for match in self._GOOGLE_SECTION_BLOCK.finditer(cleaned):
            section_name = match.group(1).lower()
            section_body = match.group(2)

            if section_name in ("args", "arguments", "parameters", "params"):
                info.params = self._parse_entry_block(section_body)
            elif section_name.startswith("return"):
                info.returns = self._clean_text(section_body)
            elif section_name.startswith("raise"):
                info.raises = self._parse_entry_block(section_body)
            elif section_name == "deprecated":
                info.deprecation = self._clean_text(section_body)

        return info

    def _parse_numpy(self, cleaned: str, raw: str) -> DocInfo:
        """Parse a NumPy-style docstring."""
        info = DocInfo(raw=raw)

        # Summary = everything before the first section.
        first_section = self._NUMPY_SECTION.search(cleaned)
        if first_section:
            info.summary = cleaned[: first_section.start()].strip()
        else:
            info.summary = cleaned.strip()

        for match in self._NUMPY_BLOCK.finditer(cleaned):
            section_name = match.group(1).lower()
            section_body = match.group(2)

            if section_name == "parameters":
                info.params = self._parse_numpy_entries(section_body)
            elif section_name.startswith("return"):
                info.returns = self._clean_text(section_body)
            elif section_name.startswith("raise"):
                info.raises = self._parse_numpy_entries(section_body)
            elif section_name == "deprecated":
                info.deprecation = self._clean_text(section_body)

        return info

    def _parse_sphinx(self, cleaned: str, raw: str) -> DocInfo:
        """Parse a Sphinx/reST-style docstring."""
        info = DocInfo(raw=raw)

        # Summary = everything before the first :tag.
        first_tag = self._SPHINX_TAG.search(cleaned)
        if first_tag:
            info.summary = cleaned[: first_tag.start()].strip()
        else:
            info.summary = cleaned.strip()

        for match in self._SPHINX_PARAM.finditer(cleaned):
            info.params[match.group(1)] = self._clean_text(match.group(2))

        ret_match = self._SPHINX_RETURNS.search(cleaned)
        if ret_match:
            info.returns = self._clean_text(ret_match.group(1))

        for match in self._SPHINX_RAISES.finditer(cleaned):
            info.raises[match.group(1)] = self._clean_text(match.group(2))

        return info

    def _parse_plain(self, cleaned: str, raw: str) -> DocInfo:
        """Parse a plain / unstructured docstring."""
        return DocInfo(summary=cleaned.strip(), raw=raw)

    # ------------------------------------------------------------------
    # Entry-level helpers
    # ------------------------------------------------------------------

    def _parse_entry_block(self, text: str) -> dict[str, str]:
        """Parse ``name: description`` entries from a Google-style block."""
        entries: dict[str, str] = {}
        for match in self._GOOGLE_ENTRY.finditer(text):
            key = match.group(1).lstrip("*")
            entries[key] = self._clean_text(match.group(2))
        return entries

    def _parse_numpy_entries(self, text: str) -> dict[str, str]:
        """Parse entries from a NumPy-style block."""
        entries: dict[str, str] = {}
        for match in self._NUMPY_ENTRY.finditer(text):
            entries[match.group(1)] = self._clean_text(match.group(2))
        return entries

    @staticmethod
    def _clean_text(text: str) -> str:
        """Collapse whitespace and strip a text fragment."""
        return " ".join(text.split()).strip()
