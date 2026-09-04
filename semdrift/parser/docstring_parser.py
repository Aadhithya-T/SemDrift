"""
semdrift.parser.docstring_parser — Comprehensive multi-style docstring parser.

Parses Python and Java docstrings into a unified ``DocContract`` representation.
Supports five docstring conventions:

  - **Google** style  (``Args:``, ``Returns:``, ``Raises:``)
  - **NumPy** style   (``Parameters\\n----------``)
  - **Sphinx** style  (``":param x:"``, ``":returns:"``, ``":raises:"``).
  - **Javadoc** style (``@param``, ``@return``, ``@throws``)
  - **Plain** text    (summary only, no structured sections)

This module replaces ``DocExtractor`` for the V2 pipeline.
The old ``DocExtractor`` remains available for legacy Phase 1 usage.

Usage::

    parser = DocstringParser()
    contract = parser.parse("My summary.\\n\\nArgs:\\n    x: The input.")
    print(contract.summary)            # "My summary."
    print(contract.param_descriptions) # {"x": "The input."}
    print(contract.style)              # "google"
"""

from __future__ import annotations

import re
import textwrap
from typing import Dict, List

from semdrift.parser.contracts import DocContract


class DocstringParser:
    """Comprehensive multi-style docstring parser.

    Auto-detects the docstring convention and parses structured sections
    (parameters, returns, raises, examples, deprecation) into a unified
    ``DocContract`` representation.

    Supports Google, NumPy, Sphinx, Javadoc, and plain text styles.
    """

    # ------------------------------------------------------------------
    # Style detection patterns
    # ------------------------------------------------------------------

    _GOOGLE_SECTION = re.compile(
        r"^\s*(Args|Arguments|Parameters|Params|Returns?|Raises?|Yields?|"
        r"Examples?|Notes?|References?|Attributes?|Todo|Deprecated)\s*:",
        re.MULTILINE,
    )
    _NUMPY_SECTION = re.compile(
        r"^\s*(Parameters|Returns?|Raises?|Yields?|See Also|Notes?|"
        r"References?|Examples?|Attributes?|Deprecated)\s*\n\s*-{3,}",
        re.MULTILINE,
    )
    _SPHINX_TAG = re.compile(
        r"^\s*:(param|type|returns?|rtype|raises?|var|ivar|cvar)\s",
        re.MULTILINE,
    )
    _JAVADOC_TAG = re.compile(
        r"^\s*\*?\s*@(param|return|returns|throws|exception|deprecated|see|since|author|version)\s",
        re.MULTILINE,
    )

    # ------------------------------------------------------------------
    # Google-style parsing helpers
    # ------------------------------------------------------------------

    _GOOGLE_SECTION_SPLIT = re.compile(
        r"^\s*(Args|Arguments|Parameters|Params|Returns?|Raises?|"
        r"Yields?|Examples?|Deprecated)\s*:\s*\n",
        re.MULTILINE,
    )
    _GOOGLE_ENTRY = re.compile(
        r"^\s{2,}(\*{0,2}\w+)\s*(?:\(.+?\))?\s*:\s*(.+?)(?=\n\s{2,}\*{0,2}\w|\n\n|\n\S|\s*\Z)",
        re.MULTILINE | re.DOTALL,
    )

    # ------------------------------------------------------------------
    # NumPy-style parsing helpers
    # ------------------------------------------------------------------

    _NUMPY_BLOCK = re.compile(
        r"^\s*(Parameters|Returns?|Raises?|Deprecated)\s*\n\s*-{3,}\n(.*?)"
        r"(?=\n\s*\S+\s*\n\s*-{3,}|\Z)",
        re.MULTILINE | re.DOTALL,
    )
    _NUMPY_ENTRY = re.compile(
        r"^\s*(\w+)\s*:.*?\n((?:\s+(?!\s*\w+\s*:).+\n?)*)",
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
    # Javadoc parsing helpers
    # ------------------------------------------------------------------

    _JAVADOC_PARAM = re.compile(
        r"@param\s+(\w+)\s+(.+?)(?=\n\s*\*?\s*@|\Z)",
        re.DOTALL,
    )
    _JAVADOC_RETURN = re.compile(
        r"@returns?\s+(.+?)(?=\n\s*\*?\s*@|\Z)",
        re.DOTALL,
    )
    _JAVADOC_THROWS = re.compile(
        r"@(?:throws|exception)\s+(\w+)\s+(.+?)(?=\n\s*\*?\s*@|\Z)",
        re.DOTALL,
    )
    _JAVADOC_DEPRECATED = re.compile(
        r"@deprecated\s+(.+?)(?=\n\s*\*?\s*@|\Z)",
        re.DOTALL,
    )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse(self, raw_docstring: str) -> DocContract:
        """Parse a raw docstring into a structured ``DocContract``.

        Auto-detects the convention and delegates to the appropriate
        style-specific parser.

        Parameters
        ----------
        raw_docstring : str
            The raw docstring text.

        Returns
        -------
        DocContract
            Structured representation with all fields populated.
        """
        if not raw_docstring or not raw_docstring.strip():
            return DocContract(raw=raw_docstring or "")

        cleaned = textwrap.dedent(raw_docstring).strip()
        style = self._detect_style(cleaned)

        if style == "javadoc":
            return self._parse_javadoc(cleaned, raw_docstring)
        elif style == "numpy":
            return self._parse_numpy(cleaned, raw_docstring)
        elif style == "sphinx":
            return self._parse_sphinx(cleaned, raw_docstring)
        elif style == "google":
            return self._parse_google(cleaned, raw_docstring)
        else:
            return self._parse_plain(cleaned, raw_docstring)

    def normalise(self, doc: DocContract) -> str:
        """Produce a structured flat text representation.

        Outputs sections with ``[SUMMARY]``, ``[PARAMETERS]``,
        ``[RETURNS]``, ``[RAISES]`` headers — suitable for model input
        that preserves the full semantic contract.

        Parameters
        ----------
        doc : DocContract
            Output of :meth:`parse`.

        Returns
        -------
        str
            Normalised text with section markers.
        """
        parts: List[str] = []

        if doc.summary:
            parts.append(f"[SUMMARY] {doc.summary}")

        if doc.param_descriptions:
            param_lines = [
                f"{name}: {desc}" for name, desc in doc.param_descriptions.items()
            ]
            parts.append("[PARAMETERS] " + "; ".join(param_lines))

        if doc.return_description:
            parts.append(f"[RETURNS] {doc.return_description}")

        if doc.raises_descriptions:
            raise_lines = [
                f"{exc}: {desc}" for exc, desc in doc.raises_descriptions.items()
            ]
            parts.append("[RAISES] " + "; ".join(raise_lines))

        if doc.deprecation:
            parts.append(f"[DEPRECATED] {doc.deprecation}")

        return " ".join(parts).strip()

    def normalise_legacy(self, doc: DocContract) -> str:
        """Produce a flat text representation matching the old ``DocExtractor.normalise`` format.

        This is backwards-compatible with the Phase 1 model input format
        that doesn't use section markers.

        Parameters
        ----------
        doc : DocContract
            Output of :meth:`parse`.

        Returns
        -------
        str
            Clean flat text without section markers.
        """
        parts: List[str] = []

        if doc.summary:
            parts.append(doc.summary)

        if doc.param_descriptions:
            param_strs = [
                f"{name}: {desc}" for name, desc in doc.param_descriptions.items()
            ]
            parts.append("Parameters: " + "; ".join(param_strs) + ".")

        if doc.return_description:
            parts.append(f"Returns: {doc.return_description}")

        if doc.raises_descriptions:
            raise_strs = [
                f"{exc}: {desc}" for exc, desc in doc.raises_descriptions.items()
            ]
            parts.append("Raises: " + "; ".join(raise_strs) + ".")

        if doc.deprecation:
            parts.append(f"Deprecated: {doc.deprecation}")

        return " ".join(parts).strip()

    # ------------------------------------------------------------------
    # Style detection
    # ------------------------------------------------------------------

    def _detect_style(self, docstring: str) -> str:
        """Detect the docstring convention.

        Returns one of ``"google"``, ``"numpy"``, ``"sphinx"``,
        ``"javadoc"``, or ``"plain"``.
        """
        if self._JAVADOC_TAG.search(docstring):
            return "javadoc"
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

    def _parse_google(self, cleaned: str, raw: str) -> DocContract:
        """Parse a Google-style docstring."""
        doc = DocContract(raw=raw, style="google")

        # Summary = everything before the first section header.
        first_section = self._GOOGLE_SECTION.search(cleaned)
        if first_section:
            doc.summary = cleaned[:first_section.start()].strip()
        else:
            doc.summary = cleaned.strip()

        # Split into sections.
        sections = self._split_google_sections(cleaned)

        for section_name, section_body in sections.items():
            name_lower = section_name.lower()
            if name_lower in ("args", "arguments", "parameters", "params"):
                doc.param_descriptions = self._parse_google_entries(section_body)
            elif name_lower.startswith("return") or name_lower.startswith("yield"):
                doc.return_description = self._clean_text(section_body)
            elif name_lower.startswith("raise"):
                doc.raises_descriptions = self._parse_google_entries(section_body)
            elif name_lower.startswith("example"):
                doc.examples = [section_body.strip()]
            elif name_lower == "deprecated":
                doc.deprecation = self._clean_text(section_body)

        return doc

    def _parse_numpy(self, cleaned: str, raw: str) -> DocContract:
        """Parse a NumPy-style docstring."""
        doc = DocContract(raw=raw, style="numpy")

        first_section = self._NUMPY_SECTION.search(cleaned)
        if first_section:
            doc.summary = cleaned[:first_section.start()].strip()
        else:
            doc.summary = cleaned.strip()

        for match in self._NUMPY_BLOCK.finditer(cleaned):
            section_name = match.group(1).lower()
            section_body = match.group(2)

            if section_name == "parameters":
                doc.param_descriptions = self._parse_numpy_entries(section_body)
            elif section_name.startswith("return"):
                doc.return_description = self._clean_text(section_body)
            elif section_name.startswith("raise"):
                doc.raises_descriptions = self._parse_numpy_entries(section_body)
            elif section_name == "deprecated":
                doc.deprecation = self._clean_text(section_body)

        return doc

    def _parse_sphinx(self, cleaned: str, raw: str) -> DocContract:
        """Parse a Sphinx/reST-style docstring."""
        doc = DocContract(raw=raw, style="sphinx")

        first_tag = self._SPHINX_TAG.search(cleaned)
        if first_tag:
            doc.summary = cleaned[:first_tag.start()].strip()
        else:
            doc.summary = cleaned.strip()

        for match in self._SPHINX_PARAM.finditer(cleaned):
            doc.param_descriptions[match.group(1)] = self._clean_text(match.group(2))

        ret_match = self._SPHINX_RETURNS.search(cleaned)
        if ret_match:
            doc.return_description = self._clean_text(ret_match.group(1))

        for match in self._SPHINX_RAISES.finditer(cleaned):
            doc.raises_descriptions[match.group(1)] = self._clean_text(match.group(2))

        return doc

    def _parse_javadoc(self, cleaned: str, raw: str) -> DocContract:
        """Parse a Javadoc-style docstring.

        Handles ``@param``, ``@return``/``@returns``, ``@throws``/``@exception``,
        and ``@deprecated`` tags.  Strips leading ``*`` characters from
        Javadoc comment blocks.
        """
        doc = DocContract(raw=raw, style="javadoc")

        # Strip Javadoc comment markers: leading /** ... */  and  * at line starts.
        stripped = cleaned
        if stripped.startswith("/**"):
            stripped = stripped[3:]
        if stripped.endswith("*/"):
            stripped = stripped[:-2]
        stripped = re.sub(r"^\s*\*\s?", "", stripped, flags=re.MULTILINE).strip()

        # Summary = everything before the first @tag.
        first_tag = self._JAVADOC_TAG.search(stripped)
        if first_tag:
            doc.summary = stripped[:first_tag.start()].strip()
        else:
            doc.summary = stripped.strip()

        for match in self._JAVADOC_PARAM.finditer(stripped):
            doc.param_descriptions[match.group(1)] = self._clean_text(match.group(2))

        ret_match = self._JAVADOC_RETURN.search(stripped)
        if ret_match:
            doc.return_description = self._clean_text(ret_match.group(1))

        for match in self._JAVADOC_THROWS.finditer(stripped):
            doc.raises_descriptions[match.group(1)] = self._clean_text(match.group(2))

        dep_match = self._JAVADOC_DEPRECATED.search(stripped)
        if dep_match:
            doc.deprecation = self._clean_text(dep_match.group(1))

        return doc

    def _parse_plain(self, cleaned: str, raw: str) -> DocContract:
        """Parse a plain / unstructured docstring."""
        return DocContract(summary=cleaned.strip(), raw=raw, style="plain")

    # ------------------------------------------------------------------
    # Entry-level helpers
    # ------------------------------------------------------------------

    def _split_google_sections(self, text: str) -> Dict[str, str]:
        """Split a Google-style docstring into named sections."""
        sections: Dict[str, str] = {}
        matches = list(self._GOOGLE_SECTION_SPLIT.finditer(text))

        for i, match in enumerate(matches):
            section_name = match.group(1)
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            sections[section_name] = text[start:end]

        return sections

    def _parse_google_entries(self, text: str) -> Dict[str, str]:
        """Parse ``name: description`` entries from a Google-style block."""
        entries: Dict[str, str] = {}
        for match in self._GOOGLE_ENTRY.finditer(text):
            key = match.group(1).lstrip("*")
            entries[key] = self._clean_text(match.group(2))
        return entries

    def _parse_numpy_entries(self, text: str) -> Dict[str, str]:
        """Parse entries from a NumPy-style block."""
        entries: Dict[str, str] = {}
        for match in self._NUMPY_ENTRY.finditer(text):
            entries[match.group(1)] = self._clean_text(match.group(2))
        return entries

    @staticmethod
    def _clean_text(text: str) -> str:
        """Collapse whitespace and strip a text fragment."""
        return " ".join(text.split()).strip()
