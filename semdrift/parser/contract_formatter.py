"""
semdrift.parser.contract_formatter — Formats FunctionContract into model-ready JSON.

Converts ``FunctionContract`` objects (from the V2 Universal Parser) into the
standard JSON schema expected by the CodeBERT model:

.. code-block:: json

    {
      "function_id": "semdrift_parser_contracts_py::FunctionContract.has_docstring::L248-250",
      "code": "def has_docstring(self) -> bool: ...",
      "docstring": "[SUMMARY] Whether the function has a docstring. [PARAMETERS] ..."
    }

The key difference from the legacy ``ModelInputFormatter`` is that the docstring
field now uses structured section markers (``[SUMMARY]``, ``[PARAMETERS]``,
``[RETURNS]``, ``[RAISES]``) to preserve the full semantic contract.
"""

from __future__ import annotations

import json
import os
import re
from typing import List, Optional

from semdrift.parser.contracts import FunctionContract
from semdrift.parser.docstring_parser import DocstringParser


class ContractFormatter:
    """Formats ``FunctionContract`` objects into CodeBERT model-ready dicts.

    Parameters
    ----------
    base_path : str or None
        If provided, file paths in ``function_id`` are made relative to
        this directory.
    include_undocumented : bool
        Whether to include functions that have no docstring.
        Default False.
    use_structured_docstring : bool
        If True, the docstring field uses ``[SUMMARY]``, ``[PARAMETERS]``,
        etc. section markers.  If False, produces legacy flat text.
        Default True.
    """

    def __init__(
        self,
        base_path: Optional[str] = None,
        include_undocumented: bool = False,
        use_structured_docstring: bool = True,
    ) -> None:
        self.base_path = os.path.normpath(os.path.abspath(base_path)) if base_path else None
        self.include_undocumented = include_undocumented
        self.use_structured_docstring = use_structured_docstring
        self._doc_parser = DocstringParser()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def format(self, contracts: List[FunctionContract]) -> List[dict]:
        """Convert a list of ``FunctionContract`` into model-input dicts.

        Parameters
        ----------
        contracts : list[FunctionContract]
            Output from ``UniversalParser``.

        Returns
        -------
        list[dict]
            Each dict has keys ``function_id``, ``code``, ``docstring``,
            plus optional metadata keys.
        """
        records: list[dict] = []

        for contract in contracts:
            if not contract.has_docstring and not self.include_undocumented:
                continue

            function_id = self._build_function_id(contract)

            # Prepare docstring text.
            if contract.has_docstring:
                if self.use_structured_docstring:
                    docstring_text = self._doc_parser.normalise(contract.doc_contract)
                else:
                    docstring_text = self._doc_parser.normalise_legacy(contract.doc_contract)
            else:
                docstring_text = ""

            records.append({
                "function_id": function_id,
                "code": contract.source_code_without_docstring,
                "docstring": docstring_text,
                # --- Extended metadata ---
                "language": contract.language,
                "file_path": contract.file_path,
                "line_start": contract.line_start,
                "line_end": contract.line_end,
                "function_name": contract.name,
                "qualified_name": contract.qualified_name,
                "class_name": contract.class_name,
                "is_async": contract.is_async,
                "is_generator": contract.is_generator,
                "parse_status": contract.parse_status.value,
                "undocumented_params": contract.undocumented_params,
                "phantom_doc_params": contract.phantom_doc_params,
                "raw_docstring": contract.docstring_raw,
            })

        return records

    def format_minimal(self, contracts: List[FunctionContract]) -> List[dict]:
        """Produce minimal output matching the legacy 3-key schema.

        Returns dicts with only ``function_id``, ``code``, ``docstring``.
        """
        records: list[dict] = []
        for contract in contracts:
            if not contract.has_docstring and not self.include_undocumented:
                continue

            function_id = self._build_function_id(contract)

            if contract.has_docstring:
                if self.use_structured_docstring:
                    docstring_text = self._doc_parser.normalise(contract.doc_contract)
                else:
                    docstring_text = self._doc_parser.normalise_legacy(contract.doc_contract)
            else:
                docstring_text = ""

            records.append({
                "function_id": function_id,
                "code": contract.source_code_without_docstring,
                "docstring": docstring_text,
            })

        return records

    def format_to_json(
        self,
        contracts: List[FunctionContract],
        output_path: Optional[str] = None,
        indent: int = 2,
        minimal: bool = False,
    ) -> str:
        """Format to JSON string and optionally write to a file.

        Parameters
        ----------
        contracts : list[FunctionContract]
            Output from ``UniversalParser``.
        output_path : str or None
            If specified, write JSON to this file.
        indent : int
            JSON indentation level.  Default 2.
        minimal : bool
            If True, use the 3-key schema.  Default False.

        Returns
        -------
        str
            The JSON string.
        """
        if minimal:
            records = self.format_minimal(contracts)
        else:
            records = self.format(contracts)

        json_str = json.dumps(records, indent=indent, ensure_ascii=False)

        if output_path:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as fh:
                fh.write(json_str)

        return json_str

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_function_id(self, contract: FunctionContract) -> str:
        """Generate a stable, unique function identifier.

        Uses the qualified source location format for reproducible
        dataset splitting::

            {relative_file_path}::{qualified_name}::L{start}-{end}

        Examples::

            semdrift/parser/contracts.py::FunctionContract.has_docstring::L248-250
            app/services.py::UserService.fetch_user::L42-67
        """
        file_path = contract.file_path

        # Make path relative to base_path if available.
        if self.base_path:
            norm_path = os.path.normpath(file_path)
            try:
                rel_path = os.path.relpath(norm_path, self.base_path)
            except ValueError:
                rel_path = os.path.basename(norm_path)
        else:
            rel_path = os.path.basename(file_path)

        # Normalize separators.
        rel_path = rel_path.replace("\\", "/")

        return f"{rel_path}::{contract.qualified_name}::L{contract.line_start}-{contract.line_end}"
