"""
semdrift.parser.formatter — Model input formatter.

Combines the output of :class:`ASTParser` and :class:`DocExtractor` into
the JSON structure that the CodeBERT divergence-detection model expects.

Output format per function::

    {
        "function_id": "flask_app_py_get_send_file_max_age_001",
        "code": "def get_send_file_max_age(self, filename: ...) -> ...:\\n    ...",
        "docstring": "Used by send_file to determine the max_age cache value ..."
    }

The ``code`` field contains the function source **without** its docstring.
The ``docstring`` field contains the normalised documentation text.
The ``function_id`` is a deterministic, human-readable identifier.
"""

import json
import os
import re
from typing import List, Optional

from semdrift.parser.ast_parser import FunctionInfo
from semdrift.parser.doc_extractor import DocExtractor


class ModelInputFormatter:
    """Formats parsed function data into model-ready JSON records.

    This is the final stage of the parser pipeline.  It takes a list of
    :class:`FunctionInfo` objects (produced by :class:`ASTParser`), runs
    each docstring through the :class:`DocExtractor` for normalisation,
    and outputs a list of dictionaries matching the schema expected by
    the CodeBERT model.

    Parameters
    ----------
    base_path : str or None
        If provided, file paths in ``function_id`` are made relative to
        this directory.  Useful when scanning a full repository.
    include_undocumented : bool
        Whether to include functions that have no docstring.
        Default False — only documented functions are relevant for
        divergence detection.
    normalise_docstring : bool
        If True, the ``docstring`` field contains normalised flat text
        (structured parts concatenated).  If False, it contains the raw
        docstring as written in source.  Default True.
    """

    def __init__(
        self,
        base_path: Optional[str] = None,
        include_undocumented: bool = False,
        normalise_docstring: bool = True,
    ) -> None:
        self.base_path = os.path.abspath(base_path) if base_path else None
        self.include_undocumented = include_undocumented
        self.normalise_docstring = normalise_docstring
        self._doc_extractor = DocExtractor()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def format(self, functions: List[FunctionInfo]) -> List[dict]:
        """Convert a list of :class:`FunctionInfo` into model-input dicts.

        Parameters
        ----------
        functions : list[FunctionInfo]
            Output from :class:`ASTParser`.

        Returns
        -------
        list[dict]
            Each dict has keys ``function_id``, ``code``, ``docstring``.
        """
        records: list[dict] = []
        # Per-file counter to disambiguate functions with the same name.
        file_counters: dict[str, int] = {}

        for func in functions:
            # Skip undocumented functions unless explicitly requested.
            if not func.has_docstring and not self.include_undocumented:
                continue

            # Build the unique function_id.
            counter_key = func.file_path
            file_counters[counter_key] = file_counters.get(counter_key, 0) + 1
            counter = file_counters[counter_key]
            function_id = self._build_function_id(func, counter)

            # Prepare the docstring text.
            if func.has_docstring:
                if self.normalise_docstring:
                    doc_info = self._doc_extractor.extract(func.docstring)
                    docstring_text = self._doc_extractor.normalise(doc_info)
                else:
                    docstring_text = func.docstring.strip()
            else:
                docstring_text = ""

            records.append({
                "function_id": function_id,
                "code": func.source_code,
                "docstring": docstring_text,
            })

        return records

    def format_to_json(
        self,
        functions: List[FunctionInfo],
        output_path: Optional[str] = None,
        indent: int = 2,
    ) -> str:
        """Format to JSON and optionally write to a file.

        Parameters
        ----------
        functions : list[FunctionInfo]
            Output from :class:`ASTParser`.
        output_path : str or None
            If provided, write the JSON to this file path.
        indent : int
            JSON indentation level.  Default 2.

        Returns
        -------
        str
            The JSON string.
        """
        records = self.format(functions)
        json_str = json.dumps(records, indent=indent, ensure_ascii=False)

        if output_path:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as fh:
                fh.write(json_str)

        return json_str

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_function_id(self, func: FunctionInfo, counter: int) -> str:
        """Generate a unique, human-readable function identifier.

        Format::

            {file_path_stem}_{ClassName}_{function_name}_{counter:03d}

        Examples::

            flask_app_py_Flask_get_send_file_max_age_001
            utils_py_parse_config_001
        """
        # Make file path relative if base_path is set.
        if self.base_path and func.file_path.startswith(self.base_path):
            rel_path = os.path.relpath(func.file_path, self.base_path)
        else:
            rel_path = os.path.basename(func.file_path)

        # Convert path separators and dots to underscores.
        file_stem = re.sub(r"[/\\.]", "_", rel_path)

        parts = [file_stem]
        if func.class_name:
            parts.append(func.class_name)
        parts.append(func.name)
        parts.append(f"{counter:03d}")

        return "_".join(parts)
