"""Robust parser and normalizer for Python docstrings.

Extracts structured sections:
- [SUMMARY]: One-line or brief summary of the function purpose.
- [PARAMETERS]: Named parameter descriptions and type contracts.
- [RETURNS]: Documented return value and return type contracts.
- [RAISES]: Documented exception types and error conditions.

Supports Google, Sphinx, NumPy, and standard free-text docstring conventions.
"""

import re
from typing import Dict, List, Optional, Tuple


# Regex patterns for Sphinx style
RE_SPHINX_PARAM = re.compile(
    r":(?:param|arg|argument)\s+(?:([a-zA-Z_]\w*)\s+)?([a-zA-Z_]\w*)\s*:\s*(.*?)(?=\n\s*:|\Z)",
    re.IGNORECASE | re.DOTALL,
)
RE_SPHINX_RETURN = re.compile(
    r":(?:return|returns|rtype)\s*:\s*(.*?)(?=\n\s*:|\Z)",
    re.IGNORECASE | re.DOTALL,
)
RE_SPHINX_RAISES = re.compile(
    r":(?:raises?|exception)\s+([a-zA-Z_]\w*)\s*:\s*(.*?)(?=\n\s*:|\Z)",
    re.IGNORECASE | re.DOTALL,
)

# Regex patterns for Google style sections
RE_GOOGLE_ARGS = re.compile(
    r"(?:Args|Arguments|Parameters|Params)\s*:\s*\n(.*?)(?=\n\s*(?:Returns?|Raises?|Yields?|Note|Notes|Example|Examples|Warns?):|\Z)",
    re.IGNORECASE | re.DOTALL,
)
RE_GOOGLE_RETURNS = re.compile(
    r"(?:Returns?|Return)\s*:\s*\n(.*?)(?=\n\s*(?:Raises?|Yields?|Note|Notes|Example|Examples|Warns?):|\Z)",
    re.IGNORECASE | re.DOTALL,
)
RE_GOOGLE_RAISES = re.compile(
    r"(?:Raises?|Raise)\s*:\s*\n(.*?)(?=\n\s*(?:Returns?|Yields?|Note|Notes|Example|Examples|Warns?):|\Z)",
    re.IGNORECASE | re.DOTALL,
)

# Regex patterns for NumPy style sections
RE_NUMPY_ARGS = re.compile(
    r"Parameters\s*\n\s*-+\s*\n(.*?)(?=\n\s*(?:Returns?|Raises?|Yields?|See Also|Notes?|Examples?)\s*\n\s*-+|\Z)",
    re.IGNORECASE | re.DOTALL,
)
RE_NUMPY_RETURNS = re.compile(
    r"Returns\s*\n\s*-+\s*\n(.*?)(?=\n\s*(?:Raises?|Yields?|See Also|Notes?|Examples?)\s*\n\s*-+|\Z)",
    re.IGNORECASE | re.DOTALL,
)
RE_NUMPY_RAISES = re.compile(
    r"Raises\s*\n\s*-+\s*\n(.*?)(?=\n\s*(?:Returns?|Yields?|See Also|Notes?|Examples?)\s*\n\s*-+|\Z)",
    re.IGNORECASE | re.DOTALL,
)


def extract_summary(docstring: str) -> str:
    """Extract the first sentence or first paragraph as summary."""
    if not docstring:
        return ""
    
    clean = docstring.strip()
    # Split by empty lines or section headers
    lines = clean.splitlines()
    summary_lines = []
    for line in lines:
        s = line.strip()
        if not s:
            if summary_lines:
                break
            continue
        # Check if line starts a section
        if re.match(r"^(?:Args|Parameters|Returns|Raises|Notes|Examples|:param|:return|:raises)", s, re.IGNORECASE):
            break
        summary_lines.append(s)
    
    return " ".join(summary_lines)


def parse_parameters(docstring: str) -> Dict[str, str]:
    """Extract parameter names and their documented descriptions."""
    params: Dict[str, str] = {}
    if not docstring:
        return params

    # 1. Check Sphinx style: :param type name: desc OR :param name: desc
    for match in RE_SPHINX_PARAM.finditer(docstring):
        ptype, pname, pdesc = match.groups()
        param_name = pname or ptype
        if param_name:
            params[param_name] = pdesc.strip().replace("\n", " ")

    # 2. Check Google style: name (type): desc OR name: desc
    google_match = RE_GOOGLE_ARGS.search(docstring)
    if google_match:
        section_text = google_match.group(1)
        curr_param = None
        curr_desc = []
        for line in section_text.splitlines():
            m = re.match(r"^\s*([a-zA-Z_]\w*)\s*(?:\([^)]+\))?\s*:\s*(.*)", line)
            if m:
                if curr_param:
                    params[curr_param] = " ".join(curr_desc).strip()
                curr_param = m.group(1)
                curr_desc = [m.group(2)]
            elif curr_param and line.startswith("    "):
                curr_desc.append(line.strip())
        if curr_param:
            params[curr_param] = " ".join(curr_desc).strip()

    # 3. Check NumPy style: name : type\n    desc
    numpy_match = RE_NUMPY_ARGS.search(docstring)
    if numpy_match:
        section_text = numpy_match.group(1)
        curr_param = None
        curr_desc = []
        for line in section_text.splitlines():
            m = re.match(r"^\s*([a-zA-Z_]\w*)\s*:\s*(.*)", line)
            if m:
                if curr_param:
                    params[curr_param] = " ".join(curr_desc).strip()
                curr_param = m.group(1)
                curr_desc = [m.group(2)]
            elif curr_param and (line.startswith("    ") or line.startswith("\t")):
                curr_desc.append(line.strip())
        if curr_param:
            params[curr_param] = " ".join(curr_desc).strip()

    return params


def parse_returns(docstring: str) -> str:
    """Extract documented return information."""
    if not docstring:
        return ""

    # 1. Sphinx style
    sphinx_match = RE_SPHINX_RETURN.search(docstring)
    if sphinx_match:
        return sphinx_match.group(1).strip().replace("\n", " ")

    # 2. Google style
    google_match = RE_GOOGLE_RETURNS.search(docstring)
    if google_match:
        return google_match.group(1).strip().replace("\n", " ")

    # 3. NumPy style
    numpy_match = RE_NUMPY_RETURNS.search(docstring)
    if numpy_match:
        return numpy_match.group(1).strip().replace("\n", " ")

    # 4. Inline sentence check: "Returns ... "
    inline = re.search(r"^\s*(?:Returns?|Return)\s+([^\n.]+)", docstring, re.MULTILINE | re.IGNORECASE)
    if inline:
        return inline.group(1).strip()

    return ""


def parse_raises(docstring: str) -> Dict[str, str]:
    """Extract documented exceptions and conditions."""
    raises: Dict[str, str] = {}
    if not docstring:
        return raises

    # 1. Sphinx style
    for match in RE_SPHINX_RAISES.finditer(docstring):
        exc_name, exc_desc = match.groups()
        raises[exc_name] = exc_desc.strip().replace("\n", " ")

    # 2. Google style
    google_match = RE_GOOGLE_RAISES.search(docstring)
    if google_match:
        section_text = google_match.group(1)
        curr_exc = None
        curr_desc = []
        for line in section_text.splitlines():
            m = re.match(r"^\s*([a-zA-Z_]\w*)\s*:\s*(.*)", line)
            if m:
                if curr_exc:
                    raises[curr_exc] = " ".join(curr_desc).strip()
                curr_exc = m.group(1)
                curr_desc = [m.group(2)]
            elif curr_exc and line.startswith("    "):
                curr_desc.append(line.strip())
        if curr_exc:
            raises[curr_exc] = " ".join(curr_desc).strip()

    # 3. NumPy style
    numpy_match = RE_NUMPY_RAISES.search(docstring)
    if numpy_match:
        section_text = numpy_match.group(1)
        curr_exc = None
        curr_desc = []
        for line in section_text.splitlines():
            m = re.match(r"^\s*([a-zA-Z_]\w*)\s*(?:\n\s*-+)?", line)
            if m and not line.startswith("    "):
                if curr_exc:
                    raises[curr_exc] = " ".join(curr_desc).strip()
                curr_exc = m.group(1)
                curr_desc = []
            elif curr_exc:
                curr_desc.append(line.strip())
        if curr_exc:
            raises[curr_exc] = " ".join(curr_desc).strip()

    return raises


def normalize_docstring_structured(docstring: str) -> Tuple[str, Dict[str, any]]:
    """Format docstring into a normalized [SUMMARY], [PARAMETERS], [RETURNS], [RAISES] representation."""
    if not docstring:
        return "", {"summary": "", "parameters": {}, "returns": "", "raises": {}}

    summary = extract_summary(docstring)
    params = parse_parameters(docstring)
    returns = parse_returns(docstring)
    raises = parse_raises(docstring)

    sections = []
    if summary:
        sections.append(f"[SUMMARY]\n{summary}")

    if params:
        param_lines = [f"{p}: {desc}" if desc else p for p, desc in params.items()]
        sections.append("[PARAMETERS]\n" + "\n".join(param_lines))

    if returns:
        sections.append(f"[RETURNS]\n{returns}")

    if raises:
        raise_lines = [f"{exc}: {desc}" if desc else exc for exc, desc in raises.items()]
        sections.append("[RAISES]\n" + "\n".join(raise_lines))

    formatted = "\n\n".join(sections) if sections else summary

    parsed_meta = {
        "summary": summary,
        "parameters": params,
        "returns": returns,
        "raises": raises,
    }

    return formatted, parsed_meta
