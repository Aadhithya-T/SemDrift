"""
SemDrift Dataset Expansion - Taxonomy Specification
Defines the expanded semantic drift taxonomy, contradiction conditions,
required evidence, and classification rules.
"""

TAXONOMY = {
    # 1. API and Parameter Behavior
    "parameter_missing": {
        "description": "Function signature includes a parameter that is completely omitted from the docstring.",
        "contradiction": "Caller reading documentation has no knowledge of the argument, its purpose, or its behavior.",
        "evidence_required": ["AST parameter exists in def", "Parameter name absent in docstring Args/Parameters/Sphinx tags"]
    },
    "parameter_obsolete": {
        "description": "Docstring documents a parameter that was removed from or never existed in the code signature.",
        "contradiction": "Caller following documentation provides an unexpected keyword argument leading to TypeError.",
        "evidence_required": ["Docstring lists parameter", "AST signature does not accept parameter"]
    },
    "parameter_rename": {
        "description": "A parameter in the code signature was renamed, but the docstring documents the stale identifier.",
        "contradiction": "Caller supplies documented keyword argument, resulting in unexpected argument error.",
        "evidence_required": ["Stale name in docstring", "Refactored name in AST signature"]
    },
    "parameter_default_mismatch": {
        "description": "Docstring specifies a default value that differs from the actual AST signature default.",
        "contradiction": "Caller relies on documented default behavior (e.g., timeout=30 or validate=True), but code executes with a different default.",
        "evidence_required": ["Docstring default value claim", "AST node default value mismatch"]
    },
    "parameter_constraint_mismatch": {
        "description": "Docstring specifies a type/value constraint (e.g. positive integer, list of strings) that code restricts differently.",
        "contradiction": "Caller supplies valid documented input that gets rejected or behaves incompatibly.",
        "evidence_required": ["Docstring constraint tag", "AST type annotation or validation check mismatch"]
    },
    "parameter_order_mismatch": {
        "description": "Docstring lists positional arguments in an order conflicting with the code signature.",
        "contradiction": "Positional invocation following documentation binds values to wrong parameters.",
        "evidence_required": ["Order of docstring params differs from AST args list"]
    },

    # 2. Return and Output Behavior
    "return_type_mismatch": {
        "description": "Docstring claims function returns a specific type (or None), while code returns a conflicting type.",
        "contradiction": "Caller expects documented return object but receives incompatible structure.",
        "evidence_required": ["Docstring Returns tag", "AST return type annotation or return expression"]
    },
    "return_none_behavior": {
        "description": "Docstring states function returns a value, but implementation unconditionally or conditionally returns None without documentation.",
        "contradiction": "Caller downstream receives None instead of expected object, leading to AttributeError.",
        "evidence_required": ["Docstring guarantees non-None object", "AST return paths show None return"]
    },
    "return_value_semantics": {
        "description": "Docstring claims function returns a dict/mapping, but code returns list/tuple/DataFrame structure.",
        "contradiction": "Caller attempts dict key lookup on sequence or vice versa.",
        "evidence_required": ["Docstring data structure claim", "AST return construct mismatch"]
    },

    # 3. Errors and Exceptions
    "exception_undocumented": {
        "description": "Function explicitly raises a specific Exception, but docstring contains no Raises specification.",
        "contradiction": "Caller has no warning of runtime failures that require handling.",
        "evidence_required": ["AST Raise node with explicit exception class", "Docstring lacks Raises section"]
    },
    "exception_contract_mismatch": {
        "description": "Docstring specifies function raises Exception A, but implementation raises Exception B.",
        "contradiction": "Caller catches documented exception type while unexpected error escapes.",
        "evidence_required": ["Docstring @raises tag for Class A", "AST Raise node for Class B"]
    },

    # 4. Side Effects and State
    "mutation_behavior": {
        "description": "Docstring specifies an in-place mutation or pure operation, but implementation does the reverse.",
        "contradiction": "Caller assumes input object was modified in place, but code returned a copy without mutating.",
        "evidence_required": ["Docstring in-place claim", "AST returns new instance or vice versa"]
    },

    # 5. Execution and Asynchrony
    "async_execution_mismatch": {
        "description": "Docstring describes synchronous execution, but code is an async coroutine (`async def`) or returns Awaitable.",
        "contradiction": "Caller invokes function synchronously receiving unawaited coroutine object.",
        "evidence_required": ["AST AsyncFunctionDef", "Docstring describes synchronous invocation"]
    },

    # 6. Consistent Baseline
    "none": {
        "description": "Code implementation and documentation are fully synchronized across parameters, return types, exceptions, and execution contracts.",
        "contradiction": "None. Semantic contract verified.",
        "evidence_required": ["All AST parameters documented", "Return behavior matches", "Exceptions synchronized"]
    }
}
