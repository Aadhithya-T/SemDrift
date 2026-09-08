"""
semdrift.data — Data integrity, canonical label extraction, and dataset validation.
"""

from semdrift.data.integrity import (
    verify_dataset_integrity,
    compute_sha256,
    DatasetIntegrityError,
    CheckpointIntegrityError,
)
from semdrift.data.labels import (
    LABEL_STR_MAP,
    extract_training_label,
    extract_label_tuple,
    validate_dataset_split,
    validate_dataset_pipeline,
)

__all__ = [
    "verify_dataset_integrity",
    "compute_sha256",
    "DatasetIntegrityError",
    "CheckpointIntegrityError",
    "LABEL_STR_MAP",
    "extract_training_label",
    "extract_label_tuple",
    "validate_dataset_split",
    "validate_dataset_pipeline",
]
