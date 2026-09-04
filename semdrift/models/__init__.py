"""
semdrift.models — Neural Network Models & Loss Functions.

Contains PyTorch modules for Model B Joint-Encoder and Dual-Encoder architectures.
"""

from semdrift.models.joint_encoder import (
    JointEncoderModel,
    FocalLoss,
    make_collate_fn,
    extract_docstring_summary,
    SemDriftDataset,
)
from semdrift.models.dual_encoder import (
    DualEncoderModel,
    DualEncoderDataset,
)

__all__ = [
    "JointEncoderModel",
    "DualEncoderModel",
    "FocalLoss",
    "make_collate_fn",
    "extract_docstring_summary",
    "SemDriftDataset",
    "DualEncoderDataset",
]

