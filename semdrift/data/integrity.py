#!/usr/bin/env python3
"""
semdrift/data/integrity.py

Cryptographic Dataset Integrity Verification Gate.
Enforces a strict two-layer integrity chain:
  Layer A: Live files ↔ SHA256SUMS
  Layer B: SHA256SUMS ↔ manifest.yaml
Refuses execution if any file is missing, altered, or unverified.
"""

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Union
import yaml


class DatasetIntegrityError(RuntimeError):
    """Raised when dataset files fail cryptographic SHA-256 integrity verification."""
    pass


class CheckpointIntegrityError(RuntimeError):
    """Raised when a model checkpoint fails cryptographic SHA-256 integrity verification."""
    pass


def compute_sha256(path: Union[str, Path]) -> str:
    """Compute SHA-256 checksum of a file in binary mode."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Cannot compute SHA-256: file not found at {p}")
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_dataset_integrity(
    dataset_dir: Union[str, Path],
    manifest_path: Optional[Union[str, Path]] = None,
    required_files: Optional[List[str]] = None,
) -> Dict[str, str]:
    """
    Verify dataset integrity against SHA256SUMS and optionally manifest.yaml.
    
    Parameters
    ----------
    dataset_dir : str or Path
        Directory containing dataset files and SHA256SUMS.
    manifest_path : str or Path, optional
        Path to manifest.yaml. If None, looks for config/manifest.yaml in experiment root.
    required_files : list of str, optional
        Specific subset of files required to be present and verified (e.g. ['train.jsonl', 'val.jsonl']).
        If None, all files in SHA256SUMS are verified.
        
    Returns
    -------
    dict
        Dictionary mapping filename -> verified SHA-256 hash.
        
    Raises
    ------
    DatasetIntegrityError
        If checksum mismatch or integrity violation occurs.
    """
    d_dir = Path(dataset_dir).resolve()
    sums_file = d_dir / "SHA256SUMS"
    
    if not sums_file.is_file():
        raise DatasetIntegrityError(
            f"FATAL: Missing authoritative SHA256SUMS at {sums_file}. "
            "Execution refused: unverified dataset."
        )
        
    # 1. Parse SHA256SUMS
    recorded_sums: Dict[str, str] = {}
    with sums_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                parts = line.split(None, 1)
                if len(parts) == 2:
                    h, fname = parts
                    recorded_sums[fname.strip()] = h.strip().lower()
                    
    # 2. Layer A: Verify Live Files ↔ SHA256SUMS
    target_files = required_files if required_files is not None else list(recorded_sums.keys())
    verified_hashes: Dict[str, str] = {}
    
    for fname in target_files:
        if fname not in recorded_sums:
            raise DatasetIntegrityError(
                f"FATAL: Required file '{fname}' is not recorded in SHA256SUMS! "
                "Execution refused."
            )
        fpath = d_dir / fname
        if not fpath.is_file():
            raise DatasetIntegrityError(
                f"FATAL: Required dataset file missing from disk: {fpath}. "
                "Execution refused."
            )
        actual_hash = compute_sha256(fpath)
        expected_hash = recorded_sums[fname]
        if actual_hash != expected_hash:
            raise DatasetIntegrityError(
                f"FATAL: Checksum mismatch for '{fname}'!\n"
                f"  Expected (SHA256SUMS): {expected_hash}\n"
                f"  Actual   (Live File) : {actual_hash}\n"
                "Dataset has been tampered with or modified. Execution refused."
            )
        verified_hashes[fname] = actual_hash
        
    # 3. Layer B: Verify SHA256SUMS ↔ manifest.yaml (if manifest provided or discoverable)
    if manifest_path is None:
        cand = d_dir.parent / "config" / "manifest.yaml"
        if cand.is_file():
            manifest_path = cand
            
    if manifest_path is not None and Path(manifest_path).is_file():
        m_path = Path(manifest_path).resolve()
        with m_path.open("r", encoding="utf-8") as mf:
            manifest = yaml.safe_load(mf)
            
        m_dataset = manifest.get("dataset", {})
        m_checksums = m_dataset.get("checksums", {})
        
        # Check SHA256 of SHA256SUMS itself
        expected_sums_hash = m_checksums.get("sha256sums_sha256")
        if expected_sums_hash:
            actual_sums_hash = compute_sha256(sums_file)
            if actual_sums_hash != expected_sums_hash.lower():
                raise DatasetIntegrityError(
                    f"FATAL: Two-layer integrity failure! SHA256SUMS file does not match manifest.yaml!\n"
                    f"  Expected (manifest): {expected_sums_hash}\n"
                    f"  Actual   (SHA256SUMS): {actual_sums_hash}\n"
                    "Execution refused."
                )
                
        # Check individual file hashes against manifest
        hash_keys = {
            "train.jsonl": "train_sha256",
            "val.jsonl": "val_sha256",
            "verified_test.jsonl": "verified_test_sha256",
        }
        for fname, mkey in hash_keys.items():
            if fname in target_files and mkey in m_checksums:
                expected_m_hash = m_checksums[mkey].lower()
                actual_h = verified_hashes[fname]
                if actual_h != expected_m_hash:
                    raise DatasetIntegrityError(
                        f"FATAL: File '{fname}' does not match hash recorded in manifest.yaml!\n"
                        f"  Expected (manifest) : {expected_m_hash}\n"
                        f"  Actual   (Verified) : {actual_h}\n"
                        "Execution refused."
                    )
                    
    return verified_hashes


def verify_checkpoint_integrity(
    checkpoint_path: Union[str, Path],
    run_record_path: Union[str, Path],
) -> str:
    """
    Verify model checkpoint against recorded SHA-256 hash in training_run.json.
    Enforces a strict cryptographic security boundary:
    Refuses execution BEFORE torch.load() if checksum mismatch or missing.
    
    Parameters
    ----------
    checkpoint_path : str or Path
        Path to the saved PyTorch model checkpoint (.pt).
    run_record_path : str or Path
        Path to training_run.json recorded at training completion.
        
    Returns
    -------
    str
        The verified SHA-256 hash of the checkpoint.
        
    Raises
    ------
    CheckpointIntegrityError
        If the checkpoint is missing, run record is missing, or SHA-256 mismatches.
    """
    c_path = Path(checkpoint_path).resolve()
    r_path = Path(run_record_path).resolve()
    
    if not c_path.is_file():
        raise CheckpointIntegrityError(
            f"FATAL: Checkpoint file missing at {c_path}. Evaluation refused."
        )
    if not r_path.is_file():
        raise CheckpointIntegrityError(
            f"FATAL: Training run record missing at {r_path}. "
            "Checkpoint has no verified cryptographic provenance. Evaluation refused."
        )
        
    try:
        with r_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        raise CheckpointIntegrityError(
            f"FATAL: Failed to parse training run record at {r_path}: {e}"
        )
        
    expected_hash = data.get("checkpoint_sha256")
    if not expected_hash:
        raise CheckpointIntegrityError(
            f"FATAL: Training run record at {r_path} contains no verified 'checkpoint_sha256'! "
            "Evaluation refused."
        )
        
    actual_hash = compute_sha256(c_path)
    if actual_hash != expected_hash.strip().lower():
        raise CheckpointIntegrityError(
            f"FATAL: Checkpoint SHA-256 mismatch!\n"
            f"  Expected ({r_path.name}): {expected_hash}\n"
            f"  Actual   (Live File)    : {actual_hash}\n"
            "Model checkpoint has been modified or tampered with. Evaluation refused."
        )
        
    return actual_hash

