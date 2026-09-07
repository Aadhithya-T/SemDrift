# SemDrift V2 Dataset Audit Report

**Generated**: 2026-09-07T16:45:51.095962+00:00
**Canonical Directory**: `experiments/2026-09-07_clean_v2/dataset/`

============================================================
SEMDRIFT V2 DATASET AUDIT
============================================================

TRAIN
  rows:               24,229
  lineages:           13,856
  clean (label 0):    13,871
  drift (label 1):    10,358
  provenance:
    clean_grounded:              13,871
    contract_grounded_generated: 10,358
    authentic_historical_mined:  0

VALIDATION
  rows:               2,636
  lineages:           1,520
  clean (label 0):    1,520
  drift (label 1):    1,116
  provenance:
    clean_grounded:              1,520
    contract_grounded_generated: 1,116
    authentic_historical_mined:  0

VERIFIED TEST (BALANCED DIAGNOSTIC TEST SET)
  rows:               104
  lineages:           104
  clean (label 0):    52
  drift (label 1):    52
  provenance:
    clean_grounded:              52
    authentic_historical_mined:  52
    contract_grounded_generated: 0

LEAKAGE & DISJOINTNESS
  train ∩ val:        0 lineages (PASSED)
  train ∩ test:       0 lineages (PASSED)
  val ∩ test:         0 lineages (PASSED)
  purged candidates:  71 rows removed for sharing test lineages

DUPLICATES & CONSISTENCY
  exact duplicate rows: 0 (PASSED, 0 filtered)
  lineage key formula:  repo::normalized_file::qualified_name (PASSED)
  explicit schema:      100% of rows contain explicit provenance, lineage, and label (PASSED)

INTEGRITY HASHES (SHA-256)
  train.jsonl:         7d0f5c21da4d618ffdb3c7681b98ec45e25f2a7fd53fe663d6c427bf59f1f9c5
  val.jsonl:           d188b128f5f955f5565e5479d9010eabc15c8df816b63ff096a0a2486cff17a5
  verified_test.jsonl: fff74007ea28dfd2f4822f4833f8d702489404ed8ad4882f53ea307ba540ef40
  SHA256SUMS:          fec452530a1d27e97d3ef6dd99d53640fbe48e616235deec7959637de5aba64f

RESULT
  ✓ DATASET VALID & CRYPTOGRAPHICALLY LOCKED
============================================================
