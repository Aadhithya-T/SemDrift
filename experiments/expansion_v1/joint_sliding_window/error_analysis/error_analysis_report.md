# SemDrift Phase 3A: Authentic Drift Error Analysis Report

**Analysis Execution Date**: 2026-10-13  
**Auditor / Analyst**: Antigravity Assistant  
**Git Commit**: `bc0717bd51a0a20c771789028fd228d828615a3b` (`main`)  
**Artifact Directory**: [`experiments/expansion_v1/joint_sliding_window/error_analysis/`](file:///c:/Users/aadhi/OneDrive/Desktop/SemDrift/experiments/expansion_v1/joint_sliding_window/error_analysis/)

---

## 1. Executive Summary & Core Diagnostic Findings

Phase 3A investigates the fundamental divergence between SemDrift's performance on the **Expansion V1 Synthetic Benchmark** ($94.99\%$ Macro-F1, $96.37\%$ Recall) and the **Locked Authentic Diagnostic Set** ($46.15\%$ Recall, $32.69\%$ Clean False Alarm Rate).

### Key Empirical Findings:
1. **The Truncation Asymmetry Paradox Explained**:
   - On Expansion V1, sliding-window inference rescued 38 false negatives ($FN: 58 \rightarrow 20$, $+6.90\%$ Recall lift, $p = 1.69 \times 10^{-6}$) because synthetic AST mutations planted in downstream docstring paragraphs were physically inaccessible to the 192-token baseline.
   - On Authentic Historical Drift, sliding windows caught **0 additional drifts** ($24 \rightarrow 23/52$). The missed authentic drifts did **not** fail due to token truncation: the contradiction was almost always situated in the opening summary sentence. The model failed because its cross-attention representations could not discern subtle behavioral contract inversions from aligned natural language variations.
2. **The Clean False-Alarm Driver ($21 / 52 = 40.38\%$)**:
   - False positives on clean production code are heavily correlated with **docstring length** (mean $324.8$ tokens vs $101.3$ tokens for true negatives) and **internal helper delegation**. When human developers refactor implementations to delegate logic to internal helpers (e.g. `_guess_type()`, `*args, **kwargs` forwarding), the cross-encoder interprets the absence of explicit literal logic as a semantic contract violation.
3. **The Ranking Metric Collapse**:
   - While the model achieves high synthetic accuracy, its discriminatory ranking on authentic production functions collapses:
     $$\text{ROC-AUC}_{\text{diagnostic}} = \mathbf{0.5429} \quad (\text{Barely above random chance } 0.50)$$
     $$\text{PR-AUC}_{\text{diagnostic}} = \mathbf{0.5574} \quad (\text{Prevalence baseline } 0.50)$$

---

## 2. Freeze Verification & Cryptographic Hashes

To guarantee research integrity and reproducibility, all artifacts and prediction logs remain strictly frozen:

| Asset Name | Relative Path | SHA-256 Hash |
| :--- | :--- | :--- |
| **Diagnostic Test Dataset** | `experiments/2026-09-07_clean_v2/dataset/verified_test.jsonl` | `fff74007ea28dfd2f4822f4833f8d702489404ed8ad4882f53ea307ba540ef40` |
| **Phase 2 Checkpoint (joint_doc192)** | `experiments/expansion_v1/joint_doc192/joint_encoder_checkpoint.pt` | `6930634acab6448bc6753fbd1a987bcb39a6ebe50f05502992c00dfb5a9132f0` |
| **Historical Checkpoint (Clean V2)** | `experiments/2026-09-07_clean_v2/checkpoints/joint_encoder_checkpoint.pt` | `47f223de69d798664da1e4f923671325eae5f4c925e2190a0b8eca5acb5240b8` |
| **Phase 2 Predictions Export** | `experiments/expansion_v1/joint_sliding_window/phase2/authentic_diagnostic_predictions.jsonl` | `443cc84b8058b4f1462556a347e21246b21547e64e11f45d6f73def69d872e1b` |
| **Historical Predictions Export** | `experiments/2026-09-07_clean_v2/predictions/independent_predictions.jsonl` | `76061aed419e0ae7ab3a2f2ff2d2f39978f5878d118f1b111a372d38caeaed3d` |

---

## 3. Detailed Analysis of the 52 Authentic Historical Drift Cases

Full per-sample analysis across all 52 cases is exported in [`authentic_miss_analysis.csv`](file:///c:/Users/aadhi/OneDrive/Desktop/SemDrift/experiments/expansion_v1/joint_sliding_window/error_analysis/authentic_miss_analysis.csv).

### 3.1 Breakdown: Caught ($N = 24$) vs. Missed ($N = 28$)

| Dimension | Caught Cases ($N = 24$) | Missed Cases ($N = 28$) | Observational Difference |
| :--- | :---: | :---: | :--- |
| **Mean $P(\text{drift})$ (Single)** | **0.6530** (Median 0.6113) | **0.3047** (Median 0.3286) | Model is strongly confident in its misses ($P < 0.35$). |
| **Mean Docstring Tokens** | **298.6** tokens | **765.5** tokens | Missed docstrings are $2.56\times$ longer on average. |
| **Mean Code Tokens** | **911.2** tokens | **724.9** tokens | Code lengths are comparable (~700–900 tokens). |
| **Single-Window Share ($K=1$)** | 13 / 24 (54.2%) | 15 / 28 (53.6%) | Single-window share is identical across both groups. |

### 3.2 Recurring Failure Patterns in Missed Cases ($N = 28$)

```
Missed Failure Pattern Distribution:
├── Long Sphinx/Table Overload & Token Asymmetry   : 9 cases (32.1%)
├── Subtle Behavioral / Algorithmic Shift         : 7 cases (25.0%)
├── Return Structure / Generator Drift            : 6 cases (21.4%)
├── Parameter Refactor / Signature Dropped         : 3 cases (10.7%)
├── Exception/Error Handling Contract Drift       : 2 cases (7.1%)
└── Default Value Literal Mismatch                : 1 case  (3.6%)
```

#### Category 1: Subtle Behavioral & Algorithmic Inversion (7 cases)
The contradiction is semantic rather than syntactic: the docstring describes an algorithmic guarantee that the code quietly violates, but the code still contains high lexical overlap with the docstring keywords.
- **Concrete Case**: `click::ShellComplete.format_completion` (Sample 0, $P = 0.3451$)
  - *Docstring*: `Escape newlines in value and help to fix completion errors with multi-line help strings.`
  - *Code*: `if item.help: help_ = item.help.replace('\n', '\\n').replace('\t', ' '); return f"{item.type},{item.value}\t{help_}"; return f"{item.type},{item.value}"`
  - *Failure Mechanism*: The developer removed newline escaping from `item.value` while keeping it on `item.help`. Because `.replace('\n', '\\n')` is prominently present in the code, CodeBERT's bidirectional self-attention matches "Escape newlines" with the literal `.replace('\n', '\\n')` and assigns a high alignment score ($P_{\text{drift}} = 0.3451$). It fails to realize that the contract applied to *both* operands.
- **Concrete Case**: `click::_nullpager` (Sample 2, $P = 0.3041$)
  - *Docstring*: `Simply print unformatted text. This is the ultimate fallback.`
  - *Code*: `def _nullpager(stream, color=None): encoding = get_best_encoding(stream); yield stream, encoding, color`
  - *Failure Mechanism*: The print statement was deleted and the function was refactored into a generator yielding stream configuration tuples. The model predicts aligned because it has no representation of `print` vs `yield` semantic incompatibilities.

#### Category 2: Long Sphinx / RST Parameter Table Overload (9 cases)
Prevalent in SQLAlchemy and Django, where docstrings exceed 800 tokens and contain detailed ASCII/Sphinx tables explaining ORM relationship options, foreign keys, or database dialect quirks.
- **Concrete Case**: `django::join` (Sample 4, $P = 0.3227$)
  - *Docstring*: 236 tokens explaining SQL table alias generation, mentioning parameters `connection`, `reuse`, `nullable`, `join_field`.
  - *Code*: `def join(self, join, reuse=None): ...`
  - *Failure Mechanism*: `nullable` and `join_field` were removed from the signature. However, because the docstring contains hundreds of tokens discussing SQL joins, queries, and aliases that match identifiers in the 40-line function body, the cross-attention pooling dilutes the missing parameter signal.

#### Category 3: Generator / Stream vs. Concrete Return Drift (6 cases)
The docstring guarantees returning a concrete data structure (e.g. `list`, `dict`, or scalar), while the implementation returns a stream, generator, or iterator.
- **Concrete Case**: `sqlalchemy::Query.yield_per` (Sample 43, $P = 0.2814$)
  - The docstring guarantees returning instances, while the implementation modifies batch yield generators. Pretrained representations do not treat generator iteration as a return-type contradiction.

---

## 4. Detailed Analysis of the 52 Clean-Grounded Production Cases

Full per-sample analysis across all 52 clean cases is exported in [`clean_false_positive_analysis.csv`](file:///c:/Users/aadhi/OneDrive/Desktop/SemDrift/experiments/expansion_v1/joint_sliding_window/error_analysis/clean_false_positive_analysis.csv).

### 4.1 Breakdown: True Negatives ($N = 31$) vs. False Positives ($N = 21$)

| Metric | True Negatives ($N = 31$) | False Positives ($N = 21$) | Observational Difference |
| :--- | :---: | :---: | :--- |
| **Mean $P(\text{drift})$ (Single)** | **0.2827** (Median 0.3094) | **0.6356** (Median 0.6187) | False positives receive high confidence scores ($P > 0.60$). |
| **Mean Docstring Tokens** | **101.3** tokens | **324.8** tokens | Clean false positives have **$3.2\times$ longer docstrings**. |
| **Mean Code Tokens** | **419.7** tokens | **341.3** tokens | Code sizes are roughly similar (~350–420 tokens). |
| **Doc-to-Code Token Ratio** | **0.24 : 1** (Short doc, substantial code) | **0.95 : 1** (Docstring length equals code length) | High doc-to-code ratio strongly predicts false positives. |

### 4.2 Recurring False-Positive Hypotheses ($N = 21$)

```
Clean False Positive Hypothesis Distribution:
├── Lexical Asymmetry / Descriptive Prose Mismatch : 13 cases (61.9%)
├── Internal Helper Delegation / Forwarding        : 4 cases  (19.0%)
└── Long Multi-Clause Documentation Paraphrase     : 4 cases  (19.0%)
```

#### Hypothesis 1: Lexical Asymmetry in High-Level Architectural Docstrings (13 cases)
When a docstring explains *why* a function exists, its architectural role, or its interaction with external systems, it uses natural language vocabulary that does not literally appear in the Python statements.
- **Concrete Case**: `click::Context.invoke` (Sample 52, $P = 0.7061$)
  - *Docstring*: `Invokes a command. The first argument is the callback, the remaining are passed as arguments to the callback...`
  - *Code*: `def invoke(self, callback: t.Callable[..., t.Any], *args: t.Any, **kwargs: t.Any) -> t.Any:`
  - *Explanation*: The code uses generic `*args, **kwargs` forwarding, while the docstring discusses positional argument handling and context callbacks. The model flags this as drift because it expects the literal names mentioned in the docstring to appear in the AST argument list.

#### Hypothesis 2: Internal Helper Delegation (4 cases)
Clean refactoring in production code frequently moves complex logic into private helper functions (`_guess_type()`, `_build_inner()`).
- **Concrete Case**: `click::convert_type` (Sample 53, $P = 0.5367$)
  - The developer refactored type inference into an internal helper `_guess_type(val)`. The docstring continues to describe type inference accurately. But because the actual implementation statements reside inside the un-inlined helper function, the cross-encoder fails to find the logic in the immediate function body and flags drift.

#### Hypothesis 3: Type Annotation Refactoring (4 cases)
When production repositories add Python 3 type hints (`t.Optional`, `t.Union`, `t.Callable`) to legacy codebases without changing documentation prose, the cross-encoder penalizes the added syntax if the docstring lacks explicit Sphinx `:type:` tags.

---

## 5. Dataset & Training Provenance Audit: The Synthetic Shortcut

To understand why the Expansion V1 model behaves this way, we audited the generation pipeline in [`scripts/dataset_expansion/taxonomy_definitions.py`](file:///c:/Users/aadhi/OneDrive/Desktop/SemDrift/scripts/dataset_expansion/taxonomy_definitions.py):

### 5.1 AST Generator Shortcuts in Expansion V1
1. **Isolated Name Deletions**: In synthetic `parameter_missing` or `parameter_obsolete`, a parameter is deleted from the AST signature while its documentation in the docstring remains intact. This creates an **exact literal mismatch**: a token $T$ exists in the docstring's parameter block, but $T \notin \text{signature}$.
2. **Direct Negation Inversions**: In synthetic `doc_negation`, words like `not`, `never`, `false` are swapped into the first sentence.
3. **Absence of Descriptive Prose in Synthetic Samples**: Synthetic training examples rarely feature 400-token architectural narratives with high-level design philosophies or generic `*args, **kwargs` delegation.

### 5.2 The Real-World Mismatch
In production human commits:
- Developers do not randomly swap boolean keywords.
- Code changes involve helper extraction, control flow restructuring, or subtle boundary adjustments.
- Docstrings are often verbose, conceptual, or formatted with complex Sphinx directives.

**Consequence**: The cross-encoder learned that **low surface token alignment $\implies$ drift**. On clean functions with verbose prose or helper delegation, surface alignment is low $\rightarrow$ **False Positive ($32.7\%$)**. On authentic drift with high lexical overlap but subtle logic inversions, surface alignment is high $\rightarrow$ **False Negative ($53.8\%$)**.

---

## 6. Score Distributions & Calibration (Exploratory Post-Hoc Analysis)

*Disclaimer: Evaluated post-hoc on the 104 diagnostic examples. Not used for threshold tuning or model selection.*

Full distributions are archived in [`confidence_and_length_analysis.json`](file:///c:/Users/aadhi/OneDrive/Desktop/SemDrift/experiments/expansion_v1/joint_sliding_window/error_analysis/confidence_and_length_analysis.json).

```
Score Distribution Overview:
Clean True Negatives  (N=31): [===== 0.28 =====]
Authentic Missed (FN) (N=28): [====== 0.30 ======]
Clean False Positives (N=21):             [====== 0.64 ======]
Authentic Caught (TP) (N=24):              [======= 0.65 =======]
```

### Statistical Summary Table:
| Quadrant | Count | Mean $P(\text{drift})$ | Median $P(\text{drift})$ | P25 | P75 | Mean Doc Tokens | Mean Code Tokens |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Authentic Caught (TP)** | 24 | **0.6530** | 0.6113 | 0.5794 | 0.7316 | 298.6 | 911.2 |
| **Authentic Missed (FN)** | 28 | **0.3047** | 0.3286 | 0.2194 | 0.3993 | 765.5 | 724.9 |
| **Clean False Positives (FP)** | 21 | **0.6356** | 0.6187 | 0.5594 | 0.6923 | 324.8 | 341.3 |
| **Clean True Negatives (TN)** | 31 | **0.2827** | 0.3094 | 0.1863 | 0.3673 | 101.3 | 419.7 |

Notice that the score distributions of **Missed Authentic Drifts ($0.30$)** and **Clean True Negatives ($0.28$)** are virtually indistinguishable! Similarly, **Caught Authentic Drifts ($0.65$)** and **Clean False Positives ($0.64$)** overlap almost completely. 

**Threshold calibration cannot resolve this overlap**: any global threshold adjustment that catches more authentic drift will simultaneously inflate clean false alarms.

---

## 7. Limitations & Research Integrity Safeguards

1. **Non-Causal Observations**: The correlations identified (e.g. docstring length, helper delegation) are observable patterns supported by case inspection, not causal proofs.
2. **Post-Hoc Status**: The diagnostic set ($N=104$) was previously inspected. No decisions or threshold tunings have been performed on this set.
3. **No Retraining Conducted**: No models were fine-tuned or trained during Phase 3A.
