# Next Experiment Proposal: Semantic-Contract Triplet Tuning (Phase 3B)

**Proposal Date**: 2026-10-13  
**Status**: Proposal (Pending Review — Zero Retraining Performed)  
**Target Repository**: `SemDrift`  
**Prerequisites**: Phase 2 Locked Evaluation & Phase 3A Error Analysis Complete  

---

## 1. Motivation: Lessons from the Sliding-Window Experiment & Error Analysis

The empirical findings of Phase 2 and Phase 3A demonstrated:
1. **The Truncation Bottleneck is a Myth for Real Drift**:
   Expanding the context window via sliding windows gave significant gains on synthetic AST mutations (+6.9% recall on Expansion V1, $p < 10^{-5}$), but **zero gain on authentic human drift** ($24 \rightarrow 23 / 52$). The missed authentic drifts already had their contradiction in the opening tokens.
2. **The Root Bottleneck is the Surface Alignment Shortcut**:
   Cross-encoders trained solely on rule-based AST mutations (renaming parameters, deleting keywords) learn that *low lexical/structural similarity $\implies$ drift*.
   - On clean production code with architectural prose or helper delegation, literal surface alignment is low $\rightarrow$ **Massive False Alarms (32.7%–40.4%)**.
   - On authentic drift where code changes subtle logic but retains keywords, surface alignment is high $\rightarrow$ **Massive False Negatives (53.8%)**.

**Conclusion**: Retraining with more windows or MIL pooling will merely optimize window aggregation over the same flawed shortcut. The next experiment must **directly decouple surface lexical overlap from semantic contract validity**.

---

## 2. Proposed Experiment: Contrastive Semantic-Contract Tuning (CSCT)

We recommend one targeted experiment to be developed strictly using training and validation splits.

### 2.1 Core Innovations
```
                                 [Documented Function Pair]
                                             │
                      ┌──────────────────────┴──────────────────────┐
                      ▼                                             ▼
          [Structural Contract View]                     [Behavioral Implementation View]
        - Docstring Parameter Specs                    - Function Signature & AST Def
        - Docstring Return Constraints                 - AST Return Nodes & Types
        - Docstring Raises Clauses                     - Explicit Raise Statements & Assertions
                      │                                             │
                      └──────────────────────┬──────────────────────┘
                                             ▼
                          [Cross-Encoder Cross-Attention]
                                             │
                      ┌──────────────────────┴──────────────────────┐
                      ▼                                             ▼
          [Auxiliary Contract Heads]                   [Contrastive Refactoring Loss]
        - Param Contract Match: {0, 1}               - Push away: Authentic Git Drifts
        - Return Contract Match: {0, 1}              - Pull together: Clean Helper Refactorings
        - Exception Contract Match: {0, 1}             (Penalize false alarms on code churn)
```

1. **Hard Clean-Refactoring Negatives**:
   - *Problem*: The model treats any implementation churn (e.g. adding type hints, delegating to `_helper()`, using `*args, **kwargs`) as drift.
   - *Solution*: Mine clean historical git commits where code was refactored across commits $(C_{before} \rightarrow C_{after})$ but the docstring remained identical and aligned $(D)$. Train with a contrastive loss that explicitly forces $\text{Score}(D, C_{after}) \approx \text{Score}(D, C_{before})$, teaching the encoder that code refactoring without contract alteration is **aligned**.
2. **Auxiliary Multi-Task Contract Heads**:
   - Rather than relying on a single binary drift head that falls back to surface token matching, add 3 auxiliary multi-task classification heads:
     - Head 1: Signature/Parameter Completeness Match ($y_{\text{param}} \in \{0, 1\}$)
     - Head 2: Return Type & Nullability Guarantee ($y_{\text{return}} \in \{0, 1\}$)
     - Head 3: Exception & Error Handling Contract ($y_{\text{except}} \in \{0, 1\}$)
   - Total Loss: $\mathcal{L} = \mathcal{L}_{\text{focal}}(\text{drift}) + 0.3 \mathcal{L}_{\text{param}} + 0.3 \mathcal{L}_{\text{return}} + 0.3 \mathcal{L}_{\text{except}}$.
3. **Structured Section-Aware Docstring Masking**:
   - Strip boilerplate Sphinx markup, ASCII parameter tables, and license headers during preprocessing to prevent long narrative tables from diluting cross-attention.

---

## 3. Strict Confirmatory Evaluation Protocol (Protecting Test Integrity)

Because the 104-instance diagnostic dataset has already been evaluated across multiple iterations, it is now an **exploratory diagnostic set**. To make definitive, scientifically credible claims of real-world generalization, we must evaluate on a fresh, untouched set:

### 3.1 Untouched Confirmatory Test Set Specification ($N = 120$)
- **Repositories Mined**: 5 top-tier open-source Python repositories never present in the diagnostic test:
  1. `requests` (HTTP library)
  2. `pandas` (Data manipulation)
  3. `scipy` (Scientific computing)
  4. `urllib3` (HTTP client)
  5. `rich` (Terminal formatting)
- **Composition**: Exactly 60 authentic historical human drifts (mined from pull requests and commits) and 60 clean grounded production functions.
- **Integrity Protocol**:
  1. The new dataset will be assembled, verified, and sealed with a SHA-256 hash.
  2. **Zero models, hyperparameter searches, or threshold sweeps will touch this set during development.**
  3. The final model selected on the validation split will be evaluated in a **single locked pass**.

---

## 4. Ranked List of Alternative Experiments

If Contrastive Semantic-Contract Tuning (CSCT) is not preferred, here are the top 3 alternative research directions:

| Rank | Experiment Name | Core Mechanism | Pros | Cons |
|:---:|:---|:---|:---|:---|
| **1** | **CSCT (Recommended)** | Multi-task contract heads + Clean refactoring contrastive loss | Directly fixes the surface-shortcut root cause | Requires mining refactored git commit pairs |
| **2** | **Lexical-Neural Meta-Ensemble** | Logistic regression meta-learner combining TF-IDF Relational + CodeBERT | Exploits TF-IDF's high specificity (low false alarms) with CodeBERT's semantic breadth | Fast to implement, no deep retraining needed |
| **3** | **AST-Augmented CFG Dual-Encoder** | Injects control-flow graph and return-path AST nodes into code representation | Teaches CodeBERT explicit `return` and `raise` paths | Requires specialized code parsing and graph neural nets |
| **4** | **LLM-Guided Contract Decomposition** | Use Claude/GPT-4 to extract formal executable contracts (Z3/Dafny style) | Maximum semantic precision | High inference cost, outside pure neural scope |

---

## 5. Execution Workflow (Phase 3B Roadmap)

1. **Step 1: Mine Clean-Refactoring Hard Negatives** (from existing training repository lineages).
2. **Step 2: Implement Multi-Task Contract Dataset & Loss** in `semdrift/models/contract_joint_encoder.py`.
3. **Step 3: Train and Tune strictly on Expansion V1 Train & Validation Splits**.
4. **Step 4: Mine and Cryptographically Seal the Untouched Confirmatory Test Set ($N=120$)**.
5. **Step 5: Execute Single Locked Confirmatory Evaluation**.
