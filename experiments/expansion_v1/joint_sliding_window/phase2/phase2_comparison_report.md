# SemDrift Phase 2 Comparison Report: Sliding-Window Joint Encoder

**Evaluation Date**: 2026-10-12T20:19:54Z  
**Checkpoint**: `joint_encoder_checkpoint.pt` (`6930634acab6448b...`)  
**Hardware**: NVIDIA GeForce RTX 4050 Laptop GPU (cuda)  

---

## 1. Executive Summary & Success Criteria Assessment

| Criterion | Target | Baseline (Config A) | Locked Winner (Config D) | Absolute Delta | Status |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Expansion V1 Macro-F1** | $\ge 85.0\%$ | **91.91%** | **94.99%** | **+3.08%** | **PASS** |
| **Expansion V1 Drift Recall** | $\ge 85.0\%$ | **89.47%** | **96.37%** | **+6.90%** | **PASS** |
| **Authentic Drift Detected ($N=52$)** | $\ge 10/52$ | **24/52** | **23/52** | **-1** | **PASS** |
| **Clean False Positives ($N=52$)** | $\le 5/52$ | **21/52** | **17/52** | **-4** | **FAIL** |

---

## 2. Expansion V1 Test Set Performance ($N = 1,328$)

| ID | Configuration | Aggregation | $\tau$ | Accuracy | Macro-F1 | Bal Acc | Precision | Recall | F1 | Confusion Matrix |
|:---:|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **A** | Original Baseline | First window only | 0.50 | 92.17% | 91.91% | 91.78% | 91.47% | 89.47% | 90.46% | `TN=731, FP=46, FN=58, TP=493` |
| **B** | Tuned Baseline | First window only | 0.52 | 92.32% | 92.05% | 91.83% | 92.28% | 88.93% | 90.57% | `TN=736, FP=41, FN=61, TP=490` |
| **C** | Sliding Mean | Window Mean | 0.52 | 94.65% | 94.51% | 94.67% | 92.55% | 94.74% | 93.63% | `TN=735, FP=42, FN=29, TP=522` |
| **D** | **Locked Winner** | **Window Max** | **0.58** | **95.11%** | **94.99%** | **95.29%** | **92.19%** | **96.37%** | **94.23%** | `TN=732, FP=45, FN=20, TP=531` |

### McNemar's Paired Significance Test (Config D vs Config A)
- **Both Correct ($n_{11}$)**: 1212
- **Config D Unique Wins ($n_{10}$)**: 51
- **Config A Unique Wins ($n_{01}$)**: 12
- **Both Incorrect ($n_{00}$)**: 53
- **Discordant Pairs**: 63
- **Chi-Square (with continuity correction)**: **22.9206**
- **p-value**: **1.6883e-06** (Statistically Significant at $\alpha=0.05$: **Yes**)

---

## 3. Authentic Diagnostic Test Set Breakdown ($N = 104$)

### A. Authentic Historical Human Drift ($N = 52$)
| ID | Configuration | $\tau$ | Caught (TP) | Missed (FN) | Authentic Drift Recall (95% Wilson CI) |
|:---:|:---|:---:|:---:|:---:|:---|
| **A** | Original Baseline | 0.50 | 24 | 28 | **24/52 (46.15%)** [33.34% - 59.5%] |
| **B** | Tuned Baseline | 0.52 | 23 | 29 | **23/52 (44.23%)** [31.6% - 57.66%] |
| **C** | Sliding Mean | 0.52 | 24 | 28 | **24/52 (46.15%)** [33.34% - 59.5%] |
| **D** | **Locked Winner** | **0.58** | **23** | **29** | **23/52 (44.23%)** [31.6% - 57.66%] |

### B. Clean-Grounded Production Functions ($N = 52$)
| ID | Configuration | $\tau$ | False Alarms (FP) | Clean Correct (TN) | Clean FPR (95% Wilson CI) | Specificity |
|:---:|:---|:---:|:---:|:---:|:---|:---:|
| **A** | Original Baseline | 0.50 | 21 | 31 | **21/52 (40.38%)** [28.16% - 53.93%] | 59.62% |
| **B** | Tuned Baseline | 0.52 | 20 | 32 | **20/52 (38.46%)** [26.47% - 52.04%] | 61.54% |
| **C** | Sliding Mean | 0.52 | 18 | 34 | **18/52 (34.62%)** [23.15% - 48.2%] | 65.38% |
| **D** | **Locked Winner** | **0.58** | **17** | **35** | **17/52 (32.69%)** [21.52% - 46.24%] | **67.31%** |

---

## 4. Stratified Latency Benchmarking (Expansion V1 Test, $N=1,328$)

| Stratum | Definition | Sample Count | Share | Mean Latency | Median Latency | P95 Latency | Min – Max |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **$K = 1$** | $N_{doc} \le 192$ | 1134 | 85.4% | 21.83 ms | 20.26 ms | 32.23 ms | 12.43 – 72.61 ms |
| **$K = 2$** | $193 \le N_{doc} \le 320$ | 104 | 7.8% | 52.24 ms | 52.22 ms | 56.18 ms | 42.81 – 58.06 ms |
| **$K = 3$** | $321 \le N_{doc} \le 448$ | 49 | 3.7% | 80.22 ms | 81.00 ms | 85.28 ms | 67.50 – 87.36 ms |
| **$K \ge 4$** | $N_{doc} > 448$ | 41 | 3.1% | 145.97 ms | 111.00 ms | 220.37 ms | 94.96 – 569.67 ms |
| **Overall** | All test instances | 1328 | 100.0% | **30.20 ms** | **22.44 ms** | **80.68 ms** | 12.43 – 569.67 ms |
