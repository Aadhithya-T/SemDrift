# SemDrift — Semantic Drift Detection Between Code and Documentation

<p align="center">
  <strong>Can a model tell when code no longer does what its documentation promises?</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue" alt="Python 3.9+" />
  <img src="https://img.shields.io/badge/Model-CodeBERT-orange" alt="CodeBERT" />
  <img src="https://img.shields.io/badge/Framework-PyTorch-red" alt="PyTorch" />
  <img src="https://img.shields.io/badge/Status-Research%20Prototype-purple" alt="Research prototype" />
  <img src="https://img.shields.io/badge/License-MIT-green" alt="MIT License" />
</p>

**SemDrift** is a research framework for detecting semantic inconsistencies between source code and its documentation. It investigates whether machine learning models can identify functions whose implementation no longer satisfies the behavioral contract described by their docstrings.

Unlike approaches that rely exclusively on lexical similarity, SemDrift evaluates classical machine-learning baselines, pretrained representations, dual encoders, and joint CodeBERT encoders that process documentation and implementation together.

The project combines model development, dataset construction, controlled experiments, provenance and leakage checks, statistical evaluation, and a Python repository-scanning CLI.

---

## Research highlights

- **95.11% test accuracy** using the joint encoder with sliding-window max pooling on the Expansion V1 test benchmark.
- **94.99% Macro-F1** and **95.29% balanced accuracy** under the same evaluation configuration.
- A controlled joint-versus-dual encoder comparison showing that joint encoding improves performance on the earlier V1 benchmark.
- An investigation of the generalization gap between controlled benchmark performance and authentic historical documentation drift.

> **Important:** These are benchmark-specific results. High performance on controlled or generated drift examples does not establish equivalent performance on authentic historical drift. SemDrift explicitly investigates this distinction.

## Table of contents

- [The problem](#the-problem)
- [How SemDrift works](#how-semdrift-works)
- [Benchmark results](#benchmark-results)
- [Datasets and evaluation](#datasets-and-evaluation)
- [Repository structure](#repository-structure)
- [Installation](#installation)
- [Running the scanner](#running-the-scanner)
- [Reproducing the research](#reproducing-the-research)
- [Limitations](#limitations)
- [Future directions](#future-directions)
- [License](#license)

---

## The problem

Documentation drift occurs when a function's implementation changes but its documentation no longer accurately describes its behavior.

For example, a docstring might promise that a function returns a list, while the implementation now returns a generator. The function may remain syntactically valid, pass tests, and execute successfully, yet its documentation is no longer correct.

SemDrift formulates this as a binary classification problem:

| Label | Meaning |
|---|---|
| `aligned` | The documentation is consistent with the implementation. |
| `drifted` | The implementation contradicts, omits, or no longer satisfies a documented behavioral contract. |

The challenge is distinguishing genuine behavioral contradictions from harmless refactoring, helper delegation, paraphrasing, and implementation details that do not violate the documented contract.

## How SemDrift works

SemDrift compares documentation and implementation using multiple modeling approaches.

### Joint encoder

The joint encoder uses `microsoft/codebert-base` to process documentation and source code together. Self-attention allows the model to contextualize information across both inputs before making a drift prediction.

This is the primary architecture investigated in the controlled architectural comparison.

### Dual encoder

The dual encoder processes documentation and code separately before combining their representations. It serves as an architectural baseline for investigating whether joint cross-input attention improves drift detection.

### Classical and pretrained baselines

The evaluation also includes:

- TF-IDF with Logistic Regression.
- Relational lexical features.
- Zero-shot CodeBERT similarity-based classification.
- Fine-tuned dual and joint CodeBERT models.
- Joint encoder training with focal loss and hard-negative-oriented experiments.

### Sliding-window inference

The Expansion V1 experiments also investigate sliding-window inference for long inputs. Multiple windows are evaluated and their drift probabilities are aggregated using mean or max pooling.

The selected sliding-window configuration improves performance on the Expansion V1 test benchmark. However, it does not demonstrate improved authentic-drift recall on the separate 104-example diagnostic set.

### Repository scanning

The CLI extracts Python functions and docstrings, applies the available model checkpoint, ranks potential inconsistencies by predicted drift probability, and presents results for manual review. It supports terminal inspection and Markdown or JSON reporting, depending on the selected output mode.

---

## Benchmark results

### 1. Expansion V1 — joint encoder with sliding windows

The latest reported Expansion V1 sliding-window experiment evaluates **1,328 test samples**.

| Metric | Sliding max, threshold 0.58 |
|---|---:|
| Accuracy | **95.11%** |
| Macro-F1 | **94.99%** |
| Balanced accuracy | **95.29%** |
| Drift precision | 92.19% |
| Drift recall | 96.37% |
| Drift F1 | 94.23% |
| True negatives | 732 |
| False positives | 45 |
| False negatives | 20 |
| True positives | 531 |

The single-window baseline at threshold 0.50 achieved 92.17% accuracy and 91.91% Macro-F1. Sliding-window max pooling improved accuracy and drift recall on this benchmark.

A paired McNemar comparison between the sliding-window configuration and the single-window baseline yielded a continuity-corrected \(\chi^2=22.92\), \(p\approx1.69\times10^{-6}\).

These results support an improvement on the evaluated Expansion V1 test set. They should not be interpreted as evidence of equivalent performance on every category of real-world documentation drift.

### 2. Controlled joint-versus-dual encoder comparison

The earlier V1 controlled comparison uses a frozen test benchmark of **1,205 samples**. Under the same cross-entropy training objective, the joint encoder outperformed the dual encoder.

| Model | Accuracy | Macro-F1 | Drift F1 |
|---|---:|---:|---:|
| TF-IDF combined | 65.23% | 65.10% | 67.19% |
| TF-IDF relational | 70.04% | 70.04% | 70.19% |
| Dual encoder | 80.91% | 80.64% | 78.34% |
| Joint encoder | **85.06%** | **84.95%** | **83.67%** |
| Joint encoder with focal loss | **85.81%** | **85.66%** | **84.21%** |

In the controlled cross-entropy comparison, joint encoding improved binary F1 by 5.33 percentage points over the dual encoder. A paired McNemar test reported \(\chi^2=18.76\), \(p\approx1.48\times10^{-5}\).

The focal-loss model is a separate training configuration and should not be confused with the controlled cross-entropy architecture comparison.

### 3. Authentic historical drift diagnostic

The separate diagnostic set contains 104 verified examples:

- 52 authentic historical documentation-drift cases mined from open-source repository history.
- 52 clean-grounded functions.

On this diagnostic set, the earlier Clean-Slate V2 joint encoder caught **2 of 52 authentic drift cases**, while the dual encoder caught none. A later Expansion V1 joint encoder checkpoint caught 24 of 52 authentic drift cases using single-window inference.

These results belong to different model checkpoints and training configurations. They are not interchangeable measurements of one model.

The diagnostic set is small and has been reused for exploratory analysis. Its results should therefore be interpreted as evidence of a generalization challenge, not as a definitive estimate of performance across all Python repositories.

For complete historical metrics, confusion matrices, model comparisons, and dataset details, see [BENCHMARK_RESULTS.md](BENCHMARK_RESULTS.md).

---

## Datasets and evaluation

SemDrift separates controlled evaluation from real-world-grounded data construction and authentic historical diagnostics.

### V1 — controlled synthetic benchmark

The original frozen V1 benchmark contains 1,205 test examples from 10 repositories, with 597 aligned and 608 drifted examples.

Controlled mutations include changes to return behavior, parameter names, documentation negation, and documentation sentences. These examples allow repeatable architectural comparisons, but the results depend on the mutation patterns represented in the dataset.

Expansion V1 is a subsequent benchmark generation with a 1,328-sample test set. Its results are reported separately rather than mixed with the original V1 metrics.

### V2 — real-world-grounded dataset

The Clean-Slate V2 training and validation data use grounded examples constructed from real repository code and documented contracts. The training and validation splits contain 24,229 and 2,636 samples, respectively.

The independent diagnostic set contains 104 examples, including 52 authentic historical drift cases. The dataset pipeline includes lineage checks and a frozen test manifest intended to reduce train/test contamination.

**Real-world-grounded does not mean every drift example is an untouched historical human mistake.** Generated and rule-augmented examples are distinguished from authentic historical drift so that the different evaluation conditions remain visible.

### Reproducibility principles

SemDrift's experiments emphasize:

- Explicit dataset provenance and sample lineage.
- Fixed evaluation splits and recorded model configurations.
- Confusion matrices and class-specific metrics alongside accuracy.
- Statistical comparisons where paired predictions are available.
- Separate reporting for controlled benchmarks and authentic-drift diagnostics.
- Integrity checks and reproducible experiment artifacts.

The repository's benchmark reports and experiment directories provide the detailed records needed to inspect individual results.

---

## Repository structure

The repository contains the research pipeline, models, data preparation code, experiment artifacts, tests, and scanning interface.

```text
SemDrift/
├── semdrift/                 # Core package: parsing, formatting and models
├── scripts/                  # Experiment and repository-scanning scripts
│   └── scan_repo.py          # Terminal repository scanner
├── data/                     # Dataset generations and prepared splits
├── data_pipeline/            # Dataset preparation and validation
├── training/                 # Model training utilities
├── experiments/              # Checkpoints, predictions and experiment results
│   ├── expansion_v1/         # Expansion V1 model comparisons
│   └── 2026-09-07_clean_v2/  # Clean-Slate V2 artifacts
├── archive/                  # Archived experiment artifacts
├── tests/                    # Automated tests
├── BENCHMARK_RESULTS.md      # Consolidated historical benchmark report
├── requirements.txt
└── LICENSE
```

Some experiments and checkpoints may require additional artifacts that are not installed automatically.

---

## Installation

### Requirements

- Python 3.9 or newer.
- PyTorch.
- Hugging Face Transformers.
- A supported Python environment with the project dependencies installed.
- A GPU is recommended for faster transformer inference, but CPU execution may be possible.

### Set up the environment

Clone the repository and enter its directory:

```bash
git clone https://github.com/Aadhithya-T/SemDrift.git
cd SemDrift
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

On Windows:

```powershell
.venv\Scripts\Activate.ps1
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

Install the project dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The scanner requires a compatible model checkpoint. The repository currently configures the scanner to look for the Clean-Slate V2 checkpoint at:

```text
experiments/2026-09-07_clean_v2/checkpoints/joint_encoder_checkpoint.pt
```

An archived fallback checkpoint is also configured in the scanner. Confirm that the required checkpoint is present before running inference. Model weights are not guaranteed to be downloaded automatically by the installation steps above.

---

## Running the scanner

The repository includes a Python repository-scanning CLI.

Scan the current repository:

```bash
python scripts/scan_repo.py .
```

Scan a specific project directory:

```bash
python scripts/scan_repo.py /path/to/python/project
```

Use interactive inspection mode:

```bash
python scripts/scan_repo.py . --interactive
```

Set a custom decision threshold:

```bash
python scripts/scan_repo.py . --threshold 0.60
```

Export a Markdown report:

```bash
python scripts/scan_repo.py . --output markdown --output_file drift_report.md
```

The scanner can also produce JSON output using its supported output options. Use the CLI help to inspect the arguments available in the checked-out version:

```bash
python scripts/scan_repo.py --help
```

### Interpreting results

A high drift probability indicates that the model considers a function more likely to be inconsistent with its documentation. It is **not a calibrated guarantee** that a real behavioral contradiction exists.

Treat flagged functions as candidates for review. Validate them against function behavior, call sites, tests, and repository history before concluding that a defect exists.

The default CLI checkpoint belongs to the Clean-Slate V2 model lineage. Do not assume that the 95.11% Expansion V1 sliding-window benchmark result applies to the default CLI model; the checkpoint, inference procedure, and evaluation data differ.

---

## Reproducing the research

For model comparisons and historical benchmark metrics, start with:

- [Master benchmark report](BENCHMARK_RESULTS.md)
- [`experiments/`](experiments/)
- [`tests/`](tests/)

To reproduce a reported result, use the corresponding dataset split, checkpoint, preprocessing configuration, inference settings, and threshold recorded for that experiment. Do not substitute the CLI's default checkpoint for an experimental checkpoint unless the experiment explicitly specifies it.

The 95.11% Expansion V1 result uses sliding-window max pooling at threshold 0.58. It is not directly comparable to a single-window result without accounting for the inference configuration.

The authentic-drift diagnostic has already been reused for exploratory model comparisons. It should not be treated as an untouched confirmatory test set for future model selection.

---

## Limitations

SemDrift is an experimental research prototype, not a production-grade correctness verifier.

1. **Generalization:** Strong results on controlled and grounded benchmarks do not guarantee equivalent performance on authentic historical drift.
2. **False positives:** Legitimate refactoring, helper delegation, and high-level documentation can be mistaken for drift.
3. **False negatives:** Subtle behavioral changes may be missed even when the documentation explicitly describes the affected behavior.
4. **Long inputs:** Transformer sequence limits and the selected windowing strategy affect which evidence is available to the model.
5. **Threshold sensitivity:** Classification decisions depend on the chosen threshold; changing it trades off precision and recall.
6. **Language scope:** The primary empirical evaluation focuses on Python. Parser infrastructure for other languages should not be interpreted as equivalent validated model performance.
7. **Diagnostic sample size:** The 104-case authentic-drift diagnostic is valuable for error analysis but too limited to establish broad real-world performance by itself.

SemDrift should therefore be used to prioritize manual review, not to automatically rewrite documentation or certify behavioral correctness.

---

## Future directions

Potential research directions include:

- Evaluating on a larger, untouched collection of authentic historical drift.
- Improving sensitivity to behavioral contracts rather than surface-level lexical patterns.
- Distinguishing legitimate implementation refactoring from actual contract violations.
- Investigating parameter, return-value, and exception-contract supervision.
- Extending evaluation to additional languages and repository types.
- Improving calibration and reporting uncertainty alongside drift scores.

These directions remain open research questions rather than claims of completed functionality.

---

## License

SemDrift is released under the [MIT License](LICENSE).

## Acknowledgments

SemDrift uses [CodeBERT](https://huggingface.co/microsoft/codebert-base), PyTorch, Hugging Face Transformers, and the broader open-source Python ecosystem.

---

**SemDrift — measuring the gap between what code does and what its documentation promises.**
