# Changelog

All notable changes to the multi-objective feature selection project are documented in this file.

The format follows the guidance at [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html) with the understanding that pre-release iterations (v0.x.x) may involve breaking changes without a major version increment.

---

## [Unreleased]

No unreleased changes. All development is captured in the releases below.

---

## [0.2.0] - 2026-07-20

### Added

- **Community health files**: Added `CODE_OF_CONDUCT.md` (Contributor Covenant v2.1), `CONTRIBUTING.md` (contribution guidelines), `SECURITY.md` (vulnerability reporting policy), and `CITATION.cff` (citation metadata). These files establish project governance, community participation guidelines, and academic attribution framework.

---

## [0.1.0] - 2026-04-22

### Added

#### Core NSGA-II Implementation

- `src/moefs/evolution.py`: Full NSGA-II evolutionary loop implementing tournament selection, simulated binary crossover, polynomial mutation, non-dominated sorting, and crowding distance computation via the DEAP framework.
- `src/moefs/evaluation.py`: Cross-validation evaluation harness computing classification accuracy (primary objective) alongside precision, recall, and F1-score across training and test splits.
- `src/moefs/experiment.py`: Orchestration layer that wires dataset loading, the evolutionary loop, baseline comparisons, visualization, and result serialization into a single reproducible pipeline.
- `src/moefs/config.py`: Centralized configuration object parameterizing population size, generation count, crossover and mutation probabilities, classifier hyperparameter bounds, and cross-validation folds.
- `src/moefs/datasets.py`: Dataset loading interface with built-in support for the Wisconsin Breast Cancer dataset (scikit-learn) and a documented extension pattern for user-supplied tabular data.
- `src/moefs/baselines.py`: Reference baseline implementations including recursive feature elimination (RFE), PCA-based dimensionality reduction, random-forest importance thresholding, grid search, and random search for classifier hyperparameters.

#### Utilities and Visualization

- `src/moefs/utils.py`: Shared utility functions for result serialization, metric aggregation across cross-validation folds, data transformation, and numerical sanitization.
- `src/moefs/visualization.py`: Publication-grade plotting module producing Pareto front curves, evolution trajectory plots, per-method accuracy comparisons (box plots), and feature stability heatmaps.

#### Experiment Runner

- `run_experiment.py`: Executable entry point that loads the Wisconsin Breast Cancer dataset, runs the full NSGA-II optimization, evaluates all Pareto-optimal solutions, generates comparison tables against baseline methods, and exports all visualizations and numerical results to the `results/` directory.

#### Test Suite

- `tests/conftest.py`: Shared pytest fixtures providing reproducible random seeds, pre-configured classifier instances, small-scale synthetic datasets for fast test execution, and helper functions for validating Pareto front properties.
- `tests/test_evolution.py`: Unit tests covering NSGA-II initialization, selection pressure, crossover integrity, mutation correctness, and non-dominated sorting convergence properties.
- `tests/test_evaluation.py`: Tests for cross-validation metric correctness, stratification behavior, and consistency across repeated evaluations.
- `tests/test_datasets.py`: Validation of dataset shape contracts, class balance properties, and train-test split reproducibility.
- `tests/test_baselines.py`: Correctness checks for RFE dimensionality reduction, PCA component selection, and hyperparameter search space enumeration.

#### Experiment Results

- `results/all_results.csv`: Consolidated accuracy, precision, recall, and F1 metrics for all methods across all experimental replicates.
- `results/dataset_overview.csv` and `results/dataset_overview.md`: Structural summary of the Wisconsin Breast Cancer dataset including sample counts, feature dimensionality, class distribution, and descriptive statistics.
- `results/experiment_metadata.json`: Machine-readable record of all experimental parameters including random seeds, cross-validation configuration, NSGA-II hyperparameters, and environment details.
- `results/final_comparison_table.csv` and `results/final_comparison_table.md`: Final accuracy and feature count comparison across NSGA-II and all baseline methods, formatted for direct inclusion in academic manuscripts.
- `results/significance_tests.csv` and `results/significance_tests.md`: Pairwise statistical significance tests (Mann-Whitney U with Bonferroni correction) comparing method performance distributions.
- `results/summary_by_method.csv` and `results/summary_by_method.md`: Per-method summary statistics including mean accuracy, standard deviation, median feature count, and best-found configurations.
- `results/wisconsin_breast_cancer_comparison.csv`: Method-by-method comparison on the Wisconsin Breast Cancer benchmark, recording accuracy, feature count, and training time for each replicate.
- `results/wisconsin_breast_cancer_evolution_log.csv`: Per-generation log recording population diversity, front cardinality, best accuracy, and minimum feature count across the 30-generation NSGA-II run.
- `results/wisconsin_breast_cancer_nsga_permutation_importance.csv`: Permutation-based feature importance scores computed from the Pareto-optimal feature subset, measuring the accuracy drop incurred by shuffling each selected feature.
- `results/wisconsin_breast_cancer_pareto_front.csv`: Full set of non-dominated solutions discovered during evolution, each row specifying the feature mask, classifier hyperparameters, cross-validation accuracy, and feature count.

#### Infrastructure

- `requirements.txt`: Pinned dependency specification including DEAP (>=1.4.1), NumPy (>=1.24.0), pandas (>=2.0.0), scikit-learn (>=1.3.0), Matplotlib (>=3.7.0), seaborn (>=0.13.0), SciPy (>=1.10.0), Jupyter (>=1.0.0), and pytest (>=8.0.0).
- `pytest.ini`: Test runner configuration directing pytest to the `tests/` directory and enabling quiet output mode.
- `.gitignore`: Exclusion rules for Python bytecode cache, virtual environments, notebook checkpoints, test artifacts, and OS-specific metadata files.
- `.github/workflows/ci.yml`: Continuous integration definition triggering pytest on each push, ensuring test suite integrity across all contributions.

#### Documentation and Licensing

- `LICENSE`: MIT License granting permission for free use, modification, and distribution with attribution to Sourav Roy (2026).
- `README.md`: Comprehensive project documentation including problem formulation, algorithm description (non-dominated sorting, crowding distance, tournament selection, elitism), experimental setup, full results with baseline comparison table, installation instructions, usage examples, research context, and BibTeX citation entry.
- `project_demo.ipynb`: Interactive Jupyter notebook demonstrating the NSGA-II workflow from dataset loading through Pareto front visualization, enabling hands-on exploration of the optimization process.

### Changed

- No changes in this initial release.

### Fixed

- No fixes in this initial release.

---

## [0.1.1] - 2026-04-22

### Fixed

- Corrected GitHub repository link formatting in `README.md` to ensure the URL resolves without encoding errors.

---

## [0.1.2] - 2026-05-31

### Changed

- Restructured `README.md` to improve section hierarchy, relocate implementation details into appropriate subsections, and consolidate the results presentation for faster comprehension.

---

## [0.1.3] - 2026-06-11

### Changed

- Expanded algorithm documentation in `README.md` with a detailed walkthrough of the NSGA-II loop including non-dominated sorting mechanics, crowding distance computation, tournament selection criteria, and the crossover-mutation pipeline for joint feature mask and hyperparameter evolution.
- Extended baseline comparison with quantitative accuracy and feature count values for each reference method.
- Added the full evolved classifier configuration (RandomForest with n_estimators=132, max_depth=24, min_samples_split=3, min_samples_leaf=4, max_features=0.57) to the results section.
- Introduced a Research Context section linking this work to related projects (Unsupervised Confidence Estimation and Production Drift Detection) as a coherent model lifecycle pipeline.

---

## [0.1.4] - 2026-06-14

### Changed

- Refined wording in `README.md` across multiple sections for terminological precision and readability.

---

## [0.1.5] - 2026-06-15

### Changed

- Minor editorial updates to `README.md` preserving technical accuracy while improving flow.

---

## [0.1.6] - 2026-07-14

### Changed

- Polish pass on `README.md` tightening prose and correcting minor formatting inconsistencies.

---

## Notes on Versioning

This project uses pre-release versioning (v0.x.x) to reflect its active development phase. The public API encompasses the module interfaces in `src/moefs/`, the experiment runner protocol in `run_experiment.py`, and the serialized result schema. Breaking changes to any of these interfaces will be documented with appropriate migration guidance.

---

<!--
    Keep a Changelog v1.1.0
    Template reference: https://keepachangelog.com/en/1.1.0/
-->
