# Contributing to Multi-Objective Feature Selection

Thank you for your interest in contributing to Multi-Objective Feature Selection!
This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Commit Message Conventions](#commit-message-conventions)
- [Issue Reporting](#issue-reporting)
- [Feature Requests](#feature-requests)

## Code of Conduct

This project and everyone participating in it is governed by our
[Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to
uphold this code. Please report unacceptable behavior to royxforge@proton.me.

## Getting Started

1. Fork the repository on GitHub.
2. Clone your fork locally:
   ```
   git clone https://github.com/your-username/multi-objective-feature-selection.git
   cd multi-objective-feature-selection
   ```
3. Add the upstream repository:
   ```
   git remote add upstream https://github.com/royxforge/multi-objective-feature-selection.git
   ```

## Development Setup

### Prerequisites

- Python 3.10+
- pip

### Environment Setup

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Verify Installation

```bash
python -c "from sklearn.ensemble import RandomForestClassifier; print('Setup OK')"
```

## Coding Standards

- Follow [PEP 8](https://peps.python.org/pep-0008/) style guide.
- Use type annotations for all function signatures.
- Maximum line length: 88 characters (Black-compatible).
- Use descriptive variable names.
- Document all public functions with docstrings.

### Imports

Organize imports in the following order:

1. Standard library imports
2. Third-party imports
3. Local application imports

## Testing

- All experiments must be reproducible (seeded random state).
- Run the experiment before submitting changes:
  ```bash
  python run_experiment.py
  ```
- Verify that the Pareto front converges meaningfully.
- Add unit tests in `tests/` for new components:
  ```bash
  pytest tests/ -v
  ```

## Pull Request Process

1. Create a new branch from `main`:
   ```
   git checkout -b feature/your-feature-name
   ```

2. Make your changes with clear, descriptive commit messages.

3. Run the experiment to verify correctness:
   ```bash
   python run_experiment.py
   ```

4. Push your branch and open a Pull Request on GitHub.

5. In your PR description, include:
   - What the change does
   - Any relevant issue numbers
   - How you tested the change
   - Results or comparison tables if applicable

6. Request review from a maintainer.

## Commit Message Conventions

We follow conventional commit format:

```
<type>(<scope>): <description>
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

Examples:
```
feat(evolution): add SVM classifier support with RBF kernel
docs(readme): update Pareto front visualization section
refactor(experiment): extract data loading into separate module
```

## Issue Reporting

### Bug Reports

When filing a bug report, please include:

- A clear, descriptive title
- Steps to reproduce the issue
- Expected behavior and actual behavior
- Environment details (OS, Python version)
- Relevant code snippets or error logs

### Feature Requests

We welcome feature suggestions! Please include:

- A clear description of the proposed feature
- The motivation or use case
- Any relevant research or references
- Whether you are willing to implement it

Thank you for helping make Multi-Objective Feature Selection better!
