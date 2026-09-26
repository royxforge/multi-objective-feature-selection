from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from moefs.datasets import TrainOnlyFeatureExpander, _download_csv


def _test_logger() -> logging.Logger:
    logger = logging.getLogger("test.datasets")
    if not logger.handlers:
        logger.addHandler(logging.NullHandler())
    return logger


def test_train_only_feature_expander_reduces_dimensions() -> None:
    rng = np.random.default_rng(42)
    X_train = rng.normal(size=(40, 6))
    X_test = rng.normal(size=(10, 6))

    expander = TrainOnlyFeatureExpander(
        enable_feature_expansion=True,
        max_features=15,
        dataset_name="Synthetic",
        logger=_test_logger(),
    )

    transformed_train = expander.fit_transform(X_train, [f"f{i}" for i in range(X_train.shape[1])])
    transformed_test = expander.transform(X_test)

    assert transformed_train.shape[1] <= 15
    assert transformed_test.shape[1] == transformed_train.shape[1]
    assert len(expander.output_feature_names) == transformed_train.shape[1]


def test_feature_expander_disabled_keeps_original_columns() -> None:
    rng = np.random.default_rng(7)
    X_train = rng.normal(size=(30, 4))
    X_test = rng.normal(size=(8, 4))

    expander = TrainOnlyFeatureExpander(
        enable_feature_expansion=False,
        max_features=50,
        dataset_name="NoExpansion",
        logger=_test_logger(),
    )

    transformed_train = expander.fit_transform(X_train, ["a", "b", "c", "d"])
    transformed_test = expander.transform(X_test)

    assert transformed_train.shape == X_train.shape
    assert transformed_test.shape == X_test.shape
    assert expander.output_feature_names == ["a", "b", "c", "d"]


def test_mi_ranking_prefers_signal_over_large_scale_noise() -> None:
    """Label-aware pruning must not select the largest-magnitude noise term.

    x_noise has ~1000x the scale of x_signal, so raw-variance ranking picks
    x_noise and x_noise^2 (pure noise). MI ranking must keep the informative
    small-scale feature.
    """
    rng = np.random.default_rng(11)
    n = 400
    x_noise = rng.normal(0.0, 1000.0, n)
    x_signal = rng.normal(0.0, 1.0, n)
    y = (x_signal > 0).astype(int)
    X = np.column_stack([x_noise, x_signal])

    expander = TrainOnlyFeatureExpander(
        enable_feature_expansion=True,
        max_features=2,
        dataset_name="ScaleBias",
        logger=_test_logger(),
    )
    expander.fit(X, ["noise", "signal"], y=y)

    names = expander.output_feature_names
    assert len(names) == 2
    # The informative original feature must survive pruning.
    assert "signal" in names


def test_ranking_falls_back_to_variance_without_labels() -> None:
    """Legacy callers (no labels) still prune, via scale-dependent variance."""
    rng = np.random.default_rng(11)
    X = rng.normal(size=(40, 2))

    expander = TrainOnlyFeatureExpander(
        enable_feature_expansion=True,
        max_features=3,
        dataset_name="NoLabels",
        logger=_test_logger(),
    )
    expander.fit(X, ["a", "b"])

    assert len(expander.output_feature_names) == 3


def test_config_validation_catches_invalid_inputs(tmp_path: Path) -> None:
    """Config validation must fail fast on unsupported parameter values."""
    from moefs.config import ExperimentConfig

    invalid = ExperimentConfig(project_root=tmp_path, population_size=2)
    with pytest.raises(ValueError, match="population_size must be at least 4"):
        invalid.validate()


def test_download_csv_writes_cache_file(tmp_path, monkeypatch) -> None:
    expected = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
    real_read_csv = pd.read_csv

    def fake_read_csv(path, *args, **kwargs):
        if isinstance(path, str) and path.startswith("https://example.com"):
            return expected.copy()
        return real_read_csv(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", fake_read_csv)

    output_path = tmp_path / "cached.csv"
    result = _download_csv("https://example.com/mock.csv", output_path)

    assert output_path.exists()
    pd.testing.assert_frame_equal(result, expected)
    cached = real_read_csv(output_path)
    pd.testing.assert_frame_equal(cached, expected)
