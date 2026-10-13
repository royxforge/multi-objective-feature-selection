from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from moefs.datasets import (
    DATASET_LABELS,
    TrainOnlyFeatureExpander,
    _download_csv,
    available_datasets,
    load_medical_datasets,
    resolve_dataset_key,
    resolve_dataset_keys,
)
from moefs.config import ExperimentConfig


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


# --- Dataset selection (available_datasets / load_medical_datasets / config) ---


def test_available_datasets_lists_canonical_keys() -> None:
    assert available_datasets() == ["wisconsin", "pima", "heart"]


def test_select_datasets_loads_only_the_requested_one(tmp_path) -> None:
    logger = logging.getLogger("test-selection")
    bundles = load_medical_datasets(tmp_path, logger, datasets=["wisconsin"])
    assert [b.name for b in bundles] == ["Wisconsin Breast Cancer"]


def test_selection_is_case_insensitive_and_order_preserving(tmp_path) -> None:
    logger = logging.getLogger("test-selection")
    bundles = load_medical_datasets(tmp_path, logger, datasets=[" HEART ", "wisconsin"])
    assert [b.name for b in bundles] == [
        "Heart Disease (Cleveland)",
        "Wisconsin Breast Cancer",
    ]


def test_unknown_dataset_key_raises(tmp_path) -> None:
    logger = logging.getLogger("test-selection")
    with pytest.raises(ValueError, match="unknown dataset"):
        load_medical_datasets(tmp_path, logger, datasets=["nope"])


def test_empty_dataset_selection_raises(tmp_path) -> None:
    logger = logging.getLogger("test-selection")
    with pytest.raises(ValueError):
        load_medical_datasets(tmp_path, logger, datasets=["  ", ""])


def test_dataset_limit_still_applies_when_no_names_given(tmp_path) -> None:
    logger = logging.getLogger("test-selection")
    bundles = load_medical_datasets(tmp_path, logger, dataset_limit=1)
    assert [b.name for b in bundles] == ["Wisconsin Breast Cancer"]


def test_config_accepts_dataset_names() -> None:
    config = ExperimentConfig(population_size=16, generations=2, datasets=["pima"])
    config.validate()
    assert config.datasets == ["pima"]


def test_config_rejects_datasets_plus_limit() -> None:
    config = ExperimentConfig(
        population_size=16, generations=2, datasets=["pima"], dataset_limit=1
    )
    with pytest.raises(ValueError, match="mutually exclusive"):
        config.validate()


def test_config_rejects_empty_dataset_list() -> None:
    config = ExperimentConfig(population_size=16, generations=2, datasets=["  "])
    with pytest.raises(ValueError, match="at least one"):
        config.validate()


class TestDatasetNameResolution:
    """`--datasets` accepts keys, display labels and aliases."""

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("pima", "pima"),
            ("PIMA", "pima"),
            ("  heart  ", "heart"),
            ("Pima Indians Diabetes", "pima"),
            ("heart disease (cleveland)", "heart"),
            ("Wisconsin Breast Cancer", "wisconsin"),
            ("breast", "wisconsin"),
            ("diabetes", "pima"),
            ("cleveland", "heart"),
        ],
    )
    def test_names_resolve_to_canonical_keys(self, raw, expected):
        assert resolve_dataset_key(raw) == expected

    @pytest.mark.parametrize("raw", ["", "   ", "mnist", "breast cancer x"])
    def test_unknown_names_return_none(self, raw):
        assert resolve_dataset_key(raw) is None

    def test_labels_printed_by_list_datasets_are_accepted(self):
        """The listing shows labels, so copying one must not fail."""

        for key in available_datasets():
            assert resolve_dataset_key(DATASET_LABELS[key]) == key

    def test_resolution_keeps_requested_order(self):
        assert resolve_dataset_keys(["heart", "wisconsin"]) == ["heart", "wisconsin"]

    def test_duplicates_collapse_to_one_entry(self):
        assert resolve_dataset_keys(["pima", "Pima Indians Diabetes", "pima"]) == ["pima"]

    def test_blank_entries_are_ignored(self):
        assert resolve_dataset_keys(["", "pima", "  "]) == ["pima"]

    def test_empty_selection_is_rejected(self):
        with pytest.raises(ValueError, match="at least one non-empty"):
            resolve_dataset_keys(["", "   "])

    def test_unknown_name_error_lists_available_keys(self):
        with pytest.raises(ValueError) as excinfo:
            resolve_dataset_keys(["pima", "nope"])

        message = str(excinfo.value)
        assert "nope" in message
        for key in available_datasets():
            assert key in message
