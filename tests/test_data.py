import pytest
from pathlib import Path
from final_project.data import load_and_split_data, DataSplit


def test_load_and_split_data_shapes_and_stratification(synthetic_data_paths):
    features_path, target_path, _ = synthetic_data_paths

    split = load_and_split_data(
        features_path=features_path,
        target_path=target_path,
        test_size=0.20,
        random_state=42,
    )

    assert isinstance(split, DataSplit)
    total_samples = len(split.X_train) + len(split.X_test)
    assert len(split.X_train) == len(split.y_train)
    assert len(split.X_test) == len(split.y_test)
    assert abs(len(split.X_test) / total_samples - 0.20) < 0.05
    assert split.X_train.shape[1] == 27
    assert split.X_test.shape[1] == 27

    # Verify no index leakage
    train_indices = set(split.X_train.index)
    test_indices = set(split.X_test.index)
    assert train_indices.isdisjoint(test_indices)

    # Verify stratified distribution
    train_churn_rate = split.y_train.mean()
    test_churn_rate = split.y_test.mean()
    assert abs(train_churn_rate - test_churn_rate) < 0.05


def test_load_and_split_data_with_validation_split(synthetic_data_paths):
    features_path, target_path, _ = synthetic_data_paths

    split = load_and_split_data(
        features_path=features_path,
        target_path=target_path,
        test_size=0.20,
        val_size=0.10,
        random_state=42,
    )

    assert split.X_val is not None
    assert split.y_val is not None
    total_samples = len(split.X_train) + len(split.X_val) + len(split.X_test)
    assert total_samples == 100

    # Verify disjoint partitions
    train_idx = set(split.X_train.index)
    val_idx = set(split.X_val.index)
    test_idx = set(split.X_test.index)
    assert train_idx.isdisjoint(val_idx)
    assert train_idx.isdisjoint(test_idx)
    assert val_idx.isdisjoint(test_idx)

    # Proportions: roughly 70/10/20
    assert abs(len(split.X_train) / total_samples - 0.70) < 0.05
    assert abs(len(split.X_val) / total_samples - 0.10) < 0.05
    assert abs(len(split.X_test) / total_samples - 0.20) < 0.05


def test_load_and_split_data_single_class_guard(tmp_path):
    # Dataset with single class (e.g. only 0s)
    import pandas as pd
    import numpy as np
    from final_project.serving.adapter import MODEL_FEATURE_NAMES

    feat_df = pd.DataFrame({f: np.random.randn(30) for f in MODEL_FEATURE_NAMES})
    target_df = pd.DataFrame({"Churn Value": [0] * 30})

    f_path = tmp_path / "X_features.csv"
    t_path = tmp_path / "y_target.csv"
    feat_df.to_csv(f_path, index=False)
    target_df.to_csv(t_path, index=False)

    # Should not raise ValueError even with single class
    split = load_and_split_data(
        features_path=f_path,
        target_path=t_path,
        test_size=0.20,
        val_size=0.10,
        random_state=42,
    )
    assert len(split.X_train) > 0
    assert len(split.X_val) > 0
    assert len(split.X_test) > 0
