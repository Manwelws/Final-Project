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
