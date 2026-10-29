import pytest
import pandas as pd
from pathlib import Path
from final_project.data import load_and_split_data, DataSplit

def test_load_and_split_data_shapes_and_stratification():
    features_path = Path("data/processed/X_features.csv")
    target_path = Path("data/processed/y_target.csv")

    split = load_and_split_data(
        features_path=features_path,
        target_path=target_path,
        test_size=0.20,
        random_state=42,
    )

    assert isinstance(split, DataSplit)
    assert len(split.X_train) == 5634
    assert len(split.X_test) == 1409
    assert split.X_train.shape[1] == 27
    assert split.X_test.shape[1] == 27
    assert len(split.y_train) == 5634
    assert len(split.y_test) == 1409

    # Verify no index leakage
    train_indices = set(split.X_train.index)
    test_indices = set(split.X_test.index)
    assert train_indices.isdisjoint(test_indices)

    # Verify stratified distribution
    train_churn_rate = split.y_train.mean()
    test_churn_rate = split.y_test.mean()
    assert abs(train_churn_rate - test_churn_rate) < 0.01
