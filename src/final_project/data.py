from dataclasses import dataclass
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class DataSplit:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


def load_and_split_data(
    features_path: Path | str,
    target_path: Path | str,
    test_size: float = 0.20,
    random_state: int = 42,
) -> DataSplit:
    """Load features and target CSVs, performing a stratified train/test split."""
    X = pd.read_csv(features_path)
    y_df = pd.read_csv(target_path)
    
    # Extract target as 1D Series
    if "Churn Value" in y_df.columns:
        y = y_df["Churn Value"]
    else:
        y = y_df.iloc[:, 0]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    return DataSplit(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )
