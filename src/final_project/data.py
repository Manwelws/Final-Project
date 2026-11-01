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
    X_val: pd.DataFrame | None = None
    y_val: pd.Series | None = None


def load_features_and_target(
    data_dir: Path | str = "data/processed",
) -> tuple[pd.DataFrame, pd.Series]:
    """Load features and target DataFrames without splitting."""
    data_dir = Path(data_dir)
    X = pd.read_csv(data_dir / "X_features.csv")
    y_df = pd.read_csv(data_dir / "y_target.csv")
    if "Churn Value" in y_df.columns:
        y = y_df["Churn Value"]
    else:
        y = y_df.iloc[:, 0]
    return X, y


def load_and_split_data(
    features_path: Path | str,
    target_path: Path | str,
    test_size: float = 0.20,
    val_size: float = 0.0,
    random_state: int = 42,
) -> DataSplit:
    """Load features and target CSVs, performing a stratified train/test or train/val/test split."""
    X = pd.read_csv(features_path)
    y_df = pd.read_csv(target_path)
    
    # Extract target as 1D Series
    if "Churn Value" in y_df.columns:
        y = y_df["Churn Value"]
    else:
        y = y_df.iloc[:, 0]

    # Single-class guard: skip stratification when target has fewer than 2 distinct classes
    stratify = y if y.nunique() >= 2 else None

    if val_size > 0.0:
        if test_size + val_size >= 1.0:
            raise ValueError(f"test_size ({test_size}) + val_size ({val_size}) must be < 1.0")
        
        # Step 1: Hold out test partition
        X_rem, X_test, y_rem, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            stratify=stratify,
            random_state=random_state,
        )

        # Step 2: Split remaining data into train and validation
        rem_stratify = y_rem if y_rem.nunique() >= 2 else None
        relative_val_size = val_size / (1.0 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_rem,
            y_rem,
            test_size=relative_val_size,
            stratify=rem_stratify,
            random_state=random_state,
        )

        return DataSplit(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            X_val=X_val,
            y_val=y_val,
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=stratify,
        random_state=random_state,
    )

    return DataSplit(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )


def get_stratified_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.20,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Perform a stratified split on feature and target DataFrames directly."""
    return train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

