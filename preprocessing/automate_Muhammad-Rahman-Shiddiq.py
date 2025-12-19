import os
import argparse
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def load_data(csv_path: str) -> pd.DataFrame:
    """Load raw dataset from CSV."""
    return pd.read_csv(csv_path)


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Build preprocessing pipeline for numeric + categorical columns."""
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns
    cat_cols = X.select_dtypes(include=["object"]).columns

    numeric_transformer = Pipeline(steps=[
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols)
        ]
    )
    return preprocessor


def preprocess_and_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42
):
    """
    Apply preprocessing steps (same as notebook), then split train/test,
    then transform to model-ready arrays.
    """
    df_clean = df.copy()

    # 1) Drop customerID
    if "customerID" in df_clean.columns:
        df_clean.drop(columns=["customerID"], inplace=True)

    # 2) TotalCharges -> numeric, handle missing with median
    if "TotalCharges" in df_clean.columns:
        df_clean["TotalCharges"] = pd.to_numeric(df_clean["TotalCharges"], errors="coerce")
        df_clean["TotalCharges"].fillna(df_clean["TotalCharges"].median(), inplace=True)

    # 3) Drop duplicates
    df_clean.drop_duplicates(inplace=True)

    # 4) Encode target Churn
    if "Churn" not in df_clean.columns:
        raise ValueError("Kolom target 'Churn' tidak ditemukan dalam dataset.")
    df_clean["Churn"] = df_clean["Churn"].map({"Yes": 1, "No": 0})

    # 5) Split features/target
    X = df_clean.drop(columns=["Churn"])
    y = df_clean["Churn"]

    # 6) Build preprocessor based on X columns
    preprocessor = build_preprocessor(X)

    # 7) Train-test split (stratify target)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    # 8) Fit/transform
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    return X_train_processed, X_test_processed, y_train, y_test


def save_outputs(output_dir: str, X_train_processed, X_test_processed, y_train, y_test):
    """Save processed datasets to CSV."""
    os.makedirs(output_dir, exist_ok=True)

    pd.DataFrame(X_train_processed).to_csv(os.path.join(output_dir, "X_train.csv"), index=False)
    pd.DataFrame(X_test_processed).to_csv(os.path.join(output_dir, "X_test.csv"), index=False)
    pd.DataFrame(y_train).to_csv(os.path.join(output_dir, "y_train.csv"), index=False)
    pd.DataFrame(y_test).to_csv(os.path.join(output_dir, "y_test.csv"), index=False)


def main():
    parser = argparse.ArgumentParser(description="Automated preprocessing for Telco Customer Churn dataset.")
    parser.add_argument(
        "--input",
        type=str,
        default="../telco-churn_raw/WA_Fn-UseC_-Telco-Customer-Churn.csv",
        help="Path ke CSV dataset raw."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./namadataset_preprocessing",
        help="Folder output untuk hasil preprocessing."
    )
    args = parser.parse_args()

    df = load_data(args.input)
    X_train_p, X_test_p, y_train, y_test = preprocess_and_split(df)

    save_outputs(args.output, X_train_p, X_test_p, y_train, y_test)

    print("Preprocessing selesai ✅")
    print(f"Output tersimpan di: {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()