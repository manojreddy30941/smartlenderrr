from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
except ImportError:  # XGBoost is optional so the project runs with core sklearn deps.
    XGBClassifier = None


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "loan_approval_dataset.csv"
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "loan_model.joblib"
METADATA_PATH = MODELS_DIR / "metadata.joblib"

FEATURE_COLUMNS = [
    "no_of_dependents",
    "education",
    "self_employed",
    "income_annum",
    "loan_amount",
    "loan_term",
    "cibil_score",
    "residential_assets_value",
    "commercial_assets_value",
    "luxury_assets_value",
    "bank_asset_value",
]

CATEGORY_MAPS = {
    "education": {"Graduate": 0, "Not Graduate": 1},
    "self_employed": {"No": 0, "Yes": 1},
}

TARGET_MAP = {"Approved": 0, "Rejected": 1}
LABELS = {0: "Approved", 1: "Rejected"}


def load_dataset() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. Place loan_approval_dataset.csv in the data folder."
        )

    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()
    for column in df.select_dtypes(include="object").columns:
        df[column] = df[column].str.strip()
    return df


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    missing = {"loan_id", "loan_status", *FEATURE_COLUMNS}.difference(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    X = df[FEATURE_COLUMNS].copy()
    for column, mapping in CATEGORY_MAPS.items():
        X[column] = X[column].map(mapping)
        if X[column].isna().any():
            bad_values = sorted(df.loc[X[column].isna(), column].dropna().unique())
            raise ValueError(f"Unexpected values in {column}: {bad_values}")

    y = df["loan_status"].map(TARGET_MAP)
    if y.isna().any():
        bad_values = sorted(df.loc[y.isna(), "loan_status"].dropna().unique())
        raise ValueError(f"Unexpected loan_status values: {bad_values}")

    return X, y.astype(int)


def build_models() -> dict[str, Pipeline]:
    models = {
        "Logistic Regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=2000, random_state=42)),
            ]
        ),
        "Random Forest": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=250,
                        random_state=42,
                        class_weight="balanced",
                        n_jobs=1,
                    ),
                ),
            ]
        ),
    }

    if XGBClassifier is not None:
        models["XGBoost"] = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    XGBClassifier(
                        n_estimators=250,
                        max_depth=4,
                        learning_rate=0.08,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        eval_metric="logloss",
                        random_state=42,
                    ),
                ),
            ]
        )

    return models


def run_training_pipeline() -> None:
    print("Step 1: Loading and cleaning dataset...")
    df = load_dataset()
    X, y = prepare_features(df)

    print(f"Loaded {len(df):,} applications with {len(FEATURE_COLUMNS)} model features.")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print("\nStep 2: Training and comparing models...")
    results = []
    for name, pipeline in build_models().items():
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        results.append((accuracy, name, pipeline, predictions))
        print(f"-> {name}: {accuracy * 100:.2f}% accuracy")

    best_accuracy, best_name, best_pipeline, best_predictions = max(results, key=lambda item: item[0])

    print(f"\nWinner: {best_name} ({best_accuracy * 100:.2f}% accuracy)")
    print("\nClassification report:")
    print(classification_report(y_test, best_predictions, target_names=["Approved", "Rejected"]))

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(best_pipeline, MODEL_PATH)
    joblib.dump(
        {
            "feature_columns": FEATURE_COLUMNS,
            "category_maps": CATEGORY_MAPS,
            "target_map": TARGET_MAP,
            "labels": LABELS,
            "model_name": best_name,
            "accuracy": best_accuracy,
        },
        METADATA_PATH,
    )

    print("\nStep 3: Saved production artifacts.")
    print(f"Model: {MODEL_PATH}")
    print(f"Metadata: {METADATA_PATH}")


if __name__ == "__main__":
    run_training_pipeline()
