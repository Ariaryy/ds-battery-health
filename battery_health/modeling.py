from __future__ import annotations

from typing import Any

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

from .config import (
    ARTIFACT_PATH,
    BATTERY_CAPACITY_COLUMN,
    BUNDLE_VERSION,
    DATASET_PATH,
    DYNAMIC_COLUMNS,
    ENGINEERED_NUMERIC_COLUMNS,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    USAGE_CURVE,
)
from .data import clean_dataset, load_dataset
from .schemas import PredictorBundle


def train_predictor(dataset_path=DATASET_PATH, artifact_path=ARTIFACT_PATH) -> PredictorBundle:
    raw_df = load_dataset(dataset_path)
    df, _ = clean_dataset(raw_df)

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["Operating System"]),
            (
                "num",
                StandardScaler(),
                [
                    "App Usage Time (min/day)",
                    "Screen On Time (hours/day)",
                    "Data Usage (MB/day)",
                    "Number of Apps Installed",
                    BATTERY_CAPACITY_COLUMN,
                    *ENGINEERED_NUMERIC_COLUMNS,
                ],
            ),
        ]
    )

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    model, model_name = build_regressor()
    model.fit(X_train_processed, y_train)

    y_pred = model.predict(X_test_processed)
    metrics = {
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "train_rows": float(len(X_train)),
        "test_rows": float(len(X_test)),
        "xgboost_available": 1.0,
    }

    bundle = PredictorBundle(
        bundle_version=BUNDLE_VERSION,
        model=model,
        model_name=model_name,
        preprocessor=preprocessor,
        priors=build_usage_priors(X_train),
        dynamic_bounds=build_dynamic_bounds(X_train),
        metrics=metrics,
        usage_curve=USAGE_CURVE.copy(),
    )
    joblib.dump(bundle, artifact_path)
    return bundle


def get_feature_importance(bundle: PredictorBundle) -> pd.DataFrame:
    feature_names = bundle.preprocessor.get_feature_names_out()
    if not hasattr(bundle.model, "feature_importances_"):
        return pd.DataFrame({"feature": feature_names, "importance": [0.0] * len(feature_names)})
    importance = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": bundle.model.feature_importances_,
        }
    )
    return importance.sort_values("importance", ascending=False).reset_index(drop=True)


def build_regressor() -> tuple[Any, str]:
    return (
        XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
        ),
        "XGBoost Regressor",
    )


def build_usage_priors(train_df: pd.DataFrame) -> dict[str, Any]:
    return {
        "global": train_df[DYNAMIC_COLUMNS].median().to_dict(),
        "by_os": train_df.groupby("Operating System")[DYNAMIC_COLUMNS].median().to_dict(orient="index"),
    }


def build_dynamic_bounds(train_df: pd.DataFrame) -> dict[str, dict[str, float]]:
    return {
        "low": train_df[DYNAMIC_COLUMNS].quantile(0.05).to_dict(),
        "high": train_df[DYNAMIC_COLUMNS].quantile(0.95).to_dict(),
    }


def load_predictor(artifact_path=ARTIFACT_PATH, retrain: bool = False) -> PredictorBundle:
    if retrain or not artifact_path.exists():
        return train_predictor(artifact_path=artifact_path)

    bundle = joblib.load(artifact_path)
    if getattr(bundle, "bundle_version", None) != BUNDLE_VERSION:
        return train_predictor(artifact_path=artifact_path)
    return bundle


def get_evaluation_results(bundle: PredictorBundle, dataset_path=DATASET_PATH) -> tuple[dict[str, float], pd.DataFrame]:
    raw_df = load_dataset(dataset_path)
    df, _ = clean_dataset(raw_df)

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_test_processed = bundle.preprocessor.transform(X_test)
    y_pred = bundle.model.predict(X_test_processed)

    evaluation_df = X_test.copy()
    evaluation_df["actual_drain_mah"] = y_test.to_numpy()
    evaluation_df["predicted_drain_mah"] = y_pred
    evaluation_df["residual_mah"] = evaluation_df["actual_drain_mah"] - evaluation_df["predicted_drain_mah"]
    evaluation_df["absolute_error_mah"] = evaluation_df["residual_mah"].abs()

    metrics = {
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "rmse": float(root_mean_squared_error(y_test, y_pred)),
        "r2": float(r2_score(y_test, y_pred)),
        "test_rows": float(len(X_test)),
    }
    return metrics, evaluation_df.reset_index(drop=True)
