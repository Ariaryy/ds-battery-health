from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
    XGBOOST_IMPORT_ERROR: str | None = None
except Exception as error:  # pragma: no cover - depends on local system libraries
    XGBRegressor = None
    XGBOOST_AVAILABLE = False
    XGBOOST_IMPORT_ERROR = str(error)


ROOT = Path(__file__).resolve().parent
RAW_DATASET_PATH = ROOT / "datasets" / "user_behavior_dataset.csv"
CAPACITY_LOOKUP_PATH = ROOT / "datasets" / "device_battery_capacities.csv"
DATASET_PATH = ROOT / "datasets" / "user_behavior_dataset_enriched.csv"
ARTIFACT_PATH = ROOT / "battery_predictor_v5.pkl"

BUNDLE_VERSION = 5
DEFAULT_BATTERY_CAPACITY_MAH = 4600.0
MIN_USAGE_SHARE = 0.05

TARGET_COLUMN = "Battery Drain (mAh/day)"
BATTERY_CAPACITY_COLUMN = "Battery Capacity (mAh)"
APP_USAGE_PER_SCREEN_HOUR_COLUMN = "App Usage per Screen Hour"
DATA_USAGE_PER_SCREEN_HOUR_COLUMN = "Data Usage per Screen Hour"
APPS_PER_1000_MAH_COLUMN = "Apps per 1000 mAh"
SCREEN_TIME_SHARE_OF_DAY_COLUMN = "Screen Time Share of Day"
IS_HEAVY_USAGE_OUTLIER_COLUMN = "Is Heavy Usage Outlier"
IS_EFFICIENCY_OUTLIER_COLUMN = "Is Efficiency Outlier"

ENGINEERED_NUMERIC_COLUMNS = [
    APP_USAGE_PER_SCREEN_HOUR_COLUMN,
    DATA_USAGE_PER_SCREEN_HOUR_COLUMN,
    APPS_PER_1000_MAH_COLUMN,
    SCREEN_TIME_SHARE_OF_DAY_COLUMN,
    IS_HEAVY_USAGE_OUTLIER_COLUMN,
    IS_EFFICIENCY_OUTLIER_COLUMN,
]

STATIC_COLUMNS = ["Operating System", "Number of Apps Installed", BATTERY_CAPACITY_COLUMN]
DYNAMIC_COLUMNS = ["App Usage Time (min/day)", "Screen On Time (hours/day)", "Data Usage (MB/day)"]
FEATURE_COLUMNS = STATIC_COLUMNS + DYNAMIC_COLUMNS + ENGINEERED_NUMERIC_COLUMNS
NUMERIC_COLUMNS = [
    "App Usage Time (min/day)",
    "Screen On Time (hours/day)",
    "Battery Drain (mAh/day)",
    "Number of Apps Installed",
    "Data Usage (MB/day)",
    BATTERY_CAPACITY_COLUMN,
    "Age",
    "User Behavior Class",
]
RAW_SUMMARY_COLUMNS = NUMERIC_COLUMNS.copy()
CATEGORICAL_COLUMNS = ["Device Model", "Operating System", "Gender"]
USAGE_CURVE = [
    0.01,
    0.01,
    0.01,
    0.01,
    0.01,
    0.01,
    0.04,
    0.04,
    0.04,
    0.055,
    0.055,
    0.055,
    0.055,
    0.055,
    0.055,
    0.055,
    0.055,
    0.07,
    0.07,
    0.07,
    0.07,
    0.07,
    0.015,
    0.015,
]


@dataclass(frozen=True)
class DeviceSpec:
    device_model: str
    operating_system: str
    number_of_apps_installed: int
    battery_capacity_mah: float | None = None


@dataclass(frozen=True)
class UsageSnapshot:
    current_hour: float
    current_battery_pct: float
    app_usage_minutes_so_far: float
    screen_on_hours_so_far: float
    data_usage_mb_so_far: float
    starting_battery_pct: float = 100.0


@dataclass(frozen=True)
class ChargingPolicy:
    preferred_level_pct: float = 50.0
    start_charge_pct: float = 45.0
    minimum_stop_charge_pct: float = 55.0
    maximum_stop_charge_pct: float = 70.0
    charge_rate_low_band_pct_per_hour: float = 30.0
    charge_rate_mid_band_pct_per_hour: float = 18.0
    charge_rate_high_band_pct_per_hour: float = 7.0
    time_step_hours: float = 0.25


@dataclass(frozen=True)
class ChargeSession:
    start_hour: float
    start_level_pct: float
    recommended_stop_level_pct: float


@dataclass(frozen=True)
class DrainForecast:
    model_full_day_drain_mah: float
    observed_rate_full_day_drain_mah: float
    predicted_full_day_drain_mah: float
    observed_drain_so_far_mah: float
    predicted_remaining_drain_mah: float
    historical_dynamic_usage: dict[str, float]
    projected_dynamic_usage: dict[str, float]
    blended_feature_row: dict[str, Any]
    today_usage_weight: float
    battery_capacity_mah: float
    battery_capacity_source: Literal["provided", "lookup_table", "assumed_default"]
    cumulative_usage_share: float


@dataclass(frozen=True)
class ChargingPlan:
    sessions: list[ChargeSession]
    projected_lowest_battery_pct: float
    projected_end_battery_pct: float
    no_charge_lowest_battery_pct: float
    no_charge_end_battery_pct: float


@dataclass
class PredictorBundle:
    bundle_version: int
    model: Any
    model_name: str
    preprocessor: ColumnTransformer
    priors: dict[str, Any]
    dynamic_bounds: dict[str, dict[str, float]]
    metrics: dict[str, float]
    usage_curve: list[float]


@dataclass(frozen=True)
class CleaningReport:
    original_rows: int
    cleaned_rows: int
    removed_duplicate_rows: int
    rows_removed_for_missing_required_values: int
    rows_removed_for_rule_violations: int
    remaining_missing_values: int
    capped_values_count: int
    heavy_usage_outlier_rows: int
    efficiency_outlier_rows: int
    engineered_feature_count: int
    battery_capacity_coverage_pct: float


def load_dataset(dataset_path: Path = DATASET_PATH) -> pd.DataFrame:
    if dataset_path.exists():
        return pd.read_csv(dataset_path)

    raw_df = pd.read_csv(RAW_DATASET_PATH)
    if CAPACITY_LOOKUP_PATH.exists():
        capacity_df = pd.read_csv(CAPACITY_LOOKUP_PATH)
        merged = raw_df.merge(capacity_df[["Device Model", BATTERY_CAPACITY_COLUMN]], on="Device Model", how="left")
        if BATTERY_CAPACITY_COLUMN in merged.columns:
            return merged
    return raw_df


def load_capacity_lookup(capacity_lookup_path: Path = CAPACITY_LOOKUP_PATH) -> pd.DataFrame:
    if not capacity_lookup_path.exists():
        return pd.DataFrame(columns=["Device Model", BATTERY_CAPACITY_COLUMN])
    return pd.read_csv(capacity_lookup_path)


def clean_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
    cleaned = df.copy()
    original_rows = len(cleaned)

    if BATTERY_CAPACITY_COLUMN not in cleaned.columns:
        raise ValueError(
            f"Dataset is missing '{BATTERY_CAPACITY_COLUMN}'. "
            "Run `uv run python scripts/enrich_battery_capacities.py` first."
        )

    for column in CATEGORICAL_COLUMNS:
        cleaned[column] = cleaned[column].astype(str).str.strip()

    for column in NUMERIC_COLUMNS:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    battery_capacity_coverage_pct = (
        float(cleaned[BATTERY_CAPACITY_COLUMN].notna().mean() * 100.0)
        if BATTERY_CAPACITY_COLUMN in cleaned.columns
        else 0.0
    )

    deduplicated = cleaned.drop_duplicates()
    removed_duplicate_rows = original_rows - len(deduplicated)

    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    required_columns_without_engineered = STATIC_COLUMNS + DYNAMIC_COLUMNS + [TARGET_COLUMN]
    cleaned = deduplicated.dropna(subset=required_columns_without_engineered)
    rows_removed_for_missing_required_values = len(deduplicated) - len(cleaned)

    validation_mask = (
        cleaned["App Usage Time (min/day)"].between(0, 1440)
        & cleaned["Screen On Time (hours/day)"].between(0, 24)
        & cleaned["Data Usage (MB/day)"].ge(0)
        & cleaned["Number of Apps Installed"].between(1, 500)
        & cleaned[BATTERY_CAPACITY_COLUMN].between(1500, 7000)
        & cleaned[TARGET_COLUMN].between(100, 10000)
    )
    rows_removed_for_rule_violations = int((~validation_mask).sum())
    cleaned = cleaned.loc[validation_mask].copy()

    cleaned, capped_values_count = cap_outliers_iqr(
        cleaned,
        columns=[
            "App Usage Time (min/day)",
            "Screen On Time (hours/day)",
            "Data Usage (MB/day)",
            "Number of Apps Installed",
            TARGET_COLUMN,
        ],
    )
    cleaned = add_engineered_features(cleaned)

    remaining_missing_values = int(cleaned.isna().sum().sum())
    report = CleaningReport(
        original_rows=original_rows,
        cleaned_rows=len(cleaned),
        removed_duplicate_rows=removed_duplicate_rows,
        rows_removed_for_missing_required_values=rows_removed_for_missing_required_values,
        rows_removed_for_rule_violations=rows_removed_for_rule_violations,
        remaining_missing_values=remaining_missing_values,
        capped_values_count=capped_values_count,
        heavy_usage_outlier_rows=int(cleaned[IS_HEAVY_USAGE_OUTLIER_COLUMN].sum()),
        efficiency_outlier_rows=int(cleaned[IS_EFFICIENCY_OUTLIER_COLUMN].sum()),
        engineered_feature_count=len(ENGINEERED_NUMERIC_COLUMNS),
        battery_capacity_coverage_pct=round(battery_capacity_coverage_pct, 2),
    )
    return cleaned, report


def summarize_dataset(df: pd.DataFrame) -> pd.DataFrame:
    summary_columns = RAW_SUMMARY_COLUMNS + ENGINEERED_NUMERIC_COLUMNS
    summary = df[summary_columns].describe().T
    return summary[["mean", "std", "min", "25%", "50%", "75%", "max"]].round(2)


def cap_outliers_iqr(df: pd.DataFrame, columns: list[str]) -> tuple[pd.DataFrame, int]:
    capped = df.copy()
    capped_values_count = 0
    for column in columns:
        capped[column] = capped[column].astype(float)
        q1 = capped[column].quantile(0.25)
        q3 = capped[column].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - (1.5 * iqr)
        upper = q3 + (1.5 * iqr)
        below_mask = capped[column] < lower
        above_mask = capped[column] > upper
        capped_values_count += int(below_mask.sum() + above_mask.sum())
        capped.loc[below_mask, column] = lower
        capped.loc[above_mask, column] = upper
    return capped, capped_values_count


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    engineered = df.copy()
    safe_screen_on = engineered["Screen On Time (hours/day)"].clip(lower=0.1)
    safe_capacity = engineered[BATTERY_CAPACITY_COLUMN].clip(lower=1000.0)

    engineered[APP_USAGE_PER_SCREEN_HOUR_COLUMN] = engineered["App Usage Time (min/day)"] / safe_screen_on
    engineered[DATA_USAGE_PER_SCREEN_HOUR_COLUMN] = engineered["Data Usage (MB/day)"] / safe_screen_on
    engineered[APPS_PER_1000_MAH_COLUMN] = engineered["Number of Apps Installed"] / (safe_capacity / 1000.0)
    engineered[SCREEN_TIME_SHARE_OF_DAY_COLUMN] = engineered["Screen On Time (hours/day)"] / 24.0

    heavy_usage_score = (
        engineered["App Usage Time (min/day)"].rank(pct=True)
        + engineered["Screen On Time (hours/day)"].rank(pct=True)
        + engineered["Data Usage (MB/day)"].rank(pct=True)
    ) / 3.0
    if TARGET_COLUMN in engineered.columns:
        drain_efficiency_score = (engineered[TARGET_COLUMN] / safe_capacity).rank(pct=True)
    else:
        drain_efficiency_score = pd.Series([0.0] * len(engineered), index=engineered.index)

    engineered[IS_HEAVY_USAGE_OUTLIER_COLUMN] = (heavy_usage_score >= 0.95).astype(int)
    engineered[IS_EFFICIENCY_OUTLIER_COLUMN] = (drain_efficiency_score >= 0.95).astype(int)
    return engineered


def validate_snapshot(snapshot: UsageSnapshot) -> None:
    if not 0 < snapshot.current_hour <= 24:
        raise ValueError("current_hour must be between 0 and 24.")
    if not 0 <= snapshot.current_battery_pct <= 100:
        raise ValueError("current_battery_pct must be between 0 and 100.")
    if not 0 <= snapshot.starting_battery_pct <= 100:
        raise ValueError("starting_battery_pct must be between 0 and 100.")
    if snapshot.starting_battery_pct < snapshot.current_battery_pct:
        raise ValueError("starting_battery_pct must be greater than or equal to current_battery_pct.")
    if snapshot.app_usage_minutes_so_far < 0 or snapshot.screen_on_hours_so_far < 0 or snapshot.data_usage_mb_so_far < 0:
        raise ValueError("Usage values must be non-negative.")


def train_predictor(
    dataset_path: Path = DATASET_PATH,
    artifact_path: Path = ARTIFACT_PATH,
) -> PredictorBundle:
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
    }
    if XGBOOST_IMPORT_ERROR:
        metrics["xgboost_available"] = 0.0
    else:
        metrics["xgboost_available"] = 1.0

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
        return pd.DataFrame(
            {
                "feature": feature_names,
                "importance": [0.0] * len(feature_names),
            }
        )
    importance = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": bundle.model.feature_importances_,
        }
    )
    return importance.sort_values("importance", ascending=False).reset_index(drop=True)


def get_cleaning_overview(dataset_path: Path = DATASET_PATH) -> tuple[pd.DataFrame, pd.DataFrame, CleaningReport]:
    raw_df = load_dataset(dataset_path)
    cleaned_df, report = clean_dataset(raw_df)
    return raw_df, cleaned_df, report


def build_regressor() -> tuple[Any, str]:
    if XGBOOST_AVAILABLE and XGBRegressor is not None:
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

    return (
        RandomForestRegressor(
            n_estimators=300,
            max_depth=12,
            min_samples_split=4,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),
        "Random Forest Regressor",
    )


def build_usage_priors(train_df: pd.DataFrame) -> dict[str, Any]:
    priors = {
        "global": train_df[DYNAMIC_COLUMNS].median().to_dict(),
        "by_os": train_df.groupby("Operating System")[DYNAMIC_COLUMNS].median().to_dict(orient="index"),
    }
    return priors


def build_dynamic_bounds(train_df: pd.DataFrame) -> dict[str, dict[str, float]]:
    return {
        "low": train_df[DYNAMIC_COLUMNS].quantile(0.05).to_dict(),
        "high": train_df[DYNAMIC_COLUMNS].quantile(0.95).to_dict(),
    }


def load_predictor(artifact_path: Path = ARTIFACT_PATH, retrain: bool = False) -> PredictorBundle:
    if retrain or not artifact_path.exists():
        return train_predictor(artifact_path=artifact_path)

    bundle = joblib.load(artifact_path)
    if getattr(bundle, "bundle_version", None) != BUNDLE_VERSION:
        return train_predictor(artifact_path=artifact_path)
    return bundle


def normalize_device_spec(device_spec: DeviceSpec) -> dict[str, Any]:
    return {
        "Operating System": device_spec.operating_system,
        "Number of Apps Installed": device_spec.number_of_apps_installed,
        BATTERY_CAPACITY_COLUMN: resolve_battery_capacity(device_spec)[0],
    }


def resolve_battery_capacity(device_spec: DeviceSpec) -> tuple[float, Literal["provided", "lookup_table", "assumed_default"]]:
    if device_spec.battery_capacity_mah is not None:
        return device_spec.battery_capacity_mah, "provided"
    lookup_df = load_capacity_lookup()
    match = lookup_df.loc[lookup_df["Device Model"] == device_spec.device_model, BATTERY_CAPACITY_COLUMN]
    if not match.empty:
        return float(match.iloc[0]), "lookup_table"
    return DEFAULT_BATTERY_CAPACITY_MAH, "assumed_default"


def lookup_historical_usage(bundle: PredictorBundle, device_spec: DeviceSpec) -> dict[str, float]:
    if device_spec.operating_system in bundle.priors["by_os"]:
        return bundle.priors["by_os"][device_spec.operating_system]
    return bundle.priors["global"]


def cumulative_usage_share(current_hour: float, usage_curve: list[float]) -> float:
    full_hours = int(current_hour)
    fractional_hour = current_hour - full_hours

    share = sum(usage_curve[:full_hours])
    if full_hours < len(usage_curve):
        share += usage_curve[full_hours] * fractional_hour

    return clip(share, MIN_USAGE_SHARE, 1.0)


def project_usage_to_daily(
    snapshot: UsageSnapshot,
    bounds: dict[str, dict[str, float]],
    usage_curve: list[float],
) -> dict[str, float]:
    usage_share = cumulative_usage_share(snapshot.current_hour, usage_curve)
    projected = {
        "App Usage Time (min/day)": snapshot.app_usage_minutes_so_far / usage_share,
        "Screen On Time (hours/day)": snapshot.screen_on_hours_so_far / usage_share,
        "Data Usage (MB/day)": snapshot.data_usage_mb_so_far / usage_share,
    }
    return {
        feature: clip(projected[feature], bounds["low"][feature], bounds["high"][feature])
        for feature in DYNAMIC_COLUMNS
    }


def blend_dynamic_usage(
    historical_usage: dict[str, float],
    projected_usage: dict[str, float],
    current_hour: float,
    usage_curve: list[float],
) -> tuple[dict[str, float], float]:
    today_usage_weight = clip(cumulative_usage_share(current_hour, usage_curve), 0.2, 0.85)
    blended = {}
    for feature in DYNAMIC_COLUMNS:
        blended[feature] = (
            historical_usage[feature] * (1.0 - today_usage_weight)
            + projected_usage[feature] * today_usage_weight
        )
    return blended, today_usage_weight


def build_feature_row(device_spec: DeviceSpec, dynamic_usage: dict[str, float]) -> dict[str, Any]:
    feature_row = normalize_device_spec(device_spec)
    feature_row.update(dynamic_usage)
    feature_row = add_engineered_features(pd.DataFrame([feature_row])).iloc[0].to_dict()
    return feature_row


def predict_daily_drain(bundle: PredictorBundle, feature_row: dict[str, Any]) -> float:
    df_input = pd.DataFrame([feature_row], columns=FEATURE_COLUMNS)
    X_processed = bundle.preprocessor.transform(df_input)
    return float(bundle.model.predict(X_processed)[0])


def forecast_drain(
    bundle: PredictorBundle,
    device_spec: DeviceSpec,
    snapshot: UsageSnapshot,
) -> DrainForecast:
    validate_snapshot(snapshot)

    battery_capacity_mah, battery_capacity_source = resolve_battery_capacity(device_spec)
    historical_usage = lookup_historical_usage(bundle, device_spec)
    usage_share = cumulative_usage_share(snapshot.current_hour, bundle.usage_curve)
    projected_usage = project_usage_to_daily(snapshot, bundle.dynamic_bounds, bundle.usage_curve)
    blended_usage, today_usage_weight = blend_dynamic_usage(
        historical_usage=historical_usage,
        projected_usage=projected_usage,
        current_hour=snapshot.current_hour,
        usage_curve=bundle.usage_curve,
    )
    feature_row = build_feature_row(device_spec, blended_usage)
    model_full_day_drain_mah = predict_daily_drain(bundle, feature_row)

    observed_drain_so_far_mah = battery_capacity_mah * (
        (snapshot.starting_battery_pct - snapshot.current_battery_pct) / 100.0
    )
    observed_rate_full_day_drain_mah = observed_drain_so_far_mah / usage_share
    observed_weight = clip(usage_share, 0.15, 0.8)
    predicted_full_day_drain_mah = (
        model_full_day_drain_mah * (1.0 - observed_weight)
        + observed_rate_full_day_drain_mah * observed_weight
    )
    predicted_full_day_drain_mah = max(predicted_full_day_drain_mah, observed_drain_so_far_mah)
    predicted_remaining_drain_mah = max(0.0, predicted_full_day_drain_mah - observed_drain_so_far_mah)

    return DrainForecast(
        model_full_day_drain_mah=model_full_day_drain_mah,
        observed_rate_full_day_drain_mah=observed_rate_full_day_drain_mah,
        predicted_full_day_drain_mah=predicted_full_day_drain_mah,
        observed_drain_so_far_mah=observed_drain_so_far_mah,
        predicted_remaining_drain_mah=predicted_remaining_drain_mah,
        historical_dynamic_usage=historical_usage,
        projected_dynamic_usage=projected_usage,
        blended_feature_row=feature_row,
        today_usage_weight=today_usage_weight,
        battery_capacity_mah=battery_capacity_mah,
        battery_capacity_source=battery_capacity_source,
        cumulative_usage_share=usage_share,
    )


def recommend_charging_plan(
    forecast: DrainForecast,
    snapshot: UsageSnapshot,
    policy: ChargingPolicy = ChargingPolicy(),
) -> ChargingPlan:
    remaining_drain_pct = (forecast.predicted_remaining_drain_mah / forecast.battery_capacity_mah) * 100.0
    no_charge_levels, _ = simulate_battery_levels(
        current_battery_pct=snapshot.current_battery_pct,
        current_hour=snapshot.current_hour,
        remaining_drain_pct=remaining_drain_pct,
        snapshot=snapshot,
        historical_usage=forecast.historical_dynamic_usage,
        usage_curve=USAGE_CURVE,
        policy=policy,
        allow_charging=False,
    )
    charge_levels, sessions = simulate_battery_levels(
        current_battery_pct=snapshot.current_battery_pct,
        current_hour=snapshot.current_hour,
        remaining_drain_pct=remaining_drain_pct,
        snapshot=snapshot,
        historical_usage=forecast.historical_dynamic_usage,
        usage_curve=USAGE_CURVE,
        policy=policy,
        allow_charging=True,
    )

    return ChargingPlan(
        sessions=sessions,
        projected_lowest_battery_pct=min(level for _, level in charge_levels),
        projected_end_battery_pct=charge_levels[-1][1],
        no_charge_lowest_battery_pct=min(level for _, level in no_charge_levels),
        no_charge_end_battery_pct=no_charge_levels[-1][1],
    )


def recommended_stop_level_for_session(
    current_level_pct: float,
    remaining_drain_pct_from_now: float,
    policy: ChargingPolicy,
) -> float:
    ideal_stop_level_pct = remaining_drain_pct_from_now + policy.preferred_level_pct
    bounded_stop_level_pct = clip(
        ideal_stop_level_pct,
        policy.minimum_stop_charge_pct,
        policy.maximum_stop_charge_pct,
    )
    return max(current_level_pct, bounded_stop_level_pct)


def charge_rate_for_level(level_pct: float, policy: ChargingPolicy) -> float:
    if level_pct < 50.0:
        return policy.charge_rate_low_band_pct_per_hour
    if level_pct < 80.0:
        return policy.charge_rate_mid_band_pct_per_hour
    return policy.charge_rate_high_band_pct_per_hour


def simulate_battery_levels(
    current_battery_pct: float,
    current_hour: float,
    remaining_drain_pct: float,
    snapshot: UsageSnapshot,
    historical_usage: dict[str, float],
    usage_curve: list[float],
    policy: ChargingPolicy,
    allow_charging: bool,
) -> tuple[list[tuple[float, float]], list[ChargeSession]]:
    drain_weights = build_drain_weights(
        current_hour=current_hour,
        snapshot=snapshot,
        historical_usage=historical_usage,
        time_step_hours=policy.time_step_hours,
    )
    drain_per_step = [remaining_drain_pct * weight for weight in drain_weights]

    level = current_battery_pct
    charging = False
    session_start_hour = 0.0
    session_start_level = 0.0
    sessions: list[ChargeSession] = []
    levels: list[tuple[float, float]] = [(current_hour, round(level, 2))]
    target_stop_level_pct = policy.minimum_stop_charge_pct

    for step_index, drain_pct in enumerate(drain_per_step):
        step_start_hour = current_hour + (step_index * policy.time_step_hours)
        remaining_drain_pct_from_now = sum(drain_per_step[step_index:])

        if allow_charging and not charging and level <= policy.start_charge_pct:
            charging = True
            session_start_hour = step_start_hour
            session_start_level = level
            target_stop_level_pct = recommended_stop_level_for_session(
                current_level_pct=level,
                remaining_drain_pct_from_now=remaining_drain_pct_from_now,
                policy=policy,
            )

        if charging:
            charge_rate_pct_per_hour = charge_rate_for_level(level, policy)
            charge_added = charge_rate_pct_per_hour * policy.time_step_hours
            if level + charge_added >= target_stop_level_pct:
                level = target_stop_level_pct
                charging = False
                sessions.append(
                    ChargeSession(
                        start_hour=session_start_hour,
                        start_level_pct=round(session_start_level, 2),
                        recommended_stop_level_pct=round(level, 2),
                    )
                )
            else:
                level += charge_added

        level = max(0.0, min(100.0, level - drain_pct))
        levels.append((min(24.0, step_start_hour + policy.time_step_hours), round(level, 2)))

    if not levels or levels[-1][0] < 24.0:
        levels.append((24.0, round(level, 2)))

    return levels, sessions


def build_drain_weights(
    current_hour: float,
    snapshot: UsageSnapshot,
    historical_usage: dict[str, float],
    time_step_hours: float,
) -> list[float]:
    steps = max(1, int(round((24.0 - current_hour) / time_step_hours)))
    usage_pressure = estimate_usage_pressure(snapshot, historical_usage)

    weights = []
    for step_index in range(steps):
        clock = current_hour + (step_index * time_step_hours)
        base_weight = hour_activity_multiplier(clock)
        near_term_boost = usage_pressure if step_index < int(3.0 / time_step_hours) else 1.0
        weights.append(base_weight * near_term_boost)

    total_weight = sum(weights)
    return [weight / total_weight for weight in weights]


def estimate_usage_pressure(snapshot: UsageSnapshot, historical_usage: dict[str, float]) -> float:
    observed_hourly = {
        "App Usage Time (min/day)": snapshot.app_usage_minutes_so_far / snapshot.current_hour,
        "Screen On Time (hours/day)": snapshot.screen_on_hours_so_far / snapshot.current_hour,
        "Data Usage (MB/day)": snapshot.data_usage_mb_so_far / snapshot.current_hour,
    }
    ratios = []
    for feature in DYNAMIC_COLUMNS:
        historical_hourly = max(historical_usage[feature] / 24.0, 0.001)
        ratios.append(observed_hourly[feature] / historical_hourly)

    return clip(sum(ratios) / len(ratios), 0.7, 1.8)


def hour_activity_multiplier(clock: float) -> float:
    hour = clock % 24
    if hour < 6 or hour >= 22:
        return 0.55
    if hour < 9:
        return 1.1
    if hour < 18:
        return 1.0
    return 1.2


def clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def format_hour(hour: float) -> str:
    total_minutes = int(round(hour * 60))
    clamped_minutes = max(0, min(total_minutes, 24 * 60))
    hours, minutes = divmod(clamped_minutes, 60)
    return f"{hours:02d}:{minutes:02d}"


def print_report(forecast: DrainForecast, plan: ChargingPlan, policy: ChargingPolicy, metrics: dict[str, float]) -> None:
    print(f"Model MAE: {metrics['mae']:.2f} mAh/day")
    print(
        f"Battery capacity used: {forecast.battery_capacity_mah:.1f} mAh "
        f"({forecast.battery_capacity_source})"
    )
    print(f"Expected usage-share observed by now: {forecast.cumulative_usage_share:.2f}")
    print(f"Model-only full-day drain: {forecast.model_full_day_drain_mah:.1f} mAh")
    print(f"Observed-rate full-day drain: {forecast.observed_rate_full_day_drain_mah:.1f} mAh")
    print(f"Predicted full-day drain: {forecast.predicted_full_day_drain_mah:.1f} mAh")
    print(f"Observed drain so far: {forecast.observed_drain_so_far_mah:.1f} mAh")
    print(f"Predicted remaining drain: {forecast.predicted_remaining_drain_mah:.1f} mAh")
    print(f"Today-usage weight: {forecast.today_usage_weight:.2f}")
    print()
    print("Daily feature blend used for prediction:")
    for feature in DYNAMIC_COLUMNS:
        print(
            f"  {feature}: typical={forecast.historical_dynamic_usage[feature]:.1f}, "
            f"projected={forecast.projected_dynamic_usage[feature]:.1f}, "
            f"blended={forecast.blended_feature_row[feature]:.1f}"
        )
    print()
    print(
        f"No-charge forecast: low={plan.no_charge_lowest_battery_pct:.1f}%, "
        f"end={plan.no_charge_end_battery_pct:.1f}%"
    )
    print(
        f"Recommended plan: low={plan.projected_lowest_battery_pct:.1f}%, "
        f"end={plan.projected_end_battery_pct:.1f}%"
    )
    print(
        f"Charging starts near {policy.start_charge_pct:.0f}% and the assistant aims to keep "
        f"battery around {policy.preferred_level_pct:.0f}% without exceeding "
        f"{policy.maximum_stop_charge_pct:.0f}% unless you override it."
    )
    print()

    if not plan.sessions:
        print("Recommendation: do not charge today unless usage changes materially.")
        return

    print("Recommended charging sessions:")
    for index, session in enumerate(plan.sessions, start=1):
        print(
            f"  {index}. Start charging at {format_hour(session.start_hour)} "
            f"when battery is about {session.start_level_pct:.1f}% and unplug near "
            f"{session.recommended_stop_level_pct:.1f}%."
        )


def main() -> None:
    bundle = load_predictor()

    device_spec = DeviceSpec(
        device_model="Xiaomi Mi 11",
        operating_system="Android",
        number_of_apps_installed=85,
    )
    snapshot = UsageSnapshot(
        current_hour=13.0,
        current_battery_pct=38.0,
        starting_battery_pct=100.0,
        app_usage_minutes_so_far=320.0,
        screen_on_hours_so_far=5.8,
        data_usage_mb_so_far=1350.0,
    )
    policy = ChargingPolicy()

    forecast = forecast_drain(bundle=bundle, device_spec=device_spec, snapshot=snapshot)
    plan = recommend_charging_plan(forecast=forecast, snapshot=snapshot, policy=policy)
    print_report(forecast=forecast, plan=plan, policy=policy, metrics=bundle.metrics)


if __name__ == "__main__":
    main()
