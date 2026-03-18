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
DATASET_PATH = ROOT / "datasets" / "user_behavior_dataset.csv"
ARTIFACT_PATH = ROOT / "battery_predictor_v4.pkl"

BUNDLE_VERSION = 4
DEFAULT_BATTERY_CAPACITY_MAH = 4600.0
MIN_USAGE_SHARE = 0.05

TARGET_COLUMN = "Battery Drain (mAh/day)"
STATIC_COLUMNS = ["Device Model", "Operating System", "Number of Apps Installed"]
DYNAMIC_COLUMNS = ["App Usage Time (min/day)", "Screen On Time (hours/day)", "Data Usage (MB/day)"]
FEATURE_COLUMNS = STATIC_COLUMNS + DYNAMIC_COLUMNS
NUMERIC_COLUMNS = [
    "App Usage Time (min/day)",
    "Screen On Time (hours/day)",
    "Battery Drain (mAh/day)",
    "Number of Apps Installed",
    "Data Usage (MB/day)",
    "Age",
    "User Behavior Class",
]
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
    stop_charge_pct: float = 55.0
    charge_rate_low_band_pct_per_hour: float = 30.0
    charge_rate_mid_band_pct_per_hour: float = 18.0
    charge_rate_high_band_pct_per_hour: float = 7.0
    time_step_hours: float = 0.25


@dataclass(frozen=True)
class ChargeSession:
    start_hour: float
    stop_hour: float
    start_level_pct: float
    stop_level_pct: float


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
    battery_capacity_source: Literal["provided", "assumed_default"]
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
    remaining_missing_values: int


def load_dataset(dataset_path: Path = DATASET_PATH) -> pd.DataFrame:
    return pd.read_csv(dataset_path)


def clean_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
    cleaned = df.copy()
    original_rows = len(cleaned)

    for column in CATEGORICAL_COLUMNS:
        cleaned[column] = cleaned[column].astype(str).str.strip()

    for column in NUMERIC_COLUMNS:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    deduplicated = cleaned.drop_duplicates()
    removed_duplicate_rows = original_rows - len(deduplicated)

    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    cleaned = deduplicated.dropna(subset=required_columns)
    rows_removed_for_missing_required_values = len(deduplicated) - len(cleaned)

    remaining_missing_values = int(cleaned.isna().sum().sum())
    report = CleaningReport(
        original_rows=original_rows,
        cleaned_rows=len(cleaned),
        removed_duplicate_rows=removed_duplicate_rows,
        rows_removed_for_missing_required_values=rows_removed_for_missing_required_values,
        remaining_missing_values=remaining_missing_values,
    )
    return cleaned, report


def summarize_dataset(df: pd.DataFrame) -> pd.DataFrame:
    summary = df[NUMERIC_COLUMNS].describe().T
    return summary[["mean", "std", "min", "25%", "50%", "75%", "max"]].round(2)


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
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["Device Model", "Operating System"]),
            ("num", StandardScaler(), ["App Usage Time (min/day)", "Screen On Time (hours/day)", "Data Usage (MB/day)", "Number of Apps Installed"]),
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
        "by_device_os": {},
    }
    for (device_model, operating_system), group in train_df.groupby(["Device Model", "Operating System"]):
        priors["by_device_os"][f"{device_model}|||{operating_system}"] = group[DYNAMIC_COLUMNS].median().to_dict()
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
        "Device Model": device_spec.device_model,
        "Operating System": device_spec.operating_system,
        "Number of Apps Installed": device_spec.number_of_apps_installed,
    }


def resolve_battery_capacity(device_spec: DeviceSpec) -> tuple[float, Literal["provided", "assumed_default"]]:
    if device_spec.battery_capacity_mah is not None:
        return device_spec.battery_capacity_mah, "provided"
    return DEFAULT_BATTERY_CAPACITY_MAH, "assumed_default"


def lookup_historical_usage(bundle: PredictorBundle, device_spec: DeviceSpec) -> dict[str, float]:
    device_key = f"{device_spec.device_model}|||{device_spec.operating_system}"
    if device_key in bundle.priors["by_device_os"]:
        return bundle.priors["by_device_os"][device_key]
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

    for step_index, drain_pct in enumerate(drain_per_step):
        step_start_hour = current_hour + (step_index * policy.time_step_hours)
        step_end_hour = min(24.0, step_start_hour + policy.time_step_hours)

        if allow_charging and not charging and level <= policy.start_charge_pct:
            charging = True
            session_start_hour = step_start_hour
            session_start_level = level

        if charging:
            charge_rate_pct_per_hour = charge_rate_for_level(level, policy)
            charge_added = charge_rate_pct_per_hour * policy.time_step_hours
            if level + charge_added >= policy.stop_charge_pct:
                level = policy.stop_charge_pct
                charging = False
                sessions.append(
                    ChargeSession(
                        start_hour=session_start_hour,
                        stop_hour=step_end_hour,
                        start_level_pct=round(session_start_level, 2),
                        stop_level_pct=round(level, 2),
                    )
                )
            else:
                level += charge_added

        level = max(0.0, min(100.0, level - drain_pct))
        levels.append((step_end_hour, round(level, 2)))

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
        f"Target charging band: {policy.start_charge_pct:.0f}% to "
        f"{policy.stop_charge_pct:.0f}% around {policy.preferred_level_pct:.0f}%"
    )
    print()

    if not plan.sessions:
        print("Recommendation: do not charge today unless usage changes materially.")
        return

    print("Recommended charging sessions:")
    for index, session in enumerate(plan.sessions, start=1):
        print(
            f"  {index}. Start charging at {format_hour(session.start_hour)} "
            f"when battery is about {session.start_level_pct:.1f}% and stop at "
            f"{format_hour(session.stop_hour)} near {session.stop_level_pct:.1f}%."
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
