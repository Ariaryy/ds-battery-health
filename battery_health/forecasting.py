from __future__ import annotations

from typing import Any, Literal

import pandas as pd

from .config import (
    BATTERY_CAPACITY_COLUMN,
    DEFAULT_BATTERY_CAPACITY_MAH,
    DYNAMIC_COLUMNS,
    FEATURE_COLUMNS,
    MIN_USAGE_SHARE,
)
from .data import add_engineered_features, load_capacity_lookup
from .schemas import DeviceSpec, DrainForecast, PredictorBundle, UsageSnapshot
from .utils import clip


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


def normalize_device_spec(device_spec: DeviceSpec) -> dict[str, Any]:
    return {
        "Operating System": device_spec.operating_system,
        "Number of Apps Installed": device_spec.number_of_apps_installed,
        BATTERY_CAPACITY_COLUMN: resolve_battery_capacity(device_spec)[0],
    }


def resolve_battery_capacity(
    device_spec: DeviceSpec,
) -> tuple[float, Literal["provided", "lookup_table", "assumed_default"]]:
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
    return add_engineered_features(pd.DataFrame([feature_row])).iloc[0].to_dict()


def predict_daily_drain(bundle: PredictorBundle, feature_row: dict[str, Any]) -> float:
    df_input = pd.DataFrame([feature_row], columns=FEATURE_COLUMNS)
    X_processed = bundle.preprocessor.transform(df_input)
    return float(bundle.model.predict(X_processed)[0])


def forecast_drain(bundle: PredictorBundle, device_spec: DeviceSpec, snapshot: UsageSnapshot) -> DrainForecast:
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
