from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from sklearn.compose import ColumnTransformer


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
