from __future__ import annotations

import pandas as pd

from .config import (
    APP_USAGE_PER_SCREEN_HOUR_COLUMN,
    APPS_PER_1000_MAH_COLUMN,
    BATTERY_CAPACITY_COLUMN,
    CAPACITY_LOOKUP_PATH,
    CATEGORICAL_COLUMNS,
    DATASET_PATH,
    DATA_USAGE_PER_SCREEN_HOUR_COLUMN,
    DYNAMIC_COLUMNS,
    ENGINEERED_NUMERIC_COLUMNS,
    IS_EFFICIENCY_OUTLIER_COLUMN,
    IS_HEAVY_USAGE_OUTLIER_COLUMN,
    NUMERIC_COLUMNS,
    RAW_DATASET_PATH,
    RAW_SUMMARY_COLUMNS,
    SCREEN_TIME_SHARE_OF_DAY_COLUMN,
    STATIC_COLUMNS,
    TARGET_COLUMN,
)
from .schemas import CleaningReport


def load_dataset(dataset_path=DATASET_PATH) -> pd.DataFrame:
    if dataset_path.exists():
        return pd.read_csv(dataset_path)

    raw_df = pd.read_csv(RAW_DATASET_PATH)
    if CAPACITY_LOOKUP_PATH.exists():
        capacity_df = pd.read_csv(CAPACITY_LOOKUP_PATH)
        merged = raw_df.merge(capacity_df[["Device Model", BATTERY_CAPACITY_COLUMN]], on="Device Model", how="left")
        if BATTERY_CAPACITY_COLUMN in merged.columns:
            return merged
    return raw_df


def load_capacity_lookup(capacity_lookup_path=CAPACITY_LOOKUP_PATH) -> pd.DataFrame:
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


def get_cleaning_overview(dataset_path=DATASET_PATH) -> tuple[pd.DataFrame, pd.DataFrame, CleaningReport]:
    raw_df = load_dataset(dataset_path)
    cleaned_df, report = clean_dataset(raw_df)
    return raw_df, cleaned_df, report
