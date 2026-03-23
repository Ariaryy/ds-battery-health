from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
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
