# Battery Health Dashboard

This project predicts daily battery drain using user behavior data and includes a Streamlit dashboard that walks through the full data science workflow:

- dataset overview
- data cleaning summary
- exploratory data analysis visualizations
- model training metrics and feature importance
- interactive battery drain prediction and charging recommendation

The training pipeline now enriches the source dataset with `Battery Capacity (mAh)` and uses that numeric feature instead of training directly on raw phone names. XGBoost is a required dependency for training and inference.

## Build the enriched dataset

Before retraining the model, generate the reviewed battery-capacity lookup and enriched dataset:

```bash
uv run python scripts/enrich_battery_capacities.py
```

This writes:

- `datasets/device_battery_capacities.csv`
- `datasets/user_behavior_dataset_enriched.csv`

## Run the dashboard

```bash
streamlit run streamlit_app.py
```

## Main files

- `main.py`: compatibility entry point and demo runner
- `battery_health/`: package containing configuration, preprocessing, modeling, forecasting, planning, and reporting modules
- `streamlit_app.py`: dashboard UI for the project workflow
- `datasets/user_behavior_dataset.csv`: source dataset used for training and analysis
- `datasets/user_behavior_dataset_enriched.csv`: enriched dataset with `Battery Capacity (mAh)`
- `scripts/enrich_battery_capacities.py`: scraper/enrichment pipeline for battery capacity lookup
