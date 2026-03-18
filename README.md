# Battery Health Dashboard

This project predicts daily battery drain using user behavior data and now includes a Streamlit dashboard that walks through the full data science workflow:

- dataset overview
- data cleaning summary
- exploratory data analysis visualizations
- model training metrics and feature importance
- interactive battery drain prediction and charging recommendation

## Run the dashboard

```bash
streamlit run streamlit_app.py
```

## Main files

- `main.py`: model training, bundle loading, forecasting, and charging recommendation logic
- `streamlit_app.py`: dashboard UI for the project workflow
- `datasets/user_behavior_dataset.csv`: source dataset used for training and analysis
