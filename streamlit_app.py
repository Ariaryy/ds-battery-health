from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

import main


ROOT = Path(__file__).resolve().parent

st.set_page_config(
    page_title="Battery Health Dashboard",
    page_icon="battery",
    layout="wide",
)

sns.set_theme(style="whitegrid")


def style_axis(ax: plt.Axes, title: str, xlabel: str = "", ylabel: str = "") -> None:
    ax.set_title(title, loc="left", fontsize=16, fontweight="bold", color="#153b37", pad=12)
    ax.set_xlabel(xlabel, color="#3b5853")
    ax.set_ylabel(ylabel, color="#3b5853")
    ax.grid(True, alpha=0.15)
    ax.set_facecolor("#fcfffe")
    for spine in ax.spines.values():
        spine.set_visible(False)


def annotate_bars(ax: plt.Axes, fmt: str = "{:.0f}") -> None:
    for patch in ax.patches:
        width = patch.get_width()
        height = patch.get_height()
        if width > 0:
            ax.text(
                patch.get_x() + width / 2,
                height,
                fmt.format(height),
                ha="center",
                va="bottom",
                fontsize=9,
                color="#23413d",
            )
        else:
            ax.text(
                patch.get_width(),
                patch.get_y() + patch.get_height() / 2,
                fmt.format(width),
                ha="left",
                va="center",
                fontsize=9,
                color="#23413d",
            )


@st.cache_data(show_spinner=False)
def load_project_data() -> tuple[pd.DataFrame, pd.DataFrame, main.CleaningReport]:
    return main.get_cleaning_overview()


@st.cache_resource(show_spinner=False)
def load_bundle(retrain: bool = False) -> main.PredictorBundle:
    return main.load_predictor(retrain=retrain)


def format_label(value: str) -> str:
    return value.replace("_", " ").title()


def show_project_overview(cleaned_df: pd.DataFrame, bundle: main.PredictorBundle) -> None:
    st.title("Battery Health Prediction Dashboard")
    st.write(
        "This dashboard presents the complete data science workflow for the battery-drain project: "
        "dataset inspection, data cleaning, exploratory visualization, model training results, and live prediction."
    )

    metric_columns = st.columns(4)
    metric_columns[0].metric("Rows Used", f"{len(cleaned_df)}")
    metric_columns[1].metric("Features", f"{len(main.FEATURE_COLUMNS)}")
    metric_columns[2].metric("Model MAE", f"{bundle.metrics['mae']:.2f} mAh/day")
    metric_columns[3].metric("Model", bundle.model_name)


def show_cleaning_tab(raw_df: pd.DataFrame, cleaned_df: pd.DataFrame, report: main.CleaningReport) -> None:
    st.subheader("Data Cleaning")
    st.write(
        "The preprocessing pipeline standardizes categorical text, converts numeric columns safely, "
        "removes duplicate observations, and keeps only rows that are usable for modeling."
    )

    metric_columns = st.columns(5)
    metric_columns[0].metric("Original Rows", report.original_rows)
    metric_columns[1].metric("Cleaned Rows", report.cleaned_rows)
    metric_columns[2].metric("Duplicates Removed", report.removed_duplicate_rows)
    metric_columns[3].metric("Rows Dropped", report.rows_removed_for_missing_required_values)
    metric_columns[4].metric("Missing Values Left", report.remaining_missing_values)

    left, right = st.columns(2)
    with left:
        st.caption("Raw Dataset Sample")
        st.dataframe(raw_df.head(10), use_container_width=True)
    with right:
        st.caption("Cleaned Dataset Sample")
        st.dataframe(cleaned_df.head(10), use_container_width=True)

    st.caption("Numeric Summary After Cleaning")
    st.dataframe(main.summarize_dataset(cleaned_df), use_container_width=True)


def show_visualization_tab(cleaned_df: pd.DataFrame) -> None:
    st.subheader("Exploratory Data Analysis")
    st.write(
        "Use the filters below to explore how battery drain changes across operating systems, "
        "device models, and usage behavior."
    )

    filter_col1, filter_col2 = st.columns(2)
    with filter_col1:
        selected_os = st.multiselect(
            "Filter by Operating System",
            sorted(cleaned_df["Operating System"].unique().tolist()),
            default=sorted(cleaned_df["Operating System"].unique().tolist()),
        )
    with filter_col2:
        selected_devices = st.multiselect(
            "Filter by Device Model",
            sorted(cleaned_df["Device Model"].unique().tolist()),
            default=sorted(cleaned_df["Device Model"].unique().tolist()),
        )

    filtered_df = cleaned_df[
        cleaned_df["Operating System"].isin(selected_os)
        & cleaned_df["Device Model"].isin(selected_devices)
    ].copy()

    if filtered_df.empty:
        st.warning("No data matches the selected filters.")
        return

    top_device = filtered_df.groupby("Device Model")[main.TARGET_COLUMN].mean().sort_values(ascending=False)
    top_os_drain = filtered_df.groupby("Operating System")[main.TARGET_COLUMN].mean().sort_values(ascending=False)
    corr_value = filtered_df["App Usage Time (min/day)"].corr(filtered_df[main.TARGET_COLUMN])

    metric_1, metric_2, metric_3 = st.columns(3)
    metric_1.metric("Highest Avg Drain Device", top_device.index[0], f"{top_device.iloc[0]:.1f} mAh/day")
    metric_2.metric("Highest Avg Drain OS", top_os_drain.index[0], f"{top_os_drain.iloc[0]:.1f} mAh/day")
    metric_3.metric("Usage vs Drain Correlation", f"{corr_value:.2f}")

    tabs = st.tabs(["Distribution", "Comparisons", "Relationships"])

    with tabs[0]:
        left, right = st.columns(2)

        with left:
            fig, ax = plt.subplots(figsize=(8, 4.5))
            sns.histplot(
                filtered_df[main.TARGET_COLUMN],
                bins=24,
                kde=True,
                color="#2f7c6e",
                edgecolor="white",
                ax=ax,
            )
            style_axis(ax, "Distribution of Battery Drain", "Battery Drain (mAh/day)", "Count")
            ax.axvline(filtered_df[main.TARGET_COLUMN].mean(), color="#d97706", linestyle="--", linewidth=2)
            ax.axvline(filtered_df[main.TARGET_COLUMN].median(), color="#2563eb", linestyle=":", linewidth=2)
            ax.legend(["Mean", "Median"], frameon=False)
            st.pyplot(fig, clear_figure=True)

        with right:
            fig, ax = plt.subplots(figsize=(8, 4.5))
            sns.boxplot(
                data=filtered_df,
                x="Operating System",
                y=main.TARGET_COLUMN,
                palette="Set2",
                ax=ax,
            )
            sns.stripplot(
                data=filtered_df,
                x="Operating System",
                y=main.TARGET_COLUMN,
                color="black",
                alpha=0.25,
                size=3,
                ax=ax,
            )
            style_axis(ax, "Battery Drain Spread by Operating System", "", "Battery Drain (mAh/day)")
            st.pyplot(fig, clear_figure=True)

    with tabs[1]:
        left, right = st.columns(2)

        with left:
            avg_device = (
                filtered_df.groupby("Device Model")[main.TARGET_COLUMN]
                .mean()
                .sort_values(ascending=True)
                .reset_index()
            )
            fig, ax = plt.subplots(figsize=(8, 4.8))
            sns.barplot(data=avg_device, x=main.TARGET_COLUMN, y="Device Model", palette="viridis", ax=ax)
            style_axis(ax, "Average Battery Drain by Device Model", "Battery Drain (mAh/day)", "")
            st.pyplot(fig, clear_figure=True)

        with right:
            usage_by_class = (
                filtered_df.groupby("User Behavior Class")[main.TARGET_COLUMN]
                .agg(["mean", "min", "max"])
                .reset_index()
            )
            fig, ax = plt.subplots(figsize=(8, 4.8))
            ax.plot(
                usage_by_class["User Behavior Class"],
                usage_by_class["mean"],
                marker="o",
                linewidth=2.5,
                color="#2f7c6e",
                label="Mean drain",
            )
            ax.fill_between(
                usage_by_class["User Behavior Class"],
                usage_by_class["min"],
                usage_by_class["max"],
                color="#2f7c6e",
                alpha=0.15,
                label="Min-Max range",
            )
            style_axis(ax, "Battery Drain Across Behavior Classes", "User Behavior Class", "Battery Drain (mAh/day)")
            ax.legend(frameon=False)
            st.pyplot(fig, clear_figure=True)

        st.dataframe(
            filtered_df.groupby(["Operating System", "Device Model"])[main.TARGET_COLUMN]
            .agg(["mean", "median", "count"])
            .round(2)
            .reset_index(),
            use_container_width=True,
        )

    with tabs[2]:
        left, right = st.columns(2)

        with left:
            fig, ax = plt.subplots(figsize=(8, 4.8))
            sns.regplot(
                data=filtered_df,
                x="App Usage Time (min/day)",
                y=main.TARGET_COLUMN,
                scatter_kws={"alpha": 0.45, "s": 45, "color": "#2f7c6e"},
                line_kws={"color": "#d97706", "linewidth": 2.5},
                ax=ax,
            )
            style_axis(ax, "App Usage Time vs Battery Drain", "App Usage Time (min/day)", "Battery Drain (mAh/day)")
            st.pyplot(fig, clear_figure=True)

        with right:
            fig, ax = plt.subplots(figsize=(8, 4.8))
            sns.scatterplot(
                data=filtered_df,
                x="Screen On Time (hours/day)",
                y="Data Usage (MB/day)",
                hue="Operating System",
                size=main.TARGET_COLUMN,
                sizes=(30, 220),
                alpha=0.7,
                palette="Set2",
                ax=ax,
            )
            style_axis(ax, "Screen Time vs Data Usage", "Screen On Time (hours/day)", "Data Usage (MB/day)")
            st.pyplot(fig, clear_figure=True)

        _, heatmap_col, _ = st.columns([0.18, 0.64, 0.18])
        with heatmap_col:
            fig, ax = plt.subplots(figsize=(5.2, 4.2))
            corr_cols = [
                "App Usage Time (min/day)",
                "Screen On Time (hours/day)",
                "Data Usage (MB/day)",
                "Number of Apps Installed",
                "Age",
                main.TARGET_COLUMN,
            ]
            corr = filtered_df[corr_cols].corr(numeric_only=True)
            short_labels = [
                "App\nUsage",
                "Screen\nTime",
                "Data\nUsage",
                "Apps\nInstalled",
                "Age",
                "Battery\nDrain",
            ]
            sns.heatmap(
                corr,
                annot=True,
                fmt=".2f",
                cmap="YlGnBu",
                linewidths=0.4,
                cbar=False,
                square=True,
                annot_kws={"size": 7},
                ax=ax,
            )
            style_axis(ax, "Correlation Heatmap")
            ax.set_xticklabels(short_labels, rotation=0, ha="center", fontsize=7)
            ax.set_yticklabels(short_labels, rotation=0, va="center", fontsize=7)
            ax.tick_params(axis="x", pad=8, length=0)
            ax.tick_params(axis="y", pad=4, length=0)
            fig.subplots_adjust(bottom=0.2, top=0.88, left=0.18, right=0.98)
            st.pyplot(fig, clear_figure=True, use_container_width=False)


def show_training_tab(cleaned_df: pd.DataFrame, bundle: main.PredictorBundle) -> None:
    st.subheader("Model Training")
    st.write(
        f"The modeling stage uses a {bundle.model_name} on encoded categorical features and standardized "
        "numeric behavior inputs. This section highlights performance and the features that matter most."
    )

    if getattr(main, "XGBOOST_IMPORT_ERROR", None):
        st.warning(
            "XGBoost is not available in the current environment, so the app retrained using a "
            "scikit-learn fallback model instead. This keeps the dashboard running without `libomp`."
        )

    train_col, test_col, mae_col = st.columns(3)
    train_col.metric("Training Rows", int(bundle.metrics["train_rows"]))
    test_col.metric("Test Rows", int(bundle.metrics["test_rows"]))
    mae_col.metric("Mean Absolute Error", f"{bundle.metrics['mae']:.2f} mAh/day")

    importance_df = main.get_feature_importance(bundle).head(12)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    sns.barplot(
        data=importance_df,
        x="importance",
        y="feature",
        palette=sns.color_palette("blend:#f0a35e,#1f6f66", n_colors=len(importance_df)),
        ax=ax,
    )
    style_axis(ax, "Top Feature Importances", "Importance", "")
    st.pyplot(fig, clear_figure=True)

    st.caption("Modeling Features")
    st.dataframe(cleaned_df[main.FEATURE_COLUMNS + [main.TARGET_COLUMN]].head(12), use_container_width=True)


def show_prediction_tab(cleaned_df: pd.DataFrame, bundle: main.PredictorBundle) -> None:
    st.subheader("Interactive Prediction")
    st.write(
        "Build a realistic usage snapshot and compare expected battery stress, remaining drain, "
        "and whether a charge session is likely to be needed before the end of the day."
    )

    device_models = sorted(cleaned_df["Device Model"].dropna().unique().tolist())
    operating_systems = sorted(cleaned_df["Operating System"].dropna().unique().tolist())

    form_col, summary_col = st.columns([1.1, 1.3])

    with form_col:
        with st.form("prediction_form"):
            device_model = st.selectbox("Device Model", device_models, index=min(2, len(device_models) - 1))
            operating_system = st.selectbox("Operating System", operating_systems)
            number_of_apps_installed = st.slider("Number of Apps Installed", 10, 200, 85)
            battery_capacity_mah = st.number_input("Battery Capacity (mAh)", min_value=1000.0, value=4600.0, step=100.0)

            st.markdown("**Current Usage Snapshot**")
            current_hour = st.slider("Current Hour of Day", min_value=1.0, max_value=24.0, value=13.0, step=0.5)
            current_battery_pct = st.slider("Current Battery %", 0.0, 100.0, 38.0, step=1.0)
            starting_battery_pct = st.slider("Starting Battery %", 0.0, 100.0, 100.0, step=1.0)
            app_usage_minutes_so_far = st.number_input("App Usage So Far (min)", min_value=0.0, value=320.0, step=10.0)
            screen_on_hours_so_far = st.number_input("Screen On Time So Far (hours)", min_value=0.0, value=5.8, step=0.1)
            data_usage_mb_so_far = st.number_input("Data Usage So Far (MB)", min_value=0.0, value=1350.0, step=50.0)

            submitted = st.form_submit_button("Run Prediction")

    with summary_col:
        if not submitted:
            st.info("Submit the form to generate the forecast and charging recommendation.")
            return

        device_spec = main.DeviceSpec(
            device_model=device_model,
            operating_system=operating_system,
            number_of_apps_installed=int(number_of_apps_installed),
            battery_capacity_mah=float(battery_capacity_mah),
        )
        snapshot = main.UsageSnapshot(
            current_hour=float(current_hour),
            current_battery_pct=float(current_battery_pct),
            starting_battery_pct=float(starting_battery_pct),
            app_usage_minutes_so_far=float(app_usage_minutes_so_far),
            screen_on_hours_so_far=float(screen_on_hours_so_far),
            data_usage_mb_so_far=float(data_usage_mb_so_far),
        )

        try:
            forecast = main.forecast_drain(bundle=bundle, device_spec=device_spec, snapshot=snapshot)
            plan = main.recommend_charging_plan(forecast=forecast, snapshot=snapshot)
        except ValueError as error:
            st.error(str(error))
            return

        top_metrics = st.columns(3)
        top_metrics[0].metric("Predicted Full-Day Drain", f"{forecast.predicted_full_day_drain_mah:.1f} mAh")
        top_metrics[1].metric("Remaining Drain", f"{forecast.predicted_remaining_drain_mah:.1f} mAh")
        top_metrics[2].metric("Usage Share Observed", f"{forecast.cumulative_usage_share:.2f}")

        bottom_metrics = st.columns(3)
        bottom_metrics[0].metric("No-Charge Lowest Battery", f"{plan.no_charge_lowest_battery_pct:.1f}%")
        bottom_metrics[1].metric("Recommended Lowest Battery", f"{plan.projected_lowest_battery_pct:.1f}%")
        bottom_metrics[2].metric("End-of-Day Battery", f"{plan.projected_end_battery_pct:.1f}%")

        battery_projection = pd.DataFrame(
            {
                "Scenario": ["No charge", "With recommendation", "Current level"],
                "Battery %": [
                    plan.no_charge_end_battery_pct,
                    plan.projected_end_battery_pct,
                    snapshot.current_battery_pct,
                ],
            }
        )
        fig, ax = plt.subplots(figsize=(8.2, 4.4))
        sns.barplot(
            data=battery_projection,
            x="Scenario",
            y="Battery %",
            palette=["#d86c6c", "#1f6f66", "#f0a35e"],
            ax=ax,
        )
        style_axis(ax, "End-of-Day Battery Comparison", "", "Battery %")
        annotate_bars(ax, "{:.1f}")
        st.pyplot(fig, clear_figure=True)

        st.caption("Blended Dynamic Features Used for Prediction")
        blended_df = pd.DataFrame(
            {
                "Historical": forecast.historical_dynamic_usage,
                "Projected": forecast.projected_dynamic_usage,
                "Blended": {feature: forecast.blended_feature_row[feature] for feature in main.DYNAMIC_COLUMNS},
            }
        ).round(2)
        st.dataframe(blended_df, use_container_width=True)

        if plan.sessions:
            st.success("Charging is recommended based on the current usage pattern.")
            schedule_df = pd.DataFrame(
                [
                    {
                        "Start Time": main.format_hour(session.start_hour),
                        "Stop Time": main.format_hour(session.stop_hour),
                        "Start Battery %": session.start_level_pct,
                        "Stop Battery %": session.stop_level_pct,
                    }
                    for session in plan.sessions
                ]
            )
            st.dataframe(schedule_df, use_container_width=True)
        else:
            st.success("No charging session is needed if usage stays close to the current forecast.")


def main_app() -> None:
    raw_df, cleaned_df, report = load_project_data()
    bundle = load_bundle()

    show_project_overview(cleaned_df, bundle)

    tabs = st.tabs(
        [
            "Dataset",
            "Cleaning",
            "Visualizations",
            "Training",
            "Prediction",
        ]
    )

    with tabs[0]:
        st.subheader("Dataset Overview")
        st.write(f"Dataset path: `{ROOT / 'datasets' / 'user_behavior_dataset.csv'}`")
        st.dataframe(cleaned_df, use_container_width=True)

    with tabs[1]:
        show_cleaning_tab(raw_df, cleaned_df, report)

    with tabs[2]:
        show_visualization_tab(cleaned_df)

    with tabs[3]:
        show_training_tab(cleaned_df, bundle)

    with tabs[4]:
        show_prediction_tab(cleaned_df, bundle)


if __name__ == "__main__":
    main_app()
