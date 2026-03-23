from .config import DYNAMIC_COLUMNS
from .schemas import ChargingPlan, ChargingPolicy, DrainForecast
from .utils import format_hour


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
