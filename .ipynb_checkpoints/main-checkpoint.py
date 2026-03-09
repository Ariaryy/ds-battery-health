from __future__ import annotations

import sys
from pathlib import Path
from pprint import pprint


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import main as app


def pretty_print_user_view(
    profile: dict,
    snapshot: app.UsageSnapshot,
    forecast: app.DrainForecast,
    plan: app.ChargingPlan,
) -> None:
    print("=" * 72)
    print("BATTERY DRAIN FORECAST AND CHARGING RECOMMENDATION")
    print("=" * 72)
    print()

    print("User profile")
    print("-" * 72)
    pprint(profile, sort_dicts=False)
    print()

    print("Current usage snapshot")
    print("-" * 72)
    pprint(
        {
            "current_time": app.format_hour(snapshot.current_hour),
            "current_battery_pct": f"{snapshot.current_battery_pct:.1f}%",
            "starting_battery_pct": f"{snapshot.starting_battery_pct:.1f}%",
            "app_usage_minutes_so_far": round(snapshot.app_usage_minutes_so_far, 1),
            "screen_on_hours_so_far": round(snapshot.screen_on_hours_so_far, 1),
            "data_usage_mb_so_far": round(snapshot.data_usage_mb_so_far, 1),
        },
        sort_dicts=False,
    )
    print()

    print("Forecast summary")
    print("-" * 72)
    pprint(
        {
            "model_only_full_day_drain_mah": round(forecast.model_full_day_drain_mah, 1),
            "observed_rate_full_day_drain_mah": round(forecast.observed_rate_full_day_drain_mah, 1),
            "final_predicted_full_day_drain_mah": round(forecast.predicted_full_day_drain_mah, 1),
            "observed_drain_so_far_mah": round(forecast.observed_drain_so_far_mah, 1),
            "predicted_remaining_drain_mah": round(forecast.predicted_remaining_drain_mah, 1),
        },
        sort_dicts=False,
    )
    print()

    print("Charge recommendation")
    print("-" * 72)
    if not plan.sessions:
        print("No charging session is recommended right now.")
        print(f"Expected end-of-day battery: {plan.no_charge_end_battery_pct:.1f}%")
    else:
        print(
            f"If the user does not charge, battery may drop as low as "
            f"{plan.no_charge_lowest_battery_pct:.1f}% and end near "
            f"{plan.no_charge_end_battery_pct:.1f}%."
        )
        print(
            f"With the recommended charging plan, battery should stay above "
            f"{plan.projected_lowest_battery_pct:.1f}% and end near "
            f"{plan.projected_end_battery_pct:.1f}%."
        )
        print()
        print("Recommended charging windows:")
        for index, session in enumerate(plan.sessions, start=1):
            print(
                f"  {index}. Start at {app.format_hour(session.start_hour)} "
                f"around {session.start_level_pct:.1f}% and stop at "
                f"{app.format_hour(session.stop_hour)} around {session.stop_level_pct:.1f}%."
            )
    print()
    print("=" * 72)


def main() -> None:
    bundle = app.load_predictor()

    profile = {
        "Device Model": "Xiaomi Mi 11",
        "Operating System": "Android",
        "Number of Apps Installed": 85,
    }
    snapshot = app.UsageSnapshot(
        current_hour=13.0,
        current_battery_pct=38.0,
        starting_battery_pct=100.0,
        app_usage_minutes_so_far=320.0,
        screen_on_hours_so_far=5.8,
        data_usage_mb_so_far=1350.0,
    )

    forecast = app.forecast_drain(bundle=bundle, profile=profile, snapshot=snapshot)
    plan = app.recommend_charging_plan(forecast=forecast, snapshot=snapshot)
    pretty_print_user_view(profile=profile, snapshot=snapshot, forecast=forecast, plan=plan)


if __name__ == "__main__":
    main()
