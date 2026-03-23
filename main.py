from battery_health import *  # noqa: F401,F403
from battery_health.reporting import print_report


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
