import unittest
from pprint import pprint

import pandas as pd

import main


class BatteryPlannerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bundle = main.load_predictor(retrain=True)
        cls.policy = main.ChargingPolicy()

    def run_scenario(
        self,
        device_spec: main.DeviceSpec,
        snapshot: main.UsageSnapshot,
    ) -> tuple[main.DrainForecast, main.ChargingPlan]:
        forecast = main.forecast_drain(bundle=self.bundle, device_spec=device_spec, snapshot=snapshot)
        plan = main.recommend_charging_plan(
            forecast=forecast,
            snapshot=snapshot,
            policy=self.policy,
        )
        return forecast, plan

    def print_scenario_result(
        self,
        scenario_name: str,
        device_spec: main.DeviceSpec,
        snapshot: main.UsageSnapshot,
        forecast: main.DrainForecast,
        plan: main.ChargingPlan,
    ) -> None:
        print()
        print("=" * 72)
        print(f"SCENARIO: {scenario_name}")
        print("=" * 72)
        print("Input device spec")
        print("-" * 72)
        pprint(device_spec, sort_dicts=False)
        print("Input snapshot")
        print("-" * 72)
        pprint(
            {
                "current_time": main.format_hour(snapshot.current_hour),
                "current_battery_pct": f"{snapshot.current_battery_pct:.1f}%",
                "starting_battery_pct": f"{snapshot.starting_battery_pct:.1f}%",
                "app_usage_minutes_so_far": round(snapshot.app_usage_minutes_so_far, 1),
                "screen_on_hours_so_far": round(snapshot.screen_on_hours_so_far, 1),
                "data_usage_mb_so_far": round(snapshot.data_usage_mb_so_far, 1),
            },
            sort_dicts=False,
        )
        print("Output forecast")
        print("-" * 72)
        pprint(
            {
                "battery_capacity_mah": round(forecast.battery_capacity_mah, 1),
                "battery_capacity_source": forecast.battery_capacity_source,
                "cumulative_usage_share": round(forecast.cumulative_usage_share, 3),
                "model_full_day_drain_mah": round(forecast.model_full_day_drain_mah, 1),
                "observed_rate_full_day_drain_mah": round(forecast.observed_rate_full_day_drain_mah, 1),
                "predicted_full_day_drain_mah": round(forecast.predicted_full_day_drain_mah, 1),
                "observed_drain_so_far_mah": round(forecast.observed_drain_so_far_mah, 1),
                "predicted_remaining_drain_mah": round(forecast.predicted_remaining_drain_mah, 1),
                "today_usage_weight": round(forecast.today_usage_weight, 3),
            },
            sort_dicts=False,
        )
        print("Output plan")
        print("-" * 72)
        pprint(
            {
                "no_charge_lowest_battery_pct": round(plan.no_charge_lowest_battery_pct, 1),
                "no_charge_end_battery_pct": round(plan.no_charge_end_battery_pct, 1),
                "projected_lowest_battery_pct": round(plan.projected_lowest_battery_pct, 1),
                "projected_end_battery_pct": round(plan.projected_end_battery_pct, 1),
                "sessions": [
                    {
                        "start_time": main.format_hour(session.start_hour),
                        "start_level_pct": round(session.start_level_pct, 1),
                        "recommended_stop_level_pct": round(session.recommended_stop_level_pct, 1),
                    }
                    for session in plan.sessions
                ],
            },
            sort_dicts=False,
        )

    def test_build_usage_priors_uses_train_only_rows(self) -> None:
        train_df = pd.DataFrame(
            [
                {
                    "Device Model": "Train Phone",
                    "Operating System": "Android",
                    "Number of Apps Installed": 40,
                    "App Usage Time (min/day)": 120,
                    "Screen On Time (hours/day)": 3.0,
                    "Data Usage (MB/day)": 500,
                },
                {
                    "Device Model": "Train Phone",
                    "Operating System": "Android",
                    "Number of Apps Installed": 41,
                    "App Usage Time (min/day)": 180,
                    "Screen On Time (hours/day)": 4.0,
                    "Data Usage (MB/day)": 700,
                },
            ]
        )
        priors = main.build_usage_priors(train_df)
        self.assertEqual(priors["by_os"]["Android"]["App Usage Time (min/day)"], 150.0)
        self.assertNotIn("by_device_os", priors)

    def test_cumulative_usage_share_at_known_hours(self) -> None:
        self.assertAlmostEqual(main.cumulative_usage_share(6.0, main.USAGE_CURVE), 0.06)
        self.assertAlmostEqual(main.cumulative_usage_share(11.0, main.USAGE_CURVE), 0.29)
        self.assertAlmostEqual(main.cumulative_usage_share(18.0, main.USAGE_CURVE), 0.69)
        self.assertAlmostEqual(main.cumulative_usage_share(24.0, main.USAGE_CURVE), 1.0)

    def test_projection_uses_cumulative_share_not_linear_clock(self) -> None:
        snapshot = main.UsageSnapshot(
            current_hour=11.0,
            current_battery_pct=61.0,
            starting_battery_pct=100.0,
            app_usage_minutes_so_far=170.0,
            screen_on_hours_so_far=3.9,
            data_usage_mb_so_far=700.0,
        )
        bounds = {
            "low": {
                "App Usage Time (min/day)": 0.0,
                "Screen On Time (hours/day)": 0.0,
                "Data Usage (MB/day)": 0.0,
            },
            "high": {
                "App Usage Time (min/day)": 9999.0,
                "Screen On Time (hours/day)": 9999.0,
                "Data Usage (MB/day)": 9999.0,
            },
        }
        projected = main.project_usage_to_daily(snapshot, bounds, main.USAGE_CURVE)
        self.assertAlmostEqual(projected["App Usage Time (min/day)"], 170.0 / 0.29, places=4)

    def test_today_usage_weight_follows_cumulative_share(self) -> None:
        historical = {
            "App Usage Time (min/day)": 200.0,
            "Screen On Time (hours/day)": 4.0,
            "Data Usage (MB/day)": 800.0,
        }
        projected = {
            "App Usage Time (min/day)": 400.0,
            "Screen On Time (hours/day)": 8.0,
            "Data Usage (MB/day)": 1600.0,
        }
        _, today_usage_weight = main.blend_dynamic_usage(historical, projected, 11.0, main.USAGE_CURVE)
        self.assertAlmostEqual(today_usage_weight, 0.29)

    def test_piecewise_charge_rate_changes_by_level(self) -> None:
        self.assertEqual(main.charge_rate_for_level(20.0, self.policy), 30.0)
        self.assertEqual(main.charge_rate_for_level(60.0, self.policy), 18.0)
        self.assertEqual(main.charge_rate_for_level(90.0, self.policy), 7.0)

    def test_forecast_is_physically_consistent(self) -> None:
        device_spec = main.DeviceSpec(
            device_model="Imaginary Phone X",
            operating_system="Android",
            number_of_apps_installed=65,
        )
        snapshot = main.UsageSnapshot(
            current_hour=11.0,
            current_battery_pct=61.0,
            starting_battery_pct=100.0,
            app_usage_minutes_so_far=170.0,
            screen_on_hours_so_far=3.9,
            data_usage_mb_so_far=700.0,
        )
        forecast, _ = self.run_scenario(device_spec, snapshot)
        self.assertGreaterEqual(forecast.predicted_full_day_drain_mah, forecast.observed_drain_so_far_mah)
        self.assertGreaterEqual(forecast.predicted_remaining_drain_mah, 0.0)

    def test_capacity_source_is_reported(self) -> None:
        assumed_spec = main.DeviceSpec(
            device_model="Assumed Phone",
            operating_system="Android",
            number_of_apps_installed=60,
        )
        provided_spec = main.DeviceSpec(
            device_model="Known Capacity Phone",
            operating_system="Android",
            number_of_apps_installed=60,
            battery_capacity_mah=3000.0,
        )
        snapshot = main.UsageSnapshot(
            current_hour=12.0,
            current_battery_pct=70.0,
            starting_battery_pct=100.0,
            app_usage_minutes_so_far=140.0,
            screen_on_hours_so_far=3.2,
            data_usage_mb_so_far=450.0,
        )
        assumed_forecast, _ = self.run_scenario(assumed_spec, snapshot)
        provided_forecast, _ = self.run_scenario(provided_spec, snapshot)

        self.assertEqual(assumed_forecast.battery_capacity_source, "assumed_default")
        self.assertEqual(provided_forecast.battery_capacity_source, "provided")

    def test_lookup_capacity_source_is_used_for_known_device(self) -> None:
        lookup_spec = main.DeviceSpec(
            device_model="Xiaomi Mi 11",
            operating_system="Android",
            number_of_apps_installed=60,
        )
        snapshot = main.UsageSnapshot(
            current_hour=12.0,
            current_battery_pct=70.0,
            starting_battery_pct=100.0,
            app_usage_minutes_so_far=140.0,
            screen_on_hours_so_far=3.2,
            data_usage_mb_so_far=450.0,
        )
        lookup_forecast, _ = self.run_scenario(lookup_spec, snapshot)
        self.assertEqual(lookup_forecast.battery_capacity_source, "lookup_table")
        self.assertGreater(lookup_forecast.battery_capacity_mah, 0.0)

    def test_heavy_user_requires_charging(self) -> None:
        device_spec = main.DeviceSpec(
            device_model="Xiaomi Mi 11",
            operating_system="Android",
            number_of_apps_installed=120,
        )
        snapshot = main.UsageSnapshot(
            current_hour=13.0,
            current_battery_pct=38.0,
            starting_battery_pct=100.0,
            app_usage_minutes_so_far=320.0,
            screen_on_hours_so_far=5.8,
            data_usage_mb_so_far=1350.0,
        )

        forecast, plan = self.run_scenario(device_spec, snapshot)
        self.print_scenario_result("heavy_user_requires_charging", device_spec, snapshot, forecast, plan)

        self.assertGreater(forecast.predicted_remaining_drain_mah, 0.0)
        self.assertGreaterEqual(len(plan.sessions), 1)
        self.assertLess(plan.no_charge_lowest_battery_pct, self.policy.start_charge_pct)

    def test_light_user_does_not_need_charging(self) -> None:
        device_spec = main.DeviceSpec(
            device_model="Google Pixel 5",
            operating_system="Android",
            number_of_apps_installed=42,
        )
        snapshot = main.UsageSnapshot(
            current_hour=16.0,
            current_battery_pct=72.0,
            starting_battery_pct=100.0,
            app_usage_minutes_so_far=85.0,
            screen_on_hours_so_far=2.1,
            data_usage_mb_so_far=240.0,
        )

        forecast, plan = self.run_scenario(device_spec, snapshot)
        self.print_scenario_result("light_user_does_not_need_charging", device_spec, snapshot, forecast, plan)

        self.assertEqual(plan.sessions, [])
        self.assertGreater(plan.no_charge_end_battery_pct, self.policy.start_charge_pct)

    def test_unknown_device_falls_back_to_os_priors(self) -> None:
        device_spec = main.DeviceSpec(
            device_model="Imaginary Phone X",
            operating_system="Android",
            number_of_apps_installed=65,
        )
        snapshot = main.UsageSnapshot(
            current_hour=11.0,
            current_battery_pct=61.0,
            starting_battery_pct=100.0,
            app_usage_minutes_so_far=170.0,
            screen_on_hours_so_far=3.9,
            data_usage_mb_so_far=700.0,
        )

        forecast, plan = self.run_scenario(device_spec, snapshot)
        self.print_scenario_result("unknown_device_falls_back_to_os_priors", device_spec, snapshot, forecast, plan)

        self.assertGreaterEqual(forecast.predicted_remaining_drain_mah, 0.0)
        self.assertIn("Operating System", forecast.blended_feature_row)
        self.assertIn(main.BATTERY_CAPACITY_COLUMN, forecast.blended_feature_row)
        self.assertGreaterEqual(plan.projected_end_battery_pct, 0.0)

    def test_high_battery_late_day_user_does_not_get_unnecessary_charging(self) -> None:
        device_spec = main.DeviceSpec(
            device_model="OnePlus 9",
            operating_system="Android",
            number_of_apps_installed=55,
        )
        snapshot = main.UsageSnapshot(
            current_hour=21.0,
            current_battery_pct=78.0,
            starting_battery_pct=100.0,
            app_usage_minutes_so_far=150.0,
            screen_on_hours_so_far=3.5,
            data_usage_mb_so_far=420.0,
        )

        forecast, plan = self.run_scenario(device_spec, snapshot)
        self.print_scenario_result("high_battery_late_day_no_charge", device_spec, snapshot, forecast, plan)

        self.assertEqual(plan.sessions, [])
        self.assertGreater(plan.no_charge_end_battery_pct, self.policy.start_charge_pct)

    def test_borderline_45_percent_user_starts_charging_immediately(self) -> None:
        device_spec = main.DeviceSpec(
            device_model="Google Pixel 5",
            operating_system="Android",
            number_of_apps_installed=70,
        )
        snapshot = main.UsageSnapshot(
            current_hour=15.0,
            current_battery_pct=45.0,
            starting_battery_pct=100.0,
            app_usage_minutes_so_far=220.0,
            screen_on_hours_so_far=4.5,
            data_usage_mb_so_far=900.0,
        )

        forecast, plan = self.run_scenario(device_spec, snapshot)
        self.print_scenario_result("borderline_45_percent_starts_now", device_spec, snapshot, forecast, plan)

        self.assertGreaterEqual(len(plan.sessions), 1)
        self.assertEqual(plan.sessions[0].start_hour, snapshot.current_hour)
        self.assertGreaterEqual(plan.sessions[0].recommended_stop_level_pct, self.policy.minimum_stop_charge_pct)

    def test_custom_capacity_changes_percent_conversion(self) -> None:
        assumed_spec = main.DeviceSpec(
            device_model="Xiaomi Mi 11",
            operating_system="Android",
            number_of_apps_installed=85,
        )
        provided_spec = main.DeviceSpec(
            device_model="Xiaomi Mi 11",
            operating_system="Android",
            number_of_apps_installed=85,
            battery_capacity_mah=3200.0,
        )
        snapshot = main.UsageSnapshot(
            current_hour=13.0,
            current_battery_pct=60.0,
            starting_battery_pct=100.0,
            app_usage_minutes_so_far=220.0,
            screen_on_hours_so_far=4.3,
            data_usage_mb_so_far=900.0,
        )

        assumed_forecast, _ = self.run_scenario(assumed_spec, snapshot)
        provided_forecast, _ = self.run_scenario(provided_spec, snapshot)

        self.assertNotEqual(assumed_forecast.observed_drain_so_far_mah, provided_forecast.observed_drain_so_far_mah)
        self.assertEqual(assumed_forecast.battery_capacity_source, "lookup_table")
        self.assertEqual(provided_forecast.battery_capacity_source, "provided")


if __name__ == "__main__":
    unittest.main()
