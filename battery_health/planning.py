from __future__ import annotations

from .config import DYNAMIC_COLUMNS, USAGE_CURVE
from .schemas import ChargeSession, ChargingPlan, ChargingPolicy, DrainForecast, UsageSnapshot
from .utils import clip


def recommend_charging_plan(
    forecast: DrainForecast,
    snapshot: UsageSnapshot,
    policy: ChargingPolicy = ChargingPolicy(),
) -> ChargingPlan:
    remaining_drain_pct = (forecast.predicted_remaining_drain_mah / forecast.battery_capacity_mah) * 100.0
    no_charge_levels, _ = simulate_battery_levels(
        current_battery_pct=snapshot.current_battery_pct,
        current_hour=snapshot.current_hour,
        remaining_drain_pct=remaining_drain_pct,
        snapshot=snapshot,
        historical_usage=forecast.historical_dynamic_usage,
        usage_curve=USAGE_CURVE,
        policy=policy,
        allow_charging=False,
    )
    charge_levels, sessions = simulate_battery_levels(
        current_battery_pct=snapshot.current_battery_pct,
        current_hour=snapshot.current_hour,
        remaining_drain_pct=remaining_drain_pct,
        snapshot=snapshot,
        historical_usage=forecast.historical_dynamic_usage,
        usage_curve=USAGE_CURVE,
        policy=policy,
        allow_charging=True,
    )

    return ChargingPlan(
        sessions=sessions,
        projected_lowest_battery_pct=min(level for _, level in charge_levels),
        projected_end_battery_pct=charge_levels[-1][1],
        no_charge_lowest_battery_pct=min(level for _, level in no_charge_levels),
        no_charge_end_battery_pct=no_charge_levels[-1][1],
    )


def recommended_stop_level_for_session(
    current_level_pct: float,
    remaining_drain_pct_from_now: float,
    policy: ChargingPolicy,
) -> float:
    ideal_stop_level_pct = remaining_drain_pct_from_now + policy.preferred_level_pct
    bounded_stop_level_pct = clip(
        ideal_stop_level_pct,
        policy.minimum_stop_charge_pct,
        policy.maximum_stop_charge_pct,
    )
    return max(current_level_pct, bounded_stop_level_pct)


def charge_rate_for_level(level_pct: float, policy: ChargingPolicy) -> float:
    if level_pct < 50.0:
        return policy.charge_rate_low_band_pct_per_hour
    if level_pct < 80.0:
        return policy.charge_rate_mid_band_pct_per_hour
    return policy.charge_rate_high_band_pct_per_hour


def simulate_battery_levels(
    current_battery_pct: float,
    current_hour: float,
    remaining_drain_pct: float,
    snapshot: UsageSnapshot,
    historical_usage: dict[str, float],
    usage_curve: list[float],
    policy: ChargingPolicy,
    allow_charging: bool,
) -> tuple[list[tuple[float, float]], list[ChargeSession]]:
    drain_weights = build_drain_weights(
        current_hour=current_hour,
        snapshot=snapshot,
        historical_usage=historical_usage,
        time_step_hours=policy.time_step_hours,
    )
    drain_per_step = [remaining_drain_pct * weight for weight in drain_weights]

    level = current_battery_pct
    charging = False
    session_start_hour = 0.0
    session_start_level = 0.0
    sessions: list[ChargeSession] = []
    levels: list[tuple[float, float]] = [(current_hour, round(level, 2))]
    target_stop_level_pct = policy.minimum_stop_charge_pct

    for step_index, drain_pct in enumerate(drain_per_step):
        step_start_hour = current_hour + (step_index * policy.time_step_hours)
        remaining_drain_pct_from_now = sum(drain_per_step[step_index:])

        if allow_charging and not charging and level <= policy.start_charge_pct:
            charging = True
            session_start_hour = step_start_hour
            session_start_level = level
            target_stop_level_pct = recommended_stop_level_for_session(
                current_level_pct=level,
                remaining_drain_pct_from_now=remaining_drain_pct_from_now,
                policy=policy,
            )

        if charging:
            charge_rate_pct_per_hour = charge_rate_for_level(level, policy)
            charge_added = charge_rate_pct_per_hour * policy.time_step_hours
            if level + charge_added >= target_stop_level_pct:
                level = target_stop_level_pct
                charging = False
                sessions.append(
                    ChargeSession(
                        start_hour=session_start_hour,
                        start_level_pct=round(session_start_level, 2),
                        recommended_stop_level_pct=round(level, 2),
                    )
                )
            else:
                level += charge_added

        level = max(0.0, min(100.0, level - drain_pct))
        levels.append((min(24.0, step_start_hour + policy.time_step_hours), round(level, 2)))

    if not levels or levels[-1][0] < 24.0:
        levels.append((24.0, round(level, 2)))

    return levels, sessions


def build_drain_weights(
    current_hour: float,
    snapshot: UsageSnapshot,
    historical_usage: dict[str, float],
    time_step_hours: float,
) -> list[float]:
    steps = max(1, int(round((24.0 - current_hour) / time_step_hours)))
    usage_pressure = estimate_usage_pressure(snapshot, historical_usage)

    weights = []
    for step_index in range(steps):
        clock = current_hour + (step_index * time_step_hours)
        base_weight = hour_activity_multiplier(clock)
        near_term_boost = usage_pressure if step_index < int(3.0 / time_step_hours) else 1.0
        weights.append(base_weight * near_term_boost)

    total_weight = sum(weights)
    return [weight / total_weight for weight in weights]


def estimate_usage_pressure(snapshot: UsageSnapshot, historical_usage: dict[str, float]) -> float:
    observed_hourly = {
        "App Usage Time (min/day)": snapshot.app_usage_minutes_so_far / snapshot.current_hour,
        "Screen On Time (hours/day)": snapshot.screen_on_hours_so_far / snapshot.current_hour,
        "Data Usage (MB/day)": snapshot.data_usage_mb_so_far / snapshot.current_hour,
    }
    ratios = []
    for feature in DYNAMIC_COLUMNS:
        historical_hourly = max(historical_usage[feature] / 24.0, 0.001)
        ratios.append(observed_hourly[feature] / historical_hourly)

    return clip(sum(ratios) / len(ratios), 0.7, 1.8)


def hour_activity_multiplier(clock: float) -> float:
    hour = clock % 24
    if hour < 6 or hour >= 22:
        return 0.55
    if hour < 9:
        return 1.1
    if hour < 18:
        return 1.0
    return 1.2
