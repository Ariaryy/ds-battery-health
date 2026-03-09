# Charging Assistant Implementation Explanation

## What this system is

This project is a charging assistant, not a true battery-health model.

Its job is to:

- forecast how much battery drain is likely for the rest of the day
- recommend when the user should start charging
- recommend when the user should stop charging
- keep the battery around a battery-friendly operating band centered on `50%`

It does **not** model long-term battery degradation, cycle count, depth of discharge, or state of health.

## Current realism limits

The current implementation is a defensible prototype, but some outputs can still be too aggressive for real-world use.

Two important limitations are:

- the observed-rate correction can become too large if the battery drops quickly early in the day
- the planner can recommend too many charging sessions because it optimizes tightly around the `45% -> 55%` target band without considering user convenience

Example:

- if the phone drops from `100%` to `38%` by `13:00`
- and the heuristic usage-share by `13:00` is only `0.40`
- then the observed-rate estimate becomes:

```text
observed_rate_full_day_drain_mah =
    observed_drain_so_far_mah / cumulative_usage_share
```

That can produce a very large full-day drain estimate, sometimes larger than the nominal battery capacity. This is mathematically consistent with the current model, but it may not be realistic if:

- the morning was unusually heavy
- the starting battery assumption is too simplistic
- the heuristic day-shape curve is not a good fit for that user

So the assistant should currently be treated as:

- useful for relative charge-timing guidance
- not yet trustworthy as a precise real-world battery simulator

The most likely next realism improvements are:

- damping or capping the observed-rate extrapolation
- using a shorter recent discharge window instead of only `starting_battery_pct`
- penalizing too many charging sessions
- enforcing a minimum time gap between charging sessions
- widening the practical target band when user convenience matters more than tight control

## Main design decisions

The current implementation fixes three major issues from the earlier version:

1. priors and clipping bounds are built only from the training split, so evaluation is not polluted by test-set information
2. partial-day usage is projected with a time-of-day-aware heuristic curve instead of `24 / current_hour`
3. charging is simulated with a piecewise curve instead of one constant charge speed

## Inputs

### `DeviceSpec`

The assistant takes a `DeviceSpec`:

- `device_model`
- `operating_system`
- `number_of_apps_installed`
- `battery_capacity_mah` (optional)

If battery capacity is not provided, the assistant uses a documented fallback value:

- default battery capacity: `4600.0 mAh`

The forecast output records whether the capacity was:

- `"provided"`
- `"assumed_default"`

This makes the battery-capacity assumption explicit without trying to scrape device capacities yet.

### `UsageSnapshot`

The live snapshot contains:

- current time as `current_hour`
- current battery percentage
- starting battery percentage for the day
- app usage minutes so far
- screen-on hours so far
- data usage so far

## Training flow

`train_predictor()` now works like this:

1. load the dataset
2. split into train and test
3. fit the preprocessor on `X_train`
4. train the regressor on `X_train`
5. build usage priors from `X_train` only
6. build clipping bounds from `X_train` only
7. evaluate on `X_test`

This removes the leakage that existed when priors and bounds were computed from the full dataframe.

## Historical priors

The assistant uses historical daily-usage medians at three levels:

- by `(device model, operating system)`
- by `operating system`
- global fallback

This gives the assistant a notion of typical usage when the live snapshot is still incomplete.

## Time-of-day-aware projection

The assistant cannot learn real hourly behavior from this repo because the dataset only contains daily totals.

So it uses a fixed heuristic hourly usage curve:

- hours `00-05`: `0.01` each
- hours `06-08`: `0.04` each
- hours `09-16`: `0.055` each
- hours `17-21`: `0.07` each
- hours `22-23`: `0.015` each

These weights sum to `1.0`.

The helper `cumulative_usage_share(current_hour, usage_curve)` estimates how much of a normal day’s usage should already have happened by the current time.

Example:

- at `06:00`, expected usage share is `0.06`
- at `11:00`, expected usage share is `0.29`
- at `18:00`, expected usage share is `0.69`
- at `24:00`, expected usage share is `1.00`

This share is clamped to `[0.05, 1.0]` so the projection does not explode very early in the day.

### Projection formula

For each dynamic usage feature:

```text
projected_full_day_value = usage_so_far / cumulative_usage_share
```

Then the projected value is clipped to training-derived bounds.

This replaces the old `24 / current_hour` logic.

## Blending historical and live usage

The assistant blends:

- historical daily usage for similar devices / OS
- projected full-day usage from today’s live snapshot

The blend weight is now based on how much of a normal day’s usage has likely happened:

```text
today_usage_weight = clamp(cumulative_usage_share, 0.2, 0.85)
```

This is better than using `current_hour / 24`, because it measures observed information rather than just wall-clock progress.

## Drain forecast

`forecast_drain()` produces three key estimates:

1. `model_full_day_drain_mah`
   - the regressor prediction from the blended feature row

2. `observed_rate_full_day_drain_mah`
   - the full-day drain implied by the battery loss already observed today

3. `predicted_full_day_drain_mah`
   - a blend of the two

### Observed drain so far

```text
observed_drain_so_far_mah =
    battery_capacity_mah * (starting_battery_pct - current_battery_pct) / 100
```

### Observed-rate full-day estimate

```text
observed_rate_full_day_drain_mah =
    observed_drain_so_far_mah / cumulative_usage_share
```

This is intentionally aligned to the same time-of-day curve used for usage projection.

### Final full-day estimate

The assistant blends:

- model-based estimate
- observed-rate estimate

using:

```text
observed_weight = clamp(cumulative_usage_share, 0.15, 0.8)
```

and then:

```text
predicted_full_day_drain_mah =
    model_full_day_drain_mah * (1 - observed_weight)
    + observed_rate_full_day_drain_mah * observed_weight
```

Finally it enforces:

```text
predicted_full_day_drain_mah >= observed_drain_so_far_mah
```

so the forecast cannot become physically impossible.

## Charge recommendation model

The target charging band remains:

- start charging at `45%`
- stop charging at `55%`
- preferred center around `50%`

### Piecewise charging curve

Charging is no longer simulated with one constant rate.

The assistant uses:

- `0% <= level < 50%` -> `30 pct/hour`
- `50% <= level < 80%` -> `18 pct/hour`
- `80% <= level <= 100%` -> `7 pct/hour`

This is still heuristic, but it is much more realistic than a single fixed charging speed.

### Simulation

`simulate_battery_levels()` runs forward in `15-minute` steps:

- compute drain for the step
- if charging is allowed and battery is at or below `45%`, start charging
- add charge according to the current battery band
- stop charging once `55%` is reached
- continue stepping to end of day

## Remaining-drain allocation

The assistant still uses a heuristic forward drain allocation, not a learned hourly drain model.

It uses:

- `hour_activity_multiplier()` for time-of-day behavior
- `estimate_usage_pressure()` to front-load near-term drain when current usage is heavier than the historical baseline

This is intentionally described as heuristic, not learned.

## Artifacts and compatibility

The predictor artifact is now:

- `battery_predictor_v3.pkl`

The bundle contains:

- `bundle_version`
- fitted model
- fitted preprocessor
- train-only priors
- train-only dynamic bounds
- metrics
- the heuristic usage curve

Older bundle versions are retrained automatically instead of being reused.

## Test coverage

The root `test_user_behaviors.py` covers:

- train-only priors
- cumulative usage share at known times
- projection using cumulative share
- today-usage weighting
- piecewise charging rates
- physical consistency of the forecast
- explicit capacity-source reporting
- heavy-usage scenario requiring charging
- light-usage scenario avoiding charging
- unknown-device fallback
- high-battery late-day scenario with no unnecessary charging
- borderline `45%` immediate charging
- capacity-sensitive conversion differences when a custom device capacity is provided
