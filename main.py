import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('./datasets/user_behavior_dataset.csv')

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

categorical_cols = ['Device Model', 'Operating System', 'Gender']
numerical_cols = [
    'App Usage Time (min/day)',
    'Screen On Time (hours/day)',
    'Data Usage (MB/day)',
    'Number of Apps Installed',
    'Age'
]

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
        ('num', StandardScaler(), numerical_cols)
    ]
)

X = df[categorical_cols + numerical_cols]
y = df['Battery Drain (mAh/day)']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

model_xgb = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    random_state=42
)

model_xgb.fit(X_train_processed, y_train)

from sklearn.metrics import mean_absolute_error
y_pred = model_xgb.predict(X_test_processed)
print("MAE:", mean_absolute_error(y_test, y_pred))

import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler

joblib.dump(model_xgb, "battery_model.pkl")
joblib.dump(preprocessor, "preprocessor.pkl")

print("✅ Model and preprocessor saved successfully!")

import joblib
import pandas as pd

# Load model + preprocessor
model_xgb = joblib.load("battery_model.pkl")
preprocessor = joblib.load("preprocessor.pkl")


def predict_daily_drain(user_input_dict):
    df_input = pd.DataFrame([user_input_dict])

    # Transform using saved preprocessor (handles encoding + scaling automatically)
    X_processed = preprocessor.transform(df_input)

    prediction = model_xgb.predict(X_processed)[0]
    return round(prediction, 1)


# Example user
sample_user = {
    'Device Model': 'Xiaomi Mi 11',
    'Operating System': 'Android',
    'App Usage Time (min/day)': 420,
    'Screen On Time (hours/day)': 7.2,
    'Data Usage (MB/day)': 1800,
    'Number of Apps Installed': 85,
    'Age': 28,
    'Gender': 'Male'
}

print("🔮 Predicted daily drain:", predict_daily_drain(sample_user), "mAh")

def predict_remaining_drain(user_input_dict,
                            hours_elapsed,
                            battery_capacity_mah=4600):
    
    df_input = pd.DataFrame([user_input_dict])
    
    # 1️⃣ Project full-day usage proportionally
    scaling_factor = 24 / hours_elapsed
    
    df_input['App Usage Time (min/day)'] *= scaling_factor
    df_input['Screen On Time (hours/day)'] *= scaling_factor
    df_input['Data Usage (MB/day)'] *= scaling_factor
    
    # 2️⃣ Predict full-day drain
    X_processed = preprocessor.transform(df_input)
    predicted_full_day_drain = model_xgb.predict(X_processed)[0]
    
    # 3️⃣ Estimate drain so far based on time proportion
    drain_so_far_estimated = predicted_full_day_drain * (hours_elapsed / 24)
    
    # 4️⃣ Remaining drain
    remaining_drain = predicted_full_day_drain - drain_so_far_estimated
    
    # 5️⃣ Derive consistent battery %
    current_battery_pct = 100 - (drain_so_far_estimated / battery_capacity_mah) * 100
    
    return remaining_drain, predicted_full_day_drain, current_battery_pct

    import pulp

def optimize_remaining_day(predicted_remaining_drain_mah,
                           current_hour,
                           current_battery_pct,
                           battery_capacity_mah=4600,
                           target_min_pct=40,
                           penalty_per_pct=300.0):
    hours_left = 24 - current_hour
    if hours_left <= 0:
        print("No hours left to optimize.")
        return

    daily_remaining_pct = (predicted_remaining_drain_mah / battery_capacity_mah) * 100
    hourly_drain_pct = daily_remaining_pct / hours_left

    prob = pulp.LpProblem("Smart_Charging_Remaining_SOFT", pulp.LpMinimize)

    level = [pulp.LpVariable(f"level_{h}", 0, 100) for h in range(hours_left)]
    charge = [pulp.LpVariable(f"charge_{h}", 0, 38) for h in range(hours_left)]
    above_80 = [pulp.LpVariable(f"above80_{h}", lowBound=0) for h in range(hours_left)]
    end_shortfall = pulp.LpVariable("end_shortfall", 0)

    # Objective
    prob += (pulp.lpSum(above_80) + 0.01 * pulp.lpSum(level)
             + penalty_per_pct * end_shortfall)

    for h in range(hours_left):
        prev = current_battery_pct if h == 0 else level[h-1]
        prob += above_80[h] >= level[h] - 80
        prob += above_80[h] >= 0
        prob += level[h] == prev - hourly_drain_pct + charge[h]

    # Soft target: penalize ending below target_min_pct
    prob += end_shortfall >= target_min_pct - level[hours_left-1]
    prob += end_shortfall >= 0

    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    print("Solver Status:", pulp.LpStatus[prob.status])

    optimal_levels = [round(v.value(), 1) for v in level]
    optimal_charges = [round(v.value(), 2) for v in charge]
    print("\n📅 Remaining-day battery schedule:")
    for i in range(hours_left):
        real_hour = current_hour + i
        lvl = optimal_levels[i]
        chg = optimal_charges[i]
        if chg > 0:
            print(f"{real_hour:02d}:00 → {lvl}% 🔌 Charge: {chg}")
        else:
            print(f"{real_hour:02d}:00 → {lvl}%")

    print("\nTotal charge scheduled:", round(sum(optimal_charges),2))
    if sum(optimal_charges) > 0:
        print("⚡ Recommendation: Charging required.")
    else:
        print("✅ Recommendation: No charging needed.")

def optimize_battery_health(predicted_remaining_drain_mah,
                           current_hour,
                           current_battery_pct,
                           battery_capacity_mah=4600,
                           optimal_range=(40, 70),
                           min_acceptable=30,
                           penalty_weights=None):
    """
    Optimize for battery LONGEVITY, not just daily survival
    """
    if penalty_weights is None:
        penalty_weights = {
            'above_80': 500.0,
            'below_40': 300.0,
            'outside_optimal': 50.0,
            'end_shortfall': 1000.0,
            'fast_charging': 100.0,
            'charge_cycles': 150.0
        }
    
    hours_left = 24 - current_hour
    if hours_left <= 0:
        return None, None
    
    daily_remaining_pct = (predicted_remaining_drain_mah / battery_capacity_mah) * 100
    hourly_drain_pct = daily_remaining_pct / hours_left
    
    prob = pulp.LpProblem("Battery_Longevity_Optimization", pulp.LpMinimize)
    
    # Decision variables
    level = [pulp.LpVariable(f"level_{h}", 0, 100) for h in range(hours_left)]
    charge = [pulp.LpVariable(f"charge_{h}", 0, 20) for h in range(hours_left)]
    
    # Binary variables for charge sessions
    is_charging = [pulp.LpVariable(f"charging_{h}", cat='Binary') for h in range(hours_left)]
    
    # Penalty variables
    above_80 = [pulp.LpVariable(f"above80_{h}", lowBound=0) for h in range(hours_left)]
    below_40 = [pulp.LpVariable(f"below40_{h}", lowBound=0) for h in range(hours_left)]
    outside_optimal = [pulp.LpVariable(f"outside_opt_{h}", lowBound=0) for h in range(hours_left)]
    end_shortfall = pulp.LpVariable("end_shortfall", lowBound=0)
    
    # OBJECTIVE: Multi-factor battery health optimization
    # Fixed: Use linear penalty for fast charging instead of quadratic
    prob += (
        penalty_weights['above_80'] * pulp.lpSum(above_80) +
        penalty_weights['below_40'] * pulp.lpSum(below_40) +
        penalty_weights['outside_optimal'] * pulp.lpSum(outside_optimal) +
        penalty_weights['end_shortfall'] * end_shortfall +
        penalty_weights['fast_charging'] * pulp.lpSum(charge) +  # Linear penalty
        penalty_weights['charge_cycles'] * pulp.lpSum(is_charging)
    )
    
    # CONSTRAINTS
    for h in range(hours_left):
        prev = current_battery_pct if h == 0 else level[h-1]
        
        # Battery level dynamics
        prob += level[h] == prev - hourly_drain_pct + charge[h]
        
        # Above 80% penalty
        prob += above_80[h] >= level[h] - 80
        prob += above_80[h] >= 0
        
        # Below 40% penalty
        prob += below_40[h] >= 40 - level[h]
        prob += below_40[h] >= 0
        
        # Outside optimal range (40-70%) penalty
        prob += outside_optimal[h] >= level[h] - optimal_range[1]  # Above 70%
        prob += outside_optimal[h] >= optimal_range[0] - level[h]  # Below 40%
        prob += outside_optimal[h] >= 0
        
        # Link charging to binary variable
        prob += charge[h] <= 20 * is_charging[h]  # If not charging, charge = 0
        prob += charge[h] >= 0.1 * is_charging[h]  # If charging, must charge at least 0.1%
    
    # End-of-day constraint (soft)
    prob += end_shortfall >= min_acceptable - level[hours_left-1]
    prob += end_shortfall >= 0
    
    # Solve
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    
    # Extract and display results
    if pulp.LpStatus[prob.status] == 'Optimal':
        optimal_levels = [round(v.value(), 1) for v in level]
        optimal_charges = [round(v.value(), 2) for v in charge]
        
        print("\n🔋 BATTERY LONGEVITY-OPTIMIZED SCHEDULE:")
        print("=" * 60)
        
        charge_sessions = []
        in_session = False
        session_start = None
        
        for i in range(hours_left):
            real_hour = current_hour + i
            lvl = optimal_levels[i]
            chg = optimal_charges[i]
            
            # Track charge sessions
            if chg > 0.1 and not in_session:
                in_session = True
                session_start = real_hour
            elif chg <= 0.1 and in_session:
                charge_sessions.append((session_start, real_hour - 1))
                in_session = False
            
            # Color code by health zone
            if lvl > 80:
                zone = "⚠️ HIGH (BAD FOR BATTERY)"
            elif lvl >= 70:
                zone = "🟡 ELEVATED"
            elif lvl >= 40:
                zone = "✅ OPTIMAL"
            elif lvl >= 30:
                zone = "🟠 LOW"
            else:
                zone = "🔴 CRITICAL"
            
            charge_indicator = f"🔌 +{chg}%" if chg > 0 else ""
            print(f"{real_hour:02d}:00 → {lvl:5.1f}% {zone:25} {charge_indicator}")
        
        # Close any open session
        if in_session:
            charge_sessions.append((session_start, 23))
        
        # Summary
        print("\n" + "=" * 60)
        print(f"📊 HEALTH METRICS:")
        print(f"   • Total charging sessions: {len(charge_sessions)}")
        print(f"   • Total charge added: {sum(optimal_charges):.1f}%")
        print(f"   • Time in optimal range (40-70%): {sum(1 for l in optimal_levels if 40 <= l <= 70)}/{hours_left} hours")
        print(f"   • Time above 80%: {sum(1 for l in optimal_levels if l > 80)}/{hours_left} hours")
        print(f"   • Final battery level: {optimal_levels[-1]:.1f}%")
        
        if charge_sessions:
            print(f"\n⚡ RECOMMENDED CHARGING WINDOWS:")
            for start, end in charge_sessions:
                print(f"   • {start:02d}:00 - {end:02d}:00")
        else:
            print("\n✅ No charging needed - battery will stay healthy!")
        
        return optimal_levels, optimal_charges
    else:
        print(f"❌ Optimization failed: {pulp.LpStatus[prob.status]}")
        return None, None

current_hour = 6

usage_so_far = {
    'Device Model': 'Xiaomi Mi 11',
    'Operating System': 'Android',
    'App Usage Time (min/day)': 550,   # heavy usage till 1 PM
    'Screen On Time (hours/day)': 8.5,
    'Data Usage (MB/day)': 2200,
    'Number of Apps Installed': 130,
    'Age': 28,
    'Gender': 'Male'
}


remaining_drain, full_day_drain, current_battery_pct = predict_remaining_drain(
    usage_so_far,
    hours_elapsed=current_hour
)

print("Predicted Full-Day Drain:", round(full_day_drain, 1), "mAh")
print("Remaining Drain:", round(remaining_drain, 1), "mAh")

optimize_battery_health(
    predicted_remaining_drain_mah=remaining_drain,
    current_hour=current_hour,
    current_battery_pct=current_battery_pct
)

