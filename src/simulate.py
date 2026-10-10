from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t

from src.allocate import get_all_allocations, load_data
from src.config import (
    CENTRAL_STORE_DELAY_HOURS,
    HANDLING_TIME_MAX_HOURS,
    HANDLING_TIME_MIN_HOURS,
    MAX_BORROW_KM,
    NUM_SIMULATION_RUNS,
    PEAK_WEEK_MULTIPLIER,
    RANDOM_SEED,
    ROAD_DISTANCE_FACTOR,
    RURAL_DEADLINE_HOURS,
    RURAL_SPEED_KMH,
    URBAN_DEADLINE_HOURS,
    URBAN_SPEED_KMH,
)
from src.data import DEPOTS, haversine_km

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def calculate_distances(df):
    """Calculate road-adjusted distances from every transformer to each depot."""
    depot_latitudes = np.array([depot["lat"] for depot in DEPOTS])
    depot_longitudes = np.array([depot["lon"] for depot in DEPOTS])
    distances = haversine_km(
        df["lat"].to_numpy()[:, None], df["lon"].to_numpy()[:, None],
        depot_latitudes[None, :], depot_longitudes[None, :],
    )
    distances *= ROAD_DISTANCE_FACTOR
    depot_names = [depot["name"] for depot in DEPOTS]
    if set(depot_names) != set(get_all_allocations(df)["Equal split"]):
        raise ValueError("Depot names in data.py and allocate.py do not match.")
    allocation_order = list(get_all_allocations(df)["Equal split"])
    depot_indices = [depot_names.index(name) for name in allocation_order]
    return distances[:, depot_indices], allocation_order


def get_period_probabilities(df, total_spares, scenario, peak_multiplier):
    """Scale synthetic failure probabilities to the selected scenario."""
    risk = df["true_failure_prob"].to_numpy(dtype=float)
    if scenario == "normal":
        target_failures = 0.70 * total_spares
    elif scenario == "peak":
        target_failures = 0.70 * total_spares * peak_multiplier
    else:
        raise ValueError(f"Unknown scenario: {scenario}")
    if peak_multiplier <= 0:
        raise ValueError("peak_multiplier must be greater than zero.")
    if np.any(risk < 0) or not np.all(np.isfinite(risk)):
        raise ValueError("Failure probabilities contain invalid values.")
    if risk.sum() <= 0:
        return np.zeros(len(df), dtype=float)
    probabilities = risk * (target_failures / risk.sum())
    return np.clip(probabilities, 0.0, 1.0)


def simulate_one_strategy(df, allocation, failed_indices, handling_times,
                          distances, area_types, depot_names):
    """Simulate a strategy; borrowing is allowed only within MAX_BORROW_KM."""
    remaining = dict(allocation)
    restoration_times = []
    within_deadline = []
    stock_outs = 0

    for transformer_index in failed_indices:
        area_type = area_types[transformer_index]
        if area_type == "rural":
            speed, deadline = RURAL_SPEED_KMH, RURAL_DEADLINE_HOURS
        else:
            speed, deadline = URBAN_SPEED_KMH, URBAN_DEADLINE_HOURS

        # Road-adjusted distances are already sorted nearest first.
        depot_order = np.argsort(distances[transformer_index])
        selected_depot = None
        for depot_index in depot_order:
            distance_km = distances[transformer_index, depot_index]
            if distance_km > MAX_BORROW_KM:
                # All remaining depots are farther away.
                break
            depot_name = depot_names[depot_index]
            if remaining[depot_name] > 0:
                selected_depot = depot_index
                remaining[depot_name] -= 1
                break

        if selected_depot is None:
            # Stock-out means no depot within the allowed radius had a spare.
            restoration_time = CENTRAL_STORE_DELAY_HOURS
            stock_outs += 1
        else:
            travel_time = distances[transformer_index, selected_depot] / speed
            restoration_time = travel_time + handling_times[transformer_index]

        restoration_times.append(restoration_time)
        within_deadline.append(restoration_time <= deadline)

    if not restoration_times:
        return {
            "failures": 0,
            "restored_within_deadline_pct": 100.0,
            "mean_restoration_hours": 0.0,
            "p95_restoration_hours": 0.0,
            "stock_outs": 0,
        }
    return {
        "failures": len(restoration_times),
        "restored_within_deadline_pct": 100.0 * np.mean(within_deadline),
        "mean_restoration_hours": float(np.mean(restoration_times)),
        "p95_restoration_hours": float(np.percentile(restoration_times, 95)),
        "stock_outs": stock_outs,
    }


def paired_mean_ci(differences, confidence=0.95):
    """Student-t confidence interval for a vector of paired run differences."""
    values = np.asarray(differences, dtype=float)
    n = len(values)
    mean = float(values.mean())
    if n < 2:
        return mean, np.nan, np.nan
    standard_error = values.std(ddof=1) / np.sqrt(n)
    critical = t.ppf((1 + confidence) / 2, df=n - 1)
    return mean, float(mean - critical * standard_error), float(mean + critical * standard_error)


def run_simulation(df=None, num_runs=NUM_SIMULATION_RUNS, seed=RANDOM_SEED,
                   peak_multiplier=PEAK_WEEK_MULTIPLIER, total_spares=None):
    """Compare strategies on paired failures, processing order and handling times."""
    if num_runs < 1:
        raise ValueError("num_runs must be at least 1.")
    if df is None:
        df = load_data()
    if total_spares is None:
        total_spares = sum(get_all_allocations(df)["Equal split"].values())
    if total_spares < 1:
        raise ValueError("total_spares must be at least 1.")

    allocations = get_all_allocations(df, total_spares)
    distances, depot_names = calculate_distances(df)
    area_types = df["area_type"].to_numpy()
    rng = np.random.default_rng(seed)
    records = []

    for scenario in ("normal", "peak"):
        probabilities = get_period_probabilities(df, total_spares, scenario, peak_multiplier)
        print(f"{scenario.title()} scenario: actual expected failures per run = {probabilities.sum():.4f}")
        for run_number in range(1, num_runs + 1):
            # Draw failures, processing order and handling times once, then reuse for all strategies.
            random_values = rng.random(len(df))
            failed_indices = np.flatnonzero(random_values < probabilities)
            rng.shuffle(failed_indices)
            handling_times = rng.uniform(
                HANDLING_TIME_MIN_HOURS, HANDLING_TIME_MAX_HOURS, size=len(df)
            )
            for strategy, allocation in allocations.items():
                metrics = simulate_one_strategy(
                    df, allocation, failed_indices, handling_times,
                    distances, area_types, depot_names,
                )
                records.append({
                    "scenario": scenario,
                    "strategy": strategy,
                    "run": run_number,
                    **metrics,
                })

    raw = pd.DataFrame(records)
    paired = raw.pivot(index=["scenario", "run"], columns="strategy",
                       values="restored_within_deadline_pct")
    paired_rows = []
    for scenario in ("normal", "peak"):
        scenario_pairs = paired.loc[scenario]
        diff_prop = scenario_pairs["Risk-based"] - scenario_pairs["Proportional count"]
        diff_equal = scenario_pairs["Risk-based"] - scenario_pairs["Equal split"]
        paired_rows.append({
            "scenario": scenario,
            "comparison": "Risk-based − Proportional count",
            "mean_difference_pp": paired_mean_ci(diff_prop)[0],
            "ci95_low_pp": paired_mean_ci(diff_prop)[1],
            "ci95_high_pp": paired_mean_ci(diff_prop)[2],
        })
        paired_rows.append({
            "scenario": scenario,
            "comparison": "Risk-based − Equal split",
            "mean_difference_pp": paired_mean_ci(diff_equal)[0],
            "ci95_low_pp": paired_mean_ci(diff_equal)[1],
            "ci95_high_pp": paired_mean_ci(diff_equal)[2],
        })
    paired_summary = pd.DataFrame(paired_rows)

    summary = raw.groupby(["scenario", "strategy"]).agg(
        restored_mean_pct=("restored_within_deadline_pct", "mean"),
        restored_std_pct=("restored_within_deadline_pct", "std"),
        mean_stock_outs=("stock_outs", "mean"),
    ).reset_index()
    summary[["restored_mean_pct", "restored_std_pct", "mean_stock_outs"]] = summary[
        ["restored_mean_pct", "restored_std_pct", "mean_stock_outs"]
    ].round(3)

    output_path = PROJECT_ROOT / "data" / "simulation_results.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw.to_csv(output_path, index=False)
    print(f"\nMonte Carlo runs per strategy/scenario: {num_runs}")
    print(f"Peak multiplier: {peak_multiplier:.2f}")
    print(f"Maximum borrowing road distance: {MAX_BORROW_KM} km")
    print("Restocking during a simulated period: No")
    print(f"Saved run-level results: {output_path}")
    print("\nDeadline restoration percentage and stock-outs (mean across runs):")
    print(summary.to_string(index=False))
    print("\nPaired differences in deadline restoration percentage (percentage points):")
    print(paired_summary.round(3).to_string(index=False))
    return raw, summary, paired_summary


if __name__ == "__main__":
    run_simulation()
