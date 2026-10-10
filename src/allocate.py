
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import NUM_SPARES, RANDOM_SEED


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "transformers.csv"

DEPOT_NAMES = [
    "Jaipur",
    "Alwar",
    "Tonk",
    "Dausa",
    "Bharatpur",
    "Jhalawar",
    "Kota",
    "Sawai Madhopur",
]


def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "Scored transformers.csv not found. Run "
            "`python -m src.data` and `python -m src.model` first."
        )

    df = pd.read_csv(DATA_PATH)

    required = ["nearest_depot", "predicted_risk"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}. Run `python -m src.model`.")

    if df["predicted_risk"].isna().any():
        raise ValueError("predicted_risk contains missing values.")

    return df


def _largest_remainder_allocation(weights, total_spares=NUM_SPARES):
    """Convert nonnegative weights into integer allocations summing exactly to total."""
    weights = np.asarray(weights, dtype=float)

    if total_spares < 0:
        raise ValueError("total_spares cannot be negative.")

    if len(weights) == 0:
        if total_spares == 0:
            return np.array([], dtype=int)
        raise ValueError("Cannot allocate spares without depots.")

    weights = np.clip(weights, 0, None)

    if weights.sum() == 0:
        weights = np.ones(len(weights), dtype=float)

    quotas = weights / weights.sum() * total_spares
    allocation = np.floor(quotas).astype(int)
    remaining = total_spares - int(allocation.sum())

    # Stable tie-breaking keeps results reproducible.
    order = np.argsort(-(quotas - allocation), kind="stable")
    allocation[order[:remaining]] += 1

    return allocation


def equal_split(df=None, total_spares=NUM_SPARES):
    """Distribute spares equally among the eight depots."""
    return dict(
        zip(
            DEPOT_NAMES,
            _largest_remainder_allocation(
                np.ones(len(DEPOT_NAMES)), total_spares
            ).tolist(),
        )
    )


def proportional_count(df=None, total_spares=NUM_SPARES):
    """Allocate spares in proportion to transformers nearest to each depot."""
    if df is None:
        df = load_data()

    counts = (
        df["nearest_depot"]
        .value_counts()
        .reindex(DEPOT_NAMES, fill_value=0)
        .to_numpy(dtype=float)
    )

    allocation = _largest_remainder_allocation(counts, total_spares)
    return dict(zip(DEPOT_NAMES, allocation.tolist()))


def risk_based(df=None, total_spares=NUM_SPARES):
    """
    Allocate spares one at a time to the depot with the largest shortfall:
    sum(predicted risk) minus spares already allocated there.
    """
    if df is None:
        df = load_data()

    expected_failures = (
        df.groupby("nearest_depot")["predicted_risk"]
        .sum()
        .reindex(DEPOT_NAMES, fill_value=0.0)
        .to_dict()
    )

    allocation = {name: 0 for name in DEPOT_NAMES}

    for _ in range(total_spares):
        depot = max(
            DEPOT_NAMES,
            key=lambda name: (
                expected_failures[name] - allocation[name],
                -DEPOT_NAMES.index(name),
            ),
        )
        allocation[depot] += 1

    return allocation


def get_all_allocations(df=None, total_spares=NUM_SPARES):
    if df is None:
        df = load_data()

    return {
        "Equal split": equal_split(df, total_spares),
        "Proportional count": proportional_count(df, total_spares),
        "Risk-based": risk_based(df, total_spares),
    }


def print_allocations(df=None, total_spares=NUM_SPARES):
    if df is None:
        df = load_data()

    allocations = get_all_allocations(df, total_spares)

    table = pd.DataFrame(allocations).reindex(DEPOT_NAMES)
    table.index.name = "Depot"
    table.loc["TOTAL"] = table.sum(axis=0)

    print(f"\nSpare allocation comparison ({total_spares} spares)")
    print(table.to_string())
    print("\nAllocation totals:")
    for strategy, allocation in allocations.items():
        assert sum(allocation.values()) == total_spares
        print(f"{strategy}: {sum(allocation.values())}")

    return allocations


if __name__ == "__main__":
    print_allocations()
