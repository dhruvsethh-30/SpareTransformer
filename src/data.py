from pathlib import Path

import numpy as np
import pandas as pd

from src.config import NUM_TRANSFORMERS, RANDOM_SEED
# Eight assumed depot locations in Rajasthan.
# Unequal weights control how transformers are distributed around depot towns.
DEPOTS = [
    {"name": "Jaipur", "lat": 26.9124, "lon": 75.7873, "weight": 0.25},
    {"name": "Alwar", "lat": 27.5500, "lon": 76.6300, "weight": 0.17},
    {"name": "Tonk", "lat": 26.1667, "lon": 75.7885, "weight": 0.10},
    {"name": "Dausa", "lat": 26.8900, "lon": 76.3350, "weight": 0.09},
    {"name": "Bharatpur", "lat": 27.2152, "lon": 77.5030, "weight": 0.12},
    {"name": "Jhalawar", "lat": 24.5967, "lon": 76.1640, "weight": 0.07},
    {"name": "Kota", "lat": 25.2138, "lon": 75.8648, "weight": 0.13},
    {"name": "Sawai Madhopur", "lat": 26.0173, "lon": 76.5026, "weight": 0.07},
]


def generate_locations(n=NUM_TRANSFORMERS, seed=RANDOM_SEED):
    """Generate synthetic transformer coordinates around Rajasthan depot towns."""
    rng = np.random.default_rng(seed)

    weights = np.array([d["weight"] for d in DEPOTS], dtype=float)
    weights /= weights.sum()

    depot_indices = rng.choice(len(DEPOTS), size=n, p=weights)

    latitudes = np.empty(n)
    longitudes = np.empty(n)
    area_types = np.empty(n, dtype=object)
    for i, depot_index in enumerate(depot_indices):
        depot = DEPOTS[depot_index]

        # Spread transformers around each town to simulate a service region.
        # These are synthetic coordinates, not actual utility asset locations.
        rural = rng.random() < 0.65
        spread = rng.uniform(0.04, 0.18) if rural else rng.uniform(0.005, 0.04)
        latitudes[i] = depot["lat"] + rng.normal(0, spread)
        longitudes[i] = depot["lon"] + rng.normal(
            0, spread / max(np.cos(np.radians(depot["lat"])), 0.5)
        )
        area_types[i] = "rural" if rural else "urban"
    return pd.DataFrame(
        {
            "transformer_id": [f"TR-{i + 1:04d}" for i in range(n)],
            "lat": latitudes,
            "lon": longitudes,
            "generated_depot": [DEPOTS[i]["name"] for i in depot_indices],
            "area_type": area_types,
        }
    )


def haversine_km(lat1, lon1, lat2, lon2):
    """Calculate great-circle distance in kilometres; supports NumPy arrays."""
    earth_radius_km = 6371.0
    lat1 = np.radians(np.asarray(lat1, dtype=float))
    lon1 = np.radians(np.asarray(lon1, dtype=float))
    lat2 = np.radians(np.asarray(lat2, dtype=float))
    lon2 = np.radians(np.asarray(lon2, dtype=float))

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    )
    a = np.clip(a, 0.0, 1.0)

    return 2 * earth_radius_km * np.arcsin(np.sqrt(a))


def assign_nearest_depot(transformers):
    """Assign each transformer to its nearest depot by geographic distance."""
    depot_latitudes = np.array([d["lat"] for d in DEPOTS])
    depot_longitudes = np.array([d["lon"] for d in DEPOTS])
    depot_names = np.array([d["name"] for d in DEPOTS])

    distances = haversine_km(
        transformers["lat"].to_numpy()[:, None],
        transformers["lon"].to_numpy()[:, None],
        depot_latitudes[None, :],
        depot_longitudes[None, :],
    )
    nearest_indices = np.argmin(distances, axis=1)

    result = transformers.copy()
    result["nearest_depot"] = depot_names[nearest_indices]
    result["nearest_depot_distance_km"] = distances[
        np.arange(len(result)), nearest_indices
    ]

    return result


def sigmoid(x):
    """Numerically stable sigmoid."""
    x = np.clip(np.asarray(x, dtype=float), -40, 40)
    return 1.0 / (1.0 + np.exp(-x))


def tune_intercept(linear_predictor, target_mean=0.11):
    """Tune the intercept so the mean true probability equals target_mean."""
    low, high = -30.0, 30.0

    for _ in range(100):
        mid = (low + high) / 2
        mean_probability = sigmoid(linear_predictor + mid).mean()

        if mean_probability < target_mean:
            low = mid
        else:
            high = mid

    return (low + high) / 2


def generate_dataset():
    rng = np.random.default_rng(RANDOM_SEED)
    transformers = generate_locations()
    n = len(transformers)

    # Synthetic equipment condition variables.
    transformers["age_years"] = rng.integers(1, 31, size=n)
    transformers["load_pct"] = rng.uniform(30, 110, size=n).round(2)
    # Temperature rises with load, with additional measurement/environment noise.
    temperature = (
        43.0
        + 0.38 * (transformers["load_pct"].to_numpy() - 30)
        + rng.normal(0, 6.0, size=n)
    )
    transformers["oil_temp_c"] = np.clip(temperature, 40, 95).round(2)

    # Regional fleet-condition differences are assumptions for this synthetic model,
    # not measured facts about utility fleets in these towns.
    older_hotter = {"Kota", "Bharatpur", "Alwar"}
    younger_cooler = {"Tonk", "Dausa", "Jhalawar", "Sawai Madhopur"}
    depot = transformers["generated_depot"]
    age_shift = np.where(depot.isin(older_hotter), 6,
                         np.where(depot.isin(younger_cooler), -4, 0))
    temp_shift = age_shift.copy()
    transformers["age_years"] = np.clip(
        transformers["age_years"].to_numpy(dtype=int) + age_shift, 1, 30
    ).astype(int)
    transformers["oil_temp_c"] = np.clip(
        transformers["oil_temp_c"].to_numpy(dtype=float) + temp_shift, 40, 95
    ).round(2)

    # All three factors contribute to synthetic failure risk.
    # Coefficients are illustrative, not calibrated engineering parameters.
    age = transformers["age_years"].to_numpy(dtype=float)
    load = transformers["load_pct"].to_numpy(dtype=float)
    temp = transformers["oil_temp_c"].to_numpy(dtype=float)

    linear_predictor = (
        0.055 * age
        + 0.018 * (load - 70)
        + 0.025 * (temp - 65)
    )

    intercept = tune_intercept(linear_predictor, target_mean=0.11)
    transformers["true_failure_prob"] = sigmoid(
        linear_predictor + intercept
    )

    # Sample observed outcomes from the hidden probabilities.
    transformers["failed"] = rng.binomial(
        1, transformers["true_failure_prob"].to_numpy()
    )

    # Crucially, nearest depot is based on coordinates, not generated_depot.
    transformers = assign_nearest_depot(transformers)
    output_path = Path(__file__).resolve().parent.parent / "data" / "transformers.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    transformers.to_csv(output_path, index=False)
    print(f"Rows: {len(transformers)}")
    print(f"Mean true failure probability: {transformers['true_failure_prob'].mean():.4f}")
    print(f"Observed failure rate: {transformers['failed'].mean():.4f}")
    print("\nUrban/rural counts:")
    print(transformers["area_type"].value_counts().to_string())
    print("\nTransformers per nearest depot:")
    print(
        transformers["nearest_depot"]
        .value_counts()
        .reindex([d["name"] for d in DEPOTS], fill_value=0)
        .to_string()
    )
    print(f"\nSaved to: {output_path}")
    return transformers


if __name__ == "__main__":
    generate_dataset()
