
# config.py
# All project settings are kept here so we can change them easily.

RANDOM_SEED = 42  # Keeps random results reproducible.

NUM_TRANSFORMERS = 2000  # Number of transformers in our synthetic district.
NUM_DEPOTS = 8  # Number of spare-storage depots.
NUM_SPARES = 100  # Total spare transformers available.

RURAL_SPEED_KMH = 30  # Assumed average rural travel speed.
URBAN_SPEED_KMH = 40  # Assumed average urban travel speed.
ROAD_DISTANCE_FACTOR = 1.3  # Road distance is estimated from straight-line distance.

HANDLING_TIME_MIN_HOURS = 3  # Minimum time to handle and replace a transformer.
HANDLING_TIME_MAX_HOURS = 6  # Maximum time to handle and replace a transformer.
CENTRAL_STORE_DELAY_HOURS = 12  # Extra delay if no depot has a spare.

URBAN_DEADLINE_HOURS = 8  # Assumed urban restoration deadline from the project brief.
RURAL_DEADLINE_HOURS = 24  # Assumed rural restoration deadline from the project brief.

ANNUAL_FAILURE_RATE = 0.11  # Target average synthetic annual failure rate.
NUM_SIMULATION_RUNS = 500  # Number of Monte Carlo scenarios to simulate.
PEAK_WEEK_MULTIPLIER = 2.0  # Peak-week stress scenario increases failure probabilities.
