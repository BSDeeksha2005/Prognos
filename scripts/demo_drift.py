"""Demonstrate sensor drift flags using a deterministic synthetic shift."""

import numpy as np

from rul.drift import flag_drift


def main() -> None:
    """Print PSI and KS output for a shifted sensor and a stable sensor."""
    rng = np.random.default_rng(42)
    reference = {"sensor_1": rng.normal(0, 1, 1000), "sensor_2": rng.normal(0, 1, 1000)}
    current = {"sensor_1": rng.normal(1.5, 1, 1000), "sensor_2": rng.normal(0, 1, 1000)}
    print(flag_drift(reference, current))


if __name__ == "__main__":
    main()
