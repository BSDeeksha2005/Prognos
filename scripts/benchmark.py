"""Measure local API request latency percentiles."""

import statistics
import time

from fastapi.testclient import TestClient

from api.main import app


def main() -> None:
    """Report p50 and p95 latency for a fixed valid request."""
    payload = {"window": [[0.0] * 21 for _ in range(30)]}
    with TestClient(app) as client:
        samples = []
        for _ in range(200):
            started = time.perf_counter()
            response = client.post("/predict", json=payload)
            if response.status_code != 200:
                raise RuntimeError(
                    f"Prediction endpoint returned {response.status_code}: {response.text}"
                )
            samples.append((time.perf_counter() - started) * 1000)
    ordered = sorted(samples)
    print(
        f"p50_ms={statistics.median(ordered):.3f} "
        f"p95_ms={ordered[int(0.95 * (len(ordered) - 1))]:.3f}"
    )


if __name__ == "__main__":
    main()
