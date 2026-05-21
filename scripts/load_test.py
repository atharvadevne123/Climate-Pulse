"""Simple load test for the Climate-Pulse prediction endpoint."""
from __future__ import annotations

import concurrent.futures
import json
import statistics
import time
import urllib.request

BASE_URL = "http://localhost:8000"

PAYLOAD = json.dumps({
    "station_id": "LOADTEST_001",
    "temperature": 22.5,
    "precipitation": 3.2,
    "humidity": 65.0,
    "pressure": 1012.5,
    "wind_speed": 18.0,
    "cloud_cover": 40.0,
    "month": 6.0,
    "day_of_year": 160.0,
}).encode()


def single_request() -> tuple[int, float]:
    start = time.perf_counter()
    req = urllib.request.Request(
        f"{BASE_URL}/api/v1/predict",
        data=PAYLOAD,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            resp.read()
            status = resp.status
    except Exception:
        status = 500
    elapsed = (time.perf_counter() - start) * 1000
    return status, elapsed


def run_load_test(n_requests: int = 100, concurrency: int = 10) -> None:
    print(f"Load test: {n_requests} requests, {concurrency} concurrent workers")
    latencies: list[float] = []
    errors = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(single_request) for _ in range(n_requests)]
        for f in concurrent.futures.as_completed(futures):
            status, elapsed = f.result()
            latencies.append(elapsed)
            if status != 200:
                errors += 1

    latencies.sort()
    print(f"  Requests:  {n_requests}")
    print(f"  Errors:    {errors}")
    print(f"  p50 (ms):  {statistics.median(latencies):.1f}")
    print(f"  p95 (ms):  {latencies[int(len(latencies) * 0.95)]:.1f}")
    print(f"  p99 (ms):  {latencies[int(len(latencies) * 0.99)]:.1f}")
    print(f"  max (ms):  {max(latencies):.1f}")


if __name__ == "__main__":
    run_load_test()
