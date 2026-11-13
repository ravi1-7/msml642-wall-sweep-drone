"""Time simulated flights to size the NSGA-II run.

Run: uv run python scripts/time_one_flight.py [--seconds 90] [--workers 12]
"""

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor

from wallsweep.sim import hover

TOTAL_FLIGHTS = 40 * 5 * 50  # population x walls per candidate x generations


def one(seconds: float) -> float:
    t = time.perf_counter()
    hover(seconds)
    return time.perf_counter() - t


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seconds", type=float, default=90.0, help="simulated length of one flight")
    p.add_argument("--workers", type=int, default=os.cpu_count())
    a = p.parse_args()

    single = one(a.seconds)
    print(f"one {a.seconds:.0f} s flight: {single:.2f} s wall-clock ({a.seconds / single:.0f}x real time)")

    n = a.workers * 2
    t = time.perf_counter()
    with ProcessPoolExecutor(a.workers) as pool:
        list(pool.map(one, [a.seconds] * n))
    per_flight = (time.perf_counter() - t) / n
    print(f"{a.workers} workers: {per_flight:.2f} s per flight on average")
    print(f"full NSGA-II run ({TOTAL_FLIGHTS} flights): ~{TOTAL_FLIGHTS * per_flight / 3600:.1f} h "
          f"(1 worker: ~{TOTAL_FLIGHTS * single / 3600:.1f} h)")
