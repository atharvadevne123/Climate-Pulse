"""Seed the database with synthetic weather observations for development."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import UTC, datetime, timedelta

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database import SessionLocal, WeatherObservation, init_db

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def seed(n_stations: int = 5, n_days: int = 90) -> int:
    """Insert synthetic weather observations for n_stations over n_days."""
    init_db()
    db = SessionLocal()
    rng = np.random.default_rng(0)
    station_ids = [f"STATION_{i:03d}" for i in range(n_stations)]
    base_time = datetime.now(UTC) - timedelta(days=n_days)
    records_created = 0

    for station_id in station_ids:
        for day_offset in range(n_days):
            ts = base_time + timedelta(days=day_offset)
            month = ts.month
            temp = 15 + 12 * float(np.sin(2 * np.pi * month / 12)) + float(rng.normal(0, 2))
            obs = WeatherObservation(
                station_id=station_id,
                timestamp=ts,
                temperature=round(temp, 2),
                precipitation=round(float(max(0, rng.normal(2, 3))), 3),
                humidity=round(float(rng.uniform(30, 95)), 1),
                pressure=round(float(rng.normal(1013, 8)), 1),
                wind_speed=round(float(abs(rng.normal(15, 6))), 1),
                cloud_cover=round(float(rng.uniform(0, 100)), 1),
            )
            db.add(obs)
            records_created += 1

    db.commit()
    db.close()
    logger.info("seed_data: inserted %d observations for %d stations", records_created, n_stations)
    return records_created


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed Climate-Pulse database with synthetic observations.")
    parser.add_argument("--stations", type=int, default=5, help="Number of stations to seed (default: 5)")
    parser.add_argument("--days", type=int, default=90, help="Number of days of history per station (default: 90)")
    args = parser.parse_args()
    total = seed(n_stations=args.stations, n_days=args.days)
    print(f"Seeded {total} weather observations ({args.stations} stations × {args.days} days).")
