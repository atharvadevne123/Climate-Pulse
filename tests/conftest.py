"""Shared pytest fixtures for Climate-Pulse tests."""

from __future__ import annotations

import os
from collections.abc import Generator

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

os.environ.setdefault("DATABASE_URL", "sqlite:///./test_climate_pulse.db")
os.environ.setdefault("MODEL_DIR", "/tmp/test_climate_models")

from app.database import Base, get_db  # noqa: E402
from app.main import app  # noqa: E402

TEST_DB_URL = "sqlite:///./test_climate_pulse.db"
test_engine = create_engine(TEST_DB_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)


@pytest.fixture(scope="session", autouse=True)
def setup_test_db():
    Base.metadata.create_all(bind=test_engine)
    yield
    Base.metadata.drop_all(bind=test_engine)


@pytest.fixture
def db() -> Generator[Session, None, None]:
    connection = test_engine.connect()
    transaction = connection.begin()
    session = sessionmaker(autocommit=False, autoflush=False, bind=connection)()
    try:
        yield session
    finally:
        session.close()
        transaction.rollback()
        connection.close()


@pytest.fixture
def client(db) -> Generator[TestClient, None, None]:
    def override_get_db():
        try:
            yield db
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture
def sample_weather_payload() -> dict:
    return {
        "station_id": "STATION_001",
        "temperature": 22.5,
        "precipitation": 3.2,
        "humidity": 65.0,
        "pressure": 1012.5,
        "wind_speed": 18.0,
        "cloud_cover": 40.0,
        "month": 6.0,
        "day_of_year": 160.0,
    }


@pytest.fixture
def sample_df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "temperature": rng.normal(15, 8, 100),
            "precipitation": np.abs(rng.normal(2, 3, 100)),
            "humidity": rng.uniform(30, 95, 100),
            "pressure": rng.normal(1013, 10, 100),
            "wind_speed": np.abs(rng.normal(15, 8, 100)),
            "cloud_cover": rng.uniform(0, 100, 100),
            "month": rng.integers(1, 13, 100).astype(float),
            "day_of_year": rng.integers(1, 366, 100).astype(float),
        }
    )
