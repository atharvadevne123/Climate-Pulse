"""Standalone model retraining script for CLI use."""
from __future__ import annotations

import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.model import train_models

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    logger.info("retrain.main: starting model retraining")
    metrics = train_models()
    logger.info("retrain.main: completed metrics=%s", metrics)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
