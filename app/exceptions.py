"""Custom exception handlers for Climate-Pulse FastAPI app."""
from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


async def prediction_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle unexpected prediction errors."""
    logger.error("exceptions.prediction_error_handler: %s", exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal prediction error", "type": type(exc).__name__},
    )


async def validation_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle validation errors with context."""
    logger.warning("exceptions.validation_error_handler: %s", exc)
    return JSONResponse(
        status_code=422,
        content={"detail": str(exc)},
    )


def register_exception_handlers(app: FastAPI) -> None:
    """Attach all custom exception handlers to the application."""
    app.add_exception_handler(ValueError, validation_error_handler)
    logger.info("exceptions.register_exception_handlers: registered")
