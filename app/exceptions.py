"""Custom exception handlers for Climate-Pulse FastAPI app."""
from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


async def prediction_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle unexpected prediction errors with a structured 500 response.

    Args:
        request: Incoming FastAPI request.
        exc: The raised exception.

    Returns:
        JSON response with status 500 and error details.
    """
    logger.error("exceptions.prediction_error_handler: %s", exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal prediction error", "type": type(exc).__name__},
    )


async def validation_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle validation errors with a structured 422 response.

    Args:
        request: Incoming FastAPI request.
        exc: The raised ValueError (or subclass).

    Returns:
        JSON response with status 422 and the error message.
    """
    logger.warning("exceptions.validation_error_handler: %s", exc)
    return JSONResponse(
        status_code=422,
        content={"detail": str(exc)},
    )


async def key_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle missing-key errors with a structured 400 response.

    Args:
        request: Incoming FastAPI request.
        exc: The raised KeyError.

    Returns:
        JSON response with status 400 and missing key name.
    """
    logger.warning("exceptions.key_error_handler: missing key %s", exc)
    return JSONResponse(
        status_code=400,
        content={"detail": f"Missing required field: {exc}"},
    )


def register_exception_handlers(app: FastAPI) -> None:
    """Attach all custom exception handlers to the application.

    Args:
        app: FastAPI application instance to register handlers on.
    """
    app.add_exception_handler(ValueError, validation_error_handler)
    app.add_exception_handler(KeyError, key_error_handler)
    logger.info("exceptions.register_exception_handlers: registered")
