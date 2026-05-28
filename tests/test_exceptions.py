"""Tests for custom exception handlers."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.exceptions import register_exception_handlers, validation_error_handler


class TestRegisterExceptionHandlers:
    def test_registers_without_error(self):
        app = FastAPI()
        register_exception_handlers(app)

    def test_registered_handlers_respond(self):
        app = FastAPI()
        register_exception_handlers(app)

        @app.get("/test-error")
        async def raise_value_error():
            raise ValueError("bad input")

        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/test-error")
        assert resp.status_code == 422

    def test_registered_handler_returns_json(self):
        app = FastAPI()
        register_exception_handlers(app)

        @app.get("/test-error")
        async def raise_value_error():
            raise ValueError("test error")

        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/test-error")
        assert resp.headers.get("content-type", "").startswith("application/json")


class TestValidationErrorHandler:
    @pytest.mark.asyncio
    async def test_returns_422_json_response(self):
        request = MagicMock()
        exc = ValueError("invalid value")
        response = await validation_error_handler(request, exc)
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_error_detail_in_response(self):
        import json
        request = MagicMock()
        exc = ValueError("specific error message")
        response = await validation_error_handler(request, exc)
        body = json.loads(response.body)
        assert "detail" in body
