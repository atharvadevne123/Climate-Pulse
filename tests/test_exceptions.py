"""Tests for custom exception handlers."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.exceptions import key_error_handler, register_exception_handlers, validation_error_handler


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
        request = MagicMock()
        exc = ValueError("specific error message")
        response = await validation_error_handler(request, exc)
        body = json.loads(response.body)
        assert "detail" in body

    @pytest.mark.asyncio
    async def test_error_message_preserved(self):
        request = MagicMock()
        exc = ValueError("custom message here")
        response = await validation_error_handler(request, exc)
        body = json.loads(response.body)
        assert "custom message here" in body["detail"]


class TestKeyErrorHandler:
    @pytest.mark.asyncio
    async def test_returns_400_status(self):
        request = MagicMock()
        exc = KeyError("missing_field")
        response = await key_error_handler(request, exc)
        assert response.status_code == 400

    @pytest.mark.asyncio
    async def test_detail_in_body(self):
        request = MagicMock()
        exc = KeyError("my_key")
        response = await key_error_handler(request, exc)
        body = json.loads(response.body)
        assert "detail" in body

    def test_key_error_handler_registered(self):
        app = FastAPI()
        register_exception_handlers(app)

        # If KeyError handler registered, raising KeyError in a route triggers a 400
        @app.get("/raise-key-error")
        async def raise_key_error():
            raise KeyError("test_key")

        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/raise-key-error")
        assert resp.status_code == 400
