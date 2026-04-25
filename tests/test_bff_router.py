import pytest
import respx
import httpx
from fastapi.testclient import TestClient

BFF_SECRET = "test-bff-secret"
KEYCLOAK_TOKEN_URL = "http://keycloak:8180/realms/agent-hub/protocol/openid-connect/token"


@pytest.fixture(autouse=True)
def set_env(monkeypatch):
    monkeypatch.setenv("BFF_SECRET", BFF_SECRET)
    monkeypatch.setenv("KEYCLOAK_URL", "http://keycloak:8180")
    monkeypatch.setenv("KEYCLOAK_REALM", "agent-hub")
    monkeypatch.setenv("BFF_CLIENT_ID", "agent-hub-bff")
    monkeypatch.setenv("BFF_CLIENT_SECRET", "client-secret")


@pytest.fixture
def client():
    from fastapi import FastAPI
    from bff_router import router
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def test_token_exchange_forbidden_without_secret(client):
    resp = client.post("/bff/token-exchange", json={
        "code": "abc", "code_verifier": "xyz",
        "redirect_uri": "https://example.com/auth/callback"
    })
    assert resp.status_code == 403


@respx.mock
def test_token_exchange_success(client):
    respx.post(KEYCLOAK_TOKEN_URL).mock(return_value=httpx.Response(200, json={
        "access_token": "at", "refresh_token": "rt", "expires_in": 3600
    }))
    resp = client.post(
        "/bff/token-exchange",
        json={"code": "abc", "code_verifier": "xyz",
              "redirect_uri": "https://example.com/auth/callback"},
        headers={"X-BFF-Secret": BFF_SECRET},
    )
    assert resp.status_code == 200
    assert resp.json() == {"access_token": "at", "refresh_token": "rt", "expires_in": 3600}


@respx.mock
def test_token_exchange_forwards_keycloak_error(client):
    respx.post(KEYCLOAK_TOKEN_URL).mock(
        return_value=httpx.Response(400, json={"error": "invalid_grant"})
    )
    resp = client.post(
        "/bff/token-exchange",
        json={"code": "bad", "code_verifier": "xyz",
              "redirect_uri": "https://example.com/auth/callback"},
        headers={"X-BFF-Secret": BFF_SECRET},
    )
    assert resp.status_code == 400


def test_token_refresh_forbidden_without_secret(client):
    resp = client.post("/bff/token-refresh", json={"refresh_token": "rt"})
    assert resp.status_code == 403


@respx.mock
def test_token_refresh_success(client):
    respx.post(KEYCLOAK_TOKEN_URL).mock(return_value=httpx.Response(200, json={
        "access_token": "at2", "refresh_token": "rt2", "expires_in": 3600
    }))
    resp = client.post(
        "/bff/token-refresh",
        json={"refresh_token": "old-rt"},
        headers={"X-BFF-Secret": BFF_SECRET},
    )
    assert resp.status_code == 200
    assert resp.json()["access_token"] == "at2"


@respx.mock
def test_token_refresh_returns_401_on_keycloak_failure(client):
    respx.post(KEYCLOAK_TOKEN_URL).mock(
        return_value=httpx.Response(401, json={"error": "invalid_token"})
    )
    resp = client.post(
        "/bff/token-refresh",
        json={"refresh_token": "expired"},
        headers={"X-BFF-Secret": BFF_SECRET},
    )
    assert resp.status_code == 401
