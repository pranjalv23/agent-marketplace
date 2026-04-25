import hmac as _hmac
import os
import httpx
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

router = APIRouter(prefix="/bff", tags=["bff"])

_KEYCLOAK_URL = lambda: os.getenv("KEYCLOAK_URL", "http://keycloak:8180")
_KEYCLOAK_REALM = lambda: os.getenv("KEYCLOAK_REALM", "agent-hub")
_BFF_SECRET = lambda: os.getenv("BFF_SECRET", "")
_CLIENT_ID = lambda: os.getenv("BFF_CLIENT_ID", "agent-hub-bff")
_CLIENT_SECRET = lambda: os.getenv("BFF_CLIENT_SECRET", "")


def _check_secret(request: Request) -> None:
    expected = _BFF_SECRET()
    provided = request.headers.get("X-BFF-Secret", "")
    if not expected or not _hmac.compare_digest(provided, expected):
        raise HTTPException(status_code=403, detail="Forbidden")


def _token_url() -> str:
    return f"{_KEYCLOAK_URL()}/realms/{_KEYCLOAK_REALM()}/protocol/openid-connect/token"


class TokenExchangeRequest(BaseModel):
    code: str
    code_verifier: str
    redirect_uri: str


class TokenRefreshRequest(BaseModel):
    refresh_token: str


@router.post("/token-exchange")
async def token_exchange(body: TokenExchangeRequest, request: Request):
    _check_secret(request)
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            _token_url(),
            data={
                "grant_type": "authorization_code",
                "client_id": _CLIENT_ID(),
                "client_secret": _CLIENT_SECRET(),
                "code": body.code,
                "code_verifier": body.code_verifier,
                "redirect_uri": body.redirect_uri,
            },
        )
    if not resp.is_success:
        raise HTTPException(status_code=400, detail="Token exchange failed")
    data = resp.json()
    result = {
        "access_token": data["access_token"],
        "refresh_token": data["refresh_token"],
        "expires_in": data["expires_in"],
    }
    if "id_token" in data:
        result["id_token"] = data["id_token"]
    return result


@router.post("/token-refresh")
async def token_refresh(body: TokenRefreshRequest, request: Request):
    _check_secret(request)
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            _token_url(),
            data={
                "grant_type": "refresh_token",
                "client_id": _CLIENT_ID(),
                "client_secret": _CLIENT_SECRET(),
                "refresh_token": body.refresh_token,
            },
        )
    if not resp.is_success:
        raise HTTPException(status_code=401, detail="Token refresh failed")
    data = resp.json()
    return {
        "access_token": data["access_token"],
        "refresh_token": data["refresh_token"],
        "expires_in": data["expires_in"],
    }
