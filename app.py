import json
import logging
import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone

import bcrypt
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from jose import JWTError, jwt as _jwt
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.httpsredirect import HTTPSRedirectMiddleware

import httpx

_MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
_auth_client: AsyncIOMotorClient | None = None


def _users_collection():
    global _auth_client
    if _auth_client is None:
        _auth_client = AsyncIOMotorClient(_MONGO_URI)
    return _auth_client["agent_auth"]["users"]

_JWT_SECRET = os.getenv("AUTH_JWT_SECRET", "change-me-in-production")
_JWT_ALGORITHM = "HS256"


def _create_access_token(user_id: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(hours=1)
    return _jwt.encode({"sub": user_id, "exp": expire, "type": "access"}, _JWT_SECRET, algorithm=_JWT_ALGORITHM)


def _create_refresh_token(user_id: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(days=7)
    return _jwt.encode({"sub": user_id, "exp": expire, "type": "refresh"}, _JWT_SECRET, algorithm=_JWT_ALGORITHM)


def _decode_token(token: str) -> str | None:
    """Decode an access token. Rejects refresh tokens to prevent type confusion."""
    try:
        payload = _jwt.decode(token, _JWT_SECRET, algorithms=[_JWT_ALGORITHM])
        if payload.get("type") != "access":
            return None
        return payload.get("sub")
    except JWTError:
        return None


def _decode_refresh_token(token: str) -> str | None:
    """Decode a refresh token only."""
    try:
        payload = _jwt.decode(token, _JWT_SECRET, algorithms=[_JWT_ALGORITHM])
        if payload.get("type") != "refresh":
            return None
        return payload.get("sub")
    except JWTError:
        return None

from config import AGENT_URLS
from router.registry import AgentRegistry
from router.router_agent import EmbeddingRouter
from router.a2a_caller import AgentCaller

load_dotenv()

class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        doc = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            doc["exc"] = self.formatException(record.exc_info)
        return json.dumps(doc, ensure_ascii=False)

_handler = logging.StreamHandler()
_handler.setFormatter(_JsonFormatter())
logging.root.setLevel(os.getenv("LOG_LEVEL", "INFO").upper())
logging.root.addHandler(_handler)
logger = logging.getLogger("marketplace.api")
limiter = Limiter(key_func=get_remote_address)

class _RequestIDMiddleware(BaseHTTPMiddleware):
    """Generate or propagate X-Request-ID for every request."""

    async def dispatch(self, request, call_next):
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


registry = AgentRegistry(AGENT_URLS)
router = EmbeddingRouter()
caller = AgentCaller()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await _users_collection().create_index("email", unique=True)
    await registry.refresh()
    await router.build_index(registry.get_cards())
    logger.info("Marketplace started — %d agent(s) registered", len(registry.get_cards()))
    yield
    await caller.close()
    logger.info("Marketplace shutdown")


app = FastAPI(
    title="Agent Marketplace",
    description="Router that discovers specialist agents via A2A and delegates queries.",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

if os.getenv("HTTPS_REDIRECT", "").lower() == "true":
    app.add_middleware(HTTPSRedirectMiddleware)

_raw_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:5173")
_allowed_origins = [o.strip() for o in _raw_origins.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(_RequestIDMiddleware)


class QueryRequest(BaseModel):
    query: str
    session_id: str | None = None
    response_format: str | None = None
    model_id: str | None = None

    model_config = {"json_schema_extra": {"examples": [
        {"query": "Analyze RELIANCE.NS stock", "session_id": None, "response_format": "detailed", "model_id": None},
        {"query": "Find papers on transformer architectures", "session_id": None, "response_format": "summary", "model_id": None},
    ]}}


class QueryResponse(BaseModel):
    query: str
    routed_to: str
    reasoning: str
    response: str


class DirectQueryRequest(BaseModel):
    query: str
    session_id: str | None = None
    response_format: str | None = None
    model_id: str | None = None


class DirectQueryResponse(BaseModel):
    agent_id: str
    query: str
    response: str


@app.post("/query", response_model=QueryResponse)
@limiter.limit("30/minute")
async def query(request: Request, body: QueryRequest):
    """Route a query to the best agent via embedding-based routing."""
    logger.info("POST /query — query='%s'", body.query[:100])

    if not registry.get_cards():
        raise HTTPException(status_code=503, detail="No agents available. Try POST /agents/refresh.")

    decision = await router.route(body.query)

    raw = request.headers.get("Authorization", "").removeprefix("Bearer ").strip()
    user_id = _decode_token(raw) if raw else None

    # Determine mode based on agent name
    mode = None
    if decision.agent_name == "financial-agent":
        mode = "financial_analyst"
    elif decision.agent_name == "research-agent":
        mode = "researcher"

    agent_url = registry.get_url(decision.agent_name)
    if not agent_url:
        raise HTTPException(
            status_code=404,
            detail=f"Agent '{decision.agent_name}' not found. Available: {list(AGENT_URLS.keys())}",
        )

    response_text = await caller.call_agent(
        agent_url, body.query, body.session_id, user_id=user_id, mode=mode,
        request_id=request.state.request_id,
    )

    return QueryResponse(
        query=body.query,
        routed_to=decision.agent_name,
        reasoning=decision.reasoning,
        response=response_text,
    )


@app.post("/agents/{agent_id}/query", response_model=DirectQueryResponse)
@limiter.limit("30/minute")
async def direct_query(agent_id: str, request: Request, body: DirectQueryRequest):
    """Call a specific agent directly via A2A, bypassing the router."""
    logger.info("POST /agents/%s/query — query='%s'", agent_id, body.query[:100])

    agent_url = registry.get_url(agent_id)
    if not agent_url:
        raise HTTPException(
            status_code=404,
            detail=f"Agent '{agent_id}' not found. Available: {list(AGENT_URLS.keys())}",
        )

    raw = request.headers.get("Authorization", "").removeprefix("Bearer ").strip()
    user_id = _decode_token(raw) if raw else None

    # Determine mode based on agent id
    mode = None
    if agent_id == "financial-agent":
        mode = "financial_analyst"
    elif agent_id == "research-agent":
        mode = "researcher"

    response_text = await caller.call_agent(
        agent_url, body.query, body.session_id, user_id=user_id, mode=mode,
        request_id=request.state.request_id,
    )

    return DirectQueryResponse(
        agent_id=agent_id,
        query=body.query,
        response=response_text,
    )


@app.post("/agents/{agent_id}/query/stream")
@limiter.limit("30/minute")
async def direct_query_stream(agent_id: str, body: DirectQueryRequest, request: Request):
    """Stream a response from a specific agent, bypassing the router."""
    logger.info("POST /agents/%s/query/stream — query='%s'", agent_id, body.query[:100])

    agent_url = registry.get_url(agent_id)
    if not agent_url:
        raise HTTPException(
            status_code=404,
            detail=f"Agent '{agent_id}' not found. Available: {list(AGENT_URLS.keys())}",
        )

    raw = request.headers.get("Authorization", "").removeprefix("Bearer ").strip()
    user_id = _decode_token(raw) if raw else None

    _request_id = request.state.request_id
    async def event_stream():
        async for chunk in caller.stream_agent(agent_url, body.query, body.session_id,
                                               response_format=body.response_format,
                                               model_id=body.model_id,
                                               user_id=user_id,
                                               request_id=_request_id):
            yield f"data: {json.dumps({'text': chunk})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/query/stream")
@limiter.limit("30/minute")
async def query_stream(body: QueryRequest, request: Request):
    """Route a query to the best agent and stream the response as SSE.

    Uses the agent's /ask/stream endpoint directly for real-time streaming.
    Each SSE event contains {"text": "..."} with a token chunk.
    The stream ends with a [DONE] sentinel.
    """
    logger.info("POST /query/stream — query='%s'", body.query[:100])

    if not registry.get_cards():
        raise HTTPException(status_code=503, detail="No agents available. Try POST /agents/refresh.")

    decision = await router.route(body.query)

    agent_url = registry.get_url(decision.agent_name)
    if not agent_url:
        raise HTTPException(
            status_code=404,
            detail=f"Agent '{decision.agent_name}' not found. Available: {list(registry.get_cards().keys())}",
        )

    raw = request.headers.get("Authorization", "").removeprefix("Bearer ").strip()
    user_id = _decode_token(raw) if raw else None

    _request_id = request.state.request_id
    async def event_stream():
        # Send routing metadata as the first event
        yield f"data: {json.dumps({'routed_to': decision.agent_name, 'reasoning': decision.reasoning})}\n\n"

        # Proxy the agent's SSE stream
        async for chunk in caller.stream_agent(agent_url, body.query, body.session_id,
                                               response_format=body.response_format,
                                               model_id=body.model_id,
                                               user_id=user_id,
                                               request_id=_request_id):
            yield f"data: {json.dumps({'text': chunk})}\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/agents")
async def list_agents():
    """List all registered agents with their Agent Cards."""
    return {"agents": registry.get_cards()}


@app.post("/agents/refresh")
async def refresh_agents():
    """Re-fetch Agent Cards and rebuild the embedding index.

    Skips re-embedding when agent cards are unchanged (returns status 'no_change').
    """
    changed = await registry.refresh()
    cards = registry.get_cards()
    if changed:
        await router.build_index(cards)
    return {
        "status": "refreshed" if changed else "no_change",
        "agents_available": len(cards),
        "agent_ids": list(cards.keys()),
    }


AVAILABLE_MODELS = [
    {"id": "azure/gpt-4o-mini",      "label": "GPT-4o mini",      "provider": "Azure AI Foundry"},
    {"id": "azure/gpt-4.1-mini",     "label": "GPT-4.1 mini",     "provider": "Azure AI Foundry"},
    {"id": "azure/llama-4-maverick", "label": "Llama 4 Maverick", "provider": "Azure AI Foundry"},
    {
        "id": "azure/gpt-oss-120b",
        "label": "GPT-OSS 120B",
        "provider": "Azure AI Foundry",
        "warning": "Tool calls may fail (Harmony format)",
    },
]


@app.get("/models")
async def list_models():
    """List available LLM models that can be selected from the frontend."""
    return {"models": AVAILABLE_MODELS}


# ── File proxy endpoints (upload / download / list) ──

@app.post("/agents/{agent_id}/upload/{upload_type}")
async def proxy_upload(agent_id: str, upload_type: str, file: UploadFile = File(...), session_id: str = Form(...)):
    """Proxy a file upload to the target agent."""
    agent_url = registry.get_url(agent_id)
    if not agent_url:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found.")

    file_bytes = await file.read()
    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(
            f"{agent_url}/upload/{upload_type}",
            files={"file": (file.filename, file_bytes, file.content_type)},
            data={"session_id": session_id},
        )

    if resp.status_code >= 400:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)
    return resp.json()


@app.get("/agents/{agent_id}/download/{file_id}")
async def proxy_download(agent_id: str, file_id: str):
    """Proxy a file download from the target agent."""
    agent_url = registry.get_url(agent_id)
    if not agent_url:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found.")

    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.get(f"{agent_url}/download/{file_id}")

    if resp.status_code >= 400:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)

    return Response(
        content=resp.content,
        media_type=resp.headers.get("content-type", "application/octet-stream"),
        headers={"Content-Disposition": resp.headers.get("content-disposition", "")},
    )


@app.get("/agents/{agent_id}/files/{session_id}")
async def proxy_list_files(agent_id: str, session_id: str):
    """Proxy a file listing request to the target agent."""
    agent_url = registry.get_url(agent_id)
    if not agent_url:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found.")

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(f"{agent_url}/files/{session_id}")

    if resp.status_code >= 400:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)
    return resp.json()


@app.get("/agents/{agent_id}/charts/{ticker}")
async def proxy_charts(agent_id: str, ticker: str, request: Request):
    """Proxy chart data request to the target agent."""
    agent_url = registry.get_url(agent_id)
    if not agent_url:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found.")
    params = dict(request.query_params)
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(f"{agent_url}/charts/{ticker}", params=params)
    if resp.status_code >= 400:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)
    return resp.json()


def _parse_error(resp) -> str:
    try:
        return resp.json()
    except Exception:
        return resp.text or f"Agent returned {resp.status_code}"


class RegisterRequest(BaseModel):
    email: str
    password: str


class LoginRequest(BaseModel):
    email: str
    password: str


@app.post("/auth/register")
@limiter.limit("10/minute")
async def register(request: Request, body: RegisterRequest):
    col = _users_collection()
    email = body.email.lower().strip()
    if await col.find_one({"email": email}):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Email already registered")
    user_id = uuid.uuid4().hex
    await col.insert_one({
        "user_id": user_id,
        "email": email,
        "password_hash": bcrypt.hashpw(body.password.encode(), bcrypt.gensalt()).decode(),
        "created_at": datetime.now(timezone.utc),
    })
    logger.info("Registered user email='%s'", email)
    return {
        "user_id": user_id,
        "email": email,
        "token": _create_access_token(user_id),
        "refresh_token": _create_refresh_token(user_id),
    }


@app.post("/auth/login")
@limiter.limit("10/minute")
async def login(request: Request, body: LoginRequest):
    col = _users_collection()
    user = await col.find_one({"email": body.email.lower().strip()}, {"_id": 0})
    if not user or not bcrypt.checkpw(body.password.encode(), user["password_hash"].encode()):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid email or password")
    logger.info("Login user='%s'", user["user_id"])
    return {
        "user_id": user["user_id"],
        "email": user["email"],
        "token": _create_access_token(user["user_id"]),
        "refresh_token": _create_refresh_token(user["user_id"]),
    }


class RefreshRequest(BaseModel):
    refresh_token: str


@app.post("/auth/refresh")
@limiter.limit("20/minute")
async def refresh_token_endpoint(request: Request, body: RefreshRequest):
    user_id = _decode_refresh_token(body.refresh_token)
    if not user_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired refresh token")
    logger.info("Token refresh for user='%s'", user_id)
    return {
        "token": _create_access_token(user_id),
        "refresh_token": _create_refresh_token(user_id),
    }


@app.get("/agents/{agent_id}/history/me")
async def proxy_history(agent_id: str, http_request: Request):
    """Verify JWT here, then forward user_id to the agent via X-User-Id header."""
    raw = http_request.headers.get("Authorization", "").removeprefix("Bearer ").strip()
    user_id = _decode_token(raw) if raw else None
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")
    agent_url = registry.get_url(agent_id)
    if not agent_url:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found.")
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(f"{agent_url}/history/user/me",
                                headers={"X-User-Id": user_id})
    if resp.status_code >= 400:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)
    return resp.json()


@app.get("/health")
async def health():
    return {"status": "ok", "service": "agent-marketplace", "agents": len(registry.get_cards())}


if __name__ == "__main__":
    port = int(os.getenv("PORT", 9000))
    ssl_certfile = os.getenv("SSL_CERTFILE") or None
    ssl_keyfile = os.getenv("SSL_KEYFILE") or None
    uvicorn.run(app, host="0.0.0.0", port=port, ssl_certfile=ssl_certfile, ssl_keyfile=ssl_keyfile)
