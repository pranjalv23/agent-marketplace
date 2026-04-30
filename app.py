from agent_sdk.secrets.akv import load_akv_secrets
load_akv_secrets()

import asyncio
import json
import logging
import os
import uuid
from contextlib import asynccontextmanager

import httpx
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.httpsredirect import HTTPSRedirectMiddleware
from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware

_INTERNAL_HEADERS = {"X-Internal-API-Key": os.getenv("INTERNAL_API_KEY")} if os.getenv("INTERNAL_API_KEY") else {}

from config import AGENT_URLS
from bff_router import router as bff_router
from router.registry import AgentRegistry
from router.router_agent import EmbeddingRouter, LowConfidenceError
from router.a2a_caller import AgentCaller
from router.proxy import create_proxy_router
from router.streaming import build_marketplace_sse_stream

load_dotenv()

from agent_sdk.logging import configure_logging
from agent_sdk.llm_services.model_registry import list_models as _sdk_list_models
from agent_sdk.auth import KeycloakJWTMiddleware
configure_logging("marketplace")
logger = logging.getLogger("marketplace.api")


def _get_rate_limit_key(request: Request) -> str:
    user_id = getattr(request.state, "user_id", None)
    if user_id:
        return f"user:{user_id}"
    return get_remote_address(request)


_REDIS_URL = os.getenv("REDIS_URL", "")
limiter = Limiter(
    key_func=_get_rate_limit_key,
    **({"storage_uri": _REDIS_URL} if _REDIS_URL else {}),
)


class _RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


class _SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        return response


registry = AgentRegistry(AGENT_URLS, internal_headers=_INTERNAL_HEADERS)
router = EmbeddingRouter()
caller = AgentCaller()

_proxy_client: httpx.AsyncClient | None = None


_REGISTRY_REFRESH_INTERVAL = int(os.getenv("REGISTRY_REFRESH_INTERVAL", "60"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    from agent_sdk.observability import init_sentry
    init_sentry("marketplace")
    global _proxy_client
    _proxy_client = httpx.AsyncClient(headers=_INTERNAL_HEADERS, timeout=120.0)
    await registry.refresh()
    await router.build_index(registry.get_cards())
    await router.init()
    logger.info("Marketplace started — %d agent(s) registered", len(registry.get_cards()))

    async def _refresh_loop():
        while True:
            await asyncio.sleep(_REGISTRY_REFRESH_INTERVAL)
            try:
                changed = await registry.refresh()
                if changed:
                    await router.build_index(registry.get_cards())
                    await router.clear_routing_cache()
                    logger.info("Periodic refresh: registry changed — rebuilt routing index (%d agents)", len(registry.get_cards()))
            except Exception:
                logger.exception("Periodic registry refresh failed — will retry in %ds", _REGISTRY_REFRESH_INTERVAL)

    refresh_task = asyncio.create_task(_refresh_loop())

    yield

    refresh_task.cancel()
    try:
        await refresh_task
    except asyncio.CancelledError:
        pass

    await router.close()
    await caller.close()
    await _proxy_client.aclose()
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

app.add_middleware(ProxyHeadersMiddleware, trusted_hosts="*")

_raw_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:5173")
_allowed_origins = [o.strip() for o in _raw_origins.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-Internal-API-Key", "X-User-Id", "X-Request-ID"],
)
app.add_middleware(_RequestIDMiddleware)
app.add_middleware(KeycloakJWTMiddleware)
app.add_middleware(_SecurityHeadersMiddleware)

app.include_router(bff_router)
app.include_router(create_proxy_router(registry, lambda: _proxy_client))


# ── Models ──

class QueryRequest(BaseModel):
    query: str
    session_id: str | None = None
    response_format: str | None = None
    model_id: str | None = None
    watchlist_id: str | None = None
    as_of_date: str | None = None


class QueryResponse(BaseModel):
    query: str
    routed_to: str
    reasoning: str
    response: str
    routing_confidence: float | None = None
    low_confidence: bool = False


class DirectQueryRequest(BaseModel):
    query: str
    session_id: str | None = None
    response_format: str | None = None
    model_id: str | None = None
    mode: str | None = None
    watchlist_id: str | None = None
    as_of_date: str | None = None


class DirectQueryResponse(BaseModel):
    agent_id: str
    query: str
    response: str


def _resolve_mode(agent_card: dict | None, agent_name: str, body_mode: str | None = None) -> str | None:
    """Resolve execution mode from body, agent card metadata, or known agent defaults."""
    if body_mode:
        return body_mode
    if agent_card:
        mode = agent_card.get("metadata", {}).get("mode")
        if mode:
            return mode
    if agent_name == "financial-agent":
        return "financial_analyst"
    if agent_name == "research-agent":
        return "researcher"
    return None


# ── Core routing endpoints ──

@app.post("/query", response_model=QueryResponse)
@limiter.limit("30/minute")
async def query(request: Request, body: QueryRequest):
    """Route a query to the best agent via embedding-based routing."""
    logger.info("POST /query — query='%s'", body.query[:100])

    if not registry.get_cards():
        raise HTTPException(status_code=503, detail="No agents available. Try POST /agents/refresh.")

    try:
        decision = await router.route_with_cache(body.query)
    except LowConfidenceError as e:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Your query doesn't clearly match any available agent (best match: '{e.best_agent}', "
                f"confidence: {e.best_score:.2f}). Try rephrasing or be more specific."
            ),
        )

    agent_url = registry.get_url(decision.agent_name)
    if not agent_url:
        raise HTTPException(
            status_code=404,
            detail=f"Agent '{decision.agent_name}' not found. Available: {list(AGENT_URLS.keys())}",
        )

    user_id = request.state.user_id
    agent_card = registry.get_card(decision.agent_name)
    mode = _resolve_mode(agent_card, decision.agent_name)

    response_text = await caller.call_agent(
        agent_url, body.query, body.session_id, user_id=user_id, mode=mode,
        request_id=request.state.request_id,
        watchlist_id=body.watchlist_id, as_of_date=body.as_of_date,
    )

    from agent_sdk.config import settings as sdk_settings
    is_low_confidence = decision.confidence < sdk_settings.min_routing_confidence
    return QueryResponse(
        query=body.query,
        routed_to=decision.agent_name,
        reasoning=decision.reasoning,
        response=response_text,
        routing_confidence=decision.confidence,
        low_confidence=is_low_confidence,
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

    user_id = request.state.user_id
    agent_card = registry.get_card(agent_id)
    mode = _resolve_mode(agent_card, agent_id, body.mode)

    response_text = await caller.call_agent(
        agent_url, body.query, body.session_id, user_id=user_id, mode=mode,
        request_id=request.state.request_id,
        watchlist_id=body.watchlist_id, as_of_date=body.as_of_date,
    )

    return DirectQueryResponse(agent_id=agent_id, query=body.query, response=response_text)


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

    user_id = request.state.user_id
    agent_card = registry.get_card(agent_id)
    supports_streaming = agent_card and agent_card.get("capabilities", {}).get("streaming", False)
    mode = _resolve_mode(agent_card, agent_id, body.mode)
    _request_id = request.state.request_id

    async def _source():
        if supports_streaming:
            async for chunk in caller.stream_agent(
                agent_url, body.query, body.session_id,
                response_format=body.response_format, model_id=body.model_id,
                mode=mode, user_id=user_id, request_id=_request_id,
                watchlist_id=body.watchlist_id, as_of_date=body.as_of_date,
            ):
                yield chunk
        else:
            response_text = await caller.call_agent(
                agent_url, body.query, body.session_id, user_id=user_id,
                mode=mode, request_id=_request_id,
                watchlist_id=body.watchlist_id, as_of_date=body.as_of_date,
            )
            yield response_text

    return StreamingResponse(
        build_marketplace_sse_stream(_source()),
        media_type="text/event-stream",
    )


@app.post("/query/stream")
@limiter.limit("30/minute")
async def query_stream(body: QueryRequest, request: Request):
    """Route a query to the best agent and stream the response as SSE."""
    logger.info("POST /query/stream — query='%s'", body.query[:100])

    if not registry.get_cards():
        raise HTTPException(status_code=503, detail="No agents available. Try POST /agents/refresh.")

    try:
        decision = await router.route_with_cache(body.query)
    except LowConfidenceError as e:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Your query doesn't clearly match any available agent (best match: '{e.best_agent}', "
                f"confidence: {e.best_score:.2f}). Try rephrasing or be more specific."
            ),
        )

    agent_url = registry.get_url(decision.agent_name)
    if not agent_url:
        raise HTTPException(
            status_code=404,
            detail=f"Agent '{decision.agent_name}' not found. Available: {list(registry.get_cards().keys())}",
        )

    user_id = request.state.user_id
    agent_card = registry.get_card(decision.agent_name)
    supports_streaming = agent_card and agent_card.get("capabilities", {}).get("streaming", False)
    mode = _resolve_mode(agent_card, decision.agent_name)
    _request_id = request.state.request_id

    from agent_sdk.config import settings as sdk_settings
    _is_low_confidence = decision.confidence < sdk_settings.min_routing_confidence
    preamble = f"data: {json.dumps({'routed_to': decision.agent_name, 'reasoning': decision.reasoning, 'routing_confidence': decision.confidence, 'low_confidence': _is_low_confidence})}\n\n"

    async def _source():
        if supports_streaming:
            async for chunk in caller.stream_agent(
                agent_url, body.query, body.session_id,
                response_format=body.response_format, model_id=body.model_id,
                mode=mode, user_id=user_id, request_id=_request_id,
                watchlist_id=body.watchlist_id, as_of_date=body.as_of_date,
            ):
                yield chunk
        else:
            response_text = await caller.call_agent(
                agent_url, body.query, body.session_id, user_id=user_id,
                mode=mode, request_id=_request_id,
                watchlist_id=body.watchlist_id, as_of_date=body.as_of_date,
            )
            yield response_text

    return StreamingResponse(
        build_marketplace_sse_stream(_source(), preamble=preamble),
        media_type="text/event-stream",
    )


# ── Agent management endpoints ──

@app.get("/agents/status")
async def agents_status():
    """Fan out /health checks to all registered agents and return their status."""
    async def _check(agent_id: str, base_url: str):
        t0 = asyncio.get_running_loop().time()
        try:
            async with httpx.AsyncClient(timeout=3.0) as c:
                r = await c.get(f"{base_url}/health", headers=_INTERNAL_HEADERS)
            latency_ms = round((asyncio.get_running_loop().time() - t0) * 1000)
            return agent_id, {"status": "ok" if r.is_success else "error", "latencyMs": latency_ms}
        except Exception:
            return agent_id, {"status": "error", "latencyMs": None}

    results = await asyncio.gather(*[_check(aid, url) for aid, url in AGENT_URLS.items()])
    return dict(results)


@app.get("/agents")
async def list_agents():
    """List all registered agents with their Agent Cards."""
    return {"agents": registry.get_cards()}


@app.post("/agents/refresh")
async def refresh_agents():
    """Re-fetch Agent Cards and rebuild the embedding index."""
    changed = await registry.refresh()
    cards = registry.get_cards()
    if changed:
        await router.build_index(cards)
        await router.clear_routing_cache()
        logger.info("Agent registry changed — routing and embedding caches cleared")
    return {
        "status": "refreshed" if changed else "no_change",
        "agents_available": len(cards),
        "agent_ids": list(cards.keys()),
    }


@app.get("/models")
async def get_models():
    """List available LLM models that can be selected from the frontend."""
    return {"models": _sdk_list_models()}


@app.get("/health")
async def health():
    return {"status": "ok", "service": "agent-marketplace", "agents": len(registry.get_cards())}


if __name__ == "__main__":
    port = int(os.getenv("PORT", 9000))
    ssl_certfile = os.getenv("SSL_CERTFILE") or None
    ssl_keyfile = os.getenv("SSL_KEYFILE") or None
    uvicorn.run(app, host="0.0.0.0", port=port, ssl_certfile=ssl_certfile, ssl_keyfile=ssl_keyfile)
