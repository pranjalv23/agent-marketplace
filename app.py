import json
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from jose import JWTError, jwt as _jwt
from pydantic import BaseModel

import httpx

_JWT_SECRET = os.getenv("AUTH_JWT_SECRET", "change-me-in-production")
_JWT_ALGORITHM = "HS256"


def _create_token(user_id: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(days=30)
    return _jwt.encode({"sub": user_id, "exp": expire}, _JWT_SECRET, algorithm=_JWT_ALGORITHM)


def _decode_token(token: str) -> str | None:
    try:
        payload = _jwt.decode(token, _JWT_SECRET, algorithms=[_JWT_ALGORITHM])
        return payload.get("sub")
    except JWTError:
        return None

from config import AGENT_URLS
from router.registry import AgentRegistry
from router.router_agent import EmbeddingRouter
from router.a2a_caller import AgentCaller

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("marketplace.api")

registry = AgentRegistry(AGENT_URLS)
router = EmbeddingRouter()
caller = AgentCaller()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await registry.refresh()
    await router.build_index(registry.get_cards())
    logger.info("Marketplace started — %d agent(s) registered", len(registry.get_cards()))
    yield
    logger.info("Marketplace shutdown")


app = FastAPI(
    title="Agent Marketplace",
    description="Router that discovers specialist agents via A2A and delegates queries.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
async def query(request: QueryRequest):
    """Route a query to the best agent via embedding-based routing."""
    logger.info("POST /query — query='%s'", request.query[:100])

    if not registry.get_cards():
        raise HTTPException(status_code=503, detail="No agents available. Try POST /agents/refresh.")

    decision = await router.route(request.query)

    agent_url = registry.get_url(decision.agent_name)
    if not agent_url:
        raise HTTPException(
            status_code=404,
            detail=f"Agent '{decision.agent_name}' not found. Available: {list(registry.get_cards().keys())}",
        )

    response_text = await caller.call_agent(agent_url, request.query, request.session_id)

    return QueryResponse(
        query=request.query,
        routed_to=decision.agent_name,
        reasoning=decision.reasoning,
        response=response_text,
    )


@app.post("/agents/{agent_id}/query", response_model=DirectQueryResponse)
async def direct_query(agent_id: str, request: DirectQueryRequest):
    """Call a specific agent directly via A2A, bypassing the router."""
    logger.info("POST /agents/%s/query — query='%s'", agent_id, request.query[:100])

    agent_url = registry.get_url(agent_id)
    if not agent_url:
        raise HTTPException(
            status_code=404,
            detail=f"Agent '{agent_id}' not found. Available: {list(AGENT_URLS.keys())}",
        )

    response_text = await caller.call_agent(agent_url, request.query, request.session_id)

    return DirectQueryResponse(
        agent_id=agent_id,
        query=request.query,
        response=response_text,
    )


@app.post("/agents/{agent_id}/query/stream")
async def direct_query_stream(agent_id: str, request: DirectQueryRequest, http_request: Request):
    """Stream a response from a specific agent, bypassing the router."""
    logger.info("POST /agents/%s/query/stream — query='%s'", agent_id, request.query[:100])

    agent_url = registry.get_url(agent_id)
    if not agent_url:
        raise HTTPException(
            status_code=404,
            detail=f"Agent '{agent_id}' not found. Available: {list(AGENT_URLS.keys())}",
        )

    raw = http_request.headers.get("Authorization", "").removeprefix("Bearer ").strip()
    user_id = _decode_token(raw) if raw else None

    async def event_stream():
        async for chunk in caller.stream_agent(agent_url, request.query, request.session_id,
                                               response_format=request.response_format,
                                               model_id=request.model_id,
                                               user_id=user_id):
            yield f"data: {json.dumps({'text': chunk})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/query/stream")
async def query_stream(request: QueryRequest, http_request: Request):
    """Route a query to the best agent and stream the response as SSE.

    Uses the agent's /ask/stream endpoint directly for real-time streaming.
    Each SSE event contains {"text": "..."} with a token chunk.
    The stream ends with a [DONE] sentinel.
    """
    logger.info("POST /query/stream — query='%s'", request.query[:100])

    if not registry.get_cards():
        raise HTTPException(status_code=503, detail="No agents available. Try POST /agents/refresh.")

    decision = await router.route(request.query)

    agent_url = registry.get_url(decision.agent_name)
    if not agent_url:
        raise HTTPException(
            status_code=404,
            detail=f"Agent '{decision.agent_name}' not found. Available: {list(registry.get_cards().keys())}",
        )

    raw = http_request.headers.get("Authorization", "").removeprefix("Bearer ").strip()
    user_id = _decode_token(raw) if raw else None

    async def event_stream():
        # Send routing metadata as the first event
        yield f"data: {json.dumps({'routed_to': decision.agent_name, 'reasoning': decision.reasoning})}\n\n"

        # Proxy the agent's SSE stream
        async for chunk in caller.stream_agent(agent_url, request.query, request.session_id,
                                               response_format=request.response_format,
                                               model_id=request.model_id,
                                               user_id=user_id):
            yield f"data: {json.dumps({'text': chunk})}\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/agents")
async def list_agents():
    """List all registered agents with their Agent Cards."""
    return {"agents": registry.get_cards()}


@app.post("/agents/refresh")
async def refresh_agents():
    """Re-fetch Agent Cards and rebuild the embedding index."""
    await registry.refresh()
    cards = registry.get_cards()
    await router.build_index(cards)
    return {"status": "refreshed", "agents_available": len(cards), "agent_ids": list(cards.keys())}


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


def _parse_error(resp) -> str:
    try:
        return resp.json()
    except Exception:
        return resp.text or f"Agent returned {resp.status_code}"


@app.post("/auth/register")
async def proxy_register(http_request: Request):
    """Proxy registration to the financial agent, then attach a JWT token here."""
    body = await http_request.json()
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(f"{AGENT_URLS['financial-agent']}/auth/register", json=body)
    if resp.status_code >= 400:
        raise HTTPException(status_code=resp.status_code, detail=_parse_error(resp))
    data = resp.json()
    return {**data, "token": _create_token(data["user_id"])}


@app.post("/auth/login")
async def proxy_login(http_request: Request):
    """Proxy login to the financial agent, then attach a JWT token here."""
    body = await http_request.json()
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(f"{AGENT_URLS['financial-agent']}/auth/login", json=body)
    if resp.status_code >= 400:
        raise HTTPException(status_code=resp.status_code, detail=_parse_error(resp))
    data = resp.json()
    return {**data, "token": _create_token(data["user_id"])}


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
    uvicorn.run(app, host="0.0.0.0", port=port)
