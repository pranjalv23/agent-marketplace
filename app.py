import logging
from contextlib import asynccontextmanager

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import os

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

    model_config = {"json_schema_extra": {"examples": [
        {"query": "Analyze RELIANCE.NS stock", "session_id": None},
        {"query": "Find papers on transformer architectures", "session_id": None},
    ]}}


class QueryResponse(BaseModel):
    query: str
    routed_to: str
    reasoning: str
    response: str


class DirectQueryRequest(BaseModel):
    query: str
    session_id: str | None = None


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


@app.get("/health")
async def health():
    return {"status": "ok", "service": "agent-marketplace", "agents": len(registry.get_cards())}


if __name__ == "__main__":
    port = int(os.getenv("PORT", 9000))
    uvicorn.run(app, host="0.0.0.0", port=port)
