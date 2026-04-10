# NOTE (AR-1 resolved): Streaming now uses A2A message/stream instead of
# direct /ask/stream SSE. The marketplace checks agent capabilities and falls
# back to message/send (non-streaming) if the agent doesn't support streaming.

import asyncio
import json
import logging
import uuid
from collections.abc import AsyncIterator

import httpx

import os

from agent_sdk.config import settings

logger = logging.getLogger("marketplace.a2a_caller")


class AgentCaller:
    """Sends tasks to agents via the A2A protocol and supports SSE streaming."""

    def __init__(self):
        # Re-use connection pool for latency benefits
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=10.0))

    async def close(self):
        await self._client.aclose()

    async def call_agent(self, agent_url: str, query: str, session_id: str | None = None,
                   user_id: str | None = None, mode: str | None = None,
                   request_id: str | None = None, watchlist_id: str | None = None,
                   as_of_date: str | None = None) -> str:
        """
        Send a message/send request to an A2A agent and return the response text.

        Args:
            agent_url: Base URL of the A2A agent (e.g., http://localhost:9001)
            query: The user's query text
            session_id: Optional session ID for conversation continuity
            user_id: Optional user ID for personalization
            mode: Optional agent execution mode
            request_id: Optional request correlation ID (forwarded as X-Request-ID)
        """
        task_id = uuid.uuid4().hex
        session_id = session_id or uuid.uuid4().hex

        # A2A message/send JSON-RPC request
        payload = {
            "jsonrpc": "2.0",
            "id": task_id,
            "method": "message/send",
            "params": {
                "id": task_id,
                "sessionId": session_id,
                "message": {
                    "messageId": uuid.uuid4().hex,
                    "role": "user",
                    "parts": [{"type": "text", "text": query}],
                },
                "metadata": {
                    "user_id": user_id,
                    "mode": mode,
                    "watchlist_id": watchlist_id,
                    "as_of_date": as_of_date,
                },
                "acceptedOutputModes": ["text"],
            },
        }

        a2a_endpoint = f"{agent_url}/a2a/"
        headers: dict[str, str] = {}
        if request_id:
            headers["X-Request-ID"] = request_id
        if internal_key := os.getenv("INTERNAL_API_KEY"):
            headers["X-Internal-API-Key"] = internal_key
        logger.info("Calling A2A agent at %s — task_id='%s'", a2a_endpoint, task_id)

        _MAX_RETRIES = settings.a2a_max_retries
        data = None
        for attempt in range(_MAX_RETRIES):
            try:
                response = await self._client.post(a2a_endpoint, json=payload, headers=headers)
                if response.status_code == 429:
                    # Special handling for rate limits: use the Retry-After header if present
                    retry_after = response.headers.get("Retry-After")
                    backoff = int(retry_after) if retry_after and retry_after.isdigit() else 2 ** attempt
                    logger.warning(
                        "A2A call rate limited (attempt %d/%d) — retrying in %ds",
                        attempt + 1, _MAX_RETRIES, backoff,
                    )
                    await asyncio.sleep(backoff)
                    continue

                response.raise_for_status()
                data = response.json()
                break
            except (httpx.TransportError, httpx.TimeoutException) as e:
                if attempt == _MAX_RETRIES - 1:
                    raise
                backoff = 2 ** attempt
                logger.warning(
                    "A2A call failed (attempt %d/%d): %s — retrying in %ds",
                    attempt + 1, _MAX_RETRIES, e, backoff,
                )
                await asyncio.sleep(backoff)
            except httpx.HTTPStatusError as e:
                if e.response.status_code < 500 and e.response.status_code != 429:
                    raise
                if attempt == _MAX_RETRIES - 1:
                    raise
                backoff = 2 ** attempt
                logger.warning(
                    "A2A call HTTP %d (attempt %d/%d) — retrying in %ds",
                    e.response.status_code, attempt + 1, _MAX_RETRIES, backoff,
                )
                await asyncio.sleep(backoff)

        # Extract the response text from A2A result
        result = data.get("result", {})
        artifacts = result.get("artifacts", [])

        texts = []
        for artifact in artifacts:
            for part in artifact.get("parts", []):
                if part.get("type") == "text":
                    texts.append(part["text"])

        # Fallback: check history
        if not texts:
            history = result.get("history", [])
            for msg in reversed(history):
                if msg.get("role") == "agent":
                    for part in msg.get("parts", []):
                        if part.get("type") == "text":
                            texts.append(part["text"])
                    break

        response_text = "\n".join(texts) if texts else "No response from agent."
        logger.info("A2A call complete — response length: %d chars", len(response_text))
        return response_text

    async def stream_agent(self, agent_url: str, query: str, session_id: str | None = None,
                           response_format: str | None = None,
                           model_id: str | None = None,
                           user_id: str | None = None,
                           request_id: str | None = None,
                           watchlist_id: str | None = None,
                           as_of_date: str | None = None) -> AsyncIterator[str]:
        """
        Call an agent via the A2A protocol's message/stream method.
        """
        session_id = session_id or uuid.uuid4().hex
        a2a_endpoint = f"{agent_url}/a2a/"

        payload = {
            "jsonrpc": "2.0",
            "id": uuid.uuid4().hex,
            "method": "message/stream",
            "params": {
                "id": uuid.uuid4().hex,
                "sessionId": session_id,
                "message": {
                    "messageId": uuid.uuid4().hex,
                    "role": "user",
                    "parts": [{"type": "text", "text": query}],
                },
                "metadata": {
                    "user_id": user_id,
                    "model_id": model_id,
                    "watchlist_id": watchlist_id,
                    "as_of_date": as_of_date,
                },
                "acceptedOutputModes": ["text"],
            },
        }

        headers: dict[str, str] = {}
        if request_id:
            headers["X-Request-ID"] = request_id
        if internal_key := os.getenv("INTERNAL_API_KEY"):
            headers["X-Internal-API-Key"] = internal_key

        logger.info("Streaming via A2A from %s — session='%s'", a2a_endpoint, session_id)

        async with self._client.stream("POST", a2a_endpoint, json=payload, headers=headers) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:].strip()
                if data == "[DONE]":
                    return
                try:
                    parsed = json.loads(data)
                    # A2A streaming typically wraps chunks in a 'text' field or similar
                    if isinstance(parsed, dict) and "text" in parsed:
                        yield parsed["text"]
                    elif isinstance(parsed, str):
                        yield parsed
                except json.JSONDecodeError:
                    yield data
