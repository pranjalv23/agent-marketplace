import logging
import os

from langchain_openai import OpenAIEmbeddings
from pydantic import BaseModel, Field

from agent_sdk.config import settings

logger = logging.getLogger("marketplace.router")


class LowConfidenceError(Exception):
    """Raised when no agent matches the query with sufficient confidence."""
    def __init__(self, best_score: float, best_agent: str):
        self.best_score = best_score
        self.best_agent = best_agent
        super().__init__(
            f"No agent matched with sufficient confidence "
            f"(best: '{best_agent}' @ {best_score:.3f}, threshold: {settings.min_routing_confidence})"
        )


class RoutingDecision(BaseModel):
    """Structured output for routing decisions."""
    agent_name: str = Field(description="The agent_id to route the query to")
    reasoning: str = Field(description="Brief explanation of why this agent was chosen")


class EmbeddingRouter:
    """Routes queries to the best agent using embedding similarity.

    Replaces the LLM-based router — embedding lookup is ~50ms vs ~500ms-1.5s
    for an LLM call, and the result is deterministic.
    """

    def __init__(self):
        self._embeddings = OpenAIEmbeddings(
            base_url=os.environ["AZURE_AI_FOUNDRY_ENDPOINT"],
            api_key=os.environ["AZURE_AI_FOUNDRY_API_KEY"],
            model="text-embedding-3-small",
        )
        self._agent_embeddings: dict[str, list[float]] = {}
        self._agent_descriptions: dict[str, str] = {}

    async def build_index(self, cards: dict[str, dict]) -> None:
        """Compute and cache unit-vector embeddings for all agent descriptions + skills.

        Unit vectors are precomputed so route() only needs a dot product (no norm
        recomputation per query — eliminates O(n) redundant sqrt calls).
        """
        self._agent_embeddings = {}
        self._agent_descriptions = {}

        for agent_id, card in cards.items():
            self._agent_descriptions[agent_id] = self._card_to_text(agent_id, card)

        if not self._agent_descriptions:
            logger.warning("No agent cards to index")
            return

        texts = list(self._agent_descriptions.values())
        agent_ids = list(self._agent_descriptions.keys())
        embeddings = await self._embeddings.aembed_documents(texts)

        for agent_id, embedding in zip(agent_ids, embeddings):
            # Store as unit vector — dot(unit_a, unit_b) == cosine_similarity(a, b)
            norm = sum(x * x for x in embedding) ** 0.5
            self._agent_embeddings[agent_id] = [x / norm for x in embedding] if norm > 0 else embedding

        logger.info("Built embedding index for %d agent(s)", len(self._agent_embeddings))

    async def route(self, query: str) -> RoutingDecision:
        """Embed the query and return the closest agent by cosine similarity."""
        if not self._agent_embeddings:
            raise ValueError("No agents indexed. Call build_index() first.")

        raw_query_embedding = await self._embeddings.aembed_query(query)
        # Normalise query embedding to unit vector for pure dot-product similarity
        qnorm = sum(x * x for x in raw_query_embedding) ** 0.5
        query_embedding = [x / qnorm for x in raw_query_embedding] if qnorm > 0 else raw_query_embedding

        best_agent = None
        best_score = -1.0

        for agent_id, agent_unit_vec in self._agent_embeddings.items():
            # Dot product of two unit vectors equals their cosine similarity
            score = sum(x * y for x, y in zip(query_embedding, agent_unit_vec))
            if score > best_score:
                best_score = score
                best_agent = agent_id

        if best_score < settings.min_routing_confidence:
            logger.warning(
                "Low-confidence routing: best agent='%s', score=%.3f (threshold=%.2f) — rejecting",
                best_agent, best_score, settings.min_routing_confidence,
            )
            raise LowConfidenceError(best_score=best_score, best_agent=best_agent)

        reasoning = (
            f"Matched '{best_agent}' with similarity {best_score:.3f} "
            f"(description: '{self._agent_descriptions[best_agent][:80]}...')"
        )
        logger.info("Routing decision: agent='%s', score=%.3f", best_agent, best_score)
        return RoutingDecision(agent_name=best_agent, reasoning=reasoning)

    @staticmethod
    def _card_to_text(agent_id: str, card: dict) -> str:
        """Flatten an agent card into a single text string for embedding."""
        name = card.get("name", agent_id)
        desc = card.get("description", "")
        skills = card.get("skills", [])
        skill_parts = []
        for s in skills:
            tags = ", ".join(s.get("tags", []))
            skill_parts.append(f"{s.get('name', '')}: {s.get('description', '')} [{tags}]")
        skills_text = "; ".join(skill_parts)
        return f"{name}. {desc}. Skills: {skills_text}"

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)
