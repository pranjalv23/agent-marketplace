import asyncio
import logging

import httpx

logger = logging.getLogger("marketplace.registry")


class AgentRegistry:
    """Fetches and caches Agent Cards from registered A2A agents."""

    def __init__(self, agent_urls: dict[str, str]):
        self._agent_urls = agent_urls
        self._cards: dict[str, dict] = {}

    async def refresh(self):
        """Fetch Agent Cards from all registered agent URLs in parallel."""
        logger.info("Refreshing agent cards from %d agent(s)", len(self._agent_urls))
        self._cards = {}

        async with httpx.AsyncClient(timeout=10.0) as client:
            async def _fetch(agent_id: str, base_url: str) -> tuple[str, dict | None]:
                try:
                    url = f"{base_url}/.well-known/agent.json"
                    response = await client.get(url)
                    response.raise_for_status()
                    card = response.json()
                    logger.info("Fetched card for '%s': %s", agent_id, card.get("name", "unknown"))
                    return agent_id, card
                except Exception as e:
                    logger.error("Failed to fetch card for '%s' at %s: %s", agent_id, base_url, e)
                    return agent_id, None

            results = await asyncio.gather(
                *[_fetch(aid, url) for aid, url in self._agent_urls.items()]
            )

        for agent_id, card in results:
            if card is not None:
                self._cards[agent_id] = card

        logger.info("Registry refreshed — %d/%d agents available",
                     len(self._cards), len(self._agent_urls))

    def get_cards(self) -> dict[str, dict]:
        return dict(self._cards)

    def get_routing_context(self) -> str:
        """Format all agent descriptions + skills as text for the router LLM."""
        if not self._cards:
            return "No agents available."

        lines = []
        for agent_id, card in self._cards.items():
            name = card.get("name", agent_id)
            desc = card.get("description", "No description")
            skills = card.get("skills", [])
            skill_strs = []
            for s in skills:
                tags = ", ".join(s.get("tags", []))
                skill_strs.append(f"  - {s.get('name', 'unknown')}: {s.get('description', '')} [tags: {tags}]")
            skills_block = "\n".join(skill_strs) if skill_strs else "  (no skills listed)"
            lines.append(f"Agent ID: {agent_id}\nName: {name}\nDescription: {desc}\nSkills:\n{skills_block}")

        return "\n\n".join(lines)

    def get_url(self, agent_id: str) -> str | None:
        """Get the base URL for an agent."""
        return self._agent_urls.get(agent_id)
