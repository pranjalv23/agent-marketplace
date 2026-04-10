import asyncio
import hashlib
import json
import logging

import httpx

logger = logging.getLogger("marketplace.registry")


def _validate_card(agent_id: str, card: dict) -> bool:
    """Validate that an agent card has the required fields for routing."""
    for field in ("name", "description"):
        if not card.get(field):
            logger.error(
                "Agent card for '%s' rejected: missing or empty required field '%s'",
                agent_id, field,
            )
            return False
    skills = card.get("skills", [])
    if not isinstance(skills, list):
        logger.error("Agent card for '%s' rejected: 'skills' must be a list", agent_id)
        return False
    for i, skill in enumerate(skills):
        if not isinstance(skill, dict) or not skill.get("name") or not skill.get("description"):
            logger.warning(
                "Agent card for '%s' skill[%d] is missing 'name' or 'description' — skipping skill",
                agent_id, i,
            )
    return True


class AgentRegistry:
    """Fetches and caches Agent Cards from registered A2A agents."""

    def __init__(self, agent_urls: dict[str, str]):
        self._agent_urls = agent_urls
        self._cards: dict[str, dict] = {}
        self._cards_hash: str = ""

    def _hash(self, cards: dict[str, dict]) -> str:
        """Stable MD5 of the full cards dict for change detection."""
        return hashlib.md5(json.dumps(cards, sort_keys=True).encode()).hexdigest()

    async def refresh(self) -> bool:
        """Fetch Agent Cards from all registered agent URLs in parallel.

        Returns:
            True if cards changed since the last refresh, False if unchanged.
        """
        logger.info("Refreshing agent cards from %d agent(s)", len(self._agent_urls))
        new_cards: dict[str, dict] = {}

        async with httpx.AsyncClient(timeout=30.0) as client:
            async def _fetch(agent_id: str, base_url: str) -> tuple[str, dict | None]:
                url = f"{base_url}/a2a/.well-known/agent.json"
                for attempt in range(3):
                    try:
                        response = await client.get(url)
                        response.raise_for_status()
                        card = response.json()
                        logger.info("Fetched card for '%s': %s", agent_id, card.get("name", "unknown"))
                        return agent_id, card
                    except Exception as e:
                        if attempt < 2:
                            logger.warning("Attempt %d/3 failed for '%s': %s — retrying in 5s",
                                           attempt + 1, agent_id, e)
                            await asyncio.sleep(5)
                        else:
                            logger.error("Failed to fetch card for '%s' at %s after 3 attempts: %s",
                                         agent_id, base_url, e)
                return agent_id, None

            results = await asyncio.gather(
                *[_fetch(aid, url) for aid, url in self._agent_urls.items()]
            )

        for agent_id, card in results:
            if card is not None and _validate_card(agent_id, card):
                new_cards[agent_id] = card

        new_hash = self._hash(new_cards)
        if new_hash == self._cards_hash:
            logger.info(
                "Agent cards unchanged (hash=%s) — skipping re-index", new_hash[:8]
            )
            return False

        self._cards = new_cards
        self._cards_hash = new_hash
        logger.info(
            "Registry refreshed — %d/%d agents available (hash=%s)",
            len(self._cards), len(self._agent_urls), new_hash[:8],
        )
        return True

    def get_cards(self) -> dict[str, dict]:
        return dict(self._cards)

    def get_card(self, agent_id: str) -> dict | None:
        """Get a single agent's card by ID."""
        return self._cards.get(agent_id)

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
