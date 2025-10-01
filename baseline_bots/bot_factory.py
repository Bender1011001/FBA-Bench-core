"""
Bot Factory for creating baseline bot instances.
"""

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class BotFactory:
    """
    Factory class for creating baseline bot instances.
    Used by tests and integration scenarios.
    """

    _registered_bots: Dict[str, type] = {}

    @classmethod
    def register_bot(cls, name: str, bot_class: type) -> None:
        """Register a bot class with the factory."""
        cls._registered_bots[name] = bot_class
        logger.debug(f"Registered bot: {name}")

    @classmethod
    def create_bot(cls, bot_type: str, config: Optional[Dict[str, Any]] = None) -> Optional[Any]:
        """
        Create a bot instance by type name.

        Args:
            bot_type: Name of the bot type to create
            config: Optional configuration dict for the bot

        Returns:
            Bot instance or None if bot type not found
        """
        try:
            if bot_type == "greedy_script_bot":
                from baseline_bots.greedy_script_bot import GreedyScriptBot

                return GreedyScriptBot()

            # Check registered bots
            if bot_type in cls._registered_bots:
                bot_class = cls._registered_bots[bot_type]
                if config:
                    return bot_class(config)
                else:
                    return bot_class()

            logger.warning(f"Unknown bot type: {bot_type}")
            return None

        except Exception as e:
            logger.error(f"Failed to create bot {bot_type}: {e}")
            return None

    @classmethod
    def list_available_bots(cls) -> list[str]:
        """Return list of available bot types."""
        available = ["greedy_script_bot"]
        available.extend(cls._registered_bots.keys())
        return list(set(available))  # Remove duplicates

    @classmethod
    def is_bot_available(cls, bot_type: str) -> bool:
        """Check if a bot type is available."""
        return bot_type in cls.list_available_bots()


# Auto-register the greedy script bot
try:
    from baseline_bots.greedy_script_bot import GreedyScriptBot

    BotFactory.register_bot("greedy_script_bot", GreedyScriptBot)
except ImportError:
    logger.warning("Could not auto-register GreedyScriptBot")

# Auto-register the OpenRouter bot
try:
    from baseline_bots.openrouter_bot import OpenRouterBot

    BotFactory.register_bot("openrouter_bot", OpenRouterBot)
except ImportError:
    logger.warning("Could not auto-register OpenRouterBot")
