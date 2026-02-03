"""
Базовый класс для tools агентов.
Адаптировано из sgr-agent-core.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, ClassVar

from pydantic import BaseModel

from api.agents.registry import ToolRegistry

if TYPE_CHECKING:
    from api.agents.agent_definition import AgentConfig
    from api.agents.models import AgentContext


logger = logging.getLogger(__name__)


class ToolRegistryMixin:
    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        if cls.__name__ not in ("BaseTool", "MCPBaseTool"):
            ToolRegistry.register(cls, name=cls.tool_name)


class BaseTool(BaseModel, ToolRegistryMixin):
    """Class to provide tool handling capabilities."""

    tool_name: ClassVar[str] = None
    description: ClassVar[str] = None

    async def __call__(self, context: AgentContext, config: AgentConfig, **kwargs) -> str:
        """The result should be a string or dumped JSON."""
        raise NotImplementedError("Execute method must be implemented by subclass")

    def __init_subclass__(cls, **kwargs) -> None:
        cls.tool_name = cls.tool_name or cls.__name__.lower()
        cls.description = cls.description or cls.__doc__ or ""
        super().__init_subclass__(**kwargs)


class MCPBaseTool(BaseTool):
    """Base model for MCP Tool schema."""

    _client: ClassVar[object | None] = None

    async def __call__(self, context: AgentContext, config: AgentConfig, **kwargs) -> str:
        # MCP support будет добавлен позже при необходимости
        logger.warning(f"MCP tool {self.tool_name} called but MCP not implemented")
        return f"Error: MCP tools not implemented yet"
