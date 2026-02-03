"""
Agent Factory для динамического создания агентов из определений.
Адаптировано из sgr-agent-core.
"""

import logging
from typing import Type, TypeVar

import httpx
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam

from api.agents.agent_definition import AgentDefinition, LLMConfig
from api.agents.base_agent import BaseAgent
from api.agents.registry import AgentRegistry, ToolRegistry
from loguru import logger

Agent = TypeVar("Agent", bound=BaseAgent)


class AgentFactory:
    """Factory for creating agent instances from definitions.

    Use AgentRegistry and ToolRegistry to look up agent classes by name
    and create instances with the appropriate configuration.
    """

    @classmethod
    def _create_client(cls, llm_config: LLMConfig) -> AsyncOpenAI:
        """Create OpenAI client from configuration.

        Args:
            llm_config: LLM configuration

        Returns:
            Configured AsyncOpenAI client
        """
        client_kwargs = {"base_url": llm_config.base_url, "api_key": llm_config.api_key}
        if llm_config.proxy:
            client_kwargs["http_client"] = httpx.AsyncClient(proxy=llm_config.proxy)

        return AsyncOpenAI(**client_kwargs)

    @classmethod
    async def create(cls, agent_def: AgentDefinition, task_messages: list[ChatCompletionMessageParam]) -> Agent:
        """Create an agent instance from a definition.

        Args:
            agent_def: Agent definition with configuration (classes already resolved)
            task_messages: Task messages in OpenAI ChatCompletionMessageParam format

        Returns:
            Created agent instance

        Raises:
            ValueError: If agent creation fails
        """
        BaseClass: Type[Agent] | None = (
            AgentRegistry.get(agent_def.base_class) if isinstance(agent_def.base_class, str) else agent_def.base_class
        )
        if BaseClass is None:
            error_msg = (
                f"Agent base class '{agent_def.base_class}' not found in registry.\n"
                f"Available base classes: {', '.join([c.__name__ for c in AgentRegistry.list_items()])}\n"
                f"To fix this issue:\n"
                f"  - Check that '{agent_def.base_class}' is spelled correctly in your configuration\n"
                f"  - Ensure the custom agent classes are imported before creating agents "
                f"(otherwise they won't be registered)"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # MCP tools пока не поддерживаются
        # mcp_tools: list = await MCP2ToolConverter.build_tools_from_mcp(agent_def.mcp)

        tools = []
        for tool in agent_def.tools:
            if isinstance(tool, str):
                tool_class = ToolRegistry.get(tool)
                if tool_class is None:
                    error_msg = (
                        f"Tool '{tool}' not found in registry.\nAvailable tools: "
                        f"{', '.join([c.__name__ for c in ToolRegistry.list_items()])}\n"
                        f"  - Ensure the custom tool classes are imported before creating agents "
                        f"(otherwise they won't be registered)"
                    )
                    logger.error(error_msg)
                    raise ValueError(error_msg)
            else:
                tool_class = tool
            tools.append(tool_class)

        try:
            agent = BaseClass(
                task_messages=task_messages,
                def_name=agent_def.name,
                toolkit=tools,
                openai_client=cls._create_client(agent_def.llm),
                agent_config=agent_def,
            )
            logger.info(
                f"Created agent '{agent_def.name}' "
                f"using base class '{BaseClass.__name__}' "
                f"with {len(agent.toolkit)} tools"
            )
            return agent
        except Exception as e:
            logger.error(f"Failed to create agent '{agent_def.name}': {e}", exc_info=True)
            raise ValueError(f"Failed to create agent: {e}") from e
