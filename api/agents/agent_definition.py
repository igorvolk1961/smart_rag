"""
Определения конфигурации агентов.
Адаптировано из sgr-agent-core.
"""

import importlib.util
import inspect
import logging
import os
from functools import cached_property
from pathlib import Path
from typing import Any, Self

from pydantic import BaseModel, Field, FilePath, ImportString, computed_field, field_validator, model_validator

logger = logging.getLogger(__name__)


class LLMConfig(BaseModel, extra="allow"):
    api_key: str | None = Field(default=None, description="API key")
    base_url: str = Field(default="https://api.openai.com/v1", description="Base URL")
    model: str = Field(default="gpt-4o-mini", description="Model to use")
    max_tokens: int = Field(default=8000, description="Maximum number of output tokens")
    temperature: float = Field(default=0.4, ge=0.0, le=1.0, description="Generation temperature")
    proxy: str | None = Field(
        default=None, description="Proxy URL (e.g., socks5://127.0.0.1:1081 or http://127.0.0.1:8080)"
    )

    def to_openai_client_kwargs(self) -> dict[str, Any]:
        return self.model_dump(exclude={"api_key", "base_url", "proxy"})


class SearchConfig(BaseModel, extra="allow"):
    tavily_api_key: str | None = Field(default=None, description="Tavily API key")
    tavily_api_base_url: str = Field(default="https://api.tavily.com", description="Tavily API base URL")

    max_searches: int = Field(default=4, ge=0, description="Maximum number of searches")
    max_results: int = Field(default=10, ge=1, description="Maximum number of search results")
    content_limit: int = Field(default=3500, gt=0, description="Content character limit per source")


class PromptsConfig(BaseModel, extra="allow"):
    system_prompt_file: FilePath | None = Field(
        default=None,
        description="Path to system prompt file",
    )
    initial_user_request_file: FilePath | None = Field(
        default=None,
        description="Path to initial user request file",
    )
    clarification_response_file: FilePath | None = Field(
        default=None,
        description="Path to clarification response file",
    )
    system_prompt_str: str | None = None
    initial_user_request_str: str | None = None
    clarification_response_str: str | None = None

    @computed_field
    @cached_property
    def system_prompt(self) -> str:
        if self.system_prompt_str:
            return self.system_prompt_str
        if self.system_prompt_file:
            return self._load_prompt_file(self.system_prompt_file)
        return ""  # Будет использован дефолтный промпт

    @computed_field
    @cached_property
    def initial_user_request(self) -> str:
        if self.initial_user_request_str:
            return self.initial_user_request_str
        if self.initial_user_request_file:
            return self._load_prompt_file(self.initial_user_request_file)
        return ""  # Будет использован дефолтный шаблон

    @computed_field
    @cached_property
    def clarification_response(self) -> str:
        if self.clarification_response_str:
            return self.clarification_response_str
        if self.clarification_response_file:
            return self._load_prompt_file(self.clarification_response_file)
        return ""  # Будет использован дефолтный шаблон

    @staticmethod
    def _load_prompt_file(file_path: str | Path | None) -> str:
        """Load prompt content from a file."""
        if file_path is None:
            return ""
        return Path(file_path).read_text(encoding="utf-8")

    def __repr__(self) -> str:
        return (
            f"PromptsConfig(system_prompt='{self.system_prompt[:100] if self.system_prompt else 'None'}...', "
            f"initial_user_request='{self.initial_user_request[:100] if self.initial_user_request else 'None'}...', "
            f"clarification_response='{self.clarification_response[:100] if self.clarification_response else 'None'}...')"
        )


class ExecutionConfig(BaseModel, extra="allow"):
    """Execution parameters and limits for agents.

    You can add any additional fields as needed.
    """

    max_clarifications: int = Field(default=3, ge=0, description="Maximum number of clarifications")
    max_iterations: int = Field(default=10, gt=0, description="Maximum number of iterations")
    mcp_context_limit: int = Field(default=15000, gt=0, description="Maximum context length from MCP server response")
    max_retries: int = Field(default=3, ge=0, description="Maximum number of LLM call retries")

    logs_dir: str | None = Field(
        default="logs", description="Directory for saving bot logs. Set to None or empty string to disable logging."
    )
    reports_dir: str = Field(default="reports", description="Directory for saving reports")


class MCPConfig(BaseModel, extra="allow"):
    """MCP (Model Context Protocol) configuration.
    
    Упрощенная версия без зависимостей от fastmcp.
    """

    mcpServers: dict[str, Any] = Field(default_factory=dict, description="MCP servers configuration")


class AgentConfig(BaseModel):
    llm: LLMConfig = Field(default_factory=LLMConfig, description="LLM settings")
    search: SearchConfig | None = Field(default=None, description="Search settings")
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig, description="Execution settings")
    prompts: PromptsConfig = Field(default_factory=PromptsConfig, description="Prompts settings")
    mcp: MCPConfig = Field(default_factory=MCPConfig, description="MCP settings")


class AgentDefinition(AgentConfig):
    """Definition of a custom agent.

    Agents can override global settings by providing:
    - llm: dict with keys matching LLMConfig (api_key, base_url, model, etc.)
    - prompts: dict with keys matching PromptsConfig (system_prompt_file, etc.)
    - ExecutionConfig: execution parameters and limits
    - tools: list of tool names to include
    """

    name: str = Field(description="Unique agent name/ID")
    # ToDo: not sure how to type this properly and avoid circular imports
    base_class: type[Any] | ImportString | str = Field(description="Agent class name")
    tools: list[type[Any] | str] = Field(default_factory=list, description="List of tool names to include")

    @field_validator("base_class", mode="before")
    def base_class_import_points_to_file(cls, v: Any) -> Any:
        """Ensure ImportString based base_class points to an existing file to
        catch a FileError and not interpret it as str class_name.

        A dotted path indicates an import string (e.g.,
        dir.agent.MyAgent). We use importlib to automatically search for
        the module in sys.path.
        """
        if isinstance(v, str) and "." in v:
            module_parts = v.split(".")
            if len(module_parts) >= 2:
                # Get module path (everything except the class name)
                module_path = ".".join(module_parts[:-1])
                # Use importlib to find module in sys.path automatically
                spec = importlib.util.find_spec(module_path)
                if spec is None or spec.origin is None:
                    file_path = Path(*module_parts[:-1]).with_suffix(".py")
                    raise FileNotFoundError(
                        f"base_class import '{v}' points to '{file_path}', "
                        f"but the file could not be found in sys.path"
                    )
        return v

    @model_validator(mode="before")
    def default_config_override_validator(cls, data):
        # Упрощенная версия без GlobalConfig - используем дефолтные значения
        if not isinstance(data.get("llm"), BaseModel):
            data["llm"] = data.get("llm", {})
        if not isinstance(data.get("search"), BaseModel):
            data["search"] = data.get("search")
        if not isinstance(data.get("prompts"), BaseModel):
            data["prompts"] = data.get("prompts", {})
        if not isinstance(data.get("execution"), BaseModel):
            data["execution"] = data.get("execution", {})
        if not isinstance(data.get("mcp"), BaseModel):
            data["mcp"] = data.get("mcp", {})
        return data

    @model_validator(mode="after")
    def necessary_fields_validator(self) -> Self:
        if self.llm.api_key is None:
            raise ValueError(f"LLM API key is not provided for agent '{self.name}'")
        # Проверяем search API key только если search настроен
        # (для интернет-поиска проверка выполняется в agent_adapter)
        if self.search and self.search.tavily_api_key is None:
            raise ValueError(f"Search API key is not provided for agent '{self.name}' when search is enabled")
        if not self.tools:
            raise ValueError(f"Tools are not provided for agent '{self.name}'")
        return self

    @field_validator("base_class", mode="after")
    def base_class_is_agent(cls, v: Any) -> type[Any]:
        from api.agents.base_agent import BaseAgent

        if inspect.isclass(v) and not issubclass(v, BaseAgent):
            raise TypeError("Imported base_class must be a subclass of BaseAgent")
        return v

    def __str__(self) -> str:
        base_class_name = self.base_class.__name__ if isinstance(self.base_class, type) else self.base_class
        tool_names = [t.__name__ if isinstance(t, type) else t for t in self.tools]
        return (
            f"AgentDefinition(name='{self.name}', "
            f"base_class={base_class_name}, "
            f"tools={tool_names}, "
            f"execution={self.execution}), "
        )


class Definitions(BaseModel):
    agents: dict[str, AgentDefinition] = Field(
        default_factory=dict, description="Dictionary of agent definitions by name"
    )
