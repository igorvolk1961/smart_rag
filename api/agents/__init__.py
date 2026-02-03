"""
Агенты на базе sgr-agent-core.
"""

# Импортируем все необходимые классы для регистрации
from api.agents.base_agent import BaseAgent
from api.agents.base_tool import BaseTool
from api.agents.sgr_tool_calling_agent import SGRToolCallingAgent
from api.agents.tools import FinalAnswerTool, ReasoningTool, WebSearchTool, RAGTool
from api.agents.agent_factory import AgentFactory
from api.agents.models import AgentContext, AgentStatesEnum, SourceData, SearchResult

__all__ = [
    "BaseAgent",
    "BaseTool",
    "SGRToolCallingAgent",
    "FinalAnswerTool",
    "ReasoningTool",
    "WebSearchTool",
    "RAGTool",
    "AgentFactory",
    "AgentContext",
    "AgentStatesEnum",
    "SourceData",
    "SearchResult",
]
