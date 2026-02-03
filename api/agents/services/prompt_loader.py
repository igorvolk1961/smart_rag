"""
Загрузчик промптов для агентов.
Адаптировано из sgr-agent-core.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from api.agents.agent_definition import PromptsConfig
    from api.agents.base_tool import BaseTool


# Дефолтный системный промпт из sgr-agent-core для агентов с поиском
DEFAULT_AGENT_SYSTEM_PROMPT = """<MAIN_TASK_GUIDELINES>
You are an expert researcher with adaptive planning and schema-guided-reasoning capabilities. You get the research task and you neeed to do research and genrete answer
</MAIN_TASK_GUIDELINES>

<DATE_GUIDELINES>
PAY ATTENTION TO THE DATE INSIDE THE USER REQUEST
DATE FORMAT: YYYY-MM-DD HH:MM:SS (ISO 8601)
IMPORTANT: The date above is in YYYY-MM-DD format (Year-Month-Day). For example, 2025-10-03 means October 3rd, 2025, NOT March 10th.
</DATE_GUIDELINES>

<IMPORTANT_LANGUAGE_GUIDELINES>: Detect the language from user request and use this LANGUAGE for all responses, searches, and result finalanswertool
LANGUAGE ADAPTATION: Always respond and create reports in the SAME LANGUAGE as the user's request.
If user writes in Russian - respond in Russian, if in English - respond in English.
</IMPORTANT_LANGUAGE_GUIDELINES>:

<CORE_PRINCIPLES>:
1. Memorize plan you generated in first step and follow the task inside your plan.
1. Adapt plan when new data contradicts initial assumptions
2. Search queries in SAME LANGUAGE as user request
3. Final Answer ENTIRELY in SAME LANGUAGE as user request
</CORE_PRINCIPLES>

<REASONING_GUIDELINES>
ADAPTIVITY: Actively change plan when discovering new data.
ANALYSIS EXTRACT DATA: Always analyze data that you took in extractpagecontenttool
</REASONING_GUIDELINES>

<PRECISION_GUIDELINES>
CRITICAL FOR FACTUAL ACCURACY:
When answering questions about specific dates, numbers, versions, or names:
1. EXACT VALUES: Extract the EXACT value from sources (day, month, year for dates; precise numbers for quantities)
2. VERIFY YEAR: If question mentions a specific year (e.g., "in 2022"), verify extracted content is about that SAME year
3. CROSS-VERIFICATION: When sources provide contradictory information, prefer:
   - Official sources and primary documentation over secondary sources
   - Search result snippets that DIRECTLY answer the question over extracted page content
   - Multiple independent sources confirming the same fact
4. DATE PRECISION: Pay special attention to exact dates - day matters (October 21 ≠ October 22)
5. NUMBER PRECISION: For numbers/versions, exact match required (6.88b ≠ 6.88c, Episode 31 ≠ Episode 32)
6. SNIPPET PRIORITY: If search snippet clearly states the answer, trust it unless extract proves it wrong
7. TEMPORAL VALIDATION: When extracting page content, check if the page shows data for correct time period
</PRECISION_GUIDELINES>

<AGENT_TOOL_USAGE_GUIDELINES>:
{available_tools}
</AGENT_TOOL_USAGE_GUIDELINES>"""


class PromptLoader:
    """Загрузчик промптов для агентов."""

    @staticmethod
    def get_system_prompt(toolkit: list[type], prompts_config: "PromptsConfig") -> str:
        """
        Получить системный промпт для агента.
        Использует системный промпт из sgr-agent-core по умолчанию.
        """
        # Если задан кастомный промпт, используем его
        if prompts_config.system_prompt_str:
            system_prompt = prompts_config.system_prompt_str
            # Добавляем информацию о доступных tools, если есть placeholder
            if "{available_tools}" in system_prompt:
                tool_descriptions = "\n".join([
                    f"{i}. {tool.tool_name}: {tool.description or ''}" 
                    for i, tool in enumerate(toolkit, start=1)
                ])
                system_prompt = system_prompt.format(available_tools=tool_descriptions)
            return system_prompt
        
        # Используем дефолтный системный промпт из sgr-agent-core
        tool_descriptions = "\n".join([
            f"{i}. {tool.tool_name}: {tool.description or ''}" 
            for i, tool in enumerate(toolkit, start=1)
        ])
        return DEFAULT_AGENT_SYSTEM_PROMPT.format(available_tools=tool_descriptions)

    @staticmethod
    def get_initial_user_request(task_messages: list, prompts_config: "PromptsConfig") -> str:
        """Получить начальный запрос пользователя."""
        if prompts_config.initial_user_request_str:
            return prompts_config.initial_user_request_str
        
        # Извлекаем последнее сообщение пользователя
        if task_messages:
            last_message = task_messages[-1]
            if isinstance(last_message, dict) and last_message.get("role") == "user":
                return last_message.get("content", "")
        
        return "Please help me with my request."

    @staticmethod
    def get_clarification_template(messages: list, prompts_config: "PromptsConfig") -> str:
        """Получить шаблон для запроса уточнений."""
        if prompts_config.clarification_response_str:
            return prompts_config.clarification_response_str
        
        return "The user has provided clarification. Please continue with the task."
