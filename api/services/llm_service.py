"""
Сервис для работы с LLM через OpenAI API.
"""

import json
import re
from typing import Dict, Optional, Any
from functools import lru_cache
from loguru import logger
import openai
from openai import OpenAI

from api.models.llm_models import (
    OpenAIConfig,
    LLMRequest,
    AssistantRequest,
    AssistantResponse,
)
from api.exceptions import ServiceError
from api.services.agent_adapter import create_agent_definition_from_request
from api.agents.agent_factory import AgentFactory
from api.agents.models import AgentStatesEnum
import json
import re


class ConfigCache:
    """Кэш для хранения конфигураций OpenAI API."""
    
    def __init__(self):
        self._cache: Dict[str, OpenAI] = {}
        self._configs: Dict[str, OpenAIConfig] = {}
    
    def _get_cache_key(self, config: OpenAIConfig) -> str:
        """Генерация ключа кэша на основе конфигурации."""
        # Используем api_key как основу ключа (первые 10 символов для безопасности)
        key_prefix = config.api_key[:10] if len(config.api_key) > 10 else config.api_key
        base_url = config.base_url or "https://api.openai.com/v1"
        return f"{key_prefix}_{base_url}"
    
    def get_client(self, config: OpenAIConfig) -> OpenAI:
        """
        Получение клиента OpenAI из кэша или создание нового.
        
        Args:
            config: Конфигурация OpenAI API
            
        Returns:
            Клиент OpenAI
        """
        cache_key = self._get_cache_key(config)
        
        if cache_key not in self._cache:
            logger.info(f"Создание нового клиента OpenAI для ключа: {cache_key[:20]}...")
            
            client_kwargs = {
                "api_key": config.api_key,
            }
            
            if config.base_url:
                client_kwargs["base_url"] = config.base_url
            
            if config.organization:
                client_kwargs["organization"] = config.organization
            
            if config.timeout:
                client_kwargs["timeout"] = config.timeout
            
            if config.max_retries is not None:
                client_kwargs["max_retries"] = config.max_retries
            
            client = OpenAI(**client_kwargs)
            self._cache[cache_key] = client
            self._configs[cache_key] = config
            
            logger.info(f"Клиент OpenAI создан и закэширован")
        else:
            logger.debug(f"Использование закэшированного клиента OpenAI")
        
        return self._cache[cache_key]
    
    def clear_cache(self) -> None:
        """Очистка кэша клиентов."""
        logger.info("Очистка кэша клиентов OpenAI")
        self._cache.clear()
        self._configs.clear()
    
    def get_cache_size(self) -> int:
        """Получение размера кэша."""
        return len(self._cache)


# Глобальный экземпляр кэша
_config_cache = ConfigCache()


class LLMService:
    """Сервис для работы с LLM."""
    
    def __init__(self):
        self.cache = _config_cache

    async def simple_llm_call(
        self, request: AssistantRequest, context: Optional[Dict[str, Any]] = None
    ) -> dict[str, Any]:
        """
        Простой вызов LLM без агента (без интернета и базы знаний).
        Строит LLMRequest из AssistantRequest и вызывает generate_response.
        context передаётся из роута (userPost и др.).
        Возвращает словарь с полем content или error.
        """
        context = context or {}
        openai_config = OpenAIConfig(
            api_key=request.llm_api_key,
            base_url=request.llm_url,
        )
        chat_messages = context.get("chat_messages")
        has_history = chat_messages and len(chat_messages) > 0
        if has_history:
            messages = [{"role": m["role"], "content": m["content"]} for m in chat_messages]
            messages.append({"role": "user", "content": request.current_message})
        elif request.system_prompt:
            system_prompt = request.system_prompt
            user_post = ""
            user_info = context.get("userInfo")
            if user_info and isinstance(user_info, dict):
                user_post = user_info.get("userPost") or ""
            # Замена только {userPost}, без .format(), чтобы не трогать фигурные скобки в JSON
            system_prompt = system_prompt.replace("{userPost}", f"Моя должность - {user_post}.")
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": request.current_message},
            ]
        else:
            messages = [{"role": "user", "content": request.current_message}]
        extra_params = {"n": request.n} if request.n != 1 else None
        context["messages_sent"] = messages
        llm_request = LLMRequest(
            openai_config=openai_config,
            messages=messages,
            model=request.llm_model_name,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            extra_params=extra_params,
        )
        return await self.generate_response(llm_request)

    async def agent_call(
        self, request: AssistantRequest, context: Optional[Dict[str, Any]] = None
    ) -> dict[str, Any]:
        """
        Вызов с агентом через sgr-agent-core (интернет и/или база знаний).
        Streaming используется только внутри агента для передачи данных между фазами.
        Наружу возвращается финальный результат после завершения всех потоков агента.
        Агент работает в изолированном контексте - получает только текущее сообщение пользователя,
        без истории чата. История чата сохраняется после получения ответа от агента.
        context передаётся из роута (userPost и др.).
        Возвращает словарь с полем content или error.
        """
        context = context or {}
        
        try:
            # Создаем AgentDefinition из request
            agent_def = create_agent_definition_from_request(request, context)
            
            # Агент работает в изолированном контексте - передаем только текущее сообщение
            task_messages = [{"role": "user", "content": request.current_message}]
            
            # Сохраняем только текущее сообщение для последующего сохранения истории
            context["messages_sent"] = task_messages
            
            # Создаем агента
            agent = await AgentFactory.create(agent_def, task_messages)
            
            # Передаем параметры в custom_context агента для использования в RAGTool
            custom_context = {}
            if request.file_irv_ids:
                custom_context["file_irv_ids"] = request.file_irv_ids
            if request.vdb_url:
                custom_context["vdb_url"] = request.vdb_url
            # Параметры эмбеддингов для RAG поиска
            if request.embed_api_key:
                custom_context["embed_api_key"] = request.embed_api_key
            if request.embed_url:
                custom_context["embed_url"] = request.embed_url
            if request.embed_model_name:
                custom_context["embed_model_name"] = request.embed_model_name
            
            if custom_context:
                agent._context.custom_context = custom_context
            
            # Запускаем агента и ждем завершения выполнения
            # Streaming используется только внутри агента для передачи данных между reasoning и action фазами
            execution_result = await agent.execute()
            
            # Проверяем состояние агента
            if agent._context.state == AgentStatesEnum.COMPLETED:
                # Агент успешно завершил работу
                final_answer = agent._context.execution_result or execution_result or ""
                
                # Извлекаем chat_title и chat_summary из финального ответа, если возможно
                # (можно попробовать распарсить структурированный ответ, как в generate_response)
                chat_title = None
                chat_summary = None
                
                # Пытаемся извлечь структурированные данные из ответа
                try:
                    text = final_answer.strip()
                    json_block = re.match(r"^```json\s*\n?(.*)\n?```\s*$", text, re.DOTALL | re.IGNORECASE)
                    if json_block:
                        parsed = json.loads(json_block.group(1).strip())
                    elif text.startswith("{"):
                        parsed = json.loads(text)
                    else:
                        parsed = None
                    
                    if isinstance(parsed, dict):
                        chat_title = parsed.get("chat_title")
                        chat_summary = parsed.get("chat_summary")
                        # Если есть answer, используем его вместо полного ответа
                        answer = parsed.get("answer")
                        if isinstance(answer, str) and answer.strip():
                            final_answer = answer
                except (json.JSONDecodeError, TypeError, KeyError):
                    pass
                
                result = {"content": final_answer}
                if chat_title:
                    result["chat_title"] = chat_title
                if chat_summary:
                    result["chat_summary"] = chat_summary
                
                return result
            elif agent._context.state == AgentStatesEnum.FAILED:
                return {
                    "error": "Агент завершил работу с ошибкой",
                    "detail": agent._context.execution_result or "Неизвестная ошибка",
                    "code": "agent_failed",
                }
            else:
                return {
                    "error": "Агент не завершил работу",
                    "detail": f"Состояние агента: {agent._context.state}",
                    "code": "agent_incomplete",
                }
            
        except ValueError as e:
            logger.error(f"Ошибка создания агента: {e}")
            return {
                "error": "Ошибка создания агента",
                "detail": str(e),
                "code": "agent_creation_error",
            }
        except RuntimeError as e:
            logger.error(f"Ошибка выполнения агента: {e}")
            return {
                "error": "Ошибка выполнения агента",
                "detail": str(e),
                "code": "agent_execution_error",
            }
        except Exception as e:
            logger.exception("Ошибка при вызове агента")
            return {
                "error": "Ошибка при вызове агента",
                "detail": str(e),
                "code": "agent_error",
            }

    async def generate_response(
        self,
        request: LLMRequest
    ) -> dict[str, Any]:
        """
        Генерация ответа от LLM с повторными попытками при отсутствии answer в структурированном ответе.
        
        Args:
            request: Запрос к LLM
            
        Returns:
            Ответ от LLM
            
        Raises:
            Exception: При ошибке обращения к API или если answer не найден после всех попыток
        """
        # Получаем клиент из кэша
        client = self.cache.get_client(request.openai_config)
        
        # Подготавливаем параметры запроса
        completion_params = {
            "model": request.model,
            "temperature": request.temperature,
        }
        
        if request.max_tokens:
            completion_params["max_tokens"] = request.max_tokens

        if not request.messages:
            return {
                "error": "Ошибка валидации запроса",
                "detail": "Поле messages обязательно для вызова LLM.",
                "code": "missing_messages",
            }
        completion_params["messages"] = request.messages

        if request.extra_params:
            completion_params.update(request.extra_params)
        
        max_retry_count = request.max_retry_count or 3
        last_error: Optional[Exception] = None
        
        # Повторяем запрос до max_retry_count раз, пока не найдём answer
        for attempt in range(max_retry_count):
            try:
                logger.info(f"Отправка запроса к модели {request.model} (попытка {attempt + 1}/{max_retry_count})")
                
                # Выполняем запрос
                response = client.chat.completions.create(**completion_params)
                
                # Извлекаем ответ
                if not response.choices or len(response.choices) == 0:
                    if attempt < max_retry_count - 1:
                        logger.warning(f"Пустой ответ от провайдера LLM (попытка {attempt + 1}/{max_retry_count})")
                        last_error = None
                        continue
                    else:
                        return {
                            "error": "Пустой ответ от провайдера LLM",
                            "detail": "API вернул ответ без choices.",
                            "code": "empty_response",
                        }
                
                choice = response.choices[0]
                content = choice.message.content or ""
                chat_title: Optional[str] = None
                chat_summary: Optional[str] = None
                content_text: Optional[str] = None
                
                # Парсим структурированный ответ
                try:
                    text = content.strip()
                    json_block = re.match(r"^```json\s*\n?(.*)\n?```\s*$", text, re.DOTALL | re.IGNORECASE)
                    if json_block:
                        parsed = json.loads(json_block.group(1).strip())
                    elif text.startswith("{"):
                        parsed = json.loads(text)
                    else:
                        parsed = None
                    
                    if isinstance(parsed, dict):
                        # Извлекаем метаданные диалога
                        chat_title = parsed.get("chat_title") if isinstance(parsed.get("chat_title"), str) else None
                        chat_summary = parsed.get("chat_summary") if isinstance(parsed.get("chat_summary"), str) else None
                        # Извлекаем только answer для сохранения в контекст (без reasoning)
                        answer = parsed.get("answer")
                        if isinstance(answer, str) and answer.strip():
                            content_text = answer
                except (json.JSONDecodeError, TypeError, KeyError) as e:
                    logger.debug(f"Ошибка парсинга JSON ответа: {e}")
                
                # Если answer найден, возвращаем ответ
                if content_text:
                    usage = None
                    if response.usage:
                        usage = {
                            "prompt_tokens": response.usage.prompt_tokens,
                            "completion_tokens": response.usage.completion_tokens,
                            "total_tokens": response.usage.total_tokens
                        }
                    
                    logger.info(f"Получен ответ от модели {request.model} с answer (попытка {attempt + 1})")
                    
                    result = {"content": content_text}
                    if chat_title:
                        result["chat_title"] = chat_title
                    if chat_summary:
                        result["chat_summary"] = chat_summary
                    return result
                
                # Если answer не найден и это не последняя попытка, продолжаем цикл
                if attempt < max_retry_count - 1:
                    logger.warning(f"Ответ не содержит поле answer, повторная попытка ({attempt + 1}/{max_retry_count})")
                    last_error = None
                    continue
                else:
                    # Последняя попытка - сохраняем ошибку
                    last_error = ServiceError(
                        error="Ответ LLM не содержит обязательное поле answer",
                        detail=f"После {max_retry_count} попыток не удалось получить поле answer в структурированном ответе. Полученный ответ: {content[:200]}",
                        status_code=502,
                        code="missing_answer_field",
                    )
                    break
                    
            except ServiceError as e:
                # ServiceError возвращаем как словарь с error (не повторяем)
                return {
                    "error": e.error,
                    "detail": e.detail,
                    "code": e.code,
                }
            except openai.AuthenticationError as e:
                # Ошибки аутентификации не повторяем
                logger.error(f"Ошибка аутентификации LLM API: {e}")
                return {
                    "error": "Ошибка аутентификации",
                    "detail": "Неверный или недействительный API-ключ LLM. Проверьте llm_api_key.",
                    "code": "llm_auth_error",
                }
            except openai.RateLimitError as e:
                # Ошибки лимита не повторяем
                logger.error(f"Превышен лимит запросов LLM: {e}")
                return {
                    "error": "Превышен лимит запросов",
                    "detail": "Провайдер LLM ограничил частоту запросов. Повторите попытку позже.",
                    "code": "rate_limit",
                }
            except openai.BadRequestError as e:
                # Ошибки валидации запроса не повторяем
                logger.error(f"Неверный запрос к LLM API: {e}")
                return {
                    "error": "Неверный запрос к LLM",
                    "detail": f"Провайдер отклонил запрос: {e!s}. Проверьте модель и параметры.",
                    "code": "bad_request",
                }
            except (openai.APIConnectionError, openai.APITimeoutError, openai.OpenAIError) as e:
                # Ошибки соединения/таймаута/общие ошибки LLM - повторяем, если есть попытки
                last_error = e
                if attempt < max_retry_count - 1:
                    logger.warning(f"Ошибка при запросе к LLM (попытка {attempt + 1}/{max_retry_count}): {e}")
                    continue
                else:
                    # Последняя попытка - возвращаем словарь с error
                    if isinstance(e, openai.APIConnectionError):
                        return {
                            "error": "Ошибка соединения с провайдером",
                            "detail": f"Не удалось подключиться к LLM API: {e!s}. Проверьте llm_url и доступность сервиса.",
                            "code": "connection_error",
                        }
                    elif isinstance(e, openai.APITimeoutError):
                        return {
                            "error": "Таймаут запроса",
                            "detail": "Провайдер LLM не ответил вовремя. Повторите запрос.",
                            "code": "timeout",
                        }
                    else:
                        return {
                            "error": "Ошибка провайдера LLM",
                            "detail": str(e),
                            "code": "llm_api_error",
                        }
            except Exception as e:
                # Другие ошибки - повторяем, если есть попытки
                last_error = e
                if attempt < max_retry_count - 1:
                    logger.warning(f"Неожиданная ошибка при запросе к LLM (попытка {attempt + 1}/{max_retry_count}): {e}")
                    continue
                else:
                    logger.exception("Неожиданная ошибка при обращении к LLM")
                    return {
                        "error": "Внутренняя ошибка сервера",
                        "detail": str(e),
                        "code": "internal_error",
                    }
        
        # Если дошли сюда, значит все попытки исчерпаны
        if last_error:
            if isinstance(last_error, ServiceError):
                return {
                    "error": last_error.error,
                    "detail": last_error.detail,
                    "code": last_error.code,
                }
            return {
                "error": "Не удалось получить ответ от LLM",
                "detail": f"После {max_retry_count} попыток не удалось получить корректный ответ: {str(last_error)}",
                "code": "llm_retry_exhausted",
            }
        
        # Если last_error None, но answer не найден - это не должно произойти, но на всякий случай
        return {
            "error": "Ответ LLM не содержит обязательное поле answer",
            "detail": f"После {max_retry_count} попыток не удалось получить поле answer в структурированном ответе.",
            "code": "missing_answer_field",
        }


def get_llm_service() -> LLMService:
    """Получение экземпляра сервиса LLM."""
    return LLMService()
