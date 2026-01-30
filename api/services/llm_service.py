"""
Сервис для работы с LLM через OpenAI API.
"""

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
    ) -> AssistantResponse:
        """
        Простой вызов LLM без агента (без интернета и базы знаний).
        Строит LLMRequest из AssistantRequest и вызывает generate_response.
        context передаётся из роута (userPost и др.).
        """
        context = context or {}
        openai_config = OpenAIConfig(
            api_key=request.llm_api_key,
            base_url=request.llm_url,
        )
        has_history = request.chat_history and len(request.chat_history) > 0
        if has_history:
            messages = [{"role": m.role, "content": m.content} for m in request.chat_history]
            messages.append({"role": "user", "content": request.current_message})
        elif request.system_prompt:
            system_prompt = request.system_prompt
            user_post = ""
            user_info = context.get("userInfo")
            if user_info and isinstance(user_info, dict):
                user_post = user_info.get("userPost") or ""
            # Замена только {userPost}, без .format(), чтобы не трогать фигурные скобки в JSON
            system_prompt = system_prompt.replace("{userPost}", user_post)
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": request.current_message},
            ]
        else:
            messages = [{"role": "user", "content": request.current_message}]
        extra_params = {"n": request.n} if request.n != 1 else None
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
    ) -> AssistantResponse:
        """
        Вызов с агентом (интернет и/или база знаний).
        Реализация будет добавлена отдельно. context передаётся из роута (userPost и др.).
        """
        context = context or {}
        raise NotImplementedError(
            "agent_call: реализация будет обсуждена отдельно (internet/knowledge_base)"
        )

    async def generate_response(
        self,
        request: LLMRequest
    ) -> AssistantResponse:
        """
        Генерация ответа от LLM.
        
        Args:
            request: Запрос к LLM
            
        Returns:
            Ответ от LLM
            
        Raises:
            Exception: При ошибке обращения к API
        """
        try:
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
                raise ServiceError(
                    error="Ошибка валидации запроса",
                    detail="Поле messages обязательно для вызова LLM.",
                    status_code=400,
                    code="missing_messages",
                )
            completion_params["messages"] = request.messages

            if request.extra_params:
                completion_params.update(request.extra_params)
            
            logger.info(f"Отправка запроса к модели {request.model}")
            
            # Выполняем запрос
            response = client.chat.completions.create(**completion_params)
            
            # Извлекаем ответ
            if response.choices and len(response.choices) > 0:
                choice = response.choices[0]
                content = choice.message.content or ""
                
                # Формируем информацию об использовании токенов
                usage = None
                if response.usage:
                    usage = {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens
                    }
                
                logger.info(f"Получен ответ от модели {request.model}")
                
                return AssistantResponse(
                    content=content,
                    model=response.model,
                    usage=usage,
                    finish_reason=choice.finish_reason
                )
            else:
                raise ServiceError(
                    error="Пустой ответ от провайдера LLM",
                    detail="API вернул ответ без choices.",
                    status_code=502,
                    code="empty_response",
                )

        except openai.AuthenticationError as e:
            logger.error(f"Ошибка аутентификации LLM API: {e}")
            raise ServiceError(
                error="Ошибка аутентификации",
                detail="Неверный или недействительный API-ключ LLM. Проверьте llm_api_key.",
                status_code=401,
                code="llm_auth_error",
            ) from e
        except openai.RateLimitError as e:
            logger.error(f"Превышен лимит запросов LLM: {e}")
            raise ServiceError(
                error="Превышен лимит запросов",
                detail="Провайдер LLM ограничил частоту запросов. Повторите попытку позже.",
                status_code=429,
                code="rate_limit",
            ) from e
        except openai.APIConnectionError as e:
            logger.error(f"Ошибка соединения с LLM API: {e}")
            raise ServiceError(
                error="Ошибка соединения с провайдером",
                detail=f"Не удалось подключиться к LLM API: {e!s}. Проверьте llm_url и доступность сервиса.",
                status_code=503,
                code="connection_error",
            ) from e
        except openai.APITimeoutError as e:
            logger.error(f"Таймаут запроса к LLM: {e}")
            raise ServiceError(
                error="Таймаут запроса",
                detail="Провайдер LLM не ответил вовремя. Повторите запрос.",
                status_code=504,
                code="timeout",
            ) from e
        except openai.BadRequestError as e:
            logger.error(f"Неверный запрос к LLM API: {e}")
            raise ServiceError(
                error="Неверный запрос к LLM",
                detail=f"Провайдер отклонил запрос: {e!s}. Проверьте модель и параметры.",
                status_code=400,
                code="bad_request",
            ) from e
        except openai.OpenAIError as e:
            logger.error(f"Ошибка OpenAI/LLM API: {e}")
            status_code = 502
            if hasattr(e, "response") and e.response is not None:
                status_code = getattr(e.response, "status_code", 502)
            raise ServiceError(
                error="Ошибка провайдера LLM",
                detail=str(e),
                status_code=status_code,
                code="llm_api_error",
            ) from e
        except ServiceError:
            raise
        except Exception as e:
            logger.exception("Неожиданная ошибка при обращении к LLM")
            raise ServiceError(
                error="Внутренняя ошибка сервера",
                detail=str(e),
                status_code=500,
                code="internal_error",
            ) from e


def get_llm_service() -> LLMService:
    """Получение экземпляра сервиса LLM."""
    return LLMService()
