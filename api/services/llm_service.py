"""
Сервис для работы с LLM через OpenAI API.
"""

from typing import Dict, Optional, Any
from functools import lru_cache
from loguru import logger
import openai
from openai import OpenAI

from api.models.llm_models import OpenAIConfig, LLMRequest, LLMResponse


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
    
    async def generate_response(
        self,
        request: LLMRequest
    ) -> LLMResponse:
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
            
            # Формируем сообщения
            messages = []
            if request.message:
                messages.append({
                    "role": "user",
                    "content": request.message
                })
            
            if messages:
                completion_params["messages"] = messages
            
            # Добавляем дополнительные параметры если есть
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
                
                return LLMResponse(
                    content=content,
                    model=response.model,
                    usage=usage,
                    finish_reason=choice.finish_reason
                )
            else:
                raise ValueError("Пустой ответ от API")
                
        except openai.OpenAIError as e:
            logger.error(f"Ошибка OpenAI API: {e}")
            raise Exception(f"Ошибка при обращении к OpenAI API: {str(e)}")
        except Exception as e:
            logger.error(f"Неожиданная ошибка: {e}")
            raise


def get_llm_service() -> LLMService:
    """Получение экземпляра сервиса LLM."""
    return LLMService()
