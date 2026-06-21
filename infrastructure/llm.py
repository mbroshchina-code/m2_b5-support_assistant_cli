"""Клиент LLM с retry (tenacity) и fallback.

Все провайдеры работают через OpenAI-совместимый API.
Цепочка: primary → (если faq: заглушка) -> (если problem: primary → openrouter → fallback → заглушка)
"""

from __future__ import annotations

from collections.abc import Iterator
import os
from loguru import logger
from openai import OpenAI, RateLimitError, APIStatusError,  DefaultHttpxClient
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
import httpx
from m2_b5.config import Settings
from m2_b5.models import Category, LLMResult
from m2_b5.core.classification import heuristic_classify


# Ответ-заглушка, когда ни один провайдер не дал полезный ответ
FALLBACK_ANSWER = "Извините, что-то пошло не по плану. Пожалуйста, повторите попытку или напишите чуть позже."
FAQ_ANSWER = "Запрос не относится к проблемам. Найдите ответ в базе знаний"

def _build_client(api_key: str | None, base_url: str | None) -> OpenAI | None:
    """Создаёт чистый стандартный OpenAI клиент без скрытых перезаписей base_url."""
    if not api_key:
        return None
    return OpenAI(api_key=api_key, base_url=base_url)


class RobustLLMClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.primary = _build_client(settings.api_key, settings.base_url)
        self.openrouter = _build_client(settings.openrouter_api_key, settings.openrouter_base_url)
        self.fallback = _build_client(settings.fallback_api_key, settings.fallback_base_url)

    # ── Цепочка провайдеров ───────────────────────────────────────────

    def _provider_chain(self) -> Iterator[tuple[OpenAI, str, bool]]:
        """Отдаёт (client, model, name, used_fallback) для каждого доступного провайдера."""
        # 1. Сначала пробуем основную модель через прокси
        if self.primary is not None:
            yield self.primary, self.settings.primary_model, "primary", False
        # 2. Если упала — идем в OpenRouter через интернет
        if self.openrouter is not None and self.settings.openrouter_model:
            yield self.openrouter, self.settings.openrouter_model, "openrouter", True
        # 3. Если и OpenRouter лег — задействуем локальную Ollama    
        if self.fallback is not None and self.settings.fallback_model:
            yield self.fallback, self.settings.fallback_model, "fallback", True

    # ── Публичные методы ──────────────────────────────────────────────

    def classify(self, messages: list[dict[str, str]]) -> Category:
        """Классифицирует запрос: primary → openrouter →fallback → эвристика."""
        
        for client, model, name, used_fallback in self._provider_chain():
            try:
                raw = self._call(client, model, messages, temperature=0, max_tokens=8)
                return Category(raw.strip().lower())
            except Exception:
                logger.warning(f"Провайдер классификации {name} ({model}) недоступен.")
                continue

        return heuristic_classify(messages[-1]["content"])

    def answer(self, messages: list[dict[str, str]]) -> LLMResult:
        """Получает ответ: primary → openrouter → fallback → заглушка."""
        # 1. Сначала проверяем категорию запроса
        try:
            category = self.classify(messages)
            logger.info("Определена категория запроса: %s", category)
            
            # Если это FAQ — мгновенно отдаем заглушку без обращения к генеративной модели
            if category == "faq" or (isinstance(category, Category) and category.value == "faq"):
                return LLMResult(
                    text=FAQ_ANSWER, 
                    tokens=0, 
                    source="faq_stub", 
                    model="none", 
                    used_fallback=False
                )
        except Exception as e:
            logger.error("Ошибка при классификации запроса, продолжаем в стандартном режиме: %s", e)

        # 2. Если это "problem" или классификатор упал, идем по цепочке моделей
        for client, model, name, used_fallback in self._provider_chain():
            try:
                if used_fallback:
                    logger.info(f"Переключаюсь на резервный канал ({name}): {model}")
                text, tokens = self._answer_from(client, model, messages)
                return LLMResult(
                    text, tokens,
                    name,  # Динамически подставит "primary", "openrouter" или "fallback" в метаданные ответа
                    model, used_fallback,
                )
            except Exception as e:
                logger.warning(f"Провайдер {name} ({model}) недоступен: {e}")

        # Все провайдеры недоступны — даем заглушку
        return LLMResult(FALLBACK_ANSWER, 0, "none", "none", True)

    # ── Внутренние методы ─────────────────────────────────────────────

    def _answer_from(
        self, client: OpenAI, model: str, messages: list[dict[str, str]],
    ) -> tuple[str, int]:
        """Один ответ от провайдера. Возвращает (текст, токены)."""
        text = self._call(client, model, messages)
        return (text or FALLBACK_ANSWER), 0

    def _call(
        self,
        client: OpenAI,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 250,
    ) -> str:
        """Вызов LLM с retry через tenacity (экспоненциальная задержка)."""

        @retry(
            wait=wait_exponential(multiplier=1, min=1, max=60),
            stop=stop_after_attempt(self.settings.retry_attempts),
            retry=retry_if_exception_type((RateLimitError, APIStatusError)),
        )
        def _do() -> str:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=self.settings.request_timeout_seconds,
            )
            return (response.choices[0].message.content or "").strip()

        return _do()