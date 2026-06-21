"""Загрузка и сборка промптов для LLM.

Читает шаблоны промптов и few-shot примеры из файлов пакета
(``classifier_bags_few_shots.json``, ``service_facts.txt``, ``classifier_system_prompt.txt``, ``system_prompt.txt``, ``classifier_category_few_shots.json``) и предоставляет
функции для формирования готовых списков сообщений для ассистента.
"""

from __future__ import annotations

import json
from importlib import resources
from string import Template
from typing import Any
import copy


def _read_prompt_file(filename: str) -> str:
    return resources.files(__package__).joinpath(filename).read_text(encoding="utf-8").strip()


SERVICE_FACTS = _read_prompt_file("service_facts.txt")
SYSTEM_PROMPT_TEMPLATE = Template(_read_prompt_file("system_prompt.txt"))
CLASSIFIER_BAGS_FEW_SHOTS = json.loads(_read_prompt_file("classifier_bags_few_shots.json"))
CLASSIFIER_SYSTEM_PROMPT = _read_prompt_file("classifier_system_prompt.txt")
CLASSIFIER_CATEGORY_FEW_SHOTS = json.loads(_read_prompt_file("classifier_category_few_shots.json"))

def build_system_prompt(service_name: str) -> str:
    return SYSTEM_PROMPT_TEMPLATE.safe_substitute(
        service_name=service_name,
        service_facts=SERVICE_FACTS,
    )

def classifier_few_shot_messages() -> list[dict[str, str]]:
    return copy.deepcopy(CLASSIFIER_CATEGORY_FEW_SHOTS)

def answer_bags_few_shot_messages() -> list[dict[str, str]]:
    """Возвращает копию few-shot примеров для багов."""
    return copy.deepcopy(CLASSIFIER_BAGS_FEW_SHOTS)


def build_answer_messages(system_prompt: str, history: list[dict[str, str]], user_message: str) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
    messages.extend(answer_bags_few_shot_messages())
    messages.extend(history)
    messages.append({"role": "user", "content": user_message})
    return messages

def build_classifier_messages(user_message: str) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = [{"role": "system", "content": CLASSIFIER_SYSTEM_PROMPT}]
    messages.extend(classifier_few_shot_messages())
    messages.append({"role": "user", "content": user_message})
    return messages
