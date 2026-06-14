# m2_b5-support_assistant_cli
## Описание
Простое CLI-приложение для практики блока 2.5. Помощник отвечает только по теме поиска багов, умеет классифицировать обращения (консультация или проблема) , кешировать FAQ (одинаковые вопросы) в In-memory кэш, делать retry (tenacity) и fallback, логировать диалог и поддерживает базовые команды.
## Структура
bag_assistant_cli/
  cli.py                  # точка входа
  models.py               # все dataclass-ы и type alias-ы
  core/
    assistant.py          # сценарий диалога и команды CLI
    classification.py     # категории
  infrastructure/
    cache.py              # In-memory кэш
    llm.py                # retry (tenacity) + fallback для LLM
  prompts/
    loader.py             # загрузка prompt-файлов
    system_prompt.txt
    classifier_system_prompt.txt
    classifier_few_shots.json
    service_facts.txt
## Запуск
Из папки:

python -m support_assistant_cli
## Прочее
BAG_SERVICE_NAME — имя сервиса, по умолчанию EVA/
Без ключей приложение тоже работает: классификация работает эвристикой, при недоступности обоих провайдеров — эскалация на оператора.
