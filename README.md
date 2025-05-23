# SuzyYourEnglishAssistant

**SuzyYourEnglishAssistant** — это персональный ассистент для изучения английского языка с помощью современных моделей искусственного интеллекта. Проект позволяет проверять и исправлять грамматику, переводить тексты, проходить тесты и изучать лексику по уровням CEFR. Веб-интерфейс реализован на Streamlit, а backend — на FastAPI.

## Возможности

- **Проверка и исправление грамматики** (AI-модели HuggingFace)
- **Перевод текста** (русский ⇄ английский, Helsinki-NLP)
- **Тестирование знаний** (автоматическая генерация тестов)
- **Изучение словарного запаса** (выборка слов по уровням CEFR с переводами)
- **Интуитивный web-интерфейс** (Streamlit)

## Быстрый старт

### 1. Клонирование репозитория
```bash
git clone https://github.com/Kairatzh/SuzyYourEnglishAssistant.git
cd SuzyYourEnglishAssistant
```

### 2. Установка зависимостей

Создайте виртуальное окружение и установите зависимости:
```bash
python -m venv venv
source venv/bin/activate  # или venv\Scripts\activate для Windows
pip install -r requirements.txt
```

### 3. Запуск backend (FastAPI)
```bash
cd src/api
uvicorn main:app --reload --host 127.0.0.1 --port 8002
```

### 4. Запуск frontend (Streamlit)
```bash
cd ../..
streamlit run app.py
```

### 5. Переменные окружения

Создайте файл `.env` и укажите адрес backend:
```
BASE_URL=http://127.0.0.1:8002
```

## Используемые технологии и модели

- **Python 3.10+**
- **FastAPI** — backend API
- **Streamlit** — web-интерфейс
- **HuggingFace Transformers**
    - cointegrated/rut5-base-multitask (коррекция грамматики)
    - Helsinki-NLP/opus-mt-ru-en, opus-mt-en-ru (перевод)
- **Pandas, scikit-learn, datasets** (работа с данными и тестами)

## Структура проекта

```
SuzyYourEnglishAssistant/
│
├── app.py                  # Streamlit UI
├── src/
│   ├── api/
│   │   └── main.py         # FastAPI backend
│   └── models/             # Модули для работы с ИИ и данными
│       ├── get_words.py
│       ├── testing.py
│       ├── translator.py
│       └── train_models/
│           ├── grammar_correcter.py
│           ├── grammar_check_bert.py
│           └── evaluate_bert_grammar.py
├── data/                   # Датасеты 
├── requirements.txt
└── README.md
```

## Как внести вклад

1. Форкните репозиторий и создайте новую ветку для ваших изменений.
2. Убедитесь, что ваш код проходит тесты (добавьте свои при необходимости).
3. Создайте Pull Request с описанием изменений.

## TODO и планы развития

- Улучшить обработку ошибок и валидацию входных данных
- Добавить расширенные тесты на Pytest
- Реализовать личный кабинет пользователя и прогресс
- Интеграция с Telegram/Discord
- Расширение языковой поддержки (добавить новые языки)
- Адаптивные тесты и улучшенный UI

## Лицензия

MIT License. См. файл [LICENSE](LICENSE).

## Контакты

Автор: [Kairatzh](https://github.com/Kairatzh)

---

**SuzyYourEnglishAssistant** — твой ИИ-репетитор английского!
