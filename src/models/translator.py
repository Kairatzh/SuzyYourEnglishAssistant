from transformers import pipeline

"""
    Перевод текста.Есть два режима.Первый это с английского на русский а второй наоборот.
    Использовал уже готовую модель Helsinki в huggingface.
    функция translate сперва определяет режим потом переводят.Добавил еще обработку ошибок.
"""

translator_ru_to_en = pipeline("translation_ru_to_en", model="Helsinki-NLP/opus-mt-ru-en")
translator_en_to_ru = pipeline("translation_en_to_ru", model="Helsinki-NLP/opus-mt-en-ru")

def translate(text: str, category: str) -> str:
    try:
        if category == "ru_to_en": result = translator_ru_to_en(text)
        elif category == "en_to_ru": result = translator_en_to_ru(text)
        else: raise ValueError("category must be 'ru_to_en' or 'en_to_ru'")
        return result[0]["translation_text"]
    except Exception as e: return f"Translation error: {str(e)}"