"""
    Модуль для получение слов для заучивание.Рандомно выбираем и показываем 20 экземпляров.
    Думаю это нормальная количество за день.CEFR датасет был получен из huggingface.И я благодарен создателю этого шедевро
    датасет.
"""
"""
Модуль для получения слов для заучивания. Рандомно выбираем и показываем 20 слов.
Датасет CEFR получен с Hugging Face. Переводы сохраняются в CSV и переиспользуются.
"""

import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "translated_cefr_dataset.csv")

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Файл {DATA_PATH} не найден. Убедись, что ты уже перевёл и сохранил датасет.")

cefr_df = pd.read_csv(DATA_PATH)
cefr_df["ud_word_level"] = cefr_df["ud_word_level"].str.strip()

def get_words_by_level(cefr_level: str, n=20, dataset=cefr_df):
    filtered = dataset[dataset["ud_word_level"].str.upper() == cefr_level.upper()]
    if filtered.empty:
        return {"error": f"No words found for CEFR level '{cefr_level}'."}

    sampled = filtered.sample(n=min(n, len(filtered)), random_state=1)
    return [
        {
            "english": row.ud_word,
            "russian": row.translation,
            "level": row.ud_word_level
        }
        for _, row in sampled.iterrows()
    ]


"""
    Тут лучше бы с помощью модели транслейт перевести все 20к слов потом создать новую колонку и так работать.
    Думаю так было бы быстрее.Так как на ходу не переводим.Потом переделаю так уж и быть
"""