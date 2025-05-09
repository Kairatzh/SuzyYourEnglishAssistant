"""
    Модуль для получение слов для заучивание.Рандомно выбираем и показываем 20 экземпляров.
    Думаю это нормальная количество за день.CEFR датасет был получен из huggingface.И я благодарен создателю этого шедевро
    датасет.
"""
import pandas as pd
from src.models.translator import translate



df = pd.read_csv("hf://datasets/Alex123321/english_cefr_dataset/unified_dataset.csv")
if "Unnamed: 0" in df.columns:
    df = df.drop(["Unnamed: 0"], axis=1)

# print(df.head())
# print(df.info())
# print(df.describe())
# print(df.shape)

cefr_levels_dataset = df.drop(["ud_word_pos"], axis=1)
cefr_levels_unique = cefr_levels_dataset["ud_word_level"].unique().tolist()

def get_words_by_level(cefr_level: str, n=20, dataset=cefr_levels_dataset):
    dataset["ud_word_level"] = dataset["ud_word_level"].str.strip()
    filtered = dataset[dataset["ud_word_level"] == cefr_level]
    if filtered.empty:
        return {
            "error": f"No words found for CEFR level '{cefr_level}'.",
            # "available_levels": dataset["ud_word_level"].unique().tolist() Включу когда будет нужным в UI
        }
    words = filtered["ud_word"].sample(n=min(n, len(filtered)), random_state=1).tolist()
    translation = [translate(word, "en_to_ru") for word in words]
    return {
        "translations": translation,
        "original_words": words
    }

"""
    Тут лучше бы с помощью модели транслейт перевести все 20к слов потом создать новую колонку и так работать.
    Думаю так было бы быстрее.Так как на ходу не переводим.Потом переделаю так уж и быть
"""