"""
    Это подготовка данных, точнее создание датасета и сохранение
"""

import os
import pandas as pd
from tqdm import tqdm
from src.models.translator import translate

DATA_PATH = "SuzyAssistantEnglish/src/data/translated_cefr_dataset.csv"
RAW_HF_PATH = "hf://datasets/Alex123321/english_cefr_dataset/unified_dataset.csv"

def load_or_create_dataset():
    if os.path.exists(DATA_PATH):
        return "Уже есть созданный датасет который хранится в data/"

    print("🔄 Загружаю оригинальный CEFR датасет и перевожу...")
    df = pd.read_csv(RAW_HF_PATH)

    #Удаляем ненужные колонки
    if "Unnamed: 0" in df.columns:
        df = df.drop("Unnamed: 0", axis=1)
    if "ud_word_pos" in df.columns:
        df = df.drop(columns=["ud_word_pos"])

    df["ud_word_level"] = df["ud_word_level"].str.strip()

    tqdm.pandas(desc="📚 Перевод слов")
    df["translation"] = df["ud_word"].progress_apply(lambda x: translate(x, "en_to_ru"))

    os.makedirs("data", exist_ok=True)
    df.to_csv(DATA_PATH, index=False)
    print("✅ Переведённый датасет сохранён в", DATA_PATH) #сохраняем

    return df

if __name__ == "__main__":
    load_or_create_dataset()