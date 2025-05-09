import json
import random
from typing import List, Dict

"""
    Тестинг человека.Пока что в json есть только 30 вопросов.Думаю увеличить его до 150 чтобы рандомные слова попадались каждый разю
    В будущем думаю над изменением на проверки IELTS/writing(С помощью предобученной модели.Попробую найти или просто дообучу BERT),
    listening(text to speech и наоборот.Использую современные методы) и так далее.
    В основном код открывает размешевает и проверяет.
"""

#Загружаем все 150 вопросов из файла test_english.json.Так как повторяющиеся тесты это классика разве нет?
try:
    with open('C:/Users/User/Desktop/SuzyAssistantEnglish/src/data/test_data.json', 'r', encoding='utf-8') as f:
        all_questions = json.load(f)
except FileNotFoundError:
    print("Файл не найден. Проверьте путь.")
    all_questions = []  # или загрузи дефолтные вопросы
except json.JSONDecodeError:
    print("Ошибка при парсинге JSON.")
    all_questions = []

#Возвращает случайные 30 вопросов (без ответов).Рандомы сила
def get_random_questions(n: int = 30) -> List[Dict]:
    selected = random.sample(all_questions, n)
    return [
        {
            "id": q["id"],
            "question": q["question"],
            "options": q["options"]
        }
        for q in selected
    ]

#Проверка на ге... короче проверка
questions_dict = {q['id']: q for q in all_questions}
def check_answer(question_id: int, user_answer: str) -> Dict:
    user_answer = user_answer.lower().strip()
    question = questions_dict.get(question_id)
    if question:
        correct = question["correct"].lower()
        is_correct = user_answer == correct
        return {
            "result": "Correct" if is_correct else "Incorrect",
            "is_correct": is_correct,
            "correct_answer": correct,
            "explanation_ru": question["explanation_ru"]
        }
    return {"error": "Вопрос не найден."}