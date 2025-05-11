import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware

from src.models.testing import get_random_questions, check_answer
from src.models.bert_use_grammar import predict
from src.models.translator import translate
from src.models.get_words import get_words_by_level
from src.models.train_models.grammar_correcter import correct_grammar

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#Testing/Рандомно выбирает 30 вопросов(get_random_questions) и проверяет(check_answer)
class AnswerRequest(BaseModel):
    question_id: int
    answer: str
@app.get("/получить-вопросы")
def get_questions():
    return get_random_questions()
@app.post("/проверка-ответ")
def check_user_answer(req: AnswerRequest):
    return check_answer(req.question_id, req.answer)


#Чек грамматики/оценка грамматики
#Потом нужно дообучить и добавить апи для корректирование и генерации правильных ответов.Maybe T5 or GPT3
class GrammarText(BaseModel):
    text: str
@app.post("/проверка-грамотности")
def check_grammar(data: GrammarText):
    bert_classification = predict(data.text)
    if bert_classification == False:
        grammar_check = correct_grammar(data.text)
        return {"Original text": data.text, "Corrected text": grammar_check}
    else: return {"Corrected": "У вас с грамматикой все в порядке"}


#Бэкенд для транслейт текста.(английский-русский, русский-английский)
class TranslateText(BaseModel):
    text: str
    language: str #en to ru OR ru to en.Пока что с этими работает.
@app.post("/перевести-текст")
def translate_text(data: TranslateText):
    if data.language == "en_to_ru": result = translate(data.text, "en_to_ru") #Когда будет доступен и другие языки нам придется сделать функцию вне main.py.
    elif data.language == "ru_to_en": result = translate(data.text, "ru_to_en")
    else: return {"error": "Неверное значение 'language'. Используйте 'en-to-ru' или 'ru-to-en'."}
    return {
        "original_text": data.text,
        "translated_text": result
    }

class GetWords(BaseModel):
    cefr: str
@app.post("/получить-слова")
def get_words(cefr: GetWords):
    result = get_words_by_level(cefr.cefr)
    if "error" in result:
        return result
    return {
        "cefr_level": cefr.cefr,
        "word_count": len(result["original_words"]),
        "original_words": result["original_words"],
        "translations": result["translations"]
    }

if __name__ == "__main__":
    uvicorn.run(app, port=8002, host="127.0.0.1")
