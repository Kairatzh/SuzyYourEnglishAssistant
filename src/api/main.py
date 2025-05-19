import logging
import os
from typing import List, Dict


import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware


from src.models.testing import get_random_questions, check_answer
from src.models.bert_use_grammar import predict
from src.models.translator import translate
from src.models.get_words import get_words_by_level
from src.models.train_models.grammar_correcter import correct_grammar
from dotenv import load_dotenv


load_dotenv()


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)



app = FastAPI(title="Suzy AI Assistant API", description="API for English learning assistant")



ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:8501").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



class AnswerRequest(BaseModel):
    question_id: int
    answer: str
class TranslateRequest(BaseModel):
    text: str
    language: str
class GrammarCheckRequest(BaseModel):
    text: str
class GrammarCorrectRequest(BaseModel):
    text: str
class WordsRequest(BaseModel):
    level: str
    count: int = 20




@app.get("/get_questions", response_model=List[Dict])
async def get_questions():
    try:
        questions = get_random_questions()
        logger.info("Fetched %d questions", len(questions))
        return questions
    except Exception as e:
        logger.error("Error fetching questions: %s", str(e))
        raise HTTPException(status_code=500, detail="Failed to fetch questions")

@app.post("/check_answer", response_model=Dict)
async def check_user_answer(req: AnswerRequest):
    try:
        result = check_answer(req.question_id, req.answer)
        if "error" in result:
            logger.warning("Question not found: %d", req.question_id)
            raise HTTPException(status_code=404, detail=result["error"])
        logger.info("Checked answer for question %d: %s", req.question_id, result["result"])
        return result
    except Exception as e:
        logger.error("Error checking answer: %s", str(e))
        raise HTTPException(status_code=500, detail="Failed to check answer")





@app.post("/translate", response_model=Dict)
async def translate_text(req: TranslateRequest):
    try:
        if req.language not in ["ru_to_en", "en_to_ru"]:
            logger.warning("Invalid language: %s", req.language)
            raise HTTPException(status_code=400, detail="Invalid language. Use 'ru_to_en' or 'en_to_ru'.")
        translated = translate(req.text, req.language)
        if "error" in translated:
            logger.error("Translation error: %s", translated)
            raise HTTPException(status_code=500, detail=translated)
        logger.info("Translated text: %s -> %s", req.text[:50], translated[:50])
        return {"original_text": req.text, "translated_text": translated}
    except Exception as e:
        logger.error("Error translating text: %s", str(e))
        raise HTTPException(status_code=500, detail="Failed to translate text")





@app.post("/check_grammar", response_model=Dict)
async def check_grammar(req: GrammarCheckRequest):
    try:
        result = predict(req.text)
        logger.info("Grammar checked for text: %s", req.text[:50])
        return {"text": req.text, "result": result}
    except Exception as e:
        logger.error("Error checking grammar: %s", str(e))
        raise HTTPException(status_code=500, detail="Failed to check grammar")

@app.post("/correct_grammar", response_model=Dict)
async def correct_grammar_text(req: GrammarCorrectRequest):
    try:
        corrected = correct_grammar(req.text)
        logger.info("Grammar corrected for text: %s -> %s", req.text[:50], corrected[:50])
        return {"original_text": req.text, "corrected_text": corrected}
    except Exception as e:
        logger.error("Error correcting grammar: %s", str(e))
        raise HTTPException(status_code=500, detail="Failed to correct grammar")





@app.post("/get_words", response_model=List[Dict])
async def get_cefr_words(req: WordsRequest):
    try:
        words = get_words_by_level(req.level, req.count)
        logger.info("Fetched %d words for level %s", len(words), req.level)
        return words
    except Exception as e:
        logger.error("Error fetching words: %s", str(e))
        raise HTTPException(status_code=500, detail="Failed to fetch words")





if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8002)