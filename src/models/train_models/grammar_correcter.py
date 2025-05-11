from transformers import pipeline

"""
    Код для корректировки ошибок в грамматике.Используем уже готовые решение
"""


corrector = pipeline("text2text-generation", model="cointegrated/rut5-base-multitask")
def correct_grammar(text: str) -> str:
    result = corrector("gec: " + text, max_length=128, num_beams=4, clean_up_tokenization_spaces=True)
    return result[0]["generated_text"]
