from transformers import BertForSequenceClassification, BertTokenizer
import torch

"""
    Инференс дообученной модели.Создал отдельный модуль для удобство и инференса.
"""


"""
    Получаем уже обученный модель из проекта.Если не обучили то я настоятельно требую чтобы вы обучили и сохранили 
    модель по этой дороге.Хотя я мог уюрать его из гит игноре.Извиняюсь   
"""

path_to_model = ("C:/Users/User/Desktop/SuzyAssistantEnglish/src/models/train_models/grammar_bert_model")


""" 
    Используем модель как sequence classification. Точнеее классификация текста и берем берт токенизер для этой 
    задачи
"""
model = BertForSequenceClassification.from_pretrained(path_to_model, local_files_only=True)
tokenizer = BertTokenizer.from_pretrained(path_to_model)


def predict(text: str):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    predicted_class = torch.argmax(outputs.logits, dim=1).item()
    if predicted_class == 1: return True #Надо менять описание так ка звучить стремно
    elif predicted_class == 0: return False  # Стрем стрем

#Inference
#Файл для использование моделя BERT который был дообучен на датасете по классификации.
#Используется в главном модуле для предсказывание правильности грамматики.Не путайте с grammar_check_bert
