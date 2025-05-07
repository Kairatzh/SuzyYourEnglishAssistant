from transformers import BertForSequenceClassification, BertTokenizer
import torch

"""
    Инференс дообученной модели.Создал отдельный модуль для удобство и инференса.
"""

path_to_model = "C:/Users/User/Desktop/SuzyAssistantEnglish/src/models/train_models/grammar_bert_model"

model = BertForSequenceClassification.from_pretrained(path_to_model, local_files_only=True)
tokenizer = BertTokenizer.from_pretrained(path_to_model)

def predict(text: str):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    predicted_class = torch.argmax(outputs.logits, dim=1).item()
    if predicted_class == 1: return "С грамматикой все в порядке" #Надо менять описание так ка звучить стремно
    elif predicted_class == 0: return "Есть ошибки в грамматик.Исправьте"  # Стрем стрем

#Inference
#Файл для использование моделя BERT который был дообучен на датасете по классификации.
#Используется в главном модуле для предсказывание правильности грамматики.Не путайте с grammar_check_bert