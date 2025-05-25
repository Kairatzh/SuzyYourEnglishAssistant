from typing import Dict

import torch
from transformers import BertTokenizer, BertModel
from torch import nn

"""
 Инференс модели для оценки wrinting.
 Модель берет ЭССЕ и оценивает по шкале IELTS(5.5, 6.0, 6.5 и так далее)
"""

"""Используем уже дообученный модель"""
MODEL_PATH = "C:/Users/User/Desktop/SuzyAssistantEnglish/src/models/train_models/ielts_regression_bert"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BertRegressionModel(nn.Module):
    def __init__(self):
        super(BertRegressionModel, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.regressor = nn.Linear(768, 1)
    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        cls_output = outputs.last_hidden_state[:, 0, :]
        prediction = self.regressor(cls_output).squeeze(-1)
        return prediction

"""Загрузка модели и токенизатора."""
model = BertRegressionModel()
model.load_state_dict(torch.load(f"{MODEL_PATH}/pytorch_model.bin", map_location=DEVICE))
model.to(DEVICE)
model.eval()

tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)

"""Предсказывание """
def predict_writing(text: str) -> Dict:
    if not text.strip():
        raise ValueError("Пустой текст")

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    ).to(DEVICE)

    with torch.no_grad():
        output = model(**inputs)

    raw_score = output.item()
    rounded_score = round(raw_score * 2) / 2
    return {
        "raw_score": round(raw_score, 2),
        "rounded_score": rounded_score
    }

