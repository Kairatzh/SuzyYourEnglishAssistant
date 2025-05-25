import pandas as pd
import torch
from transformers import BertTokenizer, BertModel, Trainer, TrainingArguments
from torch import nn
import os

""" Дообучение базового берта на датасете с эссе - Ielts(5.5), Использовал датасет из гитхаба"""
MODEL_SAVE_PATH = "ielts_regression_bert"  #Путь к сохранению модели.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") #Определение девайса

""" Уже фильтрованный датасет для обучение """
dataset = pd.read_csv("C:/Users/User/Desktop/SuzyAssistantEnglish/src/data/ielts_writing_dataset.csv")
dataset = dataset[["Essay", "Overall"]]
labels = dataset["Overall"].values

""" Токенизации модели БЕРТ"""
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
encodings = tokenizer(list(dataset["Essay"]), truncation=True, padding=True, max_length=128)

""" Создание датасета удобного для Берта"""
class RegressionDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item
    def __len__(self):
        return len(self.labels)

train_dataset = RegressionDataset(encodings, labels)

""" Модель регрессии """
class BertRegressionModel(nn.Module):
    def __init__(self):
        super(BertRegressionModel, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.regressor = nn.Linear(768, 1)
    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        cls_output = outputs.last_hidden_state[:, 0, :]
        prediction = self.regressor(cls_output).squeeze(-1)
        loss = None
        if labels is not None:
            loss_fn = nn.MSELoss()
            loss = loss_fn(prediction, labels)
        return {"loss": loss, "logits": prediction}

""" Fine-tuning модели БЕРТ на уже готовом датасете"""
model = BertRegressionModel().to(DEVICE)

training_args = TrainingArguments(
    output_dir="./results_regression",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    warmup_steps=100,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    save_strategy="epoch",
    remove_unused_columns=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

trainer.train()

""" Сохранение дообученной модели на папке train_models/"""
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
#Сохранение state_dict и токенизатора
torch.save(model.state_dict(), os.path.join(MODEL_SAVE_PATH, "pytorch_model.bin"))
tokenizer.save_pretrained(MODEL_SAVE_PATH)
tokenizer.save_pretrained(MODEL_SAVE_PATH)
