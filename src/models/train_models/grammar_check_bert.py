import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch
import os
from datasets import load_dataset

"""
    Дообучение модели BERT на датасете JFLEG.
"""
MODEL_SAVE_PATH = "grammar_bert_model"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#Используем классику.А как иначе хах.Готовим датасет для обучение ой дообучение
def prepare_dataset():
    jfleg = load_dataset("jhu-clsp/jfleg")
    data = []
    for split in ["validation", "test"]:
        for example in jfleg[split]:
            if example['sentence'].strip():
                data.append({
                    "text": example['sentence'].strip(),
                    "label": 0
                })

            for correction in example['corrections']:
                if correction.strip():
                    data.append({
                        "text": correction.strip(),
                        "label": 1
                    })
    #НАконец создаем дотосет хах
    dataset = pd.DataFrame(data)
    dataset = dataset.drop_duplicates(subset=['text'])
    dataset = dataset[dataset['text'].str.len() > 0]
    dataset = dataset.sample(frac=1, random_state=42).reset_index(drop=True) #Мешаем на блендере
    return dataset

#ТокенИзейшен
class GrammarDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item
    def __len__(self):
        return len(self.labels)

def train_model():
    dataset = prepare_dataset()
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    encodings = tokenizer(list(dataset["text"]), truncation=True, padding=True, max_length=128)
    train_dataset = GrammarDataset(encodings, dataset["label"].tolist())

    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
    model.to(DEVICE)

    training_args = TrainingArguments(
        output_dir="results",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        save_strategy="epoch",
        remove_unused_columns=False
    )

    #Дообучаем BERT с кнутом и ждем час пока обучиться этот щенок

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset
    )
    trainer.train()

    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    model.save_pretrained(MODEL_SAVE_PATH) #Сохраняемм
    tokenizer.save_pretrained(MODEL_SAVE_PATH)


train_model()