from transformers import pipeline
from datasets import load_dataset
from sklearn.metrics import classification_report, confusion_matrix

"""
    Код чисто для понимание как работает сам модель с помощью метрик.
"""


classifier = pipeline("text-classification", model="grammar_bert_model", device=-1)  # -1 = CPU
def load_eval_data():
    dataset = load_dataset("jhu-clsp/jfleg", split="validation")
    texts, labels = [], []
    for example in dataset:
        if example['sentence'].strip():
            texts.append(example['sentence'].strip())
            labels.append(0)
        for corr in example['corrections']:
            if corr.strip():
                texts.append(corr.strip())
                labels.append(1)
    return texts, labels

def evaluate():
    texts, labels = load_eval_data()
    preds = []
    for text in texts:
        result = classifier(text)[0]['label']
        predicted = 1 if result == 'LABEL_1' else 0
        preds.append(predicted)

    print(f"classification report: {classification_report(labels, preds)}")
    print(f"confucion matrix: {confusion_matrix(labels, preds)}")

if __name__ == "__main__":
    evaluate()
