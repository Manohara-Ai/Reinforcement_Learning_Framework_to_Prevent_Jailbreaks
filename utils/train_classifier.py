import pandas as pd
import torch
import yaml
from torch.utils.data import Dataset
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments

with open("config/distilbert.yaml", "r") as f:
    config = yaml.safe_load(f)

pretrained_path = config["model"]["pretrained_path"]

df = pd.read_csv("datasets/train_classifier.csv")

label2id = {"negative": 0, "positive": 1}
id2label = {0: "negative", 1: "positive"}

texts = df["sentence"].tolist()
labels = [label2id[label] for label in df["sentiment"].tolist()]

tokenizer = DistilBertTokenizerFast.from_pretrained(pretrained_path)
encodings = tokenizer(texts, truncation=True, padding=True)

class SentimentDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

dataset = SentimentDataset(encodings, labels)

model = DistilBertForSequenceClassification.from_pretrained(
    pretrained_path,
    num_labels=2,
    id2label=id2label,
    label2id=label2id
)

train_args = TrainingArguments(
    output_dir=None,
    num_train_epochs=3,
    per_device_train_batch_size=8,
    logging_strategy="no",
    save_strategy="no",
    report_to=[]
)

trainer = Trainer(
    model=model,
    args=train_args,
    train_dataset=dataset,
)

trainer.train()

trainer.save_model("./sentiment_model")
tokenizer.save_pretrained("./sentiment_model")

print("Training complete. Model saved to ./sentiment_model")
