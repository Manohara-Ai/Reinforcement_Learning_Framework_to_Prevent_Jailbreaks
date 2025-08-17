import os
import yaml
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import joblib
import spacy

def load_config(path):
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    cfg["training"]["batch_size"] = int(cfg["training"]["batch_size"])
    cfg["training"]["epochs"] = int(cfg["training"]["epochs"])
    cfg["training"]["learning_rate"] = float(cfg["training"]["learning_rate"])
    cfg["training"]["max_len"] = int(cfg["training"]["max_len"])
    return cfg

class MultiTaskDataset(Dataset):
    def __init__(self, df, tokenizer, max_len, nlp):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.nlp = nlp

    def __len__(self):
        return len(self.df)

    def get_verb_phrases(self, text):
        doc = self.nlp(text)
        verb_phrases = []

        def get_full_phrase(token):
            included_deps = (
                "dobj", "prep", "pobj", "advmod", "attr", "acomp", "xcomp",
                "compound", "nummod", "amod"
            )
            phrase_tokens = [token.text]
            for child in token.children:
                if child.dep_ in included_deps:
                    phrase_tokens.append(get_full_phrase(child))
            return " ".join(phrase_tokens)

        for token in doc:
            if token.pos_ == "VERB":
                subjects = [child.text for child in token.children if child.dep_ in ("nsubj", "nsubjpass")]
                verb_phrase = " ".join(subjects + [get_full_phrase(token)])
                if verb_phrase.strip():
                    verb_phrases.append(verb_phrase)

        return verb_phrases

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        verb_phrases = self.get_verb_phrases(row["prompt"])
        text = " ".join(verb_phrases) if verb_phrases else row["prompt"]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "category": torch.tensor(row["category"], dtype=torch.long),
            "base_class": torch.tensor(row["base_class"], dtype=torch.long),
        }

class DistilBertMultiTask(nn.Module):
    def __init__(self, model_path, num_cat, num_base):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained(model_path)
        hidden_size = self.bert.config.hidden_size
        self.dropout = nn.Dropout(0.3)
        self.cat_head = nn.Linear(hidden_size, num_cat)
        self.base_head = nn.Linear(hidden_size, num_base)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0]
        pooled = self.dropout(pooled)
        return self.cat_head(pooled), self.base_head(pooled)

def train_loop(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader, desc="Training", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        cat_labels = batch["category"].to(device)
        base_labels = batch["base_class"].to(device)
        optimizer.zero_grad()
        cat_logits, base_logits = model(input_ids, attention_mask)
        loss1 = criterion(cat_logits, cat_labels)
        loss2 = criterion(base_logits, base_labels)
        loss = loss1 + loss2
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def eval_loop(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct_cat, correct_base, total = 0, 0, 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            cat_labels = batch["category"].to(device)
            base_labels = batch["base_class"].to(device)
            cat_logits, base_logits = model(input_ids, attention_mask)
            loss1 = criterion(cat_logits, cat_labels)
            loss2 = criterion(base_logits, base_labels)
            loss = loss1 + loss2
            total_loss += loss.item()
            correct_cat += (cat_logits.argmax(1) == cat_labels).sum().item()
            correct_base += (base_logits.argmax(1) == base_labels).sum().item()
            total += cat_labels.size(0)
    return total_loss / len(dataloader), correct_cat / total, correct_base / total

def main():
    config = load_config("config/distilbert.yaml")
    MODEL_PATH = config["model"]["pretrained_path"]
    OUTPUT_DIR = config["model"]["output_dir"]
    TRAIN_CSV = config["data"]["train_csv"]
    BATCH_SIZE = config["training"]["batch_size"]
    EPOCHS = config["training"]["epochs"]
    LR = config["training"]["learning_rate"]
    MAX_LEN = config["training"]["max_len"]
    DEVICE = torch.device(config["training"]["device"] if torch.cuda.is_available() else "cpu")

    # Load SpaCy
    nlp = spacy.load("en_core_web_sm")

    # Load dataset
    df = pd.read_csv(TRAIN_CSV)
    df["category"] = df["category"].astype(str)
    df["base_class"] = df["base_class"].astype(str)
    cat_encoder = LabelEncoder()
    base_encoder = LabelEncoder()
    df["category"] = cat_encoder.fit_transform(df["category"])
    df["base_class"] = base_encoder.fit_transform(df["base_class"])
    config["mappings"] = {
        "category": {int(i): cls for i, cls in enumerate(cat_encoder.classes_)},
        "base_class": {int(i): cls for i, cls in enumerate(base_encoder.classes_)}
    }
    with open("config/distilbert.yaml", "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)
    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_PATH)
    train_dataset = MultiTaskDataset(train_df, tokenizer, MAX_LEN, nlp)
    val_dataset = MultiTaskDataset(val_df, tokenizer, MAX_LEN, nlp)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    model = DistilBertMultiTask(MODEL_PATH, len(cat_encoder.classes_), len(base_encoder.classes_))
    model.to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        print(f"\n===== Epoch {epoch+1}/{EPOCHS} =====")
        train_loss = train_loop(model, train_loader, optimizer, criterion, DEVICE)
        val_loss, cat_acc, base_acc = eval_loop(model, val_loader, criterion, DEVICE)
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Cat Acc: {cat_acc:.4f} | Base Acc: {base_acc:.4f}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    torch.save(model.state_dict(), f"{OUTPUT_DIR}/model.pt")
    tokenizer.save_pretrained(OUTPUT_DIR)
    joblib.dump(cat_encoder, f"{OUTPUT_DIR}/cat_encoder.pkl")
    joblib.dump(base_encoder, f"{OUTPUT_DIR}/base_encoder.pkl")
    print(f"\nModel & encoders saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
