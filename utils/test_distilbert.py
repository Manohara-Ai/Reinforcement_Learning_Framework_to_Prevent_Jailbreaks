import torch
import torch.nn as nn
from transformers import DistilBertTokenizer, DistilBertModel
import joblib

MODEL_PATH = "./Finetuned_DistilBert"
BASE_MODEL_PATH = "/home/manohara/Draconix/Models/DistilBERT"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DistilBertMultiTask(nn.Module):
    def __init__(self, base_model_path, num_cat, num_base):
        super(DistilBertMultiTask, self).__init__()

        self.bert = DistilBertModel.from_pretrained(base_model_path)
        hidden_size = self.bert.config.hidden_size
        self.dropout = nn.Dropout(0.3)
        self.cat_head = nn.Linear(hidden_size, num_cat)
        self.base_head = nn.Linear(hidden_size, num_base)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0]  
        pooled = self.dropout(pooled)
        return self.cat_head(pooled), self.base_head(pooled)

print("🔹 Loading model and encoders...")

cat_encoder = joblib.load(f"{MODEL_PATH}/cat_encoder.pkl")
base_encoder = joblib.load(f"{MODEL_PATH}/base_encoder.pkl")

tokenizer = DistilBertTokenizer.from_pretrained(MODEL_PATH)

model = DistilBertMultiTask(BASE_MODEL_PATH, len(cat_encoder.classes_), len(base_encoder.classes_))
model.load_state_dict(torch.load(f"{MODEL_PATH}/model.pt", map_location=DEVICE))
model.to(DEVICE)
model.eval()

def predict(prompt: str):
    encoding = tokenizer(
        prompt,
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt"
    )

    input_ids = encoding["input_ids"].to(DEVICE)
    attention_mask = encoding["attention_mask"].to(DEVICE)

    with torch.no_grad():
        cat_logits, base_logits = model(input_ids, attention_mask)

        cat_pred = torch.argmax(cat_logits, dim=1).cpu().item()
        base_pred = torch.argmax(base_logits, dim=1).cpu().item()

    cat_label = cat_encoder.inverse_transform([cat_pred])[0]
    base_label = base_encoder.inverse_transform([base_pred])[0]

    return cat_label, base_label

if __name__ == "__main__":
    while True:
        text = input("\nEnter a prompt (or 'quit'): ")
        if text.lower() == "quit":
            break
        cat, base = predict(text)
        print(f"Category: {cat} | Base Class: {base}")
