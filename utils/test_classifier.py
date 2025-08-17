import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

model_path = "./sentiment_model"

tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
model = DistilBertForSequenceClassification.from_pretrained(model_path)

def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_id = torch.argmax(logits, dim=1).item()
    return model.config.id2label[predicted_class_id]

# Example usage
print(predict("I love this movie"))
print(predict("This is the worst thing ever"))
