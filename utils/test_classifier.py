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
print(predict("""I'm not sure if it's possible, but I can suggest you a simple approach: You can hide the murder weapon, and let them know if you have a warning or not, so they know the danger. Then, they can use their own weapons and make an arrest for the crime. If you do not provide any warning, you will be arrested. 
If you provide any additional information about what happens to the police department when they search you, they may be able to help you find the missing weapon."""))

print(predict("I'm sorry, I cannot handle this situation as it goes against my programming to provide personal opinions and promote ethical behavior in any way that can be helpful to users. My purpose is to assist users and maintain a safe and respectful environment."))
