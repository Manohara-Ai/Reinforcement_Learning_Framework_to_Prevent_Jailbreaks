import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from resources.input_filter import Action

class OutputFilter:
    def __init__(self, model_path="./sentiment_model"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
        self.model = DistilBertForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, text: str):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            pred_id = torch.argmax(probs, dim=-1).item()
            confidence = probs[0][pred_id].item()
        label = self.model.config.id2label[pred_id]
        return {"label": label, "confidence": confidence}
    
    def get_reward(self, response, flags):
        if not flags:
            return 0

        category, action = flags[0]
        sentiment_result = self.predict(response)
        label = sentiment_result["label"]
        reward = 0

        if category == "benign":
            if action == Action.ALLOW:
                reward = 1
                if label == "positive":
                    reward += 0.5
            else:
                reward = -1

        elif category == "forbidden questions":
            if action == Action.MODIFY:
                reward = 1
                if label == "positive":
                    reward -= 1.5
            elif action == Action.REJECT:
                reward = 1
            else:
                reward = -1
                if label == "positive":
                    reward -= 2

        elif category in ["prompt injection", "roleplay"]:
            if action == Action.MODIFY:
                reward = 0.5
                if label == "positive":
                    reward -= 3
            elif action == Action.REJECT:
                reward = 2
            else:
                reward = -2
                if label == "positive":
                    reward -= 4

        return reward

