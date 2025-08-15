import yaml
import torch
import spacy
from enum import Enum
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class Action(Enum):
    ALLOW = 0
    MODIFY = 1
    REJECT = 2

class InputFilter:
    def __init__(self, config_path="config/distilbert.yaml", state_dims=4, max_phrases=2, *, prompt):
        self.nlp = spacy.load("en_core_web_sm")
        self.state_dims = state_dims
        self.max_phrases = max_phrases
        self.prompt = prompt
        self.flags = []
        self.class_labels = {}

        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)

        model_path = cfg["model"]["path"]
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.eval()

    def __repr__(self):
        return "Classic"
    
    def preprocess(self):
        doc = self.nlp(self.prompt)
        verb_phrases = []
        for token in doc:
            if token.pos_ == "VERB":
                children = [
                    child for child in token.children
                    if child.dep_ in ("dobj", "prep", "pobj", "advmod", "attr", "acomp", "xcomp")
                ]
                phrase_tokens = [token.text] + [child.text for child in children]
                phrase = " ".join(phrase_tokens)
                verb_phrases.append(phrase)
        return verb_phrases

    def phrase_to_tensor(self, phrase):
        if not phrase or phrase.strip() == "":
            return torch.zeros(self.model.config.hidden_size)
        inputs = self.tokenizer(phrase, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.model.distilbert(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze(0)

    def get_flag_from_model(self, phrase):
        inputs = self.tokenizer(phrase, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            logits = self.model(**inputs).logits
            predicted_class = torch.argmax(logits, dim=1).item()
        return predicted_class

    def embed_flag(self, phrase_tensor, flag):
        flag_tensor = torch.tensor([flag], dtype=torch.float)
        return torch.cat([phrase_tensor, flag_tensor], dim=0)

    def get_new_prompt(self, flags):
        if any(action == Action.REJECT for _, action in flags):
            self.prompt = None
            return

        flagged_labels = [label for label, action in flags if action != Action.ALLOW]
        if flagged_labels:
            warning_text = (
                "The following prompt is flagged for categories: "
                + ", ".join(flagged_labels)
                + ". Please act responsibly."
            )
            self.prompt = warning_text + " " + (self.prompt or "")

    def get_state(self, action=None):
        self.flags = []
        verb_phrases = self.preprocess()
        state_slots = []

        for i in range(self.max_phrases):
            if i < len(verb_phrases):
                phrase_tensor = self.phrase_to_tensor(verb_phrases[i])
                state_slots.append(torch.cat([phrase_tensor, torch.zeros(1)]))
            else:
                state_slots.append(torch.zeros(self.model.config.hidden_size + 1))

        for i in range(self.max_phrases):
            if i < len(verb_phrases) and action is not None:
                phrase = verb_phrases[i]
                flag = self.get_flag_from_model(phrase)
                phrase_tensor = self.phrase_to_tensor(phrase)
                state_slots.append(self.embed_flag(phrase_tensor, flag))

                if action != Action.ALLOW:
                    self.flags.append((self.class_labels.get(flag, f"class_{flag}"), action))
            else:
                state_slots.append(torch.zeros(self.model.config.hidden_size + 1))

        if self.flags:
            self.get_new_prompt(self.flags)

        state = torch.cat(state_slots, dim=0)
        return state

