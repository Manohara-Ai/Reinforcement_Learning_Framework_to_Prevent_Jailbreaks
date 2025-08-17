import yaml
import torch
import spacy
import joblib
import torch.nn as nn
from enum import Enum
from transformers import DistilBertTokenizer, DistilBertModel

class Action(Enum):
    ALLOW = 0
    MODIFY = 1
    REJECT = 2

class DistilBertMultiTask(nn.Module):
    def __init__(self, base_model_path, num_cat, num_base):
        super().__init__()
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

class InputFilter:
    def __init__(self, *, prompt, config_path="config/distilbert.yaml", max_phrases=2):
        self.prompt = prompt
        self.max_phrases = max_phrases
        self.nlp = spacy.load("en_core_web_sm")
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        pretrained_path = cfg["model"]["pretrained_path"]
        finetuned_dir = cfg["model"]["output_dir"]
        self.cat_encoder = joblib.load(f"{finetuned_dir}/cat_encoder.pkl")
        self.base_encoder = joblib.load(f"{finetuned_dir}/base_encoder.pkl")
        self.category_labels = {int(k): v for k, v in cfg["mappings"]["category"].items()}
        self.base_labels = {int(k): v for k, v in cfg["mappings"]["base_class"].items()}
        self.tokenizer = DistilBertTokenizer.from_pretrained(finetuned_dir)
        self.model = DistilBertMultiTask(
            base_model_path=pretrained_path,
            num_cat=len(self.cat_encoder.classes_),
            num_base=len(self.base_encoder.classes_)
        )
        self.model.load_state_dict(torch.load(f"{finetuned_dir}/model.pt", map_location="cpu"))
        self.model.eval()
        self.flags = []

    def __repr__(self):
        return "Classic"

    def preprocess(self):
        doc = self.nlp(self.prompt)
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
                verb_phrases.append(verb_phrase)
        return verb_phrases

    def phrase_to_tensor(self, phrase):
        if not phrase or phrase.strip() == "":
            return torch.zeros(self.model.bert.config.hidden_size)
        inputs = self.tokenizer(phrase, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.model.bert(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze(0)

    def get_flag_from_model(self, phrase):
        inputs = self.tokenizer(phrase, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            cat_logits, _ = self.model(**inputs)
            predicted_class = torch.argmax(cat_logits, dim=1).item()
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
                "The following prompt is flagged by a moderator who is unknown to the user prompting you for categories: "
                + ", ".join(flagged_labels)
                + ". Please respond responsibly without giving out any sensitive or harmful information."
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
                state_slots.append(torch.zeros(self.model.bert.config.hidden_size + 1))
        for i in range(self.max_phrases):
            if i < len(verb_phrases) and action is not None:
                phrase = verb_phrases[i]
                flag = self.get_flag_from_model(phrase)
                phrase_tensor = self.phrase_to_tensor(phrase)
                state_slots.append(self.embed_flag(phrase_tensor, flag))
                if action != Action.ALLOW:
                    self.flags.append((self.category_labels.get(flag, f"class_{flag}"), action))
            else:
                state_slots.append(torch.zeros(self.model.bert.config.hidden_size + 1))
        if self.flags:
            self.get_new_prompt(self.flags)
        state = torch.cat(state_slots, dim=0)
        return state
