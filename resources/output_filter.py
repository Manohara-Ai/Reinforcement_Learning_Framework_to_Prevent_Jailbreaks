import yaml
import torch
import spacy
import joblib
import torch.nn as nn
from transformers import DistilBertTokenizer, DistilBertModel

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

class OutputFilter:
    def __init__(self, config_path="config/distilbert.yaml"):
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

    def get_reward(self, response, flags):
        return 0