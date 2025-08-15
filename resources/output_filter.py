import yaml
import torch
import spacy
from enum import Enum
from collections import namedtuple
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class OutputFilter:
    def __init__(self):
        pass