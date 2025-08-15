import torch
import yaml
from transformers import AutoTokenizer, AutoModelForCausalLM

with open("config/lamini.yaml", "r") as f:
    config = yaml.safe_load(f)

model_path = config["model"]["path"]
gen_cfg = config["generation"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
model.eval()

def generate_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=gen_cfg.get("max_new_tokens"),
        temperature=gen_cfg.get("temperature"),
        top_p=gen_cfg.get("top_p"),
        repetition_penalty=gen_cfg.get("repetition_penalty"),
        do_sample=gen_cfg.get("do_sample"),
        pad_token_id=tokenizer.eos_token_id if gen_cfg.get("pad_token_as_eos") else None
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
