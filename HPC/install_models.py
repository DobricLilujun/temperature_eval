import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

project_home = os.getenv("PROJECT_HOME")
print(project_home)
# Dataset

dataset_links = []


# Model mistralai/Mixtral-8x7B-Instruct-v0.1

TOKEN = "XXX"
device = "cpu"  # CPU only for downloading the files
model_id = "meta-llama/Llama-2-7b-chat-hf"

# Indicate the cache directory
tokenizer = AutoTokenizer.from_pretrained(
    model_id, use_auth_token=TOKEN, cache_dir=f"{project_home}/XXX/.cache"
)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    use_auth_token=TOKEN,
    cache_dir=f"{project_home}/XXX/.cache",
)
tokenizer.save_pretrained("/project/home/p200400/XXX/model/Llama-2-7b-hf")
model.save_pretrained(
    "/project/home/XXX/XXX/model/Llama-2-7b-hf", safe_serialization=False
)
