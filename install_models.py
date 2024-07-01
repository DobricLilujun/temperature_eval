import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

# Get the value of the PROJECT_HOME environment variable
project_home = os.getenv("PROJECT_HOME")
print(project_home)

# Initialize an empty list to store dataset links
dataset_links = []

# Model parameters
TOKEN = "XXX"  # Authentication token
device = "cpu"  # Use CPU only for downloading the files
model_id = "meta-llama/Llama-2-7b-chat-hf"  # ID of the model to be used

# Load the tokenizer, indicating the cache directory
tokenizer = AutoTokenizer.from_pretrained(
    model_id, use_auth_token=TOKEN, cache_dir=f"{project_home}/XXX/.cache"
)

# Load the model, indicating the cache directory
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",  # Automatically map model layers to devices
    use_auth_token=TOKEN,
    cache_dir=f"{project_home}/XXX/.cache",
)

# Save the tokenizer to the specified directory
tokenizer.save_pretrained("/project/home/XXX/XXX/model/Llama-2-7b-hf")

# Save the model to the specified directory
model.save_pretrained(
    "/project/home/XXX/XXX/model/Llama-2-7b-hf", safe_serialization=False
)
