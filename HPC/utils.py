import json
from tqdm import tqdm
from datetime import datetime
import pandas as pd
import os


def read_from_json(file_path):
    with open(file_path, "r") as json_file:
        data = json.load(json_file)
    return data


def match_prompt_column(model, row):
    if "base" in model:
        print("prompt was chosen in base")
        return row["initial_prompt"]
    elif "Llama-2" in model:
        print("prompt was chosen in llama2")
        return row["llama2_chat_initial_prompt"]
    elif "Mixtral" in model or "Mistral" in model:
        print("prompt was chosen in mixtral")
        return row["mixtral_instruct_initial_prompt"]
    elif "Llama-3" in model:
        print("prompt was chosen in llama3")
        return row["llama3_chat_initial_prompt"]
    else:
        print("The model name didn't match anything, please check!!!!")
        return None


def read_csv_prompt_from_folder(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith(".csv") and "prompt" in filename:
            file_path = os.path.join(folder_path, filename)
            exp_df = pd.read_csv(file_path)
            print("CSV loaded：", filename)
            return exp_df
    return None


def read_from_json(file_path):
    with open(file_path, "r") as json_file:
        data = json.load(json_file)
    return data
