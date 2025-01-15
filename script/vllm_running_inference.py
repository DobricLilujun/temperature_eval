import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime
import os
import requests
import argparse
import json


def validate_config(config):
    """Validate the configuration dictionary."""
    required_keys = ["model_name", "server_url"]
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required configuration: {key}")

    if not config["server_url"].startswith("http"):
        raise ValueError("`server_url` must be a valid URL.")

    if not os.path.exists(config["model_name"]):
        raise ValueError(f"Model path does not exist: {config['model_name']}")


def generate_text_with_vllm(config, prompt):
    """Generate text using the specified language model."""
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": config["model_name"],
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
    }
    payload.update(config["options"])
    response = requests.post(config["server_url"], headers=headers, json=payload)
    if response.status_code == 200:
        data = response.json()
        return data["choices"][0]["message"]["content"]
    else:
        raise Exception(f"Error: {response.status_code}, {response.text}")


def get_latest_file(prefix):
    """Get the most recent file based on a filename prefix."""
    files = [f for f in os.listdir() if f.startswith(prefix) and f.endswith(".jsonl")]
    return max(files, key=os.path.getmtime) if files else None


def load_checkpoint(latest_file, text_column):
    """Load checkpoint from the latest generated file."""
    if latest_file:
        translated_df = pd.read_csv(latest_file)
        translated_texts = translated_df[text_column].tolist()
        return len(translated_texts)
    return 0


def main(args):
    # ===================== Configurable Parameters =====================
    # User-configurable parameters to update
    LIST_DATASET_CSV = ["CR", "CT", "ICL", "IF", "MT", "SUMM"]  # Dataset list
    MODEL_NAME = args.model_name  # Model name
    MODEL_BASE_PATH = args.model_path_base  # Base path for the model
    DATA_PATH_TEMPLATE = f"{args.input_path_folder}/{{}}.csv"  # Data path template
    TEMPERATURE_LIST = np.round(
        np.arange(0.1, 2.0, 0.3), 1
    )  # List of temperature values
    NUM_REPETITION = args.rep  # Number of repetitions for experiments
    SEED_LIST = [
        47 + i for i in range(NUM_REPETITION)
    ]  # Seed values for reproducibility

    SERVER_URL = args.server_url
    TEXT_COLUMN = "input"  # Column name for input text
    IS_NEW_FILE = args.is_new_file  # Whether to create a new output file
    PREFIX = (
        f"vllm_exp_dataset_csv_{MODEL_NAME}_"  # Prefix for generated output file names
    )
    OUTPUT_OPTIONS = {  # Configuration options for model generation
        "temperature": 0.1,
        "max_tokens": 1024,
        "top_p": 0.9,
        "frequency_penalty": 0.0,
        "seed": 47,  # Default
        "n": 1,
    }

    # Preprocess input data
    list_df = [
        pd.read_csv(DATA_PATH_TEMPLATE.format(dataset)).assign(category=dataset)
        for dataset in LIST_DATASET_CSV
    ]

    exp_df = pd.concat(list_df, axis=0)

    cartesian_df = pd.concat(
        [exp_df.assign(temperature=temp) for temp in TEMPERATURE_LIST],
        ignore_index=True,
    )

    cartesian_df_final = pd.concat(
        [cartesian_df.assign(seed=seed) for seed in SEED_LIST], ignore_index=True
    )

    # Config dictionary
    config = {
        "model_name": MODEL_BASE_PATH + MODEL_NAME,
        "server_url": SERVER_URL,
        "prefix": PREFIX,
        "text_column": TEXT_COLUMN,
        "is_new_file": IS_NEW_FILE,
        "options": OUTPUT_OPTIONS,
    }

    # Validate the configuration
    validate_config(config)

    # Get the latest output file or start fresh
    latest_file = get_latest_file(PREFIX)
    if config.get("is_new_file", False):
        output_file = f"{PREFIX}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        start_idx= 0
    else:
        if latest_file:
            output_file = latest_file
            start_idx = (
                load_checkpoint(
                    latest_file,
                    cartesian_df_final,
                    config["text_column"],
                ))
        else:
            output_file = f"{PREFIX}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
            start_idx = 0
    print("Start From Index: ", start_idx)


# python vllm_running_inference.py --model_name Llama-3.2-3B-Instruct-awq --model_path_base /home/llama/models/base_models/ --input_path_folder /home/llama/Personal_Directories/srb/temperature_eval/data/Intermediate/ --rep 3 --server_url http://0.0.0.0:4362/v1/chat/completions

    for i, row in tqdm(
        cartesian_df_final.iloc[start_idx:].iterrows(),
        total=cartesian_df_final.shape[0] - start_idx,
        desc="Processing Rows",
    ):
        updated_row = row.copy()
        config["options"]["temperature"] = updated_row["temperature"]
        config["options"]["seed"] = updated_row["seed"]
        generated_text = generate_text_with_vllm(config, updated_row["input"])
        updated_row["generate_response"] = generated_text
        updated_dataframe = pd.DataFrame([updated_row])

        # 写入 JSON 文件
        if config.get("is_new_file", False):
            if i == start_idx:
                mode = "w"
            else:
                mode = "a"
        else:
            mode = "a"
            
        updated_dataframe.to_json(
            output_file,
            orient="records",
            lines=True,
            mode=mode,
        )

    print(f"Tasks completed. Results saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch Translation Script")
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Path to the model",
        default="Llama-3.2-1B-Instruct",
    )
    parser.add_argument(
        "--model_path_base",
        type=str,
        required=True,
        help="Path to the model",
        default="/home/snt/llm_models/",
    )
    parser.add_argument(
        "--input_path_folder",
        type=str,
        required=True,
        help="Path to the model",
        default="/home/snt/projects_lujun/temperature_eval/data/Intermediate/",
    )
    parser.add_argument(
        "--rep",
        type=int,
        required=True,
        default=3,
        help="Repetition times to create more responses",
    )
    parser.add_argument(
        "--server_url",
        type=str,
        required=True,
        help="vllm server URL",
        default="http://0.0.0.0:8000/v1/chat/completions",
    )
    parser.add_argument(
        "--is_new_file",
        action="store_true",
        help="Start a new output file",
        default=True,
    )
    args = parser.parse_args()
    main(args)


# python vllm_running_inference.py     --model_name "Llama-3.2-1B-Instruct"     --model_path_base "/home/snt/llm_models/"     --input_path_folder "/home/snt/projects_lujun/temperature_eval/data/Intermediate/"     --rep 3     --server_url "http://0.0.0.0:8000/v1/chat/completions"     --is_new_file
