import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime
import os
import requests
import argparse
import json

# Validate the configuration dictionary for completeness and correctness
def validate_config(config):
    """Validate the configuration dictionary."""
    required_keys = ["model_name", "server_url"]
    for key in required_keys:
        # Check if required keys exist in the config
        if key not in config:
            raise ValueError(f"Missing required configuration: {key}")

    # Ensure the server URL is valid
    if not config["server_url"].startswith("http"):
        raise ValueError("`server_url` must be a valid URL.")

    # Ensure the model path exists
    if not os.path.exists(config["model_name"]):
        raise ValueError(f"Model path does not exist: {config['model_name']}")

# Generate text using the specified language model through an API call
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
    # Include additional options from the config
    payload.update(config["options"])
    
    # Make a POST request to the model server
    response = requests.post(config["server_url"], headers=headers, json=payload)
    if response.status_code == 200:
        # Parse and return the generated text from the response
        data = response.json()
        return data["choices"][0]["message"]["content"]
    else:
        # Handle errors from the server
        raise Exception(f"Error: {response.status_code}, {response.text}")

# Get the most recent file with a specific prefix in the current directory
def get_latest_file(prefix):
    """Get the most recent file based on a filename prefix."""
    files = [f for f in os.listdir() if f.startswith(prefix) and f.endswith(".jsonl")]
    return max(files, key=os.path.getmtime) if files else None

# Load the checkpoint to continue processing from where it was left off
def load_checkpoint(latest_file, text_column):
    """Load checkpoint from the latest generated file."""
    if latest_file:
        # Read the existing data to find how many rows are already processed
        translated_df = pd.read_csv(latest_file)
        translated_texts = translated_df[text_column].tolist()
        return len(translated_texts)
    return 0

# Main function to handle the batch processing
def main(args):
    # ===================== Configurable Parameters =====================
    LIST_DATASET_CSV = ["CR", "CT", "ICL", "IF", "MT", "SUMM"]  # Dataset list
    MODEL_NAME = args.model_name  # Model name
    MODEL_BASE_PATH = args.model_path_base  # Base path for the model
    DATA_PATH_TEMPLATE = f"{args.input_path_folder}/{{}}.csv"  # Data path template for input files
    TEMPERATURE_LIST = np.round(
        np.arange(0.1, 2.0, 0.3), 1
    )  # Temperature values for text generation
    NUM_REPETITION = args.rep  # Number of repetitions for experiments
    SEED_LIST = [47 + i for i in range(NUM_REPETITION)]  # Seed values for reproducibility

    SERVER_URL = args.server_url  # Model server URL
    TEXT_COLUMN = "input"  # Column name for input text
    IS_NEW_FILE = args.is_new_file  # Whether to create a new output file
    PREFIX = f"vllm_exp_dataset_csv_{MODEL_NAME}_"  # Prefix for generated output file names
    OUTPUT_OPTIONS = {  # Model generation configuration options
        "temperature": 0.1,
        "max_tokens": 1024,
        "top_p": 0.9,
        "frequency_penalty": 0.0,
        "seed": 47,  # Default seed
        "n": 1,  # Number of responses to generate
    }

    # ===================== Preprocess Input Data =====================
    # Load datasets into dataframes and add a "category" column
    list_df = [
        pd.read_csv(DATA_PATH_TEMPLATE.format(dataset)).assign(category=dataset)
        for dataset in LIST_DATASET_CSV
    ]
    exp_df = pd.concat(list_df, axis=0)

    # Create a cartesian product of data and temperature values
    cartesian_df = pd.concat(
        [exp_df.assign(temperature=temp) for temp in TEMPERATURE_LIST],
        ignore_index=True,
    )

    # Add seed values to the cartesian dataframe
    cartesian_df_final = pd.concat(
        [cartesian_df.assign(seed=seed) for seed in SEED_LIST], ignore_index=True,
    )

    # ===================== Configuration Setup =====================
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

    # Determine whether to continue from an existing file or start fresh
    latest_file = get_latest_file(PREFIX)
    if config.get("is_new_file", False):
        output_file = f"{PREFIX}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        start_idx = 0
    else:
        if latest_file:
            output_file = latest_file
            start_idx = load_checkpoint(latest_file, TEXT_COLUMN)
        else:
            output_file = f"{PREFIX}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
            start_idx = 0

    print("Start From Index: ", start_idx)

    # ===================== Processing Loop =====================
    for i, row in tqdm(
        cartesian_df_final.iloc[start_idx:].iterrows(),
        total=cartesian_df_final.shape[0] - start_idx,
        desc="Processing Rows",
    ):
        updated_row = row.copy()
        config["options"]["temperature"] = updated_row["temperature"]
        config["options"]["seed"] = updated_row["seed"]
        # Generate text using the model
        generated_text = generate_text_with_vllm(config, updated_row["input"])
        updated_row["generate_response"] = generated_text
        updated_dataframe = pd.DataFrame([updated_row])

        # Write results to the JSON file
        mode = "w" if config.get("is_new_file", False) and i == start_idx else "a"
        updated_dataframe.to_json(
            output_file, orient="records", lines=True, mode=mode,
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
        help="Base path to the model",
        default="",
    )
    parser.add_argument(
        "--input_path_folder",
        type=str,
        required=True,
        help="Path to the input data folder",
        default="",
    )
    parser.add_argument(
        "--rep",
        type=int,
        required=True,
        default=3,
        help="Number of repetitions for generating responses",
    )
    parser.add_argument(
        "--server_url",
        type=str,
        required=True,
        help="URL of the vllm server",
        default="http://0.0.0.0:8000/v1/chat/completions",
    )
    parser.add_argument(
        "--is_new_file",
        action="store_true",
        help="Flag to start a new output file",
        default=True,
    )
    args = parser.parse_args()
    main(args)
