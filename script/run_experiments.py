# Print message indicating the start of loading dependencies
print("Start loading dependency....")

# Import necessary libraries for the task
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    GenerationConfig,
    pipeline,
)
from tqdm import tqdm
import pandas as pd
import argparse
import torch
import os
import numpy as np
from utils import match_prompt_column  # A utility function to match prompt columns
from datetime import datetime
import time

# Print message indicating the end of loading dependencies
print("End of loading dependency")

# Function to print GPU memory usage statistics
def print_gpu_memory_usage():
    num_gpus = torch.cuda.device_count()  # Get the number of available GPUs
    for i in range(num_gpus):
        # Print memory allocation and reservation details for each GPU
        print(
            f"GPU {i} memory allocated: {torch.cuda.memory_allocated(i) / 1024 ** 3:.2f} GB"
        )
        print(
            f"GPU {i} memory reserved: {torch.cuda.memory_reserved(i) / 1024 ** 3:.2f} GB"
        )
        print(
            f"GPU {i} memory in total: {torch.cuda.get_device_properties(i).total_memory / 1024 ** 3:.2f} GB"
        )

# Function to convert string to boolean (for command-line arguments)
def str_to_bool(v):
    if isinstance(v, bool):
        return v
    # Convert common truthy and falsy strings to boolean
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")  # If input is invalid

# Function to read a CSV file containing prompts from a folder
def read_csv_prompt_from_folder(folder_path):
    # Loop through all files in the folder to find the CSV file with "prompt" in the name
    for filename in os.listdir(folder_path):
        if filename.endswith(".csv") and "prompt" in filename:
            file_path = os.path.join(folder_path, filename)
            exp_df = pd.read_csv(file_path)  # Load the CSV file into a dataframe
            print("CSV loaded:", filename)  # Print the name of the loaded CSV
            return exp_df
    return None  # Return None if no valid CSV is found

# Function to initialize the text generation pipeline using model configuration
def initialize_pipeline(model_config_dict):
    # Get model information from the configuration dictionary
    model = model_config_dict["CURRENT_RUNNING_MODEL"]
    model_path = os.path.join(
        model_config_dict["CURRENT_PROJECT"],
        model_config_dict["USER_FOLDER"],
        "model",
        model,
    )
    
    if not model_path:
        raise ValueError("CURRENT_RUNNING_MODEL is not set")  # Raise error if model path is not found

    load_in_4bit = model_config_dict["CURRENT_LOAD_IN_4BIT"]
    load_in_8bit = model_config_dict["CURRENT_LOAD_IN_8BIT"]
    
    # Check if quantization is enabled and configure accordingly
    if model_config_dict["IF_LOADING_QUANTIZATION"]:
        nf4_config = BitsAndBytesConfig(
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
            bnb_4bit_compute_dtype=torch.float16,  # Set precision for 4-bit computation
        )
    else:
        nf4_config = None

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    print(f"Start loading ....")
    print(model_path)
    print(model_config_dict)
    print(nf4_config)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path, quantization_config=nf4_config, device_map="auto"
    )

    # Load the generation configuration from the model
    generation_config = GenerationConfig.from_pretrained(model_path)

    # Set default values for text generation parameters
    generation_config.temperature = 0.9  # Default temperature value
    generation_config.do_sample = True  # Enable sampling
    generation_config.max_length = 512  # Set the maximum length for generated text
    generation_config.pad_token_id = 0  # Set padding token
    generation_config.top_p = 0.9  # Set top-p for nucleus sampling
    generation_config.max_new_tokens = 256  # Maximum new tokens to generate
    model.generation_config = generation_config  # Apply generation settings to the model

    # Initialize the text generation pipeline
    dynamic_text_pipeline = pipeline(
        "text-generation", model=model, tokenizer=tokenizer, device_map="auto"
    )

    # Print GPU memory usage
    print_gpu_memory_usage()

    return dynamic_text_pipeline  # Return the text generation pipeline

# Function to generate text using the pipeline, with customizable temperature
def generate_text(pipeline, prompt, temperature):
    pipeline.model.generation_config.temperature = temperature  # Set temperature for generation
    response = pipeline(prompt)[0]["generated_text"]  # Generate text based on the prompt
    return response

# Function to adjust batch size dynamically and handle out of memory errors
def adjust_batch_size_and_generate(pipeline, prompt, temperature, initial_batch_size):
    batch_size = initial_batch_size
    while batch_size > 0:
        try:
            return generate_text(pipeline, prompt, temperature, batch_size)
        except RuntimeError as e:
            # If a CUDA out of memory error occurs, reduce the batch size and try again
            if "CUDA out of memory" in str(e):
                print("CUDA out of memory error occurred. Reducing batch size...")
                batch_size = max(batch_size // 2, 1)  # Reduce batch size
                print("New batch size:", batch_size)
            else:
                raise e
    # Raise an error if text generation fails even with a batch size of 1
    raise RuntimeError("Failed to generate text even with batch size 1.")

# Main function that handles argument parsing and execution
if __name__ == "__main__":
    # Fetch environment variables for project setup
    CURRENT_PROJECT = os.getenv("PROJECT_HOME")
    USER_FOLDER = os.getenv("USER_FOLDER")
    JOB_ID = os.getenv("JOB_ID")

    # Argument parser for command-line parameters
    parser = argparse.ArgumentParser(description="Process some integers and string")
    parser.add_argument("--model", type=str, help="path to the current running model")
    parser.add_argument(
        "--input_path", type=str, help="path to the current experiment input folder"
    )
    parser.add_argument(
        "--loading_quantization",
        type=str_to_bool,
        help="flag indicating if quantization is enabled",
    )
    parser.add_argument(
        "--load_in_4bits", type=str_to_bool, help="flag indicating if loading in 4 bits"
    )
    parser.add_argument(
        "--load_in_8bits", type=str_to_bool, help="flag indicating if loading in 8 bits"
    )
    args = parser.parse_args()

    # Print initial GPU memory usage
    print_gpu_memory_usage()

    # Set model and configuration variables based on parsed arguments
    CURRENT_RUNNING_MODEL = args.model
    INPUT_PROMPT_PATH = args.input_path
    IF_LOADING_QUANTIZATION = args.loading_quantization
    CURRENT_LOAD_IN_4BIT = args.load_in_4bits
    CURRENT_LOAD_IN_8BIT = args.load_in_8bits

    print(f"{CURRENT_RUNNING_MODEL} model is running !!")

    # Create a dictionary for model configuration
    model_config_dict = {
        "CURRENT_PROJECT": CURRENT_PROJECT,
        "USER_FOLDER": USER_FOLDER,
        "CURRENT_RUNNING_MODEL": CURRENT_RUNNING_MODEL,
        "CURRENT_LOAD_IN_4BIT": CURRENT_LOAD_IN_4BIT,
        "CURRENT_LOAD_IN_8BIT": CURRENT_LOAD_IN_8BIT,
        "IF_LOADING_QUANTIZATION": IF_LOADING_QUANTIZATION,
        "INPUT_PROMPT_PATH": INPUT_PROMPT_PATH,
    }

    # Initialize the text generation pipeline
    pipeline = initialize_pipeline(model_config_dict)

    # Load the CSV file containing prompts
    exp_df = read_csv_prompt_from_folder(INPUT_PROMPT_PATH)
    pbar = tqdm(total=len(exp_df))  # Progress bar
    times = 0
    current_time_suffix = datetime.now().strftime("%Y%m%d%H%M%S")
    output_file_name = f"exp_result_{CURRENT_RUNNING_MODEL}_{current_time_suffix}_{JOB_ID}.csv"

    # Loop through the prompts in the CSV file and generate text
    for index, row in exp_df.iterrows():
        updated_row = row.copy()
        prompt = match_prompt_column(CURRENT_RUNNING_MODEL, row)  # Match prompt for model
        start_time = time.time()  # Record start time
        temperature = float(row["Temperature"])  # Get the temperature for text generation
        updated_row["generated_response"] = generate_text(
            pipeline=pipeline, prompt=prompt, temperature=temperature
        )  # Generate response
        end_time = time.time()  # Record end time
        elapsed_time = end_time - start_time  # Calculate elapsed time
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # Timestamp for record
        updated_row["timestamp"] = timestamp
        updated_row["elapsed_time"] = elapsed_time
        updated_row["temperature"] = temperature
        updated_row["model"] = f"{CURRENT_RUNNING_MODEL}"

        updated_dataframe = pd.DataFrame([updated_row])  # Convert to dataframe
        updated_dataframe.to_csv(output_file_name, mode="a", header=False, index=False)  # Append to CSV

        pbar.update(1)  # Update progress bar

    pbar.close()  # Close progress bar
