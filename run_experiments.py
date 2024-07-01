print("Start loading dependency....")
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
from utils import match_prompt_column
from datetime import datetime
import time


print("End of loading dependency")


def print_gpu_memory_usage():
    num_gpus = torch.cuda.device_count()
    for i in range(num_gpus):
        print(
            f"GPU {i} memory allocated: {torch.cuda.memory_allocated(i) / 1024 ** 3:.2f} GB"
        )
        print(
            f"GPU {i} memory reserved: {torch.cuda.memory_reserved(i) / 1024 ** 3:.2f} GB"
        )
        print(
            f"GPU {i} memory in total: {torch.cuda.get_device_properties(i).total_memory / 1024 ** 3:.2f} GB"
        )


def str_to_bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def read_csv_prompt_from_folder(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith(".csv") and "prompt" in filename:
            file_path = os.path.join(folder_path, filename)
            exp_df = pd.read_csv(file_path)
            print("CSV loaded：", filename)
            return exp_df
    return None


# The design is to run the model by input the model_config_dict
def initialize_pipeline(model_config_dict):
    # Current load method
    model = model_config_dict["CURRENT_RUNNING_MODEL"]
    model_path = os.path.join(
        model_config_dict["CURRENT_PROJECT"],
        model_config_dict["USER_FOLDER"],
        "model",
        model,
    )
    if not model_path:
        raise ValueError("CURRENT_RUNNING_MODEL is not set")
    load_in_4bit = model_config_dict["CURRENT_LOAD_IN_4BIT"]
    load_in_8bit = model_config_dict["CURRENT_LOAD_IN_8BIT"]
    if model_config_dict["IF_LOADING_QUANTIZATION"]:
        nf4_config = BitsAndBytesConfig(
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
            bnb_4bit_compute_dtype=torch.float16,
        )
    else:
        nf4_config = None

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    print(f"Start loading ....")
    print(model_path)
    print(model_config_dict)
    print(nf4_config)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, quantization_config=nf4_config, device_map="auto"
    )

    generation_config = GenerationConfig.from_pretrained(model_path)

    # Set the default value of the model loading settings
    generation_config.temperature = 0.9  # Default Temperature = 0.9
    generation_config.do_sample = True  # This is used for the sampling
    generation_config.max_length = (
        512  # This is a max length for every model which can be the same
    )
    generation_config.pad_token_id = 0  # Pad token
    generation_config.top_p = 0.9  #
    generation_config.max_new_tokens = 256
    model.generation_config = generation_config  # Set the generation config
    dynamic_text_pipeline = pipeline(
        "text-generation", model=model, tokenizer=tokenizer, device_map="auto"
    )
    # Import the required module
    print_gpu_memory_usage()
    return dynamic_text_pipeline


# def generate_text(pipeline, prompt, temperature):
#     pipeline.model.generation_config.temperature = (temperature)  # important to set the temperature
#     response = pipeline(prompt)[0]["generated_text"]
#     return response


def generate_text(pipeline, prompt, temperature):
    pipeline.model.generation_config.temperature = temperature
    response = pipeline(prompt)[0]["generated_text"]
    return response


def adjust_batch_size_and_generate(pipeline, prompt, temperature, initial_batch_size):
    batch_size = initial_batch_size
    while batch_size > 0:
        try:
            return generate_text(pipeline, prompt, temperature, batch_size)
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                print("CUDA out of memory error occurred. Reducing batch size...")
                batch_size = max(batch_size // 2, 1)
                print("New batch size:", batch_size)
            else:
                raise e
    raise RuntimeError("Failed to generate text even with batch size 1.")


if __name__ == "__main__":
    CURRENT_PROJECT = os.getenv("PROJECT_HOME")
    USER_FOLDER = os.getenv("USER_FOLDER")
    JOB_ID = os.getenv("JOB_ID")
    parser = argparse.ArgumentParser(description="Process some integers and string")
    parser.add_argument("--model", type=str, help="path to the current running model")
    parser.add_argument(
        "--input_path", type=str, help="paht to the current experiment input folder"
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
    print_gpu_memory_usage()
    CURRENT_RUNNING_MODEL = args.model
    INPUT_PROMPT_PATH = args.input_path
    IF_LOADING_QUANTIZATION = args.loading_quantization
    CURRENT_LOAD_IN_4BIT = args.load_in_4bits
    CURRENT_LOAD_IN_8BIT = args.load_in_8bits
    print(f"{CURRENT_RUNNING_MODEL} model is running !!")
    model_config_dict = {
        "CURRENT_PROJECT": CURRENT_PROJECT,
        "USER_FOLDER": USER_FOLDER,
        "CURRENT_RUNNING_MODEL": CURRENT_RUNNING_MODEL,
        "CURRENT_LOAD_IN_4BIT": CURRENT_LOAD_IN_4BIT,
        "CURRENT_LOAD_IN_8BIT": CURRENT_LOAD_IN_8BIT,
        "IF_LOADING_QUANTIZATION": IF_LOADING_QUANTIZATION,
        "INPUT_PROMPT_PATH": INPUT_PROMPT_PATH,
    }

    pipeline = initialize_pipeline(model_config_dict)
    exp_df = read_csv_prompt_from_folder(
        INPUT_PROMPT_PATH
    )  # This function tries to find the csv file inside the folder
    pbar = tqdm(total=len(exp_df))
    times = 0
    current_time_suffix = datetime.now().strftime("%Y%m%d%H%M%S")
    output_file_name = (
        f"exp_result_{CURRENT_RUNNING_MODEL}_{current_time_suffix}_{JOB_ID}.csv"
    )
    for index, row in exp_df.iterrows():
        updated_row = row.copy()
        prompt = match_prompt_column(
            CURRENT_RUNNING_MODEL, row
        )  # Find the corresponding prompt
        start_time = time.time()
        temperature = float(row["Temperature"])
        updated_row["generated_response"] = generate_text(
            pipeline=pipeline, prompt=prompt, temperature=temperature
        )
        end_time = time.time()
        elapsed_time = end_time - start_time
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        updated_row["timestamp"] = timestamp
        updated_row["elapsed_time"] = elapsed_time
        updated_row["temperature"] = temperature
        updated_row["model"] = f"{CURRENT_RUNNING_MODEL}"
        updated_dataframe = pd.DataFrame([updated_row])
        if times == 0:
            updated_dataframe.to_csv(
                f"{INPUT_PROMPT_PATH}/{output_file_name}",
                index=False,
                mode="w",
                header=True,
            )
        else:
            updated_dataframe.to_csv(
                f"{INPUT_PROMPT_PATH}/{output_file_name}",
                index=False,
                mode="a",
                header=False,
            )
        pbar.update(1)
        times = times + 1
