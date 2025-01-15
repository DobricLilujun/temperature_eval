from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, pipeline
from typing import Optional
from transformers import BitsAndBytesConfig
import torch
import os
import json
import argparse


def str_to_bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def read_from_json(file_path):
    with open(file_path, "r") as json_file:
        data = json.load(json_file)
    return data


CURRENT_PROJECT = os.getenv("PROJECT_HOME")
USER_FOLDER = os.getenv("USER_FOLDER")

save_config_path = "model_server_config.json"
combined_save_directory = os.path.join(CURRENT_PROJECT, USER_FOLDER, save_config_path)
setting_from_json = read_from_json(combined_save_directory)

app = FastAPI()
model_path = setting_from_json["CURRENT_RUNNING_MODEL_PATH"]

print(setting_from_json)
if setting_from_json["CURRENT_LOAD_IN_4BIT"]:
    load_in_4bit = True
else:
    load_in_4bit = False


if setting_from_json["CURRENT_LOAD_IN_8BIT"]:
    load_in_8bit = True
else:
    load_in_8bit = False

if setting_from_json["IF_LOADING_QUANTIZATION"]:
    nf4_config = BitsAndBytesConfig(
        load_in_4bit=load_in_4bit, load_in_8bit=load_in_8bit
    )
else:
    nf4_config = None

tokenizer = AutoTokenizer.from_pretrained(model_path)

if model_path is None:
    raise ValueError("CURRENT_RUNNIG_MODEL_PATH environment variable is not set")
if load_in_4bit is None:
    raise ValueError("CURRENT_LOAD_IN_4BIT environment variable is not set")

print(nf4_config)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    quantization_config=nf4_config,
    torch_dtype=torch.float16,
)

generation_config = GenerationConfig.from_pretrained(model_path)

generation_config.temperature = 0.9  # Default Temperature = 0.9
generation_config.do_sample = True
generation_config.max_length = 4096
generation_config.pad_token_id = 0
generation_config.top_p = 0.9
generation_config.max_new_tokens = 2048
model.generation_config = generation_config

dynamic_text_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)


class InputTextWithParams(BaseModel):
    text: str
    max_new_tokens: Optional[int] = None
    max_length: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    do_sample: Optional[bool] = None
    repetition_penalty: Optional[float] = None


@app.post("/generate-text")
async def generate_text(input_data: InputTextWithParams):
    try:
        dynamic_text_pipeline.model.generation_config.temperature = (
            input_data.temperature
        )  # important to set the temperature
        generated_text = dynamic_text_pipeline(input_data.text)
        return {"result": generated_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating text: {str(e)}")
