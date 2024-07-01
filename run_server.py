import os
import sys
import torch
from uvicorn import run
import psutil
import json

import argparse

def str_to_bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def write_to_json(file_path, data):
    with open(file_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)


def find_uvicorn_processes():
    uvicorn_processes = []
    for process in psutil.process_iter(["pid", "name", "cmdline"]):
        if "uvicorn" in process.info["name"].lower() or (
            "uvicorn" in process.info["cmdline"] if process.info["cmdline"] else False
        ):
            uvicorn_processes.append(process.info)

    return uvicorn_processes


def kill_process_by_pid(pid):
    try:
        process = psutil.Process(pid)
        process.terminate()  # Alternatively, use process.kill() for a forceful termination
        print(f"Process {pid} has been terminated.")
    except psutil.NoSuchProcess:
        print(f"No process found with PID {pid}.")

# CURRENT_RUNNING_MODEL_PATH = "model/Llama-2-13b-chat-hf"
# CURRENT_RUNNING_MODEL_PATH = "model/Meta-Llama-3-8B-Instruct"
# CURRENT_LOAD_IN_4BIT= "True"
# CURRENT_LOAD_IN_8BIT= "False"
# IF_LOADING_QUANTIZATION = "True"

# CURRENT_RUNNING_MODEL_PATH = sys.argv[1]
# IF_LOADING_QUANTIZATION = sys.argv[2]
# CURRENT_LOAD_IN_4BIT= sys.argv[3]
# CURRENT_LOAD_IN_8BIT= sys.argv[4]
# host = sys.argv[5]
# port = sys.argv[6]

parser = argparse.ArgumentParser(description='Process some integers and string')
parser.add_argument('--model_path', type=str, help='path to the current running model')
parser.add_argument('--loading_quantization', type=str_to_bool, help='flag indicating if quantization is enabled')
parser.add_argument('--load_in_4bits', type=str_to_bool, help='flag indicating if loading in 4 bits')
parser.add_argument('--load_in_8bits', type=str_to_bool, help='flag indicating if loading in 8 bits')
parser.add_argument('--host', type=str, help='host address')
parser.add_argument('--port', type=int, help='port number')

args = parser.parse_args()

CURRENT_RUNNING_MODEL_PATH = args.model_path
IF_LOADING_QUANTIZATION = args.loading_quantization
CURRENT_LOAD_IN_4BIT = args.load_in_4bits
CURRENT_LOAD_IN_8BIT = args.load_in_8bits
host = args.host
port = args.port
# singularity run instance://my_server $VENV "python3.9 script/run_server.py model/Llama-2-7b-chat-hf True True False"

# CURRENT_PROJECT = "/home/llama/Personal_Directories/srb/causalllm-main/script"
CURRENT_PROJECT = os.getenv("PROJECT_HOME")
USER_FOLDER = os.getenv("USER_FOLDER")

api_params = {
    "host": host,
    "port": port
}

save_config_path = "model_server_config.json"
combined_save_directory = os.path.join(CURRENT_PROJECT, USER_FOLDER, save_config_path)

data_to_write = {
    "CURRENT_RUNNING_MODEL_PATH": CURRENT_RUNNING_MODEL_PATH,
    "CURRENT_LOAD_IN_4BIT": CURRENT_LOAD_IN_4BIT,
    "CURRENT_LOAD_IN_8BIT": CURRENT_LOAD_IN_8BIT,
    "CURRENT_PROJECT":CURRENT_PROJECT,
    "IF_LOADING_QUANTIZATION":IF_LOADING_QUANTIZATION,
    "api_params": {
        "host": api_params["host"],
        "port": api_params["port"]
    },
    "save_config_path":combined_save_directory
}
write_to_json(combined_save_directory, data_to_write)

import subprocess
directory = "script"
module = "text_generation_api_general_async"
python_executable = "python3.9"
uvicorn_command = f"{python_executable} -m uvicorn {directory}.{module}:app --host {host} --port {port}"
print (uvicorn_command)
subprocess.Popen(uvicorn_command, shell=True)
