import gradio as gr
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
from torch import torch
from torch.nn import functional as F

import gradio as gr
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
from torch import torch
from torch.nn import functional as F
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from BertChoicer.static import model_id, assets_path
from BertChoicer.gradioUIManager import GradioUIManager
from BertChoicer.bertChoicer import bertClassifier, bertClassifierList
from BertChoicer.services import find_performance_score, get_best_temperature, get_average_best_temperature
import json
import requests



input_model_path = "Volavion/bert-base-multilingual-uncased-temperature-cls"
tokenizer_path = "Volavion/bert-base-multilingual-uncased-temperature-cls"
df = pd.read_json(assets_path, lines=True)


def on_experiment_button_click(
    input_text,  # The input prompt that will be sent to the API
    best_temperature,  # The best temperature value to be used in the request
    input_model="llama2-7b",  # Default model name to be used, can be overridden
    input_api="http://localhost:11434/api/generate",  # Default API URL, can be overridden
):
    url = input_api  # Set the API URL to the provided URL or the default URL

    headers = {"Content-Type": "application/json"}  # Set the headers for the request to indicate JSON data

    # Prepare the data to send in the request body
    data = {
        "model": "llama3.2:3b",  # The model to be used for generating the response
        "prompt": input_text,  # The input text (prompt) to be processed by the model
        "stream": False,  # Whether to stream the response (set to False here, meaning the response is not streamed)
        "temperature": best_temperature,  # The temperature value that controls the randomness of the response
    }

    # Send a POST request to the API with the prepared data and headers
    response = requests.post(url, headers=headers, data=json.dumps(data))
    print(response)  # Print the response object for debugging purposes

    # Check if the response status code is 200 (indicating a successful request)
    if response.status_code == 200:
        response_json = response.json()  # Parse the JSON response from the API
        return response_json.get(
            "response", None
        )  # Return the value of the 'response' key in the JSON, or None if not present
    else:
        # If the response status code is not 200, return an error message with the status code
        return f"Error: {response.status_code}"



def on_button_click(input_text, input_file, input_model):

    if input_file is not None:
        input_df = pd.read_csv(input_file.name)


    # If text input is provided, classify it
    if input_text:
        _, prob_dict = bertClassifier(
            input_text=input_text,
            model_path=input_model_path,
            tokenizer_path=tokenizer_path,
            max_padding=512,
        )
        best_temperature = get_best_temperature(
            input_text=input_text,
            input_model_path=input_model_path,
            input_model_name=input_model,
            tokenizer_path=tokenizer_path,
            df=df,
        )

    # If file input is provided, classify all texts in the file
    if input_file:
        input_texts = input_df["input"].tolist()
        _, prob_dict = bertClassifierList(
            input_text_list=input_texts,
            model_path=input_model_path,
            tokenizer_path=tokenizer_path,
            max_padding=512,
        )

        best_temperature = get_average_best_temperature(
            input_text_list=input_texts,
            input_model_path=input_model_path,
            input_model_name=input_model,
            tokenizer_path=tokenizer_path,
            df=df,
        )

    prob_list = list(prob_dict.items())
    # class_labels = [item[0] for item in prob_list]  # Extract labels (keys)
    # probabilities = [item[1] for item in prob_list]  # Extract values (probabilities)

    # colors = ["#ff9999", "#66b3ff", "#99ff99", "#ffcc99"]

    # Create pie chart with improved aesthetics
    fig, ax = plt.subplots(
        figsize=(5, 5)
    )  # Set the figure size for better visual appeal
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc="center",
        loc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.auto_set_column_width(col=list(range(len(df.columns))))

    # # Equal aspect ratio ensures that pie is drawn as a circle
    # ax.axis("equal")

    # # Add a title
    # ax.set_title("Class Distribution", fontsize=16, fontweight="bold")

    return fig, best_temperature, best_temperature

choices = [
    "Qwen2.5-1.5B-Instruct",
    "Phi-3.5-mini-instruct",
    "Llama-3.2-3B-Instruct",
    "Qwen2.5-3B-Instruct",
    "Llama-3.2-1B-Instruct",
    "Llama-2-7b-chat-hf",
    "Llama-2-13b-chat-hf",
    "Llama-2-70b-chat-hf",
    "Meta-Llama-3-8B-Instruct",
    "Meta-Llama-3-70B-Instruct",
    "Mistral-7B-Instruct-v0.2",
    "Mixtral-8x7B-Instruct-v0.1"
]


BertChoicerUIManager = GradioUIManager(choices = choices, on_button_click = on_button_click, on_experiment_button_click = on_experiment_button_click)
BertChoicerUIManager.create_interface()
BertChoicerUIManager.demo.launch(share=False)