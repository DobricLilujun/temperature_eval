# Import necessary libraries
import gradio as gr  # Gradio is used for creating web-based UI for machine learning models
import pandas as pd  # Pandas is used for handling data in DataFrame format
from transformers import BertTokenizer, BertForSequenceClassification  # Hugging Face's transformers for BERT model and tokenizer
from torch import torch  # PyTorch for working with neural networks
from torch.nn import functional as F  # PyTorch's functional module (commonly used for activations, loss functions)
from sklearn.preprocessing import MinMaxScaler  # Used for scaling data
import matplotlib.pyplot as plt  # Matplotlib for plotting graphs

# Import custom modules from the project
from BertChoicer.static import model_id, assets_path, model_choices  # Static configurations like model IDs and asset paths
from BertChoicer.gradioUIManager import GradioUIManager  # Custom UI manager for Gradio interface
from BertChoicer.bertChoicer import bertClassifier, bertClassifierList  # Functions for BERT classification tasks
from BertChoicer.services import find_performance_score, get_best_temperature, get_average_best_temperature  # Helper functions for performance evaluation

import json  # To handle JSON data
import requests  # To make HTTP requests

# Define paths for model and tokenizer
input_model_path = "Volavion/bert-base-multilingual-uncased-temperature-cls"
tokenizer_path = "Volavion/bert-base-multilingual-uncased-temperature-cls"

# Load a JSON dataset for classification task (could be for training or evaluation)
df = pd.read_json(assets_path, lines=True)

# Function to handle the button click event in the experiment interface
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

# Function to handle button click for classification and file processing
def on_button_click(input_text, input_file, input_model):
    # If a file is uploaded, load the CSV data into a DataFrame
    if input_file is not None:
        input_df = pd.read_csv(input_file.name)

    # If text input is provided, classify it using BERT
    if input_text:
        _, prob_dict = bertClassifier(
            input_text=input_text,
            model_path=input_model_path,
            tokenizer_path=tokenizer_path,
            max_padding=512,
        )
        # Determine the best temperature for the input text
        best_temperature = get_best_temperature(
            input_text=input_text,
            bert_model_path=input_model_path,
            target_model_name=input_model,
            tokenizer_path=tokenizer_path,
            df=df,
        )

    # If file input is provided, classify all texts in the file
    if input_file:
        input_texts = input_df["input"].tolist()  # Extract all input texts from the file
        _, prob_dict = bertClassifierList(
            input_text_list=input_texts,
            model_path=input_model_path,
            tokenizer_path=tokenizer_path,
            max_padding=512,
        )

        # Determine the average best temperature across all texts in the file
        best_temperature = get_average_best_temperature(
            input_text_list=input_texts,
            bert_model_path=input_model_path,
            target_model_name=input_model,
            tokenizer_path=tokenizer_path,
            df=df,
        )

    # Extract class labels and their corresponding probabilities from the result
    prob_list = list(prob_dict.items())  # Assuming prob_dict contains the data

    class_labels = [item[0] for item in prob_list]  # Class labels
    probabilities = [item[1] for item in prob_list]  # Class probabilities

    # Create a bar chart to visualize the classification probabilities
    fig, ax = plt.subplots(figsize=(8, 5.5))  # Set the figure size
    ax.bar(class_labels, probabilities, color='skyblue')  # Create bars for each class

    # Add labels and title for better understanding
    ax.set_xlabel('Class Labels')  # X-axis label
    ax.set_ylabel('Probability')   # Y-axis label
    ax.set_title('Class Probabilities')  # Chart title

    # Rotate class labels if needed for better readability
    plt.xticks(rotation=45, ha='right')

    # Adjust layout to avoid overlapping elements
    plt.tight_layout()

    # Return the figure and temperature (adjust based on your logic)
    return fig, best_temperature, best_temperature

# Initialize Gradio UI manager and create the interface
BertChoicerUIManager = GradioUIManager(choices=model_choices, on_button_click=on_button_click, on_experiment_button_click=on_experiment_button_click)
BertChoicerUIManager.create_interface()  # Create the interface with the provided functions

# Launch the Gradio demo
BertChoicerUIManager.demo.launch(share=False)
