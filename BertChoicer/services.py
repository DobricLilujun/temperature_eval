
import requests
import json
from BertChoicer import bertChoicer
import pandas as pd

bertClassifier = bertChoicer.bertClassifier
bertClassifierList = bertChoicer.bertClassifierList

def find_performance_score(df, model_name, category, temperature):
    """
    Finds the performance score of a model based on specific criteria.

    Parameters:
        df (DataFrame): The input DataFrame containing the model data.
        model_name (str): The name of the model to filter.
        category (str): The category parameter to match.
        temperature (float): The temperature value to filter.

    Returns:
        float or None: The performance score of the first matching row, 
                       or None if no match is found.

    Raises:
        ValueError: If 'model_name', 'category', or 'temperature' values are missing or invalid.
        KeyError: If 'category' or 'temperature' columns are missing in the DataFrame.
    """
    # Step 1: Check if 'category' and 'temperature' columns exist in the DataFrame
    required_columns = ["category", "temperature"]
    for col in required_columns:
        if col not in df.columns:
            raise KeyError(f"Missing required column: {col}")

    # Step 2: Check if input values are valid (non-null and non-empty)
    if model_name is None or model_name == "":
        raise ValueError("The 'model_name' value cannot be None or empty.")
    if category is None or category == "":
        raise ValueError("The 'category' value cannot be None or empty.")
    if temperature is None:
        raise ValueError("The 'temperature' value cannot be None.")

    # Step 3: Check if 'model_name' exists in the 'model_name' column
    if model_name not in df["model_name"].values:
        print(f"Warning: '{model_name}' not found in 'model_name' column. Proceeding with empty result.")
        # Create an empty result if model_name does not exist
        if category in df["category"].values and temperature not in df["temperature"].values:
            filtered_df = df[
                (df["category"] == category)     
                & (df["temperature"] == temperature) 
            ]
            mean_value = filtered_df["accuracy"].mean()
            result = mean_value
        else:
            result = None

    elif category not in df["category"].values:
        print(f"Warning: '{category}' not found in 'category' column. Proceeding with empty result.")
        # Create an empty result if category does not exist
        result = None

    elif temperature not in df["temperature"].values:
        print(f"Warning: '{temperature}' not found in 'temperature' column. Proceeding with empty result.")
        # Create an empty result if temperature does not exist
        result = None
    else:
        # Step 4: Filter the DataFrame based on the input conditions
        filtered_df = df[
            (df["model_name"] == model_name) 
            & (df["category"] == category)     
            & (df["temperature"] == temperature) 
        ]
        result = filtered_df["accuracy"].iloc[0]

    return result



def get_best_temperature(
    input_text,  # The input prompt to be classified
    target_model_name,  # The name of the model being evaluated
    bert_model_path,  # The file path to the BERT model
    tokenizer_path,  # The file path to the tokenizer
    df,  # The performance DataFrame containing model scores
):
    """
    Determines the best temperature setting for a given input text based on model performance scores.

    Parameters:
        input_text (str): The text prompt to be classified.
        target_model_name (str): The name of the model to evaluate.
        bert_model_path (str): Path to the pre-trained BERT model.
        tokenizer_path (str): Path to the tokenizer used by the BERT model.
        df (DataFrame): Performance data containing scores for various abilities and temperatures.

    Returns:
        float or None: The temperature value with the highest score, or None if no valid temperature is found.
    """

    # Step 1: Use the BERT classifier to classify the input text and obtain probabilities for each ability
    _ , prob_dict = bertClassifier(
        input_text=input_text,           # The input text to classify
        model_path=bert_model_path,     # Path to the pre-trained model
        tokenizer_path=tokenizer_path,   # Path to the tokenizer
        max_padding=512,                 # Maximum padding length for the tokenizer
    )

    # Initialize variables to store the best temperature and its corresponding score
    best_temperature = None  # The optimal temperature value
    best_score = 0           # The highest score observed so far

    for temp in df["temperature"].unique().tolist():
        max_ability = max(prob_dict, key=prob_dict.get)
        score = find_performance_score(
                df, target_model_name, max_ability, temp  # Fetch performance score for each ability and temperature
            )
        
        if score > best_score:
            best_score = score      # Update the best score
            best_temperature = temp # Update the best temperature

    # Step 5: Return the best temperature with the highest score
    return best_temperature


# This function receives a list of input prompts and returns the best temperature by estimating the performance for each one.
def get_average_best_temperature(
    input_text_list,  # List of input text prompts to evaluate
    bert_model_path,  # Path to the pre-trained BERT model
    target_model_name,  # Target model name for classification
    tokenizer_path,  # Path to the tokenizer for BERT
    df,  # DataFrame containing performance data by temperature
):
    
    performance_dict = {}  # Dictionary to store the accumulated performance scores for each temperature

    # Loop over each unique temperature in the DataFrame
    for temp in df["temperature"].unique().tolist():
        performance_dict[temp] = 0  # Initialize performance score for the current temperature

        # For each input prompt in the input text list
        for input_text in input_text_list:
            # Evaluate the input text using a BERT classifier, getting the probability distribution for each ability
            _ , prob_dict = bertClassifier(
                input_text=input_text,  # The input text to be classified
                model_path=bert_model_path,  # Path to the BERT model
                tokenizer_path=tokenizer_path,  # Path to the tokenizer
                max_padding=512,  # Maximum padding for sequences
            )

            # Get the ability with the maximum probability (i.e., the best ability)
            max_ability = max(prob_dict, key=prob_dict.get)

            # Fetch the performance score for this ability and temperature combination
            score = find_performance_score(
                df, target_model_name, max_ability, temp  # Fetch performance score from DataFrame based on ability and temperature
            )

            # Add the score to the total for this temperature
            performance_dict[temp] += score

    # Find the temperature with the highest accumulated performance score
    max_temp = max(performance_dict, key=performance_dict.get)
    
    return max_temp  # Return the temperature with the highest score


