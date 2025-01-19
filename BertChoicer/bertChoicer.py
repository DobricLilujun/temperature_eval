import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Auto-detect GPU or fallback to CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Function for single text classification
def bertClassifier(input_text, model_path, tokenizer_path, max_padding=512):
    """
    Classify a single input text using a BERT model.

    Args:
        input_text (str): The input text to classify.
        model_path (str): Path to the saved model.
        tokenizer_path (str): Path to the tokenizer.
        max_padding (int): Maximum padding length (default is 512).

    Returns:
        tuple: Classification result string and dictionary containing category probabilities.
    """
    # Load tokenizer and model from disk
    tokenizer = AutoTokenizer.from_pretrained(model_path, do_lower_case=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()  # Set the model to evaluation mode

    # Tokenization and padding for the input text
    encoded_dict = tokenizer.encode_plus(
        input_text,
        add_special_tokens=True,  # Add [CLS] and [SEP]
        max_length=max_padding,  # Pad/truncate to the maximum length
        padding="max_length",  # Modern usage for padding
        truncation=True,  # Enable truncation
        return_attention_mask=True,  # Generate attention masks
        return_tensors="pt",  # Return PyTorch tensors
    )

    input_ids = encoded_dict["input_ids"].to(device)
    attention_mask = encoded_dict["attention_mask"].to(device)

    # Perform forward pass without gradient computation
    with torch.no_grad():
        output = model(input_ids, token_type_ids=None, attention_mask=attention_mask)

    logits = output.logits
    logits = logits.detach().cpu().numpy()  # Convert logits to a NumPy array

    # Calculate stabilized softmax probabilities
    probabilities = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probabilities /= np.sum(probabilities, axis=1, keepdims=True)

    # Map category codes to descriptive labels
    ability_mapping = {0: "CR", 1: "CT", 2: "ICL", 3: "IF", 4: "MT", 5: "SUM"}

    # Build the output string and probability dictionary
    prob_dict = {
        ability_mapping[code]: round(prob, 4)
        for code, prob in enumerate(probabilities[0])
    }
    output = " ".join([f"{cat}: {prob:.2f}" for cat, prob in prob_dict.items()])

    return output, prob_dict


# Function for batch text classification
def bertClassifierList(input_text_list, model_path, tokenizer_path, max_padding=512):
    """
    Classify a list of input texts using a BERT model.

    Args:
        input_text_list (list of str): List of input texts to classify.
        model_path (str): Path to the saved model.
        tokenizer_path (str): Path to the tokenizer.
        max_padding (int): Maximum padding length (default is 512).

    Returns:
        tuple: List of classification results for each text and aggregated category probabilities.
    """
    # Load tokenizer and model from disk
    tokenizer = AutoTokenizer.from_pretrained(model_path, do_lower_case=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()  # Set the model to evaluation mode

    # Containers for results
    all_prob_dicts = []
    all_outputs = []

    # Map category codes to descriptive labels
    ability_mapping = {0: "CR", 1: "CT", 2: "ICL", 3: "IF", 4: "MT", 5: "SUM"}

    # Iterate over each input text
    for input_text in input_text_list:
        # Tokenization and padding for the input text
        encoded_dict = tokenizer.encode_plus(
            input_text,
            add_special_tokens=True,
            max_length=max_padding,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        input_ids = encoded_dict["input_ids"].to(device)
        attention_mask = encoded_dict["attention_mask"].to(device)

        # Perform forward pass without gradient computation
        with torch.no_grad():
            output = model(input_ids, token_type_ids=None, attention_mask=attention_mask)

        logits = output.logits
        logits = logits.detach().cpu().numpy()

        # Calculate stabilized softmax probabilities
        probabilities = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probabilities /= np.sum(probabilities, axis=1, keepdims=True)

        # Build the classification probability dictionary
        prob_dict = {
            ability_mapping[code]: round(prob, 4)
            for code, prob in enumerate(probabilities[0])
        }
        all_prob_dicts.append(prob_dict)

        # Build the output string
        output_text = " ".join([f"{cat}: {prob:.2f}" for cat, prob in prob_dict.items()])
        all_outputs.append(output_text)

    # Compute the aggregated probabilities for all texts
    agg_prob_dict = {key: 0 for key in ability_mapping.values()}
    for prob_dict in all_prob_dicts:
        for key in agg_prob_dict:
            agg_prob_dict[key] += prob_dict[key]
    agg_prob_dict = {key: round(value / len(input_text_list), 4) for key, value in agg_prob_dict.items()}

    return all_outputs, agg_prob_dict


